from __future__ import annotations

import bisect
import csv
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Hashable

from modules.util.enum.ConceptDPOObjective import ConceptDPOObjective
from modules.util.enum.DPOObjective import DPOObjective

from mgds.PipelineModule import PipelineModule
from mgds.pipelineModuleTypes.SingleVariationRandomAccessPipelineModule import (
    SingleVariationRandomAccessPipelineModule,
)


class AdaptiveDPODataset(
    PipelineModule,
    SingleVariationRandomAccessPipelineModule,
):
    """Live candidate-level adaptive resampling for mixed DPO datasets.

    The ordinary dataset/index order remains the proposal distribution.  At the
    moment a sample is actually requested for training, and *before* its
    expensive split-cache/image/latent payload is fetched, the module looks up
    the candidate's current difficulty EMA. Easy candidates can be rejected and
    replaced by a strictly harder DPO pair, with harder replacements receiving
    higher draw weight.

    Only cheap aggregate metadata (pair flag, path-qualified pair key,
    effective objective, and crop resolution) is indexed at epoch start.
    Replacement decisions themselves are NOT precomputed per epoch. New
    committed difficulty observations therefore affect subsequent candidates
    in the same epoch.

    """

    # v2 tags every difficulty EMA with its objective. This prevents a pair's
    # old Sigmoid/IPO/Anchored scale from being reused as Linear-DPO difficulty
    # (or vice versa) after an objective change.
    STATE_VERSION = 2

    def __init__(
            self,
            names: list[str],
            *,
            ema_decay: float = 0.8,
            min_observations: int = 3,
            min_keep_probability: float = 0.1,
            replacement_power: float = 2.0,
            default_objective: DPOObjective | str = DPOObjective.SIGMOID,
    ):
        super().__init__()
        self.names = list(dict.fromkeys(str(name).split(".", 1)[0] for name in names))
        for required_name in (
            "dpo_is_paired",
            "dpo_pair_key",
            "crop_resolution",
            "image_path",
            "image_path_rejected",
            "concept",
        ):
            if required_name not in self.names:
                self.names.append(required_name)

        self.ema_decay = min(max(float(ema_decay), 0.0), 0.999999)
        # Require three successful observations before a pair can surrender its
        # own slot. Old/hand-edited configs may request less, but the runtime
        # warm-up floor is deliberately hard so one noisy early loss cannot
        # immediately reshape the dataset.
        self.min_observations = max(int(min_observations), 3)
        # Every pair keeps at least 25% of its original sampling opportunities,
        # regardless of how easy its EMA becomes or what an old config says.
        # Higher user-specified maintenance probabilities remain supported.
        self.min_keep_probability = min(max(float(min_keep_probability), 0.25), 1.0)
        self.replacement_power = max(float(replacement_power), 0.01)
        self.default_objective = DPOObjective(default_objective)

        self._stats: dict[str, dict[str, float | int | str]] = {}
        self._pair_key_by_index: list[str] = []
        self._paired_by_index: list[bool] = []
        self._resolution_by_index: list[Hashable] = []
        self._objective_by_index: list[str] = []
        self._dpo_indices: list[int] = []

        # Sampling structures are rebuilt lazily only after loss statistics
        # change.  A candidate lookup is then just dictionary/bisect work.
        self._sampling_generation = 0
        self._sampling_cache_generation = -1
        self._eligible_loss_by_index: dict[int, float] = {}

        # Objective-local replacement pools keep fundamentally different loss
        # scales and Linear-DPO's ranking difficulty from competing directly.
        self._eligible_by_objective: dict[str, list[tuple[float, int]]] = {}
        self._eligible_losses_by_objective: dict[str, list[float]] = {}
        self._loss_scale_by_objective: dict[str, float] = {}
        self._eligible_by_resolution: dict[Hashable, list[tuple[float, int]]] = {}
        self._eligible_losses_by_resolution: dict[Hashable, list[float]] = {}
        self._loss_scale = 0.0

        # One live decision is shared by all fields requested for the same raw
        # index.  MGDS uses num_workers=0 here, so the next raw index naturally
        # starts a fresh candidate decision.
        self._active_source_index = -1
        self._active_mapped_index = -1
        self._rand = None

        self._decision_count = 0
        self._replacement_count = 0
        self._replacement_keys: set[str] = set()
        self._lock = threading.RLock()
        self._last_epoch_summary: dict[str, Any] = {}

        # Live audit log for Adaptive DPO Dataset decisions.
        # Pair paths are written explicitly so duplicate filenames in different
        # directories remain unambiguous.
        self._current_epoch = 0
        self._csv_path = (
            Path(__file__).resolve().parents[3]
            / "adaptive_dpo_decisions.csv"
        )
        self._csv_pending: list[list[Any]] = []

        # 1 = write every decision immediately. DPO forwards are vastly more
        # expensive than this small append, so overhead should be negligible.
        self._csv_flush_every = 1

    def length(self) -> int:
        return self._get_previous_length("dpo_is_paired")

    def get_inputs(self) -> list[str]:
        return self.names

    def get_outputs(self) -> list[str]:
        return self.names

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "dpo", "paired"}
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass
        return bool(value)

    def _effective_objective(self, concept: Any) -> str:
        if hasattr(concept, "to_dict"):
            concept = concept.to_dict()
        raw = (
            concept.get("dpo_objective", ConceptDPOObjective.DEFAULT)
            if isinstance(concept, dict)
            else ConceptDPOObjective.DEFAULT
        )
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, DPOObjective):
            if raw == DPOObjective.SIGMOID:
                return str(DPOObjective.SIGMOID)
            raise ValueError(
                "Per-concept DPO objective currently supports only DEFAULT "
                f"or SIGMOID, got {raw}."
            )
        try:
            override = (
                raw
                if isinstance(raw, ConceptDPOObjective)
                else ConceptDPOObjective(str(raw or "DEFAULT").strip().upper())
            )
        except ValueError as exc:
            raise ValueError(
                "Per-concept DPO objective currently supports only DEFAULT "
                f"or SIGMOID, got {raw!r}."
            ) from exc
        return str(
            self.default_objective
            if override == ConceptDPOObjective.DEFAULT
            else DPOObjective.SIGMOID
        )

    @staticmethod
    def _normalize_pair_path(path: str) -> str:
        return os.path.normcase(
            os.path.realpath(
                os.path.abspath(os.path.expanduser(str(path)))
            )
        )

    @classmethod
    def _build_pair_key(cls, chosen_path: str, rejected_path: str) -> str:
        return (
            "dpo-pair-path-v1\n"
            f"chosen={cls._normalize_pair_path(chosen_path)}\n"
            f"rejected={cls._normalize_pair_path(rejected_path)}"
        )

    @classmethod
    def _valid_pair_key(cls, value: Any) -> str:
        """Canonicalize direct pair keys and cache composite selection keys."""
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")

        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]

        key = str(value or "").strip()
        if not key:
            return ""

        # Tolerate stringified metadata containing escaped newlines.
        if (
            r"\nchosen=" in key
            and r"\nrejected=" in key
            and "\nchosen=" not in key
        ):
            key = key.replace(
                r"\nchosen=", "\nchosen="
            ).replace(
                r"\nrejected=", "\nrejected="
            )

        # Normal source metadata begins with this.
        # Cache metadata may EMBED it inside a larger selection key.
        signature = "dpo-pair-path-v1\nchosen="
        key_start = key.find(signature)

        if key_start < 0:
            return ""

        embedded = key[key_start:]

        rejected_marker = "\nrejected="
        rejected_pos = embedded.find(rejected_marker)

        if rejected_pos < 0:
            return ""

        chosen = embedded[
            len(signature):rejected_pos
        ].strip()

        rejected = embedded[
            rejected_pos + len(rejected_marker):
        ].strip()

        # MultiResolutionDiskCache composite key can look like:
        #
        # ...dpo_pair_key=dpo-pair-path-v1
        # chosen=/...
        # rejected=/...|image_path=/...
        #
        # Strip everything after the real rejected path.
        if "|image_path=" in rejected:
            rejected = rejected.split(
                "|image_path=", 1
            )[0].strip()

        if not chosen or not rejected:
            return ""

        # Rebuild it exactly like the state-file key.
        return cls._build_pair_key(
            chosen,
            rejected,
        )

    @classmethod
    def _resolution_key(cls, value: Any) -> Hashable:
        """Turn tuple/list/tensor-ish crop resolutions into a stable key."""
        try:
            if hasattr(value, "tolist"):
                value = value.tolist()
        except Exception:
            pass

        if isinstance(value, dict):
            return tuple(
                sorted((str(key), cls._resolution_key(item)) for key, item in value.items())
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._resolution_key(item) for item in value)
        try:
            hash(value)
            return value
        except Exception:
            return repr(value)

    def observe(self, observations: list[tuple[str, float, str]]) -> None:
        """Commit detached pair-difficulty observations after a successful update.

        Unlike the first implementation, these updated EMAs become eligible for
        the very next candidate decision in the *same* epoch.
        """
        changed = False
        with self._lock:
            for pair_key, raw_loss, raw_objective in observations:
                pair_key = self._valid_pair_key(pair_key)
                if not pair_key:
                    continue
                try:
                    objective = str(DPOObjective(raw_objective))
                except ValueError:
                    continue
                try:
                    loss = float(raw_loss)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(loss) or loss < 0.0:
                    continue

                previous = self._stats.get(pair_key)
                if (
                    previous is None
                    or str(previous.get("objective", "")) != objective
                ):
                    self._stats[pair_key] = {
                        "loss_ema": loss,
                        "observations": 1,
                        "objective": objective,
                    }
                    changed = True
                    continue

                old_ema = float(previous.get("loss_ema", loss))
                count = int(previous.get("observations", 0)) + 1
                self._stats[pair_key] = {
                    "loss_ema": self.ema_decay * old_ema + (1.0 - self.ema_decay) * loss,
                    "observations": count,
                    "objective": objective,
                }
                changed = True

            if changed:
                self._sampling_generation += 1

    def _eligible_loss(
            self,
            pair_key: str,
            objective: str,
    ) -> float | None:
        stat = self._stats.get(pair_key)
        if stat is None or int(stat.get("observations", 0)) < self.min_observations:
            return None
        if str(stat.get("objective", "")) != str(objective):
            return None
        loss = float(stat.get("loss_ema", 0.0))
        if not math.isfinite(loss) or loss < 0.0:
            return None
        return loss

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) & 1:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    @staticmethod
    def _difficulty(loss: float, scale: float) -> float:
        if loss <= 0.0:
            return 0.0
        if scale <= 0.0:
            return 1.0
        return min(max(loss / (loss + scale), 0.0), 1.0)

    def _keep_probability(self, loss: float, scale: float) -> float:
        # Easy samples are aggressively replaceable, while a merely useful
        # sample quickly regains a moderate/high chance to stay in its own slot.
        difficulty = self._difficulty(loss, scale)
        # Aggressive adaptive sampling:
        # easy/median pairs should give up their slot frequently,
        # while genuinely hard pairs retain a high keep probability.
        # Strongly suppress already-easy pairs while retaining a
        # non-zero maintenance probability.
        return self.min_keep_probability + (
            1.0 - self.min_keep_probability
        ) * (difficulty ** 4.0)

    def _replacement_weight(self, loss: float, scale: float) -> float:
        difficulty = self._difficulty(loss, scale)
        return 0.01 + difficulty ** self.replacement_power

    @staticmethod
    def _weighted_choice(rand, entries: list[tuple[float, int]], weights: list[float]) -> int | None:
        if not entries:
            return None
        total = sum(max(float(weight), 0.0) for weight in weights)
        if total <= 0.0:
            return entries[rand.randrange(len(entries))][1]
        needle = rand.random() * total
        cumulative = 0.0
        for entry, weight in zip(entries, weights, strict=True):
            cumulative += max(float(weight), 0.0)
            if needle <= cumulative:
                return entry[1]
        return entries[-1][1]

    def _refresh_sampling_cache_locked(self) -> None:
        if self._sampling_cache_generation == self._sampling_generation:
            return

        eligible_loss_by_index: dict[int, float] = {}
        eligible_by_resolution: dict[Hashable, list[tuple[float, int]]] = {}
        eligible_by_objective: dict[str, list[tuple[float, int]]] = {}
        all_positive_losses: list[float] = []
        positive_losses_by_objective: dict[str, list[float]] = {}

        for index in self._dpo_indices:
            pair_key = self._pair_key_by_index[index]
            objective = self._objective_by_index[index]
            loss = self._eligible_loss(pair_key, objective)
            if loss is None:
                continue
            eligible_loss_by_index[index] = loss
            resolution = self._resolution_by_index[index]
            eligible_by_resolution.setdefault(resolution, []).append((loss, index))
            eligible_by_objective.setdefault(objective, []).append((loss, index))
            if loss > 1e-12:
                all_positive_losses.append(loss)
                positive_losses_by_objective.setdefault(objective, []).append(loss)

        eligible_losses_by_resolution: dict[Hashable, list[float]] = {}
        for resolution, entries in eligible_by_resolution.items():
            entries.sort(key=lambda item: item[0])
            eligible_losses_by_resolution[resolution] = [item[0] for item in entries]

        eligible_losses_by_objective: dict[str, list[float]] = {}
        for objective, entries in eligible_by_objective.items():
            entries.sort(key=lambda item: item[0])
            eligible_losses_by_objective[objective] = [
                loss for loss, _ in entries
            ]

        self._eligible_loss_by_index = eligible_loss_by_index
        self._eligible_by_resolution = eligible_by_resolution
        self._eligible_losses_by_resolution = eligible_losses_by_resolution

        self._eligible_by_objective = eligible_by_objective
        self._eligible_losses_by_objective = eligible_losses_by_objective
        self._loss_scale_by_objective = {
            objective: self._median(losses)
            for objective, losses in positive_losses_by_objective.items()
        }

        self._loss_scale = self._median(all_positive_losses)
        self._sampling_cache_generation = self._sampling_generation

    @staticmethod
    def _pair_paths(pair_key: str) -> tuple[str, str]:
        chosen = ""
        rejected = ""

        for line in str(pair_key or "").splitlines():
            if line.startswith("chosen="):
                chosen = line[len("chosen="):]
            elif line.startswith("rejected="):
                rejected = line[len("rejected="):]

        return chosen, rejected

    def _queue_csv_decision_locked(
            self,
            candidate_index: int,
            action: str,
            *,
            keep_probability: float | None = None,
            replacement_index: int | None = None,
    ) -> None:
        candidate_key = self._pair_key_by_index[candidate_index]
        candidate_stat = self._stats.get(candidate_key, {})
        candidate_chosen, candidate_rejected = self._pair_paths(candidate_key)

        replacement_stat: dict[str, Any] = {}
        replacement_chosen = ""
        replacement_rejected = ""

        if replacement_index is not None:
            replacement_key = self._pair_key_by_index[replacement_index]
            replacement_stat = self._stats.get(replacement_key, {})
            replacement_chosen, replacement_rejected = self._pair_paths(
                replacement_key
            )

        self._csv_pending.append([
            self._current_epoch,
            candidate_index,
            candidate_chosen,
            candidate_rejected,
            int(candidate_stat.get("observations", 0)),
            candidate_stat.get("loss_ema", ""),
            "" if keep_probability is None else keep_probability,
            1 if replacement_index is not None else 0,
            action,
            "" if replacement_index is None else replacement_index,
            replacement_chosen,
            replacement_rejected,
            (
                ""
                if replacement_index is None
                else int(replacement_stat.get("observations", 0))
            ),
            (
                ""
                if replacement_index is None
                else replacement_stat.get("loss_ema", "")
            ),
        ])

        if len(self._csv_pending) >= self._csv_flush_every:
            self._flush_csv_locked()

    def _flush_csv_locked(self) -> None:
        if not self._csv_pending:
            return

        self._csv_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = (
            not self._csv_path.exists()
            or self._csv_path.stat().st_size == 0
        )

        with self._csv_path.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)

            if write_header:
                writer.writerow([
                    "epoch",
                    "candidate_index",
                    "candidate_chosen",
                    "candidate_rejected",
                    "candidate_observations",
                    "candidate_loss_ema",
                    "keep_probability",
                    "switched",
                    "action",
                    "replacement_index",
                    "replacement_chosen",
                    "replacement_rejected",
                    "replacement_observations",
                    "replacement_loss_ema",
                ])

            writer.writerows(self._csv_pending)

        self._csv_pending.clear()

    def _select_index_locked(self, candidate_index: int) -> int:
        self._refresh_sampling_cache_locked()

        if (
            candidate_index < 0
            or candidate_index >= len(self._paired_by_index)
            or not self._paired_by_index[candidate_index]
        ):
            return candidate_index

        candidate_loss = self._eligible_loss_by_index.get(candidate_index)
        candidate_objective = self._objective_by_index[candidate_index]
        objective_scale = self._loss_scale_by_objective.get(
            candidate_objective,
            0.0,
        )
        if candidate_loss is None or objective_scale <= 1e-12:
            # Unknown/under-observed samples are always trained normally.
            self._queue_csv_decision_locked(
                candidate_index,
                "warmup_keep",
            )
            return candidate_index

        # Compare and replace only within the same effective objective.
        pool = self._eligible_by_objective.get(candidate_objective, [])
        sorted_losses = self._eligible_losses_by_objective.get(
            candidate_objective,
            [],
        )
        if len(pool) < 2:
            self._queue_csv_decision_locked(
                candidate_index,
                "no_alternative_keep",
            )
            return candidate_index

        keep_probability = self._keep_probability(candidate_loss, objective_scale)
        self._decision_count += 1
        if self._rand.random() <= keep_probability:
            self._queue_csv_decision_locked(
                candidate_index,
                "keep",
                keep_probability=keep_probability,
            )
            return candidate_index

        # A rejected candidate may only be replaced by a meaningfully
        # harder pair, rather than one whose EMA is trivially higher.
        # Only spend a replacement slot on a substantially harder pair.
        # This prevents meaningless swaps between almost identical losses.
        min_replacement_loss = max(
            candidate_loss * 1.25,
            candidate_loss + 0.25 * objective_scale,
        )

        first_harder = bisect.bisect_left(
            sorted_losses,
            min_replacement_loss,
        )
        harder_entries = pool[first_harder:]
        if not harder_entries:
            self._queue_csv_decision_locked(
                candidate_index,
                "no_harder_keep",
                keep_probability=keep_probability,
            )
            return candidate_index

        weights = [
            self._replacement_weight(loss, objective_scale)
            for loss, _ in harder_entries
        ]
        replacement_index = self._weighted_choice(self._rand, harder_entries, weights)
        if replacement_index is None:
            self._queue_csv_decision_locked(
                candidate_index,
                "replacement_draw_failed_keep",
                keep_probability=keep_probability,
            )
            return candidate_index

        self._queue_csv_decision_locked(
            candidate_index,
            "switch",
            keep_probability=keep_probability,
            replacement_index=replacement_index,
        )

        self._replacement_count += 1
        replacement_key = self._pair_key_by_index[replacement_index]
        if replacement_key:
            self._replacement_keys.add(replacement_key)
        return replacement_index

    def start(self, variation: int):
        # Flush any previous epoch audit rows first.
        with self._lock:
            self._flush_csv_locked()

        # Report the just-finished epoch before resetting runtime counters.
        if self._decision_count or self._replacement_count:
            print(
                "[OT-ADAPTIVE-DPO] "
                f"epoch={variation - 1} live_decisions={self._decision_count} "
                f"replaced={self._replacement_count} "
                f"unique_replacements={len(self._replacement_keys)}"
            )

        size = self._get_previous_length("dpo_is_paired")
        pair_key_by_index = [""] * size
        paired_by_index = [False] * size
        resolution_by_index: list[Hashable] = [None] * size
        objective_by_index = [str(self.default_objective)] * size
        dpo_indices: list[int] = []

        # Metadata-only indexing.  These fields are aggregate cache metadata in
        # the cached pipeline and are available before image/latent loading in
        # the uncached path.
        for index in range(size):
            resolution_by_index[index] = self._resolution_key(
                self._get_previous_item(variation, "crop_resolution", index)
            )
            paired = self._as_bool(
                self._get_previous_item(variation, "dpo_is_paired", index)
            )
            paired_by_index[index] = paired
            if not paired:
                continue

            concept = self._get_previous_item(
                variation,
                "concept",
                index,
            )
            objective_by_index[index] = self._effective_objective(concept)

            raw_pair_key = self._get_previous_item(
                variation,
                "dpo_pair_key",
                index,
            )

            pair_key = self._valid_pair_key(
                raw_pair_key
            )

            # Fallback for cache metadata layouts which don't expose
            # dpo_pair_key directly. These are path strings only; no
            # latent/image payload is loaded here.
            if not pair_key:
                try:
                    chosen_path = self._get_previous_item(
                        variation,
                        "image_path",
                        index,
                    )

                    rejected_path = self._get_previous_item(
                        variation,
                        "image_path_rejected",
                        index,
                    )

                    chosen_path = str(
                        chosen_path or ""
                    ).strip()

                    rejected_path = str(
                        rejected_path or ""
                    ).strip()

                    if (
                        chosen_path
                        and rejected_path
                        and self._normalize_pair_path(chosen_path)
                        != self._normalize_pair_path(rejected_path)
                    ):
                        pair_key = self._build_pair_key(
                            chosen_path,
                            rejected_path,
                        )

                except Exception:
                    pass

            pair_key_by_index[index] = pair_key

            if pair_key:
                dpo_indices.append(index)

        with self._lock:
            self._pair_key_by_index = pair_key_by_index
            self._paired_by_index = paired_by_index
            self._resolution_by_index = resolution_by_index
            self._objective_by_index = objective_by_index
            self._dpo_indices = dpo_indices

            self._current_epoch = int(variation)
            self._active_source_index = -1
            self._active_mapped_index = -1
            self._rand = self._get_rand(variation)
            self._decision_count = 0
            self._replacement_count = 0
            self._replacement_keys = set()

            # The dataset layout may change between epochs even when loss stats
            # do not, so force one cheap pool rebuild against the new indices.
            self._sampling_cache_generation = -1
            self._refresh_sampling_cache_locked()
            eligible = len(self._eligible_loss_by_index)
            scale = self._loss_scale

            self._last_epoch_summary = {
                "epoch": int(variation),
                "dpo_pairs": len(dpo_indices),
                "eligible": eligible,
                "replaced": 0,
                "scale": scale,
                "mode": "live",
            }

        paired_rows = sum(
            1 for paired in paired_by_index
            if paired
        )

        if paired_rows != len(dpo_indices):
            print(
                "[OT-ADAPTIVE-DPO] WARNING: "
                f"paired_rows={paired_rows} "
                f"but valid_pair_keys={len(dpo_indices)}. "
                "Rows without a stable pair key cannot use "
                "adaptive history."
            )

        print(
            "[OT-ADAPTIVE-DPO] "
            f"epoch={variation} "
            f"dpo_pairs={len(dpo_indices)} "
            f"eligible={eligible} "
            f"loss_scale={scale:.6g} "
            f"mode=live-before-load"
        )

    def get_item(self, index: int, requested_name: str = None) -> dict:
        # AspectBatchSorting scans crop_resolution for every candidate at epoch
        # initialization.  That scan must remain metadata-only and must NOT
        # consume a keep/replace decision.  During real item loading the first
        # non-resolution field triggers the live decision.
        if requested_name == "crop_resolution" and self._active_source_index != index:
            mapped_index = index
        else:
            with self._lock:
                if self._active_source_index != index:
                    self._active_source_index = index
                    self._active_mapped_index = self._select_index_locked(index)
                mapped_index = self._active_mapped_index

        if requested_name is not None:
            return {
                requested_name: self._get_previous_item(
                    self.current_variation,
                    requested_name,
                    mapped_index,
                )
            }

        return {
            name: self._get_previous_item(
                self.current_variation,
                name,
                mapped_index,
            )
            for name in self.names
        }

    def save_state(self, path: str) -> None:
        with self._lock:
            pairs = {
                key: {
                    "loss_ema": float(value["loss_ema"]),
                    "observations": int(value["observations"]),
                    "objective": str(value["objective"]),
                }
                for key, value in sorted(self._stats.items())
                if (
                    self._valid_pair_key(key)
                    and str(value.get("objective", ""))
                    in {str(item) for item in DPOObjective}
                )
            }

        payload = {
            "version": self.STATE_VERSION,
            "settings": {
                "ema_decay": self.ema_decay,
                "min_observations": self.min_observations,
                "min_keep_probability": self.min_keep_probability,
                "replacement_power": self.replacement_power,
            },
            "pairs": pairs,
        }
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        temp = path_obj.with_name(path_obj.name + ".tmp")
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path_obj)

    def load_state(self, path: str) -> None:
        with self._lock:
            self._stats.clear()
            self._sampling_generation += 1

        if not os.path.isfile(path):
            print(
                "[OT-ADAPTIVE-DPO] resume backup has no adaptive dataset state; "
                "starting with a cold difficulty history (legacy-compatible)."
            )
            return

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(
                "[OT-ADAPTIVE-DPO] WARNING: could not read adaptive dataset "
                f"state ({type(exc).__name__}: {exc}); starting cold."
            )
            return

        version = int(payload.get("version", -1))
        if version == 1:
            print(
                "[OT-ADAPTIVE-DPO] loading legacy v1 state without objective "
                "tags; discarding only adaptive difficulty history so model "
                "resume remains safe. New observations will create v2 state."
            )
            return
        if version != self.STATE_VERSION:
            print(
                "[OT-ADAPTIVE-DPO] WARNING: unsupported adaptive dataset state "
                f"version {payload.get('version')!r}; starting cold."
            )
            return

        pairs = payload.get("pairs", {})
        restored: dict[str, dict[str, float | int | str]] = {}
        skipped = 0
        if isinstance(pairs, dict):
            for raw_key, raw_value in pairs.items():
                key = self._valid_pair_key(raw_key)
                if not key or not isinstance(raw_value, dict):
                    skipped += 1
                    continue
                try:
                    loss_ema = float(raw_value["loss_ema"])
                    observations = int(raw_value["observations"])
                    objective = str(DPOObjective(raw_value["objective"]))
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                if (
                    not math.isfinite(loss_ema)
                    or loss_ema < 0.0
                    or observations < 1
                ):
                    skipped += 1
                    continue
                restored[key] = {
                    "loss_ema": loss_ema,
                    "observations": observations,
                    "objective": objective,
                }

        with self._lock:
            self._stats = restored
            self._sampling_generation += 1
            self._sampling_cache_generation = -1

            live_keys = {
                key
                for key in self._pair_key_by_index
                if key
            }

        matched_live = len(
            live_keys.intersection(restored)
        )

        print(
            "[OT-ADAPTIVE-DPO] restored "
            f"{len(restored)} pair loss-EMA states"
            + (f"; skipped={skipped}" if skipped else "")
            + f"; live_pair_keys={len(live_keys)}"
            + f"; matched_live={matched_live}. "
            + "Current live sampler settings are used."
        )

        if restored and live_keys and matched_live == 0:
            print(
                "[OT-ADAPTIVE-DPO] WARNING: "
                "restored state has ZERO matches with "
                "live DPO pair keys."
            )

    def state_size(self) -> int:
        with self._lock:
            return len(self._stats)

    def last_epoch_summary(self) -> dict[str, Any]:
        with self._lock:
            summary = dict(self._last_epoch_summary)
            summary.update({
                "live_decisions": self._decision_count,
                "replaced": self._replacement_count,
                "unique_replacements": len(self._replacement_keys),
                "mode": "live",
            })
            return summary
