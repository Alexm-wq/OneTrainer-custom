import time
import contextlib
import copy
import csv
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.dataLoader.dpo.AdaptiveDPODataset import AdaptiveDPODataset
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.compile_util import init_compile
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptDPOObjective import ConceptDPOObjective
from modules.util.enum.ConceptDPOReferenceMode import ConceptDPOReferenceMode
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.DPORefMode import DPORefMode
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.PrefetchIterator import PrefetchIterator
from modules.util.profiling_util import TorchMemoryRecorder, TorchProfiler
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        # cuDNN SDPA can produce invalid MHA execution plans under
        # torch.compile, especially with variable diffusion token shapes.
        # Keep Flash/Efficient/Math SDPA available, but never select cuDNN SDPA.
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        # torch._dynamo.config overrides are thread-local, so init_compile() must be called in the training thread/process.
        init_compile()

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []
        self._dpo_reference_snapshot_path = None
        self._dpo_concept_reference_snapshot_path = None
        self._dpo_reference_initialized = False
        self._resume_restore_dpo_reference = False
        self._resume_restore_dpo_concept_references = False
        self._resume_integrity_payload = None
        self._resume_saved_gradient_accumulation_steps = None
        self._gradient_accumulation_dirty = False
        self._dpo_metric_sums: dict[str, float] = {}
        self._dpo_metric_weights: dict[str, float] = {}
        # DPO gradients are accumulated in host RAM when the optimizer exposes
        # a momentum-bypass step.  This avoids a second GPU momentum buffer.
        self._dpo_bypass_cpu_grads: dict[Parameter, Tensor] = {}
        self._dpo_bypass_update_weight = 0.0
        # Diagnostic-only DPO gradient accumulator used when DPO follows the
        # optimizer's ordinary momentum path. Unlike the bypass buffer, these
        # gradients are still returned unchanged to autograd and therefore still
        # accumulate into parameter.grad exactly as before.
        self._dpo_probe_cpu_grads: dict[Parameter, Tensor] = {}
        self._dpo_gradient_csv_warned = False
        self._adaptive_dpo_dataset_module: AdaptiveDPODataset | None = None
        # Pair-loss observations are staged for the current gradient-
        # accumulation window and committed only after its optimizer step.
        self._adaptive_dpo_pending: list[tuple[str, float, str]] = []

    def __find_adaptive_dpo_dataset_module(self) -> AdaptiveDPODataset | None:
        if not bool(getattr(self.config, "rlhf_dpo_adaptive_dataset", False)):
            return None
        try:
            modules = self.data_loader.get_data_set().loading_pipeline.modules
        except Exception:
            return None
        for module in modules:
            if isinstance(module, AdaptiveDPODataset):
                return module
        return None

    def __stage_adaptive_dpo_observations(self):
        module = self._adaptive_dpo_dataset_module
        if module is None:
            return
        observations = self.model_setup.get_last_dpo_pair_losses()
        if observations:
            self._adaptive_dpo_pending.extend(observations)

    def __commit_adaptive_dpo_observations(self):
        module = self._adaptive_dpo_dataset_module
        if module is not None and self._adaptive_dpo_pending:
            module.observe(self._adaptive_dpo_pending)
        self._adaptive_dpo_pending.clear()

    def __discard_adaptive_dpo_observations(self):
        self._adaptive_dpo_pending.clear()

    def __load_adaptive_dpo_dataset_state(self, backup_path: str):
        module = self._adaptive_dpo_dataset_module
        if module is None:
            return
        module.load_state(os.path.join(
            backup_path,
            "onetrainer_dpo_adaptive_dataset.json",
        ))

    def __save_adaptive_dpo_dataset_state(self, backup_path: str):
        module = self._adaptive_dpo_dataset_module
        if module is None:
            return
        if self._adaptive_dpo_pending:
            raise RuntimeError(
                "Refusing to save Adaptive DPO Dataset state with uncommitted "
                "gradient-accumulation observations."
            )
        module.save_state(os.path.join(
            backup_path,
            "onetrainer_dpo_adaptive_dataset.json",
        ))

    def start(self):
        self.config.validate_dpo_settings()
        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.use_cache_only and self.config.clear_cache_before_training:
                print(
                    "[OT-CACHE-ONLY] Ignoring 'Clear cache before training'; "
                    "cache-only mode never deletes or rebuilds cache files."
                )
            elif self.config.clear_cache_before_training and (self.config.image_caching or self.config.text_caching):
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()
        last_backup_path = None

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
                self.__validate_resume_backup_files(last_backup_path)
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("configuring Hugging Face token")
            os.environ["HF_TOKEN"] = self.config.secrets.huggingface_token

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config

        if last_backup_path:
            self.__validate_loaded_resume_progress(last_backup_path)

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        if last_backup_path:
            if self._resume_restore_dpo_reference:
                self._dpo_reference_snapshot_path = os.path.join(
                    last_backup_path,
                    "onetrainer_dpo_reference.pt",
                )
            if self._resume_restore_dpo_concept_references:
                self._dpo_concept_reference_snapshot_path = os.path.join(
                    last_backup_path,
                    "onetrainer_dpo_concept_references.pt",
                )
            self.model_setup.load_dpo_curriculum_state(
                os.path.join(
                    last_backup_path,
                    "onetrainer_dpo_hard_pair_curriculum.json",
                ),
                self.config,
            )

        if self.config.rlhf_enabled and self.config.training_method != TrainingMethod.LORA:
            raise NotImplementedError("RLHF DPO is currently implemented for adapter training in the LoRA tab only.")

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self._adaptive_dpo_dataset_module = (
            self.__find_adaptive_dpo_dataset_module()
        )
        if (
            bool(getattr(self.config, "rlhf_dpo_adaptive_dataset", False))
            and self._adaptive_dpo_dataset_module is None
        ):
            raise RuntimeError(
                "Adaptive DPO Dataset is enabled, but its dataloader module "
                "was not created. Restart OneTrainer after installing the "
                "complete patch."
            )
        if last_backup_path:
            # Legacy backups predate this file. Missing state intentionally
            # starts cold rather than blocking resume.
            self.__load_adaptive_dpo_dataset_state(last_backup_path)

        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

        if self.config.validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

    def __concepts_use_reference_mode(
            self,
            target_mode: ConceptDPOReferenceMode,
    ) -> bool:
        concepts = self.config.concepts
        if concepts is None:
            try:
                with open(
                        self.config.concept_file_name,
                        "r",
                        encoding="utf-8",
                ) as handle:
                    concepts = json.load(handle)
            except (OSError, json.JSONDecodeError):
                concepts = []

        for concept in concepts or []:
            if isinstance(concept, dict):
                enabled = concept.get("enabled", True)
                raw = concept.get(
                    "dpo_reference_mode",
                    ConceptDPOReferenceMode.DEFAULT,
                )
            else:
                enabled = getattr(concept, "enabled", True)
                raw = getattr(
                    concept,
                    "dpo_reference_mode",
                    ConceptDPOReferenceMode.DEFAULT,
                )

            if not enabled:
                continue
            if isinstance(raw, ConceptDPOReferenceMode):
                mode = raw
            else:
                try:
                    mode = ConceptDPOReferenceMode(
                        str(raw or "DEFAULT").strip().upper()
                    )
                except ValueError:
                    continue
            if mode == target_mode:
                return True

        return False

    def __requires_gpu_existing_adapter_dpo_reference(self) -> bool:
        return (
            self.__global_requires_gpu_existing_adapter_dpo_reference()
            or self.__concepts_use_reference_mode(
                ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT
            )
        )

    def __requires_cpu_existing_adapter_dpo_reference(self) -> bool:
        return (
            self.__global_requires_cpu_existing_adapter_dpo_reference()
            or self.__concepts_use_reference_mode(
                ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT_CPU
            )
        )

    def __global_requires_gpu_existing_adapter_dpo_reference(self) -> bool:
        return (
            DPORefMode(self.config.effective_dpo_ref_mode())
            == DPORefMode.EXISTING_ADAPTER
        )

    def __global_requires_cpu_existing_adapter_dpo_reference(self) -> bool:
        return (
            DPORefMode(self.config.effective_dpo_ref_mode())
            == DPORefMode.EXISTING_ADAPTER_CPU
        )

    def __global_requires_ema_adapter_dpo_reference(self) -> bool:
        return (
            DPORefMode(self.config.effective_dpo_ref_mode())
            == DPORefMode.EMA_ADAPTER
        )

    def __concept_reference_keys(
            self,
            target_mode: ConceptDPOReferenceMode,
    ) -> list[str]:
        concepts = self.config.concepts
        if concepts is None:
            try:
                with open(
                        self.config.concept_file_name,
                        "r",
                        encoding="utf-8",
                ) as handle:
                    concepts = json.load(handle)
            except (OSError, json.JSONDecodeError):
                concepts = []

        keys: list[str] = []
        for concept in concepts or []:
            if isinstance(concept, dict):
                enabled = concept.get("enabled", True)
                seed = concept.get("seed")
                raw = concept.get(
                    "dpo_reference_mode",
                    ConceptDPOReferenceMode.DEFAULT,
                )
            else:
                enabled = getattr(concept, "enabled", True)
                seed = getattr(concept, "seed", None)
                raw = getattr(
                    concept,
                    "dpo_reference_mode",
                    ConceptDPOReferenceMode.DEFAULT,
                )
            if not enabled:
                continue
            try:
                mode = (
                    raw
                    if isinstance(raw, ConceptDPOReferenceMode)
                    else ConceptDPOReferenceMode(
                        str(raw or "DEFAULT").strip().upper()
                    )
                )
            except ValueError:
                continue
            if mode != target_mode:
                continue
            if seed is None:
                raise RuntimeError(
                    "A concept using Current Adapter Snapshot has no stable "
                    "seed. Re-save the concept before training."
                )
            keys.append(str(seed))
        return list(dict.fromkeys(keys))

    def __requires_existing_adapter_dpo_reference(self) -> bool:
        return (
            self.__requires_gpu_existing_adapter_dpo_reference()
            or self.__requires_cpu_existing_adapter_dpo_reference()
            or self.__global_requires_ema_adapter_dpo_reference()
        )

    def __validate_resume_backup_files(self, backup_path: str):
        optimizer_path = os.path.join(
            backup_path,
            "optimizer",
            "optimizer.pt",
        )
        if not os.path.isfile(optimizer_path):
            raise RuntimeError(
                "[OT-RESUME] backup is missing optimizer state: "
                f"{optimizer_path}"
            )

        args_path = os.path.join(
            backup_path,
            "onetrainer_config",
            "args.json",
        )
        saved_ga = None
        saved_args = {}
        if os.path.isfile(args_path):
            try:
                with open(args_path, "r", encoding="utf-8") as handle:
                    saved_args = json.load(handle)
                saved_ga = int(
                    saved_args.get(
                        "gradient_accumulation_steps",
                        self.config.gradient_accumulation_steps,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    "[OT-RESUME] failed to read saved training config: "
                    f"{args_path}"
                ) from exc

        if saved_ga is None or saved_ga <= 0:
            saved_ga = int(self.config.gradient_accumulation_steps)
        self._resume_saved_gradient_accumulation_steps = saved_ga

        current_ga = int(self.config.gradient_accumulation_steps)
        if (
            saved_ga != current_ga
            and os.environ.get("OT_ALLOW_RESUME_GA_CHANGE", "0") != "1"
        ):
            raise RuntimeError(
                "[OT-RESUME] gradient accumulation changed across resume: "
                f"backup={saved_ga}, current={current_ga}. Exact resume "
                "requires the same value. Set OT_ALLOW_RESUME_GA_CHANGE=1 "
                "only if this discontinuity is intentional."
            )

        integrity_path = os.path.join(
            backup_path,
            "onetrainer_resume_integrity.json",
        )
        if os.path.isfile(integrity_path):
            try:
                with open(integrity_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception as exc:
                raise RuntimeError(
                    "[OT-RESUME] failed to read resume-integrity metadata: "
                    f"{integrity_path}"
                ) from exc

            if not bool(payload.get("optimizer_boundary", False)):
                raise RuntimeError(
                    "[OT-RESUME] backup was not written at a completed "
                    "optimizer/gradient-accumulation boundary."
                )
            self._resume_integrity_payload = payload
        else:
            self._resume_integrity_payload = None
            print(
                "[OT-RESUME] legacy backup has no resume-integrity marker; "
                "the loaded global step will be checked against its saved GA."
            )

        current_global_reference_required = (
            self.config.rlhf_enabled
            and (
                self.__global_requires_gpu_existing_adapter_dpo_reference()
                or self.__global_requires_cpu_existing_adapter_dpo_reference()
                or self.__global_requires_ema_adapter_dpo_reference()
            )
        )
        saved_global_reference_mode = ""
        if self._resume_integrity_payload is not None:
            saved_global_reference_mode = str(
                self._resume_integrity_payload.get("dpo_ref_mode", "")
            )
        elif saved_args:
            saved_global_reference_mode = str(
                saved_args.get("rlhf_dpo_ref_mode", "")
            )
        saved_global_reference_required = saved_global_reference_mode in {
                str(DPORefMode.EXISTING_ADAPTER),
                str(DPORefMode.EXISTING_ADAPTER_CPU),
                str(DPORefMode.EMA_ADAPTER),
        }

        # Fixed GPU/CPU snapshots are interchangeable storage variants. An EMA
        # snapshot is a different mathematical state and must only be restored
        # into another EMA phase. Selecting Linear-DPO on a legacy checkpoint
        # therefore starts a new EMA from the resumed policy instead of trying
        # to manufacture missing history.
        current_mode = DPORefMode(self.config.effective_dpo_ref_mode())
        reference_kind_matches = (
            saved_global_reference_mode == str(DPORefMode.EMA_ADAPTER)
            if current_mode == DPORefMode.EMA_ADAPTER
            else saved_global_reference_mode in {
                str(DPORefMode.EXISTING_ADAPTER),
                str(DPORefMode.EXISTING_ADAPTER_CPU),
            }
        )

        self._resume_restore_dpo_reference = (
            current_global_reference_required
            and saved_global_reference_required
            and reference_kind_matches
        )

        if self._resume_restore_dpo_reference:
            reference_path = os.path.join(
                backup_path,
                "onetrainer_dpo_reference.pt",
            )
            if not os.path.isfile(reference_path):
                raise RuntimeError(
                    "[OT-RESUME] DPO backup is missing its saved reference: "
                    f"{reference_path}"
                )
        elif current_global_reference_required:
            if self.__global_requires_ema_adapter_dpo_reference():
                print(
                    "[OT-RLHF] starting a new Linear-DPO phase from the "
                    "adapter loaded by this legacy/non-Linear backup; a new "
                    "EMA reference will be initialized"
                )
            else:
                print(
                    "[OT-RLHF] starting a new adapter-reference phase from "
                    "the adapter loaded by this backup"
                )

        current_concept_reference_keys = set(
            self.__concept_reference_keys(
                ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT
            )
            + self.__concept_reference_keys(
                ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT_CPU
            )
        )
        saved_concept_reference_keys = set()
        if self._resume_integrity_payload is not None:
            raw_saved_keys = self._resume_integrity_payload.get(
                "dpo_concept_reference_keys",
                [],
            )
            if isinstance(raw_saved_keys, (list, tuple)):
                saved_concept_reference_keys = {
                    str(key) for key in raw_saved_keys
                }

        self._resume_restore_dpo_concept_references = bool(
            current_concept_reference_keys
            & saved_concept_reference_keys
        )
        if self._resume_restore_dpo_concept_references:
            concept_reference_path = os.path.join(
                backup_path,
                "onetrainer_dpo_concept_references.pt",
            )
            if not os.path.isfile(concept_reference_path):
                raise RuntimeError(
                    "[OT-RESUME] backup declares per-concept DPO references "
                    f"but is missing: {concept_reference_path}"
                )
        elif current_concept_reference_keys:
            print(
                "[OT-RLHF] starting new per-concept adapter-reference phase(s) "
                "from the adapter loaded by this backup"
            )

        if (
            self.config.rlhf_enabled
            and self.model_setup._dpo_hard_pair_curriculum_enabled(
                self.config
            )
        ):
            curriculum_path = os.path.join(
                backup_path,
                "onetrainer_dpo_hard_pair_curriculum.json",
            )
            if not os.path.isfile(curriculum_path):
                raise RuntimeError(
                    "[OT-RESUME] Hard-Pair Curriculum is enabled, but the "
                    "backup is missing its exact per-pair EMA state: "
                    f"{curriculum_path}"
                )

    def __validate_loaded_resume_progress(self, backup_path: str):
        global_step = int(self.model.train_progress.global_step)
        saved_ga = int(
            self._resume_saved_gradient_accumulation_steps
            or self.config.gradient_accumulation_steps
        )

        payload = self._resume_integrity_payload
        if payload is not None:
            saved_step = int(payload.get("global_step", -1))
            if saved_step != global_step:
                raise RuntimeError(
                    "[OT-RESUME] backup progress does not match its integrity "
                    f"marker: model={global_step}, marker={saved_step}"
                )
        elif saved_ga > 1 and global_step % saved_ga != 0:
            raise RuntimeError(
                "[OT-RESUME] legacy backup was captured inside a gradient-"
                "accumulation window: "
                f"global_step={global_step}, saved_GA={saved_ga}. Its in-flight "
                "gradients were never stored, so resuming it is unsafe."
            )

    def __write_resume_integrity(
            self,
            backup_path: str,
            train_progress: TrainProgress,
    ):
        payload = {
            "version": 2,
            "global_step": int(train_progress.global_step),
            "gradient_accumulation_steps": int(
                self.config.gradient_accumulation_steps
            ),
            "optimizer_boundary": True,
            "rlhf_enabled": bool(self.config.rlhf_enabled),
            "dpo_ref_mode": str(self.config.effective_dpo_ref_mode()),
            "dpo_requires_existing_adapter_reference": (
                self.__requires_existing_adapter_dpo_reference()
            ),
            "dpo_concept_reference_keys": sorted(
                self.model_setup._dpo_concept_ref_params.keys()
                | self.model_setup._dpo_concept_ref_params_cpu.keys()
            ),
        }
        path = os.path.join(
            backup_path,
            "onetrainer_resume_integrity.json",
        )
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text'] or filename.startswith('image-rlhf-') or filename.startswith('text-rlhf-')):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            ema_applied: bool,
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}"
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                self.model.to(self.temp_device)
                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")

            torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                print("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        if self.model.ema:
            #the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            #non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
            ema_applied = self.config.ema != EMAMode.OFF
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
                ema_applied = False,
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("Calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                # since validation batch size = 1
                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                # check and fix collision to display both graphs in tensorboard
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{mapping_seed_to_label[concept_seed]}",
                                            average_loss,
                                            train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        if self._gradient_accumulation_dirty:
            raise RuntimeError(
                "[OT-RESUME] refusing to write a backup inside an unfinished "
                "gradient-accumulation window."
            )

        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)
            self.model_setup.save_dpo_curriculum_state(
                os.path.join(
                    backup_path,
                    "onetrainer_dpo_hard_pair_curriculum.json",
                ),
                self.config,
            )
            self.__save_adaptive_dpo_dataset_state(backup_path)
            self.model_setup.save_dpo_reference(
                os.path.join(backup_path, "onetrainer_dpo_reference.pt")
            )
            self.model_setup.save_dpo_concept_references(
                os.path.join(
                    backup_path,
                    "onetrainer_dpo_concept_references.pt",
                )
            )
            self.__write_resume_integrity(
                backup_path,
                train_progress,
            )
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        if self._gradient_accumulation_dirty:
            raise RuntimeError(
                "[OT-TRAIN] refusing to save inside an unfinished "
                "gradient-accumulation window."
            )

        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(
            self,
            scaler,
            dpo_momentum_bypass: bool = False,
            sequential_rlhf_backward: bool = False,
    ):
        if dpo_momentum_bypass or sequential_rlhf_backward:
            if (
                self.config.optimizer.fused_back_pass
                or self.config.fused_gradient_reduce
            ):
                if dpo_momentum_bypass:
                    print(
                        "[OT-DPO] Momentum bypass disables fused back-pass and "
                        "fused gradient reduction so DPO gradients can be isolated."
                    )
                else:
                    print(
                        "[OT-RLHF] Sequential chosen Self-Flow disables fused "
                        "back-pass/reduction so supervised and DPO gradients "
                        "can be accumulated before one optimizer step."
                    )
            return

        fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)



    def __dpo_tensorboard_metric_names(self) -> set[str]:
        return {
            "objective_loss",
            "sigmoid_objective_loss",
            "sigmoid_chosen_reward",
            "sigmoid_rejected_reward",
            "sigmoid_reward_margin",
            "linear_objective_loss",
            "linear_utility",
            "linear_policy_error_gap",
            "linear_direct_accuracy",
            "linear_effective_pair_weight",
            "balanced_reject_loss",
            "balanced_chosen_budget",
            "balanced_reject_target",
            "balanced_reject_violation",
            "balanced_target_satisfied",
            "chosen_reward",
            "rejected_reward",
            "reward_margin",
            "accuracy",
            "hard_pair_curriculum_weight",
            "hard_pair_competence_ema",
            "hard_pair_margin_ema",
            "hard_pair_observations",
            "adaptive_pair_difficulty",
            "localized_active_fraction",
            "localized_mask_fraction",
            "localized_mean_weight",
            "margin_penalty_loss",
            "wrong_order_penalty_loss",
            "margin_target_violation",
            "wrong_order_violation",
            "chosen_anchor_active",
            "chosen_anchor_weight",
            "chosen_anchor_floor",
            "chosen_anchor_push_loss",
            "chosen_anchor_floor_loss",
            "chosen_anchor_aux_loss",
            "policy_auxiliary_loss",
            "chosen_supervised_weight",
            "chosen_supervised_loss",
            "total_loss",
        }

    def __accumulate_dpo_metrics(
            self,
            metrics: dict,
            pair_count: int,
    ):
        weight = float(pair_count)
        if weight <= 0:
            return

        for name, value in metrics.items():
            if (
                name in self.__dpo_tensorboard_metric_names()
                and isinstance(value, (int, float))
            ):
                self._dpo_metric_sums[name] = (
                    self._dpo_metric_sums.get(name, 0.0)
                    + float(value) * weight
                )
                self._dpo_metric_weights[name] = (
                    self._dpo_metric_weights.get(name, 0.0)
                    + weight
                )

    def __flush_dpo_tensorboard_metrics(self, global_step: int):
        if not self._dpo_metric_sums:
            return

        if hasattr(self, "tensorboard"):
            for name, total in self._dpo_metric_sums.items():
                weight = self._dpo_metric_weights.get(name, 0.0)
                if weight <= 0:
                    continue
                self.tensorboard.add_scalar(
                    f"dpo/{name}",
                    total / weight,
                    global_step,
                )

        self._dpo_metric_sums.clear()
        self._dpo_metric_weights.clear()

    def __batch_len(self, batch: dict) -> int:
        preferred_keys = (
            "concept_name",
            "concept_path",
            "image_path",
            "text",
            "dpo_is_paired",
            "latent_image",
            "latent_image_rejected",
            "image",
        )

        for key in preferred_keys:
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, (list, tuple)):
                return len(value)
            if isinstance(value, torch.Tensor):
                if key == "dpo_is_paired":
                    return int(value.numel())
                if value.ndim > 0:
                    return int(value.shape[0])

        for value in batch.values():
            if isinstance(value, (list, tuple)):
                return len(value)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])

        return int(self.config.batch_size)

    @staticmethod
    def __as_bool(value) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "dpo", "paired"}
        return bool(value)

    def __flag_indices(self, batch: dict, key: str) -> list[int]:
        batch_len = self.__batch_len(batch)
        if key not in batch:
            return []

        flags = batch[key]
        if isinstance(flags, torch.Tensor):
            values = flags.detach().cpu().flatten().tolist()
        elif isinstance(flags, (list, tuple)):
            values = list(flags)
        else:
            values = [flags] * batch_len

        if len(values) < batch_len:
            values = values + [False] * (batch_len - len(values))

        return [i for i, value in enumerate(values[:batch_len]) if self.__as_bool(value)]

    def __rlhf_dpo_indices(self, batch: dict) -> list[int]:
        # Explicit flag only. Do not infer from latent_image_rejected because normal rows carry dummy rejected latents.
        return self.__flag_indices(batch, "dpo_is_paired")

    def __normal_indices(self, batch: dict) -> list[int]:
        dpo = set(self.__rlhf_dpo_indices(batch))
        return [i for i in range(self.__batch_len(batch)) if i not in dpo]

    def __subbatch(self, batch: dict, indices: list[int]) -> dict:
        batch_len = self.__batch_len(batch)
        out = {}
        index_cache = {}

        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_len:
                device = value.device
                if device not in index_cache:
                    index_cache[device] = torch.tensor(indices, device=device, dtype=torch.long)
                out[key] = value.index_select(0, index_cache[device])
            elif isinstance(value, list) and len(value) == batch_len:
                out[key] = [value[i] for i in indices]
            elif isinstance(value, tuple) and len(value) == batch_len:
                out[key] = tuple(value[i] for i in indices)
            else:
                out[key] = value

        return out

    def __concept_type_at(self, batch: dict, index: int) -> ConceptType:
        raw = batch["concept_type"][index]
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, ConceptType):
            return raw
        return ConceptType(raw)

    def __effective_dpo_objective_at(
            self,
            batch: dict,
            index: int,
    ) -> DPOObjective:
        raw = batch.get("dpo_objective", ConceptDPOObjective.DEFAULT)
        if isinstance(raw, torch.Tensor):
            if raw.numel() == 0:
                raw = ConceptDPOObjective.DEFAULT
            elif raw.ndim == 0:
                raw = raw.detach().cpu().item()
            else:
                raw = raw.detach().cpu().flatten()[index].item()
        elif isinstance(raw, (list, tuple)):
            raw = raw[index]

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")

        if isinstance(raw, ConceptDPOObjective):
            override = raw
        elif isinstance(raw, DPOObjective):
            # Accept an explicit full objective value in hand-edited configs,
            # while the UI intentionally exposes only DEFAULT and SIGMOID.
            if raw == DPOObjective.SIGMOID:
                override = ConceptDPOObjective.SIGMOID
            else:
                raise ValueError(
                    "Per-concept DPO objective currently supports only "
                    f"DEFAULT or SIGMOID, got {raw}."
                )
        else:
            value = str(raw or "DEFAULT").strip().upper()
            try:
                override = ConceptDPOObjective(value)
            except ValueError as exc:
                raise ValueError(
                    "Per-concept DPO objective currently supports only "
                    f"DEFAULT or SIGMOID, got {raw!r}."
                ) from exc

        if override == ConceptDPOObjective.DEFAULT:
            return DPOObjective(self.config.rlhf_dpo_objective)
        return DPOObjective.SIGMOID

    def __dpo_streamed_at(self, batch: dict, index: int) -> bool:
        raw = batch.get("dpo_streamed", False)
        if isinstance(raw, torch.Tensor):
            if raw.numel() == 0:
                return False
            if raw.ndim == 0:
                raw = raw.detach().cpu().item()
            else:
                raw = raw.detach().cpu().flatten()[index].item()
        elif isinstance(raw, (list, tuple)):
            raw = raw[index]
        return self.__as_bool(raw)

    def __effective_dpo_reference_mode_at(
            self,
            batch: dict,
            index: int,
    ) -> DPORefMode:
        raw = batch.get(
            "dpo_reference_mode",
            ConceptDPOReferenceMode.DEFAULT,
        )
        if isinstance(raw, torch.Tensor):
            if raw.numel() == 0:
                raw = ConceptDPOReferenceMode.DEFAULT
            elif raw.ndim == 0:
                raw = raw.detach().cpu().item()
            else:
                raw = raw.detach().cpu().flatten()[index].item()
        elif isinstance(raw, (list, tuple)):
            raw = raw[index]

        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")

        if isinstance(raw, ConceptDPOReferenceMode):
            override = raw
        elif isinstance(raw, DPORefMode):
            return raw
        else:
            value = str(raw or "DEFAULT").strip().upper()
            try:
                override = ConceptDPOReferenceMode(value)
            except ValueError as exc:
                raise ValueError(
                    "Per-concept DPO reference supports DEFAULT, BASE_MODEL, "
                    "CURRENT_ADAPTER_SNAPSHOT, or "
                    "CURRENT_ADAPTER_SNAPSHOT_CPU, got "
                    f"{raw!r}."
                ) from exc

        if override == ConceptDPOReferenceMode.DEFAULT:
            return DPORefMode(self.config.effective_dpo_ref_mode())
        if override == ConceptDPOReferenceMode.BASE_MODEL:
            return DPORefMode.NEW_ADAPTER
        if override == ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT_CPU:
            return DPORefMode.EXISTING_ADAPTER_CPU
        return DPORefMode.EXISTING_ADAPTER

    def __dpo_reference_key_at(
            self,
            batch: dict,
            index: int,
    ) -> str | None:
        """Return a key only for an explicit per-concept snapshot override."""
        raw_mode = batch.get(
            "dpo_reference_mode",
            ConceptDPOReferenceMode.DEFAULT,
        )
        if isinstance(raw_mode, torch.Tensor):
            if raw_mode.numel() == 0:
                raw_mode = ConceptDPOReferenceMode.DEFAULT
            elif raw_mode.ndim == 0:
                raw_mode = raw_mode.detach().cpu().item()
            else:
                raw_mode = raw_mode.detach().cpu().flatten()[index].item()
        elif isinstance(raw_mode, (list, tuple)):
            raw_mode = raw_mode[index]
        if isinstance(raw_mode, bytes):
            raw_mode = raw_mode.decode("utf-8", errors="ignore")

        if isinstance(raw_mode, ConceptDPOReferenceMode):
            override = raw_mode
        elif isinstance(raw_mode, DPORefMode):
            # Full DPORefMode values are global-style overrides and have no
            # concept-owned snapshot identity.
            return None
        else:
            override = ConceptDPOReferenceMode(
                str(raw_mode or "DEFAULT").strip().upper()
            )
        if override not in {
            ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT,
            ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT_CPU,
        }:
            return None

        if "dpo_reference_key" not in batch:
            raise RuntimeError(
                "A concept requests Current Adapter Snapshot but the data "
                "loader did not emit dpo_reference_key. Restart OneTrainer "
                "after installing the complete patch."
            )
        raw_key = batch["dpo_reference_key"]
        if isinstance(raw_key, torch.Tensor):
            if raw_key.numel() == 0:
                raw_key = None
            elif raw_key.ndim == 0:
                raw_key = raw_key.detach().cpu().item()
            else:
                raw_key = raw_key.detach().cpu().flatten()[index].item()
        elif isinstance(raw_key, (list, tuple)):
            raw_key = raw_key[index]
        if isinstance(raw_key, bytes):
            raw_key = raw_key.decode("utf-8", errors="ignore")
        if raw_key is None or str(raw_key).strip() == "":
            raise RuntimeError(
                "A concept requests Current Adapter Snapshot but has no "
                "stable concept seed/reference key."
            )
        return str(raw_key)

    @staticmethod
    def __assert_dpo_resolution_homogeneous(batch: dict):
        chosen = batch.get("latent_image")
        rejected = batch.get("latent_image_rejected")
        if isinstance(chosen, torch.Tensor) and isinstance(rejected, torch.Tensor):
            if chosen.shape != rejected.shape:
                raise RuntimeError(
                    "DPO resolution mismatch: latent_image shape "
                    f"{tuple(chosen.shape)} != latent_image_rejected shape {tuple(rejected.shape)}. "
                    "The rejected image must go through the same chosen-image bucket/crop path."
                )

    def __calculate_standard_only_training_loss(
            self,
            batch: dict,
            train_progress: TrainProgress,
    ) -> Tensor:
        batch_len = self.__batch_len(batch)
        prior_pred_indices = [
            i for i in range(batch_len)
            if self.__concept_type_at(batch, i) == ConceptType.PRIOR_PREDICTION
        ]

        if len(prior_pred_indices) > 0 or (
                self.config.masked_training
                and self.config.masked_prior_preservation_weight > 0
                and self.config.training_method == TrainingMethod.LORA
        ):
            with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                prior_model_output_data = self.model_setup.predict(
                    self.model, batch, self.config, train_progress
                )
            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(dtype=model_output_data["target"].dtype)
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
        else:
            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

        return self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)

    def __calculate_mixed_rlhf_training_loss(
            self,
            batch: dict,
            train_progress: TrainProgress,
            *,
            accumulation_steps: int = 1,
            normal_backward: Callable[[Tensor], None] | None = None,
            dpo_backward: Callable[[Tensor], None] | None = None,
    ) -> tuple[Tensor | None, Tensor | None, float, bool]:
        """Return weighted normal/DPO components, optionally backpropagated.

        Flux2 may run both ordinary samples and DPO policy samples through
        Self-Flow. Self-Flow's EMA-teacher parameter swaps mean independent
        live graphs must not straddle a later normal/DPO/reference forward.
        Sequential mode therefore finishes a graph before the next such forward
        when normal and DPO data are mixed, or when multiple DPO dispatch groups
        are present. Gradients still accumulate before the same optimizer step,
        and existing count/total_items weighting is unchanged.
        """
        if not self.config.rlhf_enabled:
            return (
                self.__calculate_standard_only_training_loss(
                    batch, train_progress
                ),
                None,
                0.0,
                False,
            )

        dpo_indices = self.__rlhf_dpo_indices(batch)
        if not dpo_indices:
            return (
                self.__calculate_standard_only_training_loss(
                    batch, train_progress
                ),
                None,
                0.0,
                False,
            )

        normal_indices = self.__normal_indices(batch)
        total_items = len(dpo_indices) + len(normal_indices)
        normal_loss_sum: Tensor | None = None
        dpo_loss_sum: Tensor | None = None

        def weighted(loss: Tensor, count: int) -> Tensor:
            return loss * (count / max(total_items, 1))

        dispatch_groups: dict[
            tuple[DPOObjective, DPORefMode, str | None, bool],
            list[int],
        ] = {}
        for index in dpo_indices:
            objective = self.__effective_dpo_objective_at(batch, index)
            if objective == DPOObjective.LINEAR:
                reference_mode = DPORefMode.EMA_ADAPTER
                reference_key = None
            else:
                reference_mode = self.__effective_dpo_reference_mode_at(
                    batch,
                    index,
                )
                # A per-concept Sigmoid override under a global Linear-DPO
                # config must not silently inherit Linear's moving reference.
                # Keep the existing objective on its base-model reference
                # unless the concept explicitly selected a fixed snapshot.
                if reference_mode == DPORefMode.EMA_ADAPTER:
                    reference_mode = DPORefMode.NEW_ADAPTER
                reference_key = self.__dpo_reference_key_at(batch, index)
            streamed = self.__dpo_streamed_at(batch, index)
            dispatch_groups.setdefault(
                (objective, reference_mode, reference_key, streamed),
                [],
            ).append(index)

        accumulation_steps = max(int(accumulation_steps), 1)

        # Keep these reasons separate:
        # 1) Some model families may require a standalone chosen-supervised
        #    DPO forward.
        # 2) Self-Flow parameter swaps require live graphs to be sequentialized
        #    across normal-vs-DPO boundaries and across multiple DPO dispatch
        #    groups. A single DPO group can still backward normally after this
        #    function returns.
        externalize_chosen_supervised = bool(
            normal_backward is not None
            and dpo_backward is not None
            and self.model_setup.rlhf_chosen_supervised_requires_separate_forward(
                self.config
            )
            and any(
                self.model_setup.rlhf_chosen_supervised_weight(
                    self.config,
                    dispatch_objective,
                ) > 0.0
                for dispatch_objective, _, _, _ in dispatch_groups
            )
        )
        model_requires_graph_sequencing = (
            self.model_setup.rlhf_mixed_normal_dpo_requires_sequential_backward(
                self.config
            )
        )
        sequential_backward = bool(
            normal_backward is not None
            and dpo_backward is not None
            and (
                externalize_chosen_supervised
                or (
                    model_requires_graph_sequencing
                    and (
                        bool(normal_indices)
                        or len(dispatch_groups) > 1
                    )
                )
            )
        )

        def finish_normal_component(component: Tensor) -> Tensor:
            if not sequential_backward:
                return component
            assert normal_backward is not None
            normal_backward(component / accumulation_steps)
            self.model_setup.after_backward(
                self.model,
                self.config,
                train_progress,
            )
            return component.detach()

        def finish_dpo_component(component: Tensor) -> Tensor:
            if not sequential_backward:
                return component
            assert dpo_backward is not None
            dpo_backward(component / accumulation_steps)
            self.model_setup.after_backward(
                self.model,
                self.config,
                train_progress,
            )
            return component.detach()

        # In sequential mode even an ordinary positive subbatch must complete
        # its backward before any later EMA/reference parameter swap. This also
        # keeps peak VRAM at one training graph instead of retaining the normal
        # graph across all DPO groups.
        if normal_indices:
            normal_batch = self.__subbatch(batch, normal_indices)
            normal_loss = self.__calculate_standard_only_training_loss(
                normal_batch, train_progress
            )
            normal_component = weighted(normal_loss, len(normal_indices))
            normal_loss_sum = finish_normal_component(normal_component)
            if hasattr(self, "tensorboard"):
                self.tensorboard.add_scalar(
                    "rlhf/normal_loss",
                    float(normal_loss.detach().item()),
                    train_progress.global_step,
                )
            del normal_loss, normal_component

        # Dispatch groups are batched, never processed pair-by-pair. With full
        # Self-Flow DPO, sequential mode also prevents one dispatch group's live
        # policy graph from surviving across the next reference/EMA swap.
        for (
            objective,
            reference_mode,
            reference_key,
            streamed,
        ), objective_indices in dispatch_groups.items():
            dpo_batch = self.__subbatch(batch, objective_indices)
            self.__assert_dpo_resolution_homogeneous(dpo_batch)

            external_supervised_value = None
            if externalize_chosen_supervised:
                supervised_weight = (
                    self.model_setup.rlhf_chosen_supervised_weight(
                        self.config,
                        objective,
                    )
                )
                if supervised_weight > 0.0:
                    supervised_loss = (
                        self.model_setup.calculate_rlhf_chosen_supervised_loss(
                            self.model,
                            dpo_batch,
                            self.config,
                            train_progress,
                        )
                    )
                    external_supervised_value = float(
                        supervised_loss.detach().item()
                    )
                    supervised_component = weighted(
                        supervised_weight * supervised_loss,
                        len(objective_indices),
                    )
                    supervised_component = finish_normal_component(
                        supervised_component
                    )
                    normal_loss_sum = (
                        supervised_component
                        if normal_loss_sum is None
                        else normal_loss_sum + supervised_component
                    )
                    del supervised_loss, supervised_component

            # With an external chosen supervised value, calculate_dpo_loss()
            # reports the full objective but constructs only the pure DPO-side
            # graph. This graph is immediately differentiated before the next
            # objective/reference group can swap any live adapter parameters.
            dpo_loss = self.model_setup.calculate_dpo_loss(
                self.model,
                dpo_batch,
                self.config,
                train_progress,
                objective=objective,
                reference_mode=reference_mode,
                reference_key=reference_key,
                streamed=streamed,
                external_chosen_supervised_loss_value=(
                    external_supervised_value
                ),
            )
            component = weighted(dpo_loss, len(objective_indices))
            component = finish_dpo_component(component)
            dpo_loss_sum = (
                component
                if dpo_loss_sum is None
                else dpo_loss_sum + component
            )

            self.__stage_adaptive_dpo_observations()
            dpo_metrics = self.model_setup.get_last_dpo_metrics()
            self.__accumulate_dpo_metrics(
                dpo_metrics,
                len(objective_indices),
            )

        if normal_loss_sum is None and dpo_loss_sum is None:
            return (
                self.__calculate_standard_only_training_loss(
                    batch, train_progress
                ),
                None,
                0.0,
                False,
            )

        dpo_item_fraction = len(dpo_indices) / max(total_items, 1)
        return (
            normal_loss_sum,
            dpo_loss_sum,
            dpo_item_fraction,
            sequential_backward,
        )

    @staticmethod
    def __gradient_l2_stats(gradients) -> tuple[float, int, int]:
        total_sq = 0.0
        element_count = 0
        tensor_count = 0
        for gradient in gradients:
            if gradient is None:
                continue
            value = gradient.detach()
            if value.is_sparse:
                value = value.coalesce().values()
            total_sq += float(value.float().square().sum(dtype=torch.float64).item())
            element_count += int(value.numel())
            tensor_count += 1
        return math.sqrt(max(total_sq, 0.0)), element_count, tensor_count

    @staticmethod
    def __virtual_clip_scale(norm: float, max_norm: float | None) -> float:
        if max_norm is None or norm <= 0.0 or not math.isfinite(norm):
            return 1.0
        return min(1.0, float(max_norm) / (float(norm) + 1e-6))

    @staticmethod
    def __gradient_ratio(numerator: float, denominator: float) -> float:
        if denominator > 0.0:
            return float(numerator) / float(denominator)
        return math.inf if numerator > 0.0 else 0.0

    def __write_dpo_gradient_strength_csv(
            self,
            train_progress: TrainProgress,
            scaler,
            dpo_momentum_bypass: bool,
    ):
        if not self.config.rlhf_enabled or not multi.is_master():
            return
        if multi.is_enabled():
            if not self._dpo_gradient_csv_warned:
                print("[OT-DPO-GRAD-CSV] Multi-GPU split logging is currently disabled.")
                self._dpo_gradient_csv_warned = True
            return

        dpo_buffer = (
            self._dpo_bypass_cpu_grads
            if dpo_momentum_bypass
            else self._dpo_probe_cpu_grads
        )
        if not dpo_buffer:
            return

        grad_scale = 1.0
        if scaler is not None:
            current_scale = float(scaler.get_scale())
            if math.isfinite(current_scale) and current_scale > 0.0:
                grad_scale = 1.0 / current_scale

        normal_sq = 0.0
        dpo_sq = 0.0
        combined_sq = 0.0
        dot = 0.0
        normal_elements = 0
        dpo_elements = 0
        normal_tensors = 0
        dpo_tensors = 0

        # Stream one parameter at a time. This avoids holding a second full
        # normal-gradient copy: in the ordinary momentum path we reconstruct
        # normal = combined - captured_DPO, then immediately discard it.
        for parameter in self.parameters:
            if not parameter.requires_grad:
                continue

            dpo_cpu = dpo_buffer.get(parameter)
            total_grad = parameter.grad

            if dpo_cpu is not None:
                dpo_vec = dpo_cpu.detach().float().reshape(-1) * grad_scale
                dpo_sq += float(torch.dot(dpo_vec, dpo_vec).item())
                dpo_elements += int(dpo_vec.numel())
                dpo_tensors += 1
            else:
                dpo_vec = None

            if total_grad is None:
                normal_vec = None
                combined_vec = None
            else:
                combined_vec = total_grad.detach().float().reshape(-1).cpu() * grad_scale
                combined_sq += float(torch.dot(combined_vec, combined_vec).item())
                if dpo_momentum_bypass or dpo_vec is None:
                    normal_vec = combined_vec
                else:
                    normal_vec = combined_vec - dpo_vec

            if normal_vec is not None:
                normal_sq += float(torch.dot(normal_vec, normal_vec).item())
                normal_elements += int(normal_vec.numel())
                normal_tensors += 1
                if dpo_vec is not None:
                    dot += float(torch.dot(normal_vec, dpo_vec).item())

        normal_norm = math.sqrt(max(normal_sq, 0.0))
        dpo_norm = math.sqrt(max(dpo_sq, 0.0))
        combined_norm = math.sqrt(max(combined_sq, 0.0))
        cosine = (
            dot / (normal_norm * dpo_norm)
            if normal_norm > 0.0 and dpo_norm > 0.0
            else 0.0
        )

        max_norm = self.config.clip_grad_norm
        if dpo_momentum_bypass:
            # Normal and DPO are clipped independently because they are applied
            # by separate optimizer paths.
            normal_clip_scale = self.__virtual_clip_scale(normal_norm, max_norm)
            dpo_clip_scale = self.__virtual_clip_scale(dpo_norm, max_norm)
            dpo_update_weight = float(self._dpo_bypass_update_weight)
        else:
            # Ordinary momentum path clips the combined vector once. The same
            # scalar therefore scales both component vectors.
            combined_clip_scale = self.__virtual_clip_scale(combined_norm, max_norm)
            normal_clip_scale = combined_clip_scale
            dpo_clip_scale = combined_clip_scale
            dpo_update_weight = 1.0

        normal_effective = normal_norm * normal_clip_scale
        dpo_effective = dpo_norm * dpo_clip_scale * dpo_update_weight

        row = {
            "global_step": int(train_progress.global_step),
            "epoch": int(getattr(train_progress, "epoch", 0)),
            "epoch_step": int(getattr(train_progress, "epoch_step", 0)),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "self_flow_enabled": int(bool(getattr(self.config, "self_flow_enabled", False))),
            "no_momentum_dpo": int(bool(dpo_momentum_bypass)),
            "dpo_gradient_scale": float(getattr(self.config, "rlhf_dpo_gradient_scale", 1.0)),
            "dpo_update_weight": dpo_update_weight,
            "normal_grad_l2_preclip": normal_norm,
            "dpo_grad_l2_preclip": dpo_norm,
            "combined_grad_l2_preclip": combined_norm,
            "normal_dpo_cosine": cosine,
            "normal_clip_scale": normal_clip_scale,
            "dpo_clip_scale": dpo_clip_scale,
            "normal_grad_l2_effective": normal_effective,
            "dpo_grad_l2_effective": dpo_effective,
            "dpo_to_normal_ratio_preclip": self.__gradient_ratio(dpo_norm, normal_norm),
            "dpo_to_normal_ratio_effective": self.__gradient_ratio(dpo_effective, normal_effective),
            "normal_grad_rms_preclip": normal_norm / math.sqrt(normal_elements) if normal_elements else 0.0,
            "dpo_grad_rms_preclip": dpo_norm / math.sqrt(dpo_elements) if dpo_elements else 0.0,
            "normal_active_elements": normal_elements,
            "dpo_active_elements": dpo_elements,
            "normal_active_tensors": normal_tensors,
            "dpo_active_tensors": dpo_tensors,
        }

        output_path = os.path.join(self.config.workspace_dir, "dpo_gradient_strength.csv")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
        with open(output_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        # The diagnostic probe is scoped to one optimizer accumulation window.
        # The bypass buffer is cleared later by its own update path.
        self._dpo_probe_cpu_grads.clear()

    def __dpo_momentum_bypass_enabled(self) -> bool:
        if not self.config.rlhf_enabled or self.model is None:
            return False

        # UI/config is the authoritative control. getattr(..., True) keeps
        # legacy configs/backups compatible with the historical behavior where
        # DPO momentum bypass was enabled by default.
        enabled = bool(
            getattr(
                self.config,
                "rlhf_dpo_momentum_bypass",
                True,
            )
        )

        # Preserve OT_DPO_BYPASS_MOMENTUM=0 as an emergency command-line
        # disable, but do NOT let an environment value of 1 override a user
        # disabling the checkbox in the UI.
        env_value = os.environ.get("OT_DPO_BYPASS_MOMENTUM")
        if (
            env_value is not None
            and env_value.strip().lower() in {"0", "false", "no", "off"}
        ):
            enabled = False

        optimizer = self.model.optimizer
        has_momentum = any(
            float(group.get("momentum", 0.0)) != 0.0
            for group in optimizer.param_groups
        )

        return (
            enabled
            and has_momentum
            and bool(
                getattr(
                    optimizer,
                    "supports_dpo_momentum_bypass",
                    False,
                )
            )
        )

    def __clear_dpo_bypass_gradients(self):
        self._dpo_bypass_cpu_grads.clear()
        self._dpo_bypass_update_weight = 0.0

    def __backward_dpo_without_momentum(self, loss: Tensor):
        """Capture DPO leaf gradients in CPU FP32 and suppress .grad writes.

        A temporary leaf hook receives each incoming DPO gradient, adds it to
        the host-side accumulation buffer, then zeros that same transient GPU
        tensor before autograd accumulates it into ``parameter.grad``.  Existing
        normal gradients therefore remain untouched and no second GPU gradient
        or momentum state persists between microbatches.
        """
        had_normal_grad = {
            parameter: parameter.grad is not None
            for parameter in self.parameters
            if parameter.requires_grad
        }
        handles: list[RemovableHandle] = []

        def make_hook(parameter: Parameter):
            def capture(grad: Tensor | None):
                # A parameter may not participate in this particular DPO
                # backward pass. Some autograd/checkpointing paths report that
                # through the hook as None.
                if grad is None:
                    return None

                cpu_grad = grad.detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                existing = self._dpo_bypass_cpu_grads.get(parameter)
                if existing is None:
                    self._dpo_bypass_cpu_grads[parameter] = cpu_grad
                else:
                    existing.add_(cpu_grad)
                # Reuse the transient tensor rather than allocating zeros_like.
                grad.zero_()
                return grad
            return capture

        try:
            for parameter in self.parameters:
                if parameter.requires_grad:
                    handles.append(parameter.register_hook(make_hook(parameter)))
            loss.backward()
        finally:
            for handle in handles:
                handle.remove()

        # DPO-only parameters may have received an allocated all-zero .grad.
        # Remove those while retaining normal gradients that existed beforehand.
        for parameter, previously_present in had_normal_grad.items():
            if not previously_present:
                parameter.grad = None

    @staticmethod
    def __dpo_cpu_clip_scale(
            gradients: dict[Parameter, Tensor],
            max_norm: float | None,
    ) -> float:
        total_sq = 0.0
        for grad in gradients.values():
            if not bool(torch.isfinite(grad).all().item()):
                raise RuntimeError(
                    "DPO momentum-bypass gradient became NaN or Inf."
                )
            grad_norm = torch.linalg.vector_norm(
                grad,
                ord=2,
                dtype=torch.float64,
            )
            total_sq += float(grad_norm.item()) ** 2

        if max_norm is None:
            return 1.0

        total_norm = math.sqrt(total_sq)
        if not math.isfinite(total_norm):
            raise RuntimeError(
                "DPO momentum-bypass gradient norm became NaN or Inf."
            )
        return min(1.0, float(max_norm) / (total_norm + 1e-6))

    @torch.no_grad()
    def __apply_dpo_momentum_bypass(
            self,
            normal_grad_parameters: set[Parameter],
    ):
        if not self._dpo_bypass_cpu_grads:
            return

        update_scale = float(self._dpo_bypass_update_weight)
        if not math.isfinite(update_scale) or update_scale <= 0.0:
            raise RuntimeError(
                f"Invalid DPO momentum-bypass update scale: {update_scale!r}"
            )

        optimizer = self.model.optimizer
        if not hasattr(optimizer, "step_parameter_without_momentum"):
            raise RuntimeError(
                "The active optimizer does not implement the DPO momentum "
                "bypass step."
            )

        # Multi-GPU gradients must be averaged before clipping.  Stream one
        # parameter at a time, then return the reduced gradient to host RAM.
        if multi.is_enabled():
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    cpu_grad = self._dpo_bypass_cpu_grads.get(parameter)
                    if cpu_grad is None:
                        continue
                    parameter.grad = cpu_grad.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    multi.reduce_grads_mean(
                        [parameter],
                        self.config.gradient_reduce_precision,
                    )
                    self._dpo_bypass_cpu_grads[parameter] = (
                        parameter.grad.detach().to(
                            device="cpu",
                            dtype=torch.float32,
                        )
                    )
                    parameter.grad = None

        clip_scale = self.__dpo_cpu_clip_scale(
            self._dpo_bypass_cpu_grads,
            self.config.clip_grad_norm,
        )

        updated = 0
        try:
            for group in optimizer.param_groups:
                for i, parameter in enumerate(group["params"]):
                    cpu_grad = self._dpo_bypass_cpu_grads.get(parameter)
                    if cpu_grad is None:
                        continue
                    grad = cpu_grad.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                    if clip_scale < 1.0:
                        grad.mul_(clip_scale)

                    had_normal_grad = parameter in normal_grad_parameters
                    optimizer.step_parameter_without_momentum(
                        parameter,
                        grad,
                        group,
                        i,
                        # Apply decay exactly once.  The normal step already
                        # handled it for parameters with normal gradients.
                        apply_weight_decay=not had_normal_grad,
                        increment_state_step=not had_normal_grad,
                        update_scale=update_scale,
                    )
                    updated += 1
                    del grad
        finally:
            for parameter in self.parameters:
                parameter.grad = None
            self.__clear_dpo_bypass_gradients()

        if updated == 0:
            raise RuntimeError(
                "DPO loss produced no gradients for momentum bypass."
            )


    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

    def __ot_debug_dir(self) -> str:
        path = os.environ.get("OT_CRASH_LOG_DIR", "/workspace/ot_crash_logs").strip()
        os.makedirs(path, exist_ok=True)
        return path

    def __ot_debug_scalar(self, value):
        try:
            if hasattr(value, "detach"):
                tensor = value.detach()
                if tensor.numel() == 0:
                    return None
                if str(tensor.device).startswith("cuda") and os.environ.get("OT_CRASH_LOG_TENSOR_VALUES", "0") not in {"1", "true", "yes", "on"}:
                    return {
                        "tensor_value_skipped": "cuda tensor; set OT_CRASH_LOG_TENSOR_VALUES=1 to sync/copy values"
                    }
                return tensor.flatten()[0].cpu().item()
            return value
        except Exception as e:
            return f"<scalar_error {type(e).__name__}: {e}>"

    def __ot_debug_value_summary(self, value, max_items: int = 8, depth: int = 0):
        if depth > 3:
            return "<max_depth>"

        try:
            if hasattr(value, "detach"):
                tensor = value.detach()
                out = {
                    "type": "tensor",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "requires_grad": bool(getattr(value, "requires_grad", False)),
                    "numel": int(tensor.numel()),
                }

                # Default: do not sync/copy CUDA tensor values. For device-side
                # asserts, syncing can itself trigger the pending crash and make
                # the dump worse. Enable manually only when needed.
                want_values = os.environ.get("OT_CRASH_LOG_TENSOR_VALUES", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }

                if tensor.numel() <= max_items and (want_values or not str(tensor.device).startswith("cuda")):
                    try:
                        out["values"] = tensor.flatten().cpu().tolist()
                    except Exception as e:
                        out["values_error"] = f"{type(e).__name__}: {e}"

                if os.environ.get("OT_CRASH_LOG_TENSOR_STATS", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
                    try:
                        ft = tensor.float()
                        out["nan_count"] = int(ft.isnan().sum().cpu().item())
                        out["inf_count"] = int(ft.isinf().sum().cpu().item())
                        finite = ft[torch.isfinite(ft)]
                        if finite.numel() > 0:
                            out["min"] = float(finite.min().cpu().item())
                            out["max"] = float(finite.max().cpu().item())
                            out["mean"] = float(finite.mean().cpu().item())
                    except Exception as e:
                        out["stats_error"] = f"{type(e).__name__}: {e}"

                return out

            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")

            if isinstance(value, (str, int, float, bool)) or value is None:
                if isinstance(value, str) and len(value) > 500:
                    return value[:500] + "...<truncated>"
                return value

            if isinstance(value, (list, tuple)):
                return {
                    "type": type(value).__name__,
                    "len": len(value),
                    "items": [
                        self.__ot_debug_value_summary(x, max_items=max_items, depth=depth + 1)
                        for x in list(value)[:max_items]
                    ],
                }

            if isinstance(value, dict):
                keys = list(value.keys())
                return {
                    "type": "dict",
                    "len": len(value),
                    "keys": [str(k) for k in keys[:64]],
                    "items": {
                        str(k): self.__ot_debug_value_summary(value[k], max_items=max_items, depth=depth + 1)
                        for k in keys[:max_items]
                    },
                }

            text = str(value)
            return text[:500] + ("...<truncated>" if len(text) > 500 else "")

        except Exception as e:
            return f"<summary_error {type(e).__name__}: {e}>"

    def __ot_debug_batch_summary(self, batch: dict, train_progress=None) -> dict:
        important_keys = [
            "image_path",
            "image_path_rejected",
            "chosen_image_path",
            "rejected_image_path",
            "chosen_source_path",
            "rejected_source_path",
            "chosen_image_path_raw",
            "rejected_image_path_raw",
            "dpo_pair_key",
            "dpo_is_paired",
            "dpo_cache_mode",
            "dpo_objective",
            "dpo_reference_mode",
            "dpo_reference_key",
            "dpo_streamed",
            "crop_resolution",
            "scale_resolution",
            "loss_weight",
            "concept",
            "concept.name",
            "concept.path",
            "concept.image.enable_resolution_override",
            "concept.image.resolution_override",
            "latent_image",
            "latent_image_rejected",
            "image",
            "image_rejected",
            "text",
            "tokens",
            "tokens_1",
            "tokens_2",
            "tokens_3",
        ]

        progress = {}
        if train_progress is not None:
            for name in [
                "global_step",
                "epoch",
                "epoch_step",
                "accumulation_step",
                "sample_step",
            ]:
                if hasattr(train_progress, name):
                    try:
                        progress[name] = getattr(train_progress, name)
                    except Exception:
                        pass

        summary = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "progress": progress,
            "batch_type": type(batch).__name__,
            "batch_keys": sorted([str(k) for k in batch.keys()]) if isinstance(batch, dict) else [],
            "important": {},
            "all": {},
        }

        if not isinstance(batch, dict):
            summary["batch_repr"] = str(batch)[:1000]
            return summary

        for key in important_keys:
            if key in batch:
                summary["important"][key] = self.__ot_debug_value_summary(batch[key])

        # Also summarize every key, but keep it shape/value-safe.
        for key, value in batch.items():
            summary["all"][str(key)] = self.__ot_debug_value_summary(value)

        return summary

    def __ot_write_batch_breadcrumb(self, batch: dict, train_progress=None):
        if os.environ.get("OT_CRASH_BREADCRUMBS", "0").strip().lower() in {
            "0", "false", "off", "no", "disabled"
        }:
            return

        try:
            path = os.path.join(self.__ot_debug_dir(), "batch_breadcrumbs.jsonl")
            summary = self.__ot_debug_batch_summary(batch, train_progress)

            # Breadcrumbs are meant to identify the last batch before async CUDA
            # crashes. Keep them compact: important fields only.
            compact = {
                "time": summary.get("time"),
                "progress": summary.get("progress"),
                "important": summary.get("important"),
            }

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            print(f"[OT-CRASH-LOG] failed to write breadcrumb: {type(e).__name__}: {e}")

    def __ot_write_crash_dump(self, exc: BaseException, batch: dict, train_progress=None):
        try:
            log_dir = self.__ot_debug_dir()
            step = "unknown"
            if train_progress is not None and hasattr(train_progress, "global_step"):
                try:
                    step = str(getattr(train_progress, "global_step"))
                except Exception:
                    pass

            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = os.path.join(log_dir, f"crash_{stamp}_step_{step}")

            payload = {
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
                "batch": self.__ot_debug_batch_summary(batch, train_progress),
                "env": {
                    "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING", ""),
                    "OT_CRASH_LOG_TENSOR_VALUES": os.environ.get("OT_CRASH_LOG_TENSOR_VALUES", "0"),
                    "OT_CRASH_LOG_TENSOR_STATS": os.environ.get("OT_CRASH_LOG_TENSOR_STATS", "0"),
                },
            }

            json_path = base + ".json"
            txt_path = base + ".txt"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("OT crash dump\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"exception_type: {payload['exception_type']}\n")
                f.write(f"exception: {payload['exception']}\n\n")
                f.write("traceback:\n")
                f.write(payload["traceback"])
                f.write("\n\nimportant batch fields:\n")
                f.write(json.dumps(payload["batch"].get("important", {}), ensure_ascii=False, indent=2, default=str))
                f.write("\n\nall batch fields:\n")
                f.write(json.dumps(payload["batch"].get("all", {}), ensure_ascii=False, indent=2, default=str))

            print(f"[OT-CRASH-LOG] wrote crash dump: {txt_path}")
            print(f"[OT-CRASH-LOG] wrote crash json: {json_path}")
            print("[OT-CRASH-LOG] if this is a CUDA device-side assert, rerun with CUDA_LAUNCH_BLOCKING=1 for the real failing line")

        except Exception as log_exc:
            print(f"[OT-CRASH-LOG] failed to write crash dump: {type(log_exc).__name__}: {log_exc}")
            print("[OT-CRASH-LOG] original exception:")
            print(traceback.format_exc())


    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching all variations")
                # Cache modules now warm every configured variation on their
                # first start. Replaying every epoch would only rebuild ordering.
                self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        dpo_momentum_bypass = self.__dpo_momentum_bypass_enabled()
        sequential_rlhf_backward_possible = bool(
            self.config.rlhf_enabled
            and (
                self.model_setup.rlhf_chosen_supervised_requires_separate_forward(
                    self.config
                )
                or self.model_setup.rlhf_mixed_normal_dpo_requires_sequential_backward(
                    self.config
                )
            )
        )
        if dpo_momentum_bypass and scaler is not None:
            raise RuntimeError(
                "DPO momentum bypass currently requires BF16 or FP32 training; "
                "FP16 GradScaler isolation is not implemented safely."
            )
        fused_optimizer_step = (
            not dpo_momentum_bypass
            and not sequential_rlhf_backward_possible
            and self.config.optimizer.optimizer.supports_fused_back_pass()
            and self.config.optimizer.fused_back_pass
        )
        fused_reduce = (
            not dpo_momentum_bypass
            and not sequential_rlhf_backward_possible
            and self.config.multi_gpu
            and self.config.fused_gradient_reduce
        )
        if dpo_momentum_bypass:
            print(
                "[OT-DPO] DPO momentum bypass enabled: normal gradients use "
                "optimizer momentum; DPO gradients are accumulated in CPU "
                "FP32 and streamed through momentum-free SinkSGD updates."
            )

        self.__apply_fused_back_pass(
            scaler,
            dpo_momentum_bypass,
            sequential_rlhf_backward_possible,
        )

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        ema_loss = None
        ema_loss_steps = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            #call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                if self.config.image_caching or self.config.text_caching:
                    self.data_loader.get_data_set().start_next_epoch()
                    self.model_setup.setup_train_device(self.model, self.config)
                else:
                    self.model_setup.setup_train_device(self.model, self.config)
                    self.data_loader.get_data_set().start_next_epoch()

            if self.config.rlhf_enabled and not self._dpo_reference_initialized:
                self.model_setup.initialize_dpo_reference(
                    self.model,
                    self.config,
                    self._dpo_reference_snapshot_path,
                    force_existing_adapter=(
                        self.__global_requires_gpu_existing_adapter_dpo_reference()
                    ),
                    force_cpu_existing_adapter=(
                        self.__global_requires_cpu_existing_adapter_dpo_reference()
                    ),
                )
                self.model_setup.initialize_dpo_concept_references(
                    self.model,
                    gpu_reference_keys=self.__concept_reference_keys(
                        ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT
                    ),
                    cpu_reference_keys=self.__concept_reference_keys(
                        ConceptDPOReferenceMode.CURRENT_ADAPTER_SNAPSHOT_CPU
                    ),
                    snapshot_path=(
                        self._dpo_concept_reference_snapshot_path
                    ),
                )
                self._dpo_reference_initialized = True
                self._dpo_reference_snapshot_path = None
                self._dpo_concept_reference_snapshot_path = None

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()

            batches = self.data_loader.get_data_loader()
            if self.config.prefetch_next_batch:
                batches = PrefetchIterator(batches)
            if multi.is_master():
                batches = step_tqdm = tqdm(batches, desc="step", total=current_epoch_length,
                                 initial=train_progress.epoch_step)
            for batch in batches:
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                if not self.commands.get_stop_command() and self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        self.model.to(self.temp_device)
                        if backup:
                            self.__backup(train_progress, True, step_tqdm.write)
                        if save:
                            self.__save(train_progress, True, step_tqdm.write)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}.pickle"),
                    TorchProfiler      (enabled=False, filename=f"profile-step{train_progress.global_step}.json"),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)

                    # Normal behavior: use the batch exactly as emitted by the dataloader.
                    # Do not split/requeue/replay batches inside GenericTrainer.
                    #
                    # RLHF/DPO math is still handled by __calculate_mixed_rlhf_training_loss().
                    # This intentionally does not touch calculate_dpo_loss, native Krea DPO
                    # logp, chosen-reward anchor/floor logic, or any RL loss code.
                    self.__ot_write_batch_breadcrumb(batch, train_progress)
                    accumulation_steps = self.config.gradient_accumulation_steps

                    def backward_normal_component(component: Tensor):
                        if scaler:
                            scaler.scale(component).backward()
                        else:
                            component.backward()

                    def backward_dpo_component(component: Tensor):
                        if dpo_momentum_bypass:
                            self.__backward_dpo_without_momentum(component)
                        else:
                            probe_loss = scaler.scale(component) if scaler else component
                            self.__backward_dpo_with_gradient_probe(probe_loss)

                    try:
                        (
                            normal_loss,
                            dpo_loss,
                            dpo_item_fraction,
                            sequential_backward_done,
                        ) = self.__calculate_mixed_rlhf_training_loss(
                            batch,
                            train_progress,
                            accumulation_steps=accumulation_steps,
                            normal_backward=backward_normal_component,
                            dpo_backward=backward_dpo_component,
                        )
                    except Exception as e:
                        self.__ot_write_crash_dump(e, batch, train_progress)
                        raise

                    normal_loss = (
                        normal_loss / accumulation_steps
                        if normal_loss is not None
                        else None
                    )
                    dpo_loss = (
                        dpo_loss / accumulation_steps
                        if dpo_loss is not None
                        else None
                    )
                    if normal_loss is None and dpo_loss is None:
                        raise RuntimeError("Training batch produced no loss.")

                    self._gradient_accumulation_dirty = True
                    if dpo_momentum_bypass and dpo_loss is not None:
                        self._dpo_bypass_update_weight += (
                            float(dpo_item_fraction) / accumulation_steps
                        )

                    if not sequential_backward_done:
                        # Keep normal/Self-Flow and DPO backward calls separate so
                        # their gradient vectors can be measured independently.
                        # Both still accumulate into the same parameter.grad when
                        # No Momentum DPO is disabled, preserving the optimizer
                        # update as the exact linear sum of both components.
                        if normal_loss is not None:
                            backward_normal_component(normal_loss)
                        if dpo_loss is not None:
                            backward_dpo_component(dpo_loss)

                        self.model_setup.after_backward(
                            self.model,
                            self.config,
                            train_progress,
                        )

                    has_gradient = True
                    detached_loss = sum(
                        component.detach()
                        for component in (normal_loss, dpo_loss)
                        if component is not None
                    )
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress):
                        if fused_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        normal_grad_parameters = {
                            parameter
                            for parameter in self.parameters
                            if parameter.grad is not None
                        }

                        self.__write_dpo_gradient_strength_csv(
                            train_progress,
                            scaler,
                            dpo_momentum_bypass,
                        )

                        optimizer_step_succeeded = True
                        scaler_scale_before = (
                            float(scaler.get_scale()) if scaler else None
                        )
                        if scaler and fused_optimizer_step:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()
                        if scaler:
                            # GradScaler lowers its scale when it skips an
                            # overflowing optimizer step. Moving references
                            # must not advance when the policy did not.
                            optimizer_step_succeeded = (
                                float(scaler.get_scale())
                                >= scaler_scale_before
                            )

                        # Clear normal gradients before streaming one DPO gradient
                        # tensor at a time back to the GPU.
                        self.model.optimizer.zero_grad(set_to_none=True)
                        dpo_bypass_updated = False
                        if dpo_momentum_bypass:
                            # This update is intentionally independent of the
                            # GradScaler-controlled normal optimizer step. If
                            # the normal step overflows but the separately
                            # accumulated DPO gradients are applied, the policy
                            # still changed and its Linear-DPO EMA must follow.
                            dpo_bypass_updated = bool(
                                self._dpo_bypass_cpu_grads
                            )
                            self.__apply_dpo_momentum_bypass(
                                normal_grad_parameters
                            )
                        optimizer_step_succeeded = (
                            optimizer_step_succeeded or dpo_bypass_updated
                        )

                        lr_scheduler.step()
                        self.model_setup.commit_dpo_curriculum_state()
                        self.__commit_adaptive_dpo_observations()
                        has_gradient = False
                        self._gradient_accumulation_dirty = False
                        self.__flush_dpo_tensorboard_metrics(
                            train_progress.global_step
                        )

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError("Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation.")

                            self.tensorboard.add_scalar("loss/train_step",accumulated_loss_cpu , train_progress.global_step)
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix({
                                'loss': accumulated_loss_cpu,
                                'smooth loss': ema_loss,
                            })
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        if optimizer_step_succeeded:
                            self.model_setup.after_optimizer_step(
                                self.model,
                                self.config,
                                train_progress,
                            )
                            self.model_setup.update_dpo_ema_reference(
                                self.model,
                                self.config,
                            )

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if self.config.validation and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if (
                    self.commands.get_stop_command()
                    and not has_gradient
                ):
                    return

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if (
                self.commands.get_stop_command()
                and not has_gradient
            ):
                return

        if self._gradient_accumulation_dirty:
            self.model.optimizer.zero_grad(set_to_none=True)
            self.__clear_dpo_bypass_gradients()
            self._dpo_probe_cpu_grads.clear()
            self._gradient_accumulation_dirty = False
            self._dpo_metric_sums.clear()
            self._dpo_metric_weights.clear()
            self.model_setup.discard_dpo_curriculum_pending()
            self.__discard_adaptive_dpo_observations()
            print(
                "[OT-TRAIN] discarded an incomplete final gradient-"
                "accumulation window; no resumable backup was written for it."
            )

    def end(self):
        if self.one_step_trained:
            self.model.to(self.temp_device)

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)
                if os.path.isdir(self.config.output_model_destination) and self.config.output_model_format.is_single_file():
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )

        if self.model is not None:
            self.model.to(self.temp_device)

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
