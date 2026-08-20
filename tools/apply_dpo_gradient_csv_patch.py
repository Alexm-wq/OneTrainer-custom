from pathlib import Path


def replace_once(path: str, old: str, new: str):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    count = s.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one marker, found {count}: {old[:100]!r}")
    p.write_text(s.replace(old, new, 1), encoding="utf-8")


path = "modules/trainer/GenericTrainer.py"

replace_once(
    path,
    "import contextlib\nimport copy\nimport json\n",
    "import contextlib\nimport copy\nimport csv\nimport json\n",
)

replace_once(
    path,
    """        self._dpo_bypass_cpu_grads: dict[Parameter, Tensor] = {}\n        self._dpo_bypass_update_weight = 0.0\n        self._adaptive_dpo_dataset_module: AdaptiveDPODataset | None = None\n""",
    """        self._dpo_bypass_cpu_grads: dict[Parameter, Tensor] = {}\n        self._dpo_bypass_update_weight = 0.0\n        self._dpo_gradient_csv_warned = False\n        self._adaptive_dpo_dataset_module: AdaptiveDPODataset | None = None\n""",
)

marker = "    def __dpo_momentum_bypass_enabled(self) -> bool:\n"
methods = r'''    @staticmethod
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
            dpo_momentum_bypass: bool,
    ):
        if not self.config.rlhf_enabled or not multi.is_master():
            return
        if not dpo_momentum_bypass:
            if not self._dpo_gradient_csv_warned:
                print(
                    "[OT-DPO-GRAD-CSV] Exact split logging requires No Momentum DPO; "
                    "no rows will be written while that path is disabled/unsupported."
                )
                self._dpo_gradient_csv_warned = True
            return
        if multi.is_enabled():
            if not self._dpo_gradient_csv_warned:
                print("[OT-DPO-GRAD-CSV] Multi-GPU exact split logging is disabled.")
                self._dpo_gradient_csv_warned = True
            return
        if not self._dpo_bypass_cpu_grads:
            return

        normal_norm, normal_elements, normal_tensors = self.__gradient_l2_stats(
            parameter.grad
            for parameter in self.parameters
            if parameter.requires_grad and parameter.grad is not None
        )
        dpo_norm, dpo_elements, dpo_tensors = self.__gradient_l2_stats(
            self._dpo_bypass_cpu_grads.values()
        )

        max_norm = self.config.clip_grad_norm
        normal_clip_scale = self.__virtual_clip_scale(normal_norm, max_norm)
        dpo_clip_scale = self.__virtual_clip_scale(dpo_norm, max_norm)
        normal_postclip = normal_norm * normal_clip_scale
        dpo_postclip = dpo_norm * dpo_clip_scale

        dpo_update_weight = float(self._dpo_bypass_update_weight)
        dpo_effective = dpo_postclip * dpo_update_weight

        row = {
            "global_step": int(train_progress.global_step),
            "epoch": int(getattr(train_progress, "epoch", 0)),
            "epoch_step": int(getattr(train_progress, "epoch_step", 0)),
            "gradient_accumulation_steps": int(self.config.gradient_accumulation_steps),
            "self_flow_enabled": int(bool(getattr(self.config, "self_flow_enabled", False))),
            "dpo_gradient_scale": float(getattr(self.config, "rlhf_dpo_gradient_scale", 1.0)),
            "dpo_update_weight": dpo_update_weight,
            "normal_grad_l2_preclip": normal_norm,
            "dpo_grad_l2_preclip": dpo_norm,
            "normal_grad_l2_postclip": normal_postclip,
            "dpo_grad_l2_postclip": dpo_postclip,
            "dpo_grad_l2_effective": dpo_effective,
            "dpo_to_normal_ratio_preclip": self.__gradient_ratio(dpo_norm, normal_norm),
            "dpo_to_normal_ratio_effective": self.__gradient_ratio(dpo_effective, normal_postclip),
            "normal_grad_rms_preclip": normal_norm / math.sqrt(normal_elements) if normal_elements else 0.0,
            "dpo_grad_rms_preclip": dpo_norm / math.sqrt(dpo_elements) if dpo_elements else 0.0,
            "normal_active_elements": normal_elements,
            "dpo_active_elements": dpo_elements,
            "normal_active_tensors": normal_tensors,
            "dpo_active_tensors": dpo_tensors,
        }

        output_path = os.path.join(
            self.config.workspace_dir,
            "dpo_gradient_strength.csv",
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        write_header = not os.path.isfile(output_path) or os.path.getsize(output_path) == 0
        with open(output_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

''' + marker

p = Path(path)
s = p.read_text(encoding="utf-8")
if s.count(marker) != 1:
    raise SystemExit("GenericTrainer: DPO bypass marker mismatch")
p.write_text(s.replace(marker, methods, 1), encoding="utf-8")

replace_once(
    path,
    """                        normal_grad_parameters = {\n                            parameter\n                            for parameter in self.parameters\n                            if parameter.grad is not None\n                        }\n\n                        optimizer_step_succeeded = True\n""",
    """                        normal_grad_parameters = {\n                            parameter\n                            for parameter in self.parameters\n                            if parameter.grad is not None\n                        }\n\n                        self.__write_dpo_gradient_strength_csv(\n                            train_progress,\n                            dpo_momentum_bypass,\n                        )\n\n                        optimizer_step_succeeded = True\n""",
)

compile(Path(path).read_text(encoding="utf-8"), path, "exec")
print("Installed dpo_gradient_strength.csv logging.")
print("Restart OneTrainer for it to take effect.")
