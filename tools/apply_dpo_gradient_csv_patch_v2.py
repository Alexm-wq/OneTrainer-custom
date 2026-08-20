from pathlib import Path
import runpy


TRAINER = Path("modules/trainer/GenericTrainer.py")
V1 = Path("tools/apply_dpo_gradient_csv_patch.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


# Install the original logger first if the user has not already run it.
s = TRAINER.read_text(encoding="utf-8")
if "def __write_dpo_gradient_strength_csv(" not in s:
    runpy.run_path(str(V1), run_name="__main__")
    s = TRAINER.read_text(encoding="utf-8")

# Add a separate diagnostic DPO buffer for the ordinary momentum path.
if "self._dpo_probe_cpu_grads" not in s:
    s = replace_once(
        s,
        """        self._dpo_bypass_cpu_grads: dict[Parameter, Tensor] = {}\n        self._dpo_bypass_update_weight = 0.0\n        self._dpo_gradient_csv_warned = False\n""",
        """        self._dpo_bypass_cpu_grads: dict[Parameter, Tensor] = {}\n        self._dpo_bypass_update_weight = 0.0\n        # Diagnostic-only DPO gradient accumulator used when DPO follows the\n        # optimizer's ordinary momentum path. Unlike the bypass buffer, these\n        # gradients are still returned unchanged to autograd and therefore still\n        # accumulate into parameter.grad exactly as before.\n        self._dpo_probe_cpu_grads: dict[Parameter, Tensor] = {}\n        self._dpo_gradient_csv_warned = False\n""",
        "probe buffer init",
    )

# Add a DPO backward probe that does NOT modify gradients.
probe_marker = "    def __dpo_momentum_bypass_enabled(self) -> bool:\n"
if "def __backward_dpo_with_gradient_probe(" not in s:
    probe_method = '''    def __backward_dpo_with_gradient_probe(self, loss: Tensor):\n        """Capture DPO gradient contributions without changing normal optimizer behavior.\n\n        Hooks observe each DPO leaf gradient before PyTorch accumulates it into\n        ``parameter.grad``. The original gradient is returned unchanged, so the\n        optimizer sees exactly the same combined normal + DPO gradient as it did\n        without diagnostics. CPU FP32 copies are retained only until the next\n        optimizer update so the DPO component can be measured separately.\n        """\n        handles: list[RemovableHandle] = []\n\n        def make_hook(parameter: Parameter):\n            def capture(grad: Tensor | None):\n                if grad is None:\n                    return None\n                cpu_grad = grad.detach().to(device="cpu", dtype=torch.float32)\n                existing = self._dpo_probe_cpu_grads.get(parameter)\n                if existing is None:\n                    self._dpo_probe_cpu_grads[parameter] = cpu_grad\n                else:\n                    existing.add_(cpu_grad)\n                return grad\n            return capture\n\n        try:\n            for parameter in self.parameters:\n                if parameter.requires_grad:\n                    handles.append(parameter.register_hook(make_hook(parameter)))\n            loss.backward()\n        finally:\n            for handle in handles:\n                handle.remove()\n\n'''
    s = replace_once(s, probe_marker, probe_method + probe_marker, "probe method")

# Route every non-bypass DPO backward through the diagnostic hook.
old_wrapper = '''                    def backward_dpo_component(component: Tensor):\n                        if dpo_momentum_bypass:\n                            self.__backward_dpo_without_momentum(component)\n                        elif scaler:\n                            scaler.scale(component).backward()\n                        else:\n                            component.backward()\n'''
new_wrapper = '''                    def backward_dpo_component(component: Tensor):\n                        if dpo_momentum_bypass:\n                            self.__backward_dpo_without_momentum(component)\n                        else:\n                            probe_loss = scaler.scale(component) if scaler else component\n                            self.__backward_dpo_with_gradient_probe(probe_loss)\n'''
if old_wrapper in s:
    s = replace_once(s, old_wrapper, new_wrapper, "DPO backward wrapper")
elif new_wrapper not in s:
    raise SystemExit("DPO backward wrapper: unknown current state")

# The old non-sequential path differentiated normal+DPO as one scalar, which
# makes their contributions inseparable. Backward the two independent graphs
# separately instead. Gradient accumulation is mathematically the same sum.
old_nonseq = '''                    if not sequential_backward_done:\n                        if dpo_momentum_bypass and dpo_loss is not None:\n                            if normal_loss is not None:\n                                normal_loss.backward()\n                            self.__backward_dpo_without_momentum(dpo_loss)\n                        else:\n                            loss = (\n                                dpo_loss\n                                if normal_loss is None\n                                else normal_loss\n                                if dpo_loss is None\n                                else normal_loss + dpo_loss\n                            )\n                            if scaler:\n                                scaler.scale(loss).backward()\n                            else:\n                                loss.backward()\n\n                        self.model_setup.after_backward(\n'''
new_nonseq = '''                    if not sequential_backward_done:\n                        # Keep normal/Self-Flow and DPO backward calls separate so\n                        # their gradient vectors can be measured independently.\n                        # Both still accumulate into the same parameter.grad when\n                        # No Momentum DPO is disabled, preserving the optimizer\n                        # update as the exact linear sum of both components.\n                        if normal_loss is not None:\n                            backward_normal_component(normal_loss)\n                        if dpo_loss is not None:\n                            backward_dpo_component(dpo_loss)\n\n                        self.model_setup.after_backward(\n'''
if old_nonseq in s:
    s = replace_once(s, old_nonseq, new_nonseq, "non-sequential backward split")
elif new_nonseq not in s:
    raise SystemExit("non-sequential backward split: unknown current state")

# Replace the v1 logger with a momentum-agnostic implementation. In the normal
# momentum path parameter.grad contains normal + DPO, and the probe contains DPO,
# so normal is reconstructed exactly as (combined - DPO) parameter-by-parameter.
start = s.find("    def __write_dpo_gradient_strength_csv(\n")
end = s.find("    def __dpo_momentum_bypass_enabled(self) -> bool:\n", start)
if start < 0 or end < 0:
    raise SystemExit("gradient logger method block not found")

logger = r'''    def __write_dpo_gradient_strength_csv(
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

'''
s = s[:start] + logger + s[end:]

# Pass GradScaler state to the logger so FP16 measurements are unscaled.
old_call = '''                        self.__write_dpo_gradient_strength_csv(\n                            train_progress,\n                            dpo_momentum_bypass,\n                        )\n'''
new_call = '''                        self.__write_dpo_gradient_strength_csv(\n                            train_progress,\n                            scaler,\n                            dpo_momentum_bypass,\n                        )\n'''
if old_call in s:
    s = replace_once(s, old_call, new_call, "gradient logger call")
elif new_call not in s:
    raise SystemExit("gradient logger call: unknown current state")

# Clear any unfinished diagnostic accumulation on early/final shutdown.
cleanup_old = '''            self.__clear_dpo_bypass_gradients()\n            self._gradient_accumulation_dirty = False\n'''
cleanup_new = '''            self.__clear_dpo_bypass_gradients()\n            self._dpo_probe_cpu_grads.clear()\n            self._gradient_accumulation_dirty = False\n'''
if cleanup_old in s:
    s = replace_once(s, cleanup_old, cleanup_new, "final probe cleanup")
elif cleanup_new not in s:
    raise SystemExit("final probe cleanup: unknown current state")

compile(s, str(TRAINER), "exec")
TRAINER.write_text(s, encoding="utf-8")

print("Upgraded dpo_gradient_strength.csv logging to work with or without No Momentum DPO.")
print("CSV also includes normal_dpo_cosine and effective post-clip strength ratio.")
print("Restart OneTrainer before starting the test run.")
