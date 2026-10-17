from modules.module.quantized.mixin.QuantizedLinearMixin import QuantizedLinearMixin

from modules.module.quantized.mixin.QuantizedModuleMixin import QuantizedModuleMixin

import torch

import bitsandbytes as bnb


class LinearInt8(bnb.nn.Linear8bitLt, QuantizedLinearMixin, QuantizedModuleMixin):
    """Keep bitsandbytes' INT8 statistics attached during recursive model moves.

    nn.Module.to() recursively invokes _apply(), not child.to(). The bnb
    Linear8bitLt.to() override alone therefore cannot move its MatmulLtState
    when OneTrainer moves an entire transformer or text encoder.
    """

    @torch.no_grad()
    def quantize(self, device=None):
        if self.weight.dtype != torch.int8:
            original_device = self.weight.device
            self.weight = self.weight._quantize(device or original_device)
            self.init_8bit_state()
            self.to(original_device)

    def original_weight_shape(self):
        return self.weight.shape

    def unquantized_weight(self, dtype, device):
        if self.weight.dtype != torch.int8:
            return self.weight.detach().to(device=device, dtype=dtype)
        scale = self.weight.SCB if self.weight.SCB is not None else self.state.SCB
        return bnb.functional.int8_vectorwise_dequant(
            self.weight.to(device), scale.to(device),
        ).to(dtype)

    def _apply(self, fn, recurse=True):
        scale = self.weight.SCB if self.weight.SCB is not None else self.state.SCB
        quantized = self.weight.dtype == torch.int8
        # Preserve INT8 storage even when a parent requests another floating dtype.
        def apply_weight(tensor):
            if quantized and tensor is self.weight:
                moved = fn(tensor)
                if moved.dtype != torch.int8:
                    moved = bnb.nn.Int8Params(
                        self.weight.data.to(device=moved.device), requires_grad=False, has_fp16_weights=False,
                    )
                return moved
            return fn(tensor)

        super()._apply(apply_weight, recurse=recurse)
        if self.weight.dtype == torch.int8:
            current_scale = self.weight.SCB
            if current_scale is None and scale is not None:
                current_scale = scale.to(device=self.weight.device, dtype=torch.float32)
            self.state.CB = self.weight.data
            self.state.SCB = current_scale
            self.weight.CB = None
            self.weight.SCB = None
        return self
