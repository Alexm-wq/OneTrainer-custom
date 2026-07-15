import math
from abc import ABCMeta

from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.TimestepDistribution import TimestepDistribution

import torch
from torch import Generator, Tensor


class ModelSetupNoiseMixin(metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

        self.__weights = None
        self.__weights_key = None
        self._offset_noise_psi_schedule: Tensor | None = None

    @staticmethod
    def _sample_beta_with_generator(
            alpha: float,
            beta: float,
            batch_size: int,
            generator: Generator,
    ) -> Tensor:
        """Sample Beta(alpha, beta) without touching the global RNG.

        torch.distributions.Beta.sample() has no generator argument. DPO runs
        the reference and policy forwards separately and relies on recreating
        the same timestep/noise stream, so consuming global RNG here corrupts
        the reward comparison. A beta variate is the normalized ratio of two
        independent gamma variates; torch._standard_gamma accepts Generator.
        """
        device = generator.device
        alpha_tensor = torch.full(
            (batch_size,),
            float(alpha),
            dtype=torch.float32,
            device=device,
        )
        beta_tensor = torch.full(
            (batch_size,),
            float(beta),
            dtype=torch.float32,
            device=device,
        )

        try:
            x = torch._standard_gamma(alpha_tensor, generator=generator)
            y = torch._standard_gamma(beta_tensor, generator=generator)
        except TypeError as exc:
            raise RuntimeError(
                "This PyTorch build does not support generator-aware gamma "
                "sampling required for deterministic arbitrary Beta timesteps"
            ) from exc

        denominator = x + y
        # Extremely small concentrations can underflow both samples. Preserve
        # a finite endpoint value rather than producing NaN.
        return torch.where(
            denominator > 0,
            x / denominator.clamp_min(torch.finfo(x.dtype).tiny),
            torch.zeros_like(denominator),
        )

    def _compute_and_cache_offset_noise_psi_schedule(self, betas: Tensor) -> Tensor:
        """
        Computes the time-dependent psi_t coefficients for generalized offset noise.
        This implementation follows the paper "Generalized Diffusion Model with Adjusted Offset Noise",
        specifically Equation (34) and the logic of Algorithm 1 for the "balanced-phi_t, psi_t strategy".
        """
        if self._offset_noise_psi_schedule is not None and self._offset_noise_psi_schedule.shape[0] == betas.shape[0]:
            return self._offset_noise_psi_schedule.to(betas.device).to(torch.float64)

        betas = betas.to(torch.float64)
        T = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # From paper footnote 4: "we introduce α_0 = 1 for convenience".
        alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device, dtype=betas.dtype), alphas_cumprod[:-1]])

        # --- Start of Algorithm 1 ---
        gammas = torch.zeros(T, device=betas.device, dtype=betas.dtype)

        # Step 1: Set gamma_1 = 1
        gammas[0] = 1.0

        # This sum is `Σ_{i=1 to t-1} γ_i/√¯αᵢ₋₁` which we build iteratively.
        cumulative_sum_term = gammas[0] / torch.sqrt(alphas_cumprod_prev[0])

        # Step 2-4: Loop for t = 2 to T (in code: t = 1 to T-1)
        for t in range(1, T):
            alpha_t = alphas[t]
            alpha_cumprod_tm1 = alphas_cumprod_prev[t]

            # Denominator from the paper's formula for C_t.
            c_t_denominator = alpha_t * (1 - alpha_cumprod_tm1)
            c_t = (1 - alpha_t) * torch.sqrt(alpha_cumprod_tm1) / c_t_denominator

            # Paper's recursive formula uses the full cumulative sum.
            gammas[t] = c_t * cumulative_sum_term

            # Update the sum for the next iteration.
            cumulative_sum_term += gammas[t] / torch.sqrt(alphas_cumprod_prev[t])

        # Step 5: Calculate normalization factor psi_T
        psi_T_denominator = torch.sqrt(1 - alphas_cumprod[-1])
        psi_T = cumulative_sum_term / psi_T_denominator

        # Step 6-8: Normalize gammas
        gammas_normalized = gammas / psi_T
        # --- End of Algorithm 1 ---

        # Finally, calculate the psi schedule for all timesteps t using Equation (22)
        terms = gammas_normalized / torch.sqrt(alphas_cumprod_prev)
        s_cumulative = torch.cumsum(terms, dim=0)
        psi_schedule = s_cumulative / torch.sqrt(1 - alphas_cumprod)

        self._offset_noise_psi_schedule = psi_schedule.to(betas.device)
        return self._offset_noise_psi_schedule


    def _apply_dpo_paired_rng(self, tensor: Tensor) -> Tensor:
        # During a DPO batched forward, batch layout is [chosen; rejected].
        # The rejected half must use the same timestep/noise as the matching chosen row.
        half = getattr(self, "_dpo_paired_half", None)
        if half is not None and isinstance(tensor, torch.Tensor) and tensor.ndim > 0 and tensor.shape[0] == 2 * half:
            tensor[half:] = tensor[:half]
        return tensor

    def _create_noise(
            self,
            source_tensor: Tensor,
            config: TrainConfig,
            generator: Generator,
            timestep: Tensor | None = None,
            betas: Tensor | None = None,
    ) -> Tensor:
        noise = torch.randn(
            source_tensor.shape,
            generator=generator,
            device=config.train_device,
            dtype=source_tensor.dtype
        )

        if config.offset_noise_weight > 0:
            offset_noise = torch.randn(
                (source_tensor.shape[0], source_tensor.shape[1], *[1 for _ in range(source_tensor.ndim - 2)]),
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            # Use the time-dependent generalized method if enabled.
            # This will only be true for Diffusion models (which uses betas)
            if config.generalized_offset_noise and timestep is not None and betas is not None:
                psi_schedule = self._compute_and_cache_offset_noise_psi_schedule(betas).to(timestep.device)
                psi_t = psi_schedule[timestep]
                psi_t = psi_t.view(psi_t.shape[0], *[1 for _ in range(source_tensor.ndim - 1)])
                # Scale by the time-dependent psi_t factor
                noise = noise + (psi_t * config.offset_noise_weight * offset_noise)
            else: # Otherwise, use the normal offset noise.
                noise = noise + (config.offset_noise_weight * offset_noise)

        if config.perturbation_noise_weight > 0:
            perturbation_noise = torch.randn(
                source_tensor.shape,
                generator=generator,
                device=config.train_device,
                dtype=source_tensor.dtype
            )
            noise = noise + (config.perturbation_noise_weight * perturbation_noise)

        return self._apply_dpo_paired_rng(noise)

    @staticmethod
    def _apply_conditional_embedding_perturbation(
            embedding: Tensor | list,
            gamma: float,
            generator: Generator
    ) -> Tensor | list:
        """
        Applies Conditional Embedding Perturbation (CEP) as per Equation (8).
        Paper: "Slight Corruption in Pre-training Data Makes Better Diffusion Models"

        delta ~ U(-(gamma/sqrt(d), gamma/sqrt(d))
        """
        def _perturb_cep(tensor: Tensor) -> Tensor:
            # d denotes the dimension of c_theta(y)
            d = tensor.shape[-1]

            # gamma controls perturbation magnitude (Paper uses gamma=1.0 as default baseline)
            # Calculate scaling factor: gamma / sqrt(d)
            scale = gamma / math.sqrt(d)

            # CEP-U (Uniform) scheme
            noise = torch.rand(
                tensor.shape,
                generator=generator,
                device=tensor.device,
                dtype=tensor.dtype
            )
            perturbation = (noise * 2.0 - 1.0) * scale
            return tensor + perturbation

        if isinstance(embedding, list):
            return [_perturb_cep(emb) for emb in embedding]
        else:
            return _perturb_cep(embedding)

    def _get_timestep_discrete(
            self,
            num_train_timesteps: int,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
            shift: float = None,
    ) -> Tensor:
        if shift is None:
            shift = config.timestep_shift

        if deterministic:
            # -1 is for zero-based indexing
            return torch.tensor(
                int(num_train_timesteps * 0.5) - 1,
                dtype=torch.long,
                device=generator.device,
            ).unsqueeze(0)
        else:
            min_timestep = int(num_train_timesteps * config.min_noising_strength)
            max_timestep = int(num_train_timesteps * config.max_noising_strength)
            num_timestep = max_timestep - min_timestep

            if config.timestep_distribution in [
                TimestepDistribution.UNIFORM,
                TimestepDistribution.LOGIT_NORMAL,
                TimestepDistribution.HEAVY_TAIL,
                TimestepDistribution.BETA
            ]:
                # continuous implementations
                if config.timestep_distribution == TimestepDistribution.UNIFORM:
                    timestep = min_timestep + (max_timestep - min_timestep) \
                               * torch.rand(batch_size, generator=generator, device=generator.device)
                elif config.timestep_distribution == TimestepDistribution.LOGIT_NORMAL:
                    bias = config.noising_bias
                    scale = config.noising_weight + 1.0

                    normal = torch.normal(bias, scale, size=(batch_size,), generator=generator, device=generator.device)
                    logit_normal = normal.sigmoid()
                    timestep = logit_normal * num_timestep + min_timestep
                elif config.timestep_distribution == TimestepDistribution.HEAVY_TAIL:
                    scale = config.noising_weight

                    u = torch.rand(
                        size=(batch_size,),
                        generator=generator,
                        device=generator.device,
                    )
                    u = 1.0 - u - scale * (torch.cos(math.pi / 2.0 * u) ** 2.0 - 1.0 + u)
                    timestep = u * num_timestep + min_timestep
                elif config.timestep_distribution == TimestepDistribution.BETA:
                    # B-TTDM Configuration
                    # Noising Weight -> Alpha
                    # Noising Bias   -> Beta
                    alpha = max(1e-4, config.noising_weight)
                    beta = max(1e-4, config.noising_bias)

                    # B-TTDM Paper optimization (Section 3.3):
                    # They strictly recommend Beta=1 and Alpha < 1.
                    # When Beta=1, we can use Inverse Transform Sampling (CDF inversion)
                    # which allows us to use torch.rand with the generator
                    # CDF^(-1)(u) = u^(1/alpha)
                    if abs(beta - 1.0) < 1e-5:
                        u = torch.rand(batch_size, generator=generator, device=generator.device)
                        u = u.pow(1.0 / alpha)
                        timestep = u * num_timestep + min_timestep

                    # Inverse case: Alpha=1, Beta != 1
                    # x = 1 - u^(1/beta)
                    elif abs(alpha - 1.0) < 1e-5:
                        u = torch.rand(
                            batch_size,
                            generator=generator,
                            device=generator.device)
                        u = 1.0 - u.pow(1.0 / beta)
                        timestep = u * num_timestep + min_timestep

                    else:
                        # Arbitrary Beta values must still use the supplied
                        # generator so reference and policy DPO forwards see
                        # exactly the same timesteps.
                        u = self._sample_beta_with_generator(
                            alpha=alpha,
                            beta=beta,
                            batch_size=batch_size,
                            generator=generator,
                        )
                        timestep = u * num_timestep + min_timestep

                timestep = num_train_timesteps * shift * timestep / ((shift - 1) * timestep + num_train_timesteps)
            else:
                # Shifting a discrete distribution is done in two steps:
                # 1. Apply the inverse shift to the linspace.
                #    This moves the sample points of the function to their shifted place.
                # 2. Multiply the result with the derivative of the inverse shift function.
                #    The derivative is an approximation of the distance between sample points.
                #    Or in other words, the size of a shifted bucket in the original function.
                linspace = torch.linspace(0, 1, num_timestep)
                linspace = linspace / (shift - shift * linspace + linspace)

                linspace_derivative = torch.linspace(0, 1, num_timestep)
                linspace_derivative = shift / (shift + linspace_derivative - (linspace_derivative * shift)).pow(2)

                weights_key = (
                    config.timestep_distribution,
                    int(num_timestep),
                    float(shift),
                    float(config.noising_bias),
                    float(config.noising_weight),
                    str(generator.device),
                )
                if self.__weights is None or self.__weights_key != weights_key:
                    if config.timestep_distribution == TimestepDistribution.COS_MAP:
                        weights = 2.0 / (
                            math.pi
                            - 2.0 * math.pi * linspace
                            + 2.0 * math.pi * linspace ** 2.0
                        )
                        weights *= linspace_derivative
                    elif config.timestep_distribution == TimestepDistribution.SIGMOID:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = linspace / (
                            shift - shift * linspace + linspace
                        )
                        weights = 1 / (
                            1 + torch.exp(-weight * (weights - bias))
                        )
                        weights *= linspace_derivative
                    elif config.timestep_distribution == TimestepDistribution.INVERTED_PARABOLA:
                        bias = config.noising_bias + 0.5
                        weight = config.noising_weight

                        weights = torch.clamp(
                            -weight * ((linspace - bias) ** 2) + 2,
                            min=0.0,
                        )
                        weights *= linspace_derivative
                    else:
                        raise ValueError(
                            "Unsupported weighted timestep distribution: "
                            f"{config.timestep_distribution}"
                        )

                    self.__weights = weights.to(device=generator.device)
                    self.__weights_key = weights_key

                samples = torch.multinomial(
                    self.__weights,
                    num_samples=batch_size,
                    replacement=True,
                    generator=generator,
                ) + min_timestep
                timestep = samples.to(dtype=torch.long, device=generator.device)

            timestep = timestep.to(dtype=torch.long, device=generator.device)
        timestep = timestep.clamp(min=min_timestep, max=max_timestep - 1)
        return self._apply_dpo_paired_rng(timestep)

    def _get_timestep_continuous(
            self,
            deterministic: bool,
            generator: Generator,
            batch_size: int,
            config: TrainConfig,
    ) -> Tensor:
        if deterministic:
            return torch.full(
                size=(batch_size,),
                fill_value=0.5,
                device=generator.device,
            )
        else:
            discrete_timesteps = 10000  # Discretize to 10000 timesteps
            discrete = self._get_timestep_discrete(
                num_train_timesteps=discrete_timesteps,
                deterministic=False,
                generator=generator,
                batch_size=batch_size,
                config=config,
            ) + 1

            continuous = (discrete.float() / discrete_timesteps)
            return continuous
