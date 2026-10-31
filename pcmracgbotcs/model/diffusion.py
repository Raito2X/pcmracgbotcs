from __future__ import annotations

import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


ModelPrediction = namedtuple("ModelPrediction", ["pred_noise", "pred_x_start"])


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape):
    batch = timesteps.shape[0]
    return values.gather(-1, timesteps).reshape(
        batch,
        *((1,) * (len(shape) - 1)),
    )


def cosine_beta_schedule(timesteps: int, offset: float = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((x / timesteps) + offset) / (1 + offset) * math.pi * 0.5
    ) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion1D(nn.Module):
    """Run noising, DDPM training, and reverse sampling on single-token CAD latents."""

    def __init__(self, model: nn.Module, cfg):
        super().__init__()
        self.model = model
        self.channels = cfg.z_dim
        self.seq_length = cfg.seq_len
        self.num_timesteps = int(cfg.timesteps)
        self.guidance_start_t = int(cfg.guidance_start_t)

        betas = cosine_beta_schedule(cfg.timesteps).to(torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_bar)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alpha_bar))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alpha_bar),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alpha_bar),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alpha_bar - 1),
        )

        # Precompute posterior mean and variance coefficients for reverse diffusion.
        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar),
        )

    def _ensure_3d(self, latent: torch.Tensor):
        if latent.dim() == 2:
            return latent[:, None, :]
        if latent.dim() == 3:
            return latent
        raise ValueError(f"expected (B,Z) or (B,L,Z), got {tuple(latent.shape)}")

    @staticmethod
    def _target_tensor(target: int | torch.Tensor, batch_size: int, device):
        if torch.is_tensor(target):
            target = target.to(device=device, dtype=torch.long)
            if target.numel() == 1:
                target = target.reshape(1).repeat(batch_size)
        else:
            target = torch.full((batch_size,), int(target), device=device, dtype=torch.long)
        if target.shape != (batch_size,):
            raise ValueError(f"target must have shape ({batch_size},), got {tuple(target.shape)}")
        return target

    def q_sample(self, x_start, timesteps, noise=None):
        """Add Gaussian noise to a clean latent at the selected timesteps."""
        noise = torch.randn_like(x_start) if noise is None else noise
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape) * noise
        )

    def predict_start_from_noise(self, x_t, timesteps, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, timesteps, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, timesteps, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, timesteps):
        mean = (
            extract(self.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        variance = extract(self.posterior_variance, timesteps, x_t.shape)
        log_variance = extract(
            self.posterior_log_variance_clipped,
            timesteps,
            x_t.shape,
        )
        return mean, variance, log_variance

    def model_predictions(self, x_t, timesteps):
        noise = self.model(x_t, timesteps)
        x_start = self.predict_start_from_noise(x_t, timesteps, noise).clamp(-1.0, 1.0)
        return ModelPrediction(noise, x_start)

    @torch.no_grad()
    def p_mean_variance(self, x_t, timesteps):
        prediction = self.model_predictions(x_t, timesteps)
        mean, variance, log_variance = self.q_posterior(
            prediction.pred_x_start,
            x_t,
            timesteps,
        )
        return mean, variance, log_variance

    def classifier_gradient(self, classifier, x_t, timesteps, target, scale):
        """Compute the target-operation positive-logit gradient with respect to x_t."""
        if scale == 0:
            return torch.zeros_like(x_t)
        with torch.enable_grad():
            differentiable = x_t.detach().requires_grad_(True)
            logits = classifier(differentiable, timesteps)
            selected = F.logsigmoid(
                logits.gather(1, target.view(-1, 1)).squeeze(1)
            )
            gradient = torch.autograd.grad(selected.sum(), differentiable)[0]
        return gradient * float(scale)

    @torch.no_grad()
    def p_sample(self, x_t, timestep: int, classifier=None, target=None, guidance_scale=0.0):
        timesteps = torch.full(
            (x_t.shape[0],),
            timestep,
            device=x_t.device,
            dtype=torch.long,
        )
        mean, variance, log_variance = self.p_mean_variance(x_t, timesteps)

        if classifier is not None and target is not None:
            # Apply guidance only during the low-noise reverse steps.
            scale = guidance_scale if timestep <= self.guidance_start_t else 0.0
            mean = mean + variance * self.classifier_gradient(
                classifier,
                x_t,
                timesteps,
                target,
                scale,
            )

        noise = torch.randn_like(x_t) if timestep > 0 else 0.0
        return mean + torch.exp(0.5 * log_variance) * noise

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        classifier=None,
        target=None,
        guidance_scale=0.0,
    ):
        if classifier is not None and target is None:
            raise ValueError("classifier guidance requires a target operation")
        apply_guidance = classifier is not None and guidance_scale != 0
        if apply_guidance:
            target = self._target_tensor(target, batch_size, self.betas.device)
            classifier.eval()
            for parameter in classifier.parameters():
                parameter.requires_grad_(False)
            print(
                f"classifier guidance: scale={guidance_scale}, "
                f"t<={self.guidance_start_t}"
            )

        latent = torch.randn(
            (batch_size, self.seq_length, self.channels),
            device=self.betas.device,
        )
        for timestep in tqdm(
            reversed(range(self.num_timesteps)),
            total=self.num_timesteps,
            desc="sampling latents",
        ):
            latent = self.p_sample(
                latent,
                timestep,
                classifier=classifier if apply_guidance else None,
                target=target if apply_guidance else None,
                guidance_scale=guidance_scale,
            )
        return latent

    def p_losses(self, x_start, timesteps, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        noisy = self.q_sample(x_start, timesteps, noise)
        predicted = self.model(noisy, timesteps)
        return F.mse_loss(predicted, noise)

    def forward(self, latent):
        latent = self._ensure_3d(latent)
        if latent.size(1) != self.seq_length:
            raise ValueError(
                f"expected latent length {self.seq_length}, got {latent.size(1)}"
            )
        timesteps = torch.randint(
            0,
            self.num_timesteps,
            (latent.shape[0],),
            device=latent.device,
        )
        return self.p_losses(latent, timesteps)
