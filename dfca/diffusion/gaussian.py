"""
DFCA Crystal Gaussian Diffusion Module.

Implements Gaussian diffusion process with XY/Z anisotropic treatment for crystal structures.

IMPORTANT: This module preserves the exact original implementation from crystal_diffusion_phy.py
for checkpoint compatibility. Any changes must maintain behavior compatibility.

Scientific Contribution:
    - XY/Z anisotropic noise scaling
    - Separate loss weighting for XY vs Z coordinates
    - Physical loss decomposition (lattice XY/Z, atom XY/Z)

Noise Scaling:
    - z_noise_scale: Scale factor for atom z-coordinate noise (default 0.113001)
    - lattice_c_noise_scale: Scale factor for lattice c-axis noise (default 0.340659)

Loss Weights (Optuna-tuned):
    - ATOM_XY_FEATURE_WEIGHTS: [1.5, 2.0, 1.0, 1.0] for [period, group, x, y]
    - ATOM_Z_LOSS_WEIGHT: 0.1 (relative weight of atom z loss vs xy loss)
    - ATOM_TOTAL_LOSS_COMBINE_WEIGHT: 15.9768 (lattice + atom loss combination)
    - lattice_loss_z_weight: 0.706323 (z-axis coefficient for lattice loss)
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from diffusion.gaussian_diffusion import GaussianDiffusion, ModelMeanType, ModelVarType, LossType, \
    get_named_beta_schedule
from diffusion.timestep_sampler import create_named_schedule_sampler

# Optuna hyperparameter search results — do not change.
ATOM_XY_FEATURE_WEIGHTS = torch.tensor([1.5, 2.0, 1.0, 1.0])  # [period, group, x, y]
ATOM_Z_LOSS_WEIGHT = 0.1        # relative weight of atom z loss vs xy loss
ATOM_TOTAL_LOSS_COMBINE_WEIGHT = 15.9768  # lattice + atom loss combination coefficient


class CrystalGaussianDiffusion:
    """
    Gaussian diffusion process wrapper for crystal structures.

    Implements anisotropic XY/Z treatment:
    - Lattice: c-axis (z) has different noise scale than a/b axes (xy)
    - Atoms: z-coordinate has different noise scale than xy coordinates

    This reflects the physical anisotropy of 2D materials:
    - In-plane (XY): Strong covalent/ionic bonds, periodic
    - Out-of-plane (Z): Weak van der Waals, vacuum layer
    """

    def __init__(
            self,
            *,
            timesteps=1000,
            beta_schedule="linear",
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            z_noise_scale=0.113001,
            lattice_c_noise_scale=0.340659,
            lattice_loss_z_weight=0.706323,
    ):
        self.timesteps = timesteps
        self.z_noise_scale = z_noise_scale
        self.lattice_c_noise_scale = lattice_c_noise_scale
        self.lattice_loss_z_weight = lattice_loss_z_weight

        betas = get_named_beta_schedule(beta_schedule, timesteps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=model_mean_type,
            model_var_type=model_var_type,
            loss_type=loss_type,
        )

        self.schedule_sampler = create_named_schedule_sampler("uniform", self.diffusion)

    def q_sample(
            self,
            lattice_vectors: torch.Tensor,
            atom_features: torch.Tensor,
            t: torch.Tensor,
            noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if noise is None:
            lattice_noise = torch.randn_like(lattice_vectors)
            atom_noise = torch.randn_like(atom_features)
        else:
            lattice_noise, atom_noise = noise

        # Scale c-axis lattice noise to account for anisotropic uncertainty.
        lattice_noise_scaled = lattice_noise.clone()
        lattice_noise_scaled[:, :, 2] = lattice_noise[:, :, 2] * self.lattice_c_noise_scale

        # Scale atom z-coordinate noise separately (z is fractional, xy are periodic).
        atom_noise_scaled = atom_noise.clone()
        atom_noise_scaled[:, :, 4] = atom_noise[:, :, 4] * self.z_noise_scale

        noisy_lattice = self.diffusion.q_sample(lattice_vectors, t, noise=lattice_noise_scaled)
        noisy_atoms = self.diffusion.q_sample(atom_features, t, noise=atom_noise_scaled)

        return noisy_lattice, noisy_atoms, {
            'lattice_noise_scaled': lattice_noise_scaled,
            'atom_noise_scaled': atom_noise_scaled,
        }

    def training_losses(
            self,
            model: nn.Module,
            lattice_vectors: torch.Tensor,
            atom_features: torch.Tensor,
            t: Optional[torch.Tensor] = None,
            noise: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = lattice_vectors.shape[0]

        if t is None:
            t, weights = self.schedule_sampler.sample(batch_size, device=lattice_vectors.device)
        else:
            weights = torch.ones_like(t, dtype=torch.float)

        if noise is None:
            lattice_noise = torch.randn_like(lattice_vectors)
            atom_noise = torch.randn_like(atom_features)
        else:
            lattice_noise, atom_noise = noise

        noisy_lattice, noisy_atoms, noise_dict = self.q_sample(
            lattice_vectors, atom_features, t, noise=(lattice_noise, atom_noise)
        )

        padding_mask = (atom_features[:, :, 0] == -1)
        pred_lattice_noise, pred_atom_noise = model(noisy_lattice, noisy_atoms, t, padding_mask=padding_mask)

        target_lattice_noise = noise_dict['lattice_noise_scaled']
        target_atom_noise = noise_dict['atom_noise_scaled']

        # Compute lattice xy and z losses separately (z axis has different noise scale).
        lattice_loss_xy = torch.mean((target_lattice_noise[:, :, :2] - pred_lattice_noise[:, :, :2]) ** 2, dim=(1, 2))
        lattice_loss_z = torch.mean((target_lattice_noise[:, :, 2:] - pred_lattice_noise[:, :, 2:]) ** 2, dim=(1, 2))

        # Weighted combination with z-axis coefficient.
        lattice_loss = lattice_loss_xy + self.lattice_loss_z_weight * lattice_loss_z

        valid_mask = (atom_features[:, :, 0] != -1).float()
        xy_feature_weights = ATOM_XY_FEATURE_WEIGHTS.to(atom_noise.device)

        xy_diff = (target_atom_noise[:, :, :4] - pred_atom_noise[:, :, :4]) ** 2
        xy_diff = xy_diff * xy_feature_weights.view(1, 1, -1)
        xy_diff = xy_diff * valid_mask.unsqueeze(-1)

        num_valid = valid_mask.sum(dim=1).clamp(min=1)
        atom_loss_xy = xy_diff.sum(dim=(1, 2)) / (num_valid * xy_feature_weights.sum())

        z_diff = (target_atom_noise[:, :, 4:] - pred_atom_noise[:, :, 4:]) ** 2
        z_diff = z_diff * valid_mask.unsqueeze(-1)
        atom_loss_z = z_diff.sum(dim=(1, 2)) / num_valid

        atom_loss = atom_loss_xy + ATOM_Z_LOSS_WEIGHT * atom_loss_z
        total_loss = lattice_loss + ATOM_TOTAL_LOSS_COMBINE_WEIGHT * atom_loss
        weighted_loss = total_loss * weights

        with torch.enable_grad():
            pred_lattice_x0 = self._predict_x0_from_eps(noisy_lattice, t, pred_lattice_noise)
            pred_atom_x0 = self._predict_x0_from_eps(noisy_atoms, t, pred_atom_noise)

        return {
            "loss": weighted_loss.mean(),
            "lattice_loss": lattice_loss.mean(),
            "lattice_loss_xy": lattice_loss_xy.mean(),
            "lattice_loss_z": lattice_loss_z.mean(),
            "atom_loss": atom_loss.mean(),
            "atom_loss_xy": atom_loss_xy.mean(),
            "atom_loss_z": atom_loss_z.mean(),
            "weighted_loss": weighted_loss.mean(),
            "pred_lattice_x0": pred_lattice_x0,
            "pred_atom_x0": pred_atom_x0,
            "timesteps": t,
        }

    def p_sample_loop(
            self,
            model: nn.Module,
            batch_size: int,
            max_atoms: int,
            clip_denoised: bool = True,
            progress: bool = False,
            device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(model.parameters()).device

        lattice_shape = (batch_size, 3, 3)
        atom_shape = (batch_size, max_atoms, 5)

        lattice_noise = torch.randn(lattice_shape, device=device)
        atom_noise = torch.randn(atom_shape, device=device)

        lattice_noise[:, :, 2] = lattice_noise[:, :, 2] * self.lattice_c_noise_scale
        atom_noise[:, :, 4] = atom_noise[:, :, 4] * self.z_noise_scale

        indices = list(range(self.diffusion.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        lattice_vectors = lattice_noise
        atom_features = atom_noise

        for i in indices:
            t = torch.tensor([i] * batch_size, device=device)
            with torch.no_grad():
                pred_lattice_noise, pred_atom_noise = model(lattice_vectors, atom_features, t)
                pred_lattice_x0 = self._predict_x0_from_eps(lattice_vectors, t, pred_lattice_noise)
                pred_atom_x0 = self._predict_x0_from_eps(atom_features, t, pred_atom_noise)

                if clip_denoised:
                    pred_lattice_x0 = torch.clamp(pred_lattice_x0, -1, 1)
                    pred_atom_x0 = torch.clamp(pred_atom_x0, -1, 1)

                posterior_mean_lattice = self._q_posterior_mean(pred_lattice_x0, lattice_vectors, t)
                posterior_mean_atom = self._q_posterior_mean(pred_atom_x0, atom_features, t)

                lattice_posterior_log_variance = self._extract_into_tensor(
                    self.diffusion.posterior_log_variance_clipped, t, lattice_vectors.shape
                )
                atom_posterior_log_variance = self._extract_into_tensor(
                    self.diffusion.posterior_log_variance_clipped, t, atom_features.shape
                )

                if i == 0:
                    lattice_vectors = posterior_mean_lattice
                    atom_features = posterior_mean_atom
                else:
                    lattice_noise = torch.randn_like(lattice_vectors)
                    lattice_noise[:, :, 2] = lattice_noise[:, :, 2] * self.lattice_c_noise_scale
                    lattice_vectors = posterior_mean_lattice + torch.exp(
                        0.5 * lattice_posterior_log_variance) * lattice_noise

                    atom_noise = torch.randn_like(atom_features)
                    atom_noise[:, :, 4] = atom_noise[:, :, 4] * self.z_noise_scale
                    atom_features = posterior_mean_atom + torch.exp(0.5 * atom_posterior_log_variance) * atom_noise

        return lattice_vectors, atom_features

    def _predict_x0_from_eps(self, x_t, t, eps):
        return (
                self._extract_into_tensor(self.diffusion.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                self._extract_into_tensor(self.diffusion.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _q_posterior_mean(self, x_0, x_t, t):
        posterior_mean = (
                self._extract_into_tensor(self.diffusion.posterior_mean_coef1, t, x_t.shape) * x_0 +
                self._extract_into_tensor(self.diffusion.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)


__all__ = [
    'CrystalGaussianDiffusion',
    'ATOM_XY_FEATURE_WEIGHTS',
    'ATOM_Z_LOSS_WEIGHT',
    'ATOM_TOTAL_LOSS_COMBINE_WEIGHT',
]
