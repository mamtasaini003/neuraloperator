import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
from .fno import FNO  # base FNO implementation


class FNO_Latent(FNO, name='fno_latent'):
    """
    Latent-space variant of FNO.
    Adds latent return, latent encoding, and latent-space loss computation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_latent_loss = True  # enable latent-space loss

        # 🔹 Learnable MLP encoder (maps y_true → latent)
        # This replaces the FFT-based encode_latent()
        self.latent_encoder = nn.Sequential(
            nn.Conv2d(self.out_channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=1),
        )

    def forward(self, x, output_shape=None, return_latent: bool = False, **kwargs):
        """
        Forward pass: returns latent + output when return_latent=True,
        else returns only output (for evaluation compatibility).
        """
        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        if self.positional_embedding is not None:
            x = self.positional_embedding(x)

        x = self.lifting(x)
        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        latent_pred = x
        x = self.projection(x)
        output = x

        # 🧩 Key difference: return only output if not explicitly asking for latent
        if return_latent:
            return latent_pred, output
        else:
            return output


    def encode_latent(self, y_true: torch.Tensor) -> torch.Tensor:
        """
        Learnable latent encoding: map true output (y_true) to latent space.
        """
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        y_true = y_true.to(dtype=dtype, device=device)

        latent_true = self.latent_encoder(y_true)
        return latent_true

    def compute_loss(self, x_latent: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Combine latent-space and spatial-domain losses.
        """
        y_true = y_true.to(dtype=next(self.parameters()).dtype, device=next(self.parameters()).device)
        latent_true = self.encode_latent(y_true)

        # 1️⃣ Latent loss
        latent_loss = F.mse_loss(x_latent, latent_true)

        # 2️⃣ Spatial loss (via projection Q)
        output_pred = self.projection(x_latent)
        spatial_loss = F.mse_loss(output_pred, y_true)

        # 3️⃣ Combine both
        alpha, beta = 0.5, 0.5
        total_loss = alpha * latent_loss + beta * spatial_loss

        # For logging
        self.last_latent_loss = latent_loss.detach()
        self.last_spatial_loss = spatial_loss.detach()

        return total_loss


def partialclass(new_name, cls, *args, **kwargs):
    """Utility to dynamically create partial class variants."""
    __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return type(
        new_name,
        (cls,),
        {
            "__init__": __init__,
            "__doc__": cls.__doc__,
            "forward": cls.forward,
        },
    )


TFNO_Latent = partialclass("TFNO_Latent", FNO_Latent, factorization="Tucker")
