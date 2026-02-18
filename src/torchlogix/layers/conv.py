from typing import Union
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t, _size_3_t
from torch.nn.modules.utils import _pair, _triple
from torch.nn.functional import gumbel_softmax

from ..functional import bin_op_cnn, bin_op_cnn_walsh, gumbel_sigmoid, soft_raw, soft_walsh, hard_raw, hard_walsh, WALSH_COEFFICIENTS

import warnings

try:
    import torchlogix_cuda
except ImportError:
    warnings.warn(
        "failed to import torchlogix_cuda. CUDA features will not be available for conv layers.",
        ImportWarning,
    )

from .dense import LogicDenseCudaFunction


class LogicConv2d(nn.Module):
    """2d convolutional layer with differentiable logic operations.

    This layer implements a 2d convolution with differentiable logic operations.
    It uses a binary tree structure to combine input features using logical
    operations.
    """

    def __init__(
        self,
        in_dim: _size_2_t,
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: int = None,
        implementation: str = None,
        connections: str = "random",  # or 'random-unique'
        weight_init: str = "residual",  # "residual" or "random"
        stride: int = 1,
        padding: int = 0,
        parametrization: str = "raw", # or 'walsh'
        temperature: float = 1.0,
        forward_sampling: str = "soft" # or "hard", "gumbel_soft", or "gumbel_hard"
    ):
        """Initialize the 2d logic convolutional layer.

        Args:
            in_dim: Input dimensions (height, width)
            device: Device to run the layer on
            grad_factor: Gradient factor for the logic operations
            channels: Number of input channels
            num_kernels: Number of output kernels
            tree_depth: Depth of the binary tree
            receptive_field_size: Size of the receptive field
            implementation: Implementation type ("python" or "cuda")
            connections: Connection type: "random" or "unique". The latter will overwrite
                the tree_depth parameter and use a full binary tree of all possible connections
                within the receptive field.
            stride: Stride of the convolution
            padding: Padding of the convolution
            parametrization: Parametrization to use ("raw" or "walsh")
        """
        super().__init__()
        self.parametrization = parametrization
        self.forward_sampling = forward_sampling

        # self.tree_weights = []
        assert stride <= receptive_field_size, (
            f"Stride ({stride}) cannot be larger than receptive field size "
            f"({receptive_field_size})."
        )
        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(2**i):  # Iterate over nodes at this level
                if self.parametrization == "raw":
                    if weight_init == "residual":
                        weights = torch.zeros(
                            num_kernels, 16, device=device
                        )  # Initialize with zeros
                        weights[:, 3] = 5  # Set the fourth element (index 3) to 5
                    elif weight_init == "random":
                        weights = torch.randn(num_kernels, 16, device=device)
                elif self.parametrization == "walsh":
                    if weight_init == "residual":
                        # chose randomly from walsh_coefficients, but prefer id=10
                        walsh_coefficients_tensor = torch.tensor(list(WALSH_COEFFICIENTS.values()), device=device)
                        weights = walsh_coefficients_tensor[
                            torch.randint(0, 16, (num_kernels,), device=device)
                        ].clone()  # .clone() for safety
                        n = num_kernels // 2
                        # set half of weights to id=10 (pick index randomly)
                        indices = torch.randperm(num_kernels, device=device)
                        weights[indices[:n]] = walsh_coefficients_tensor[10]
                    elif weight_init == "random":
                        weights = torch.randn(num_kernels, 4, device=device) * 0.1
                level_weights.append(torch.nn.Parameter(weights))
            self.tree_weights.append(level_weights)
        self.in_dim = _pair(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.receptive_field_size = receptive_field_size
        self.stride = stride
        self.padding = padding
        self.connections = connections
        if connections == "random":
            self.kernel_pairs = self.get_random_receptive_field_pairs()
        elif connections == "random-unique":
            self.kernel_pairs = self.get_random_unique_receptive_field_pairs()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        self.indices = self.get_indices_from_kernel_pairs(self.kernel_pairs)
        self.temperature = temperature

        # Determine implementation
        self.implementation = implementation
        if self.implementation is None:
            if device == "cuda":
                self.implementation = "cuda"
            else:
                self.implementation = "python"

        if self.implementation == "cuda":
            self._prepare_cuda_indices()


    def forward(self, x):
        """Implement the binary tree using the pre-selected indices."""
        if self.implementation == "cuda":
            return self.forward_cuda(x)
        current_level = x
        if self.padding > 0:
            current_level = torch.nn.functional.pad(
                current_level,
                (self.padding, self.padding, self.padding, self.padding, 0, 0),
                mode="constant",
                value=0
            )

        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_c = left_indices[..., 0], left_indices[..., 1], left_indices[..., 2]
        b_h, b_w, b_c = right_indices[..., 0], right_indices[..., 1], right_indices[..., 2]
        a = current_level[:, a_c, a_h, a_w]
        b = current_level[:, b_c, b_h, b_w]

        if self.parametrization == "raw":
            weighting_func = {
                "soft": soft_raw,
                "hard": hard_raw,
                "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
                "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
            }[self.forward_sampling]

            level_weights = torch.stack(
                [weighting_func(w) for w in self.tree_weights[0]], dim=0
            )
            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                    torch.float32
                )

            current_level = bin_op_cnn(a, b, level_weights)

            # Process remaining levels
            for level in range(1, self.tree_depth + 1):
                left_indices, right_indices = self.indices[level]
                a = current_level[..., left_indices]
                b = current_level[..., right_indices]
                level_weights = torch.stack(
                    [weighting_func(w) for w in self.tree_weights[level]], dim=0
                )
                if not self.training:
                    level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                        torch.float32
                    )

                current_level = bin_op_cnn(a, b, level_weights)

        elif self.parametrization == "walsh":
            level_weights = torch.stack([w for w in self.tree_weights[0]], dim=0)
            current_level = bin_op_cnn_walsh(a, b, level_weights)
            if self.training:
                if self.forward_sampling == "soft":
                    current_level = soft_walsh(current_level, tau=self.temperature)
                elif self.forward_sampling == "hard":
                    current_level = hard_walsh(current_level, tau=self.temperature)
                elif self.forward_sampling == "gumbel_soft":
                    current_level = gumbel_sigmoid(current_level, tau=self.temperature, hard=False)
                elif self.forward_sampling == "gumbel_hard":
                    current_level = gumbel_sigmoid(current_level, tau=self.temperature, hard=True)
            else:
                current_level = (current_level > 0).to(torch.float32)

            # Process remaining levels
            for level in range(1, self.tree_depth + 1):
                left_indices, right_indices = self.indices[level]
                a = current_level[..., left_indices]
                b = current_level[..., right_indices]
                # level_weights = self.tree_weights[level]
                level_weights = torch.stack([w for w in self.tree_weights[level]], dim=0)
                current_level = bin_op_cnn_walsh(a, b, level_weights)
                if self.training:
                    current_level = torch.sigmoid(current_level / self.temperature)
                else:
                    current_level = (current_level > 0).to(torch.float32)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size) // self.stride + 1

        current_level = current_level.view(
            current_level.shape[0],
            current_level.shape[1],
            reshape_h,
            reshape_w,
        )

        return current_level


    def get_random_receptive_field_pairs(self):
        """Generate random index pairs within the receptive field for each kernel.
        May contain self connections and duplicate connections.
        """
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = 2**self.tree_depth

        all_pairs_a = []
        all_pairs_b = []

        # Generate different random pairs for each kernel
        for _ in range(self.num_kernels):
            # Randomly select positions within the receptive field
            h_indices = torch.randint(0, h_k, (2 * sample_size,), device=self.device)
            w_indices = torch.randint(0, w_k, (2 * sample_size,), device=self.device)
            c_indices = torch.randint(0, c, (2 * sample_size,), device=self.device)

            # Stack the indices
            indices = torch.stack([h_indices, w_indices, c_indices], dim=-1)

            # Split for binary tree (split the random connections)
            pairs_a = indices[:sample_size]
            pairs_b = indices[sample_size:]

            all_pairs_a.append(pairs_a)
            all_pairs_b.append(pairs_b)

        # Stack all kernel pairs: shape (num_kernels, sample_size, 3)
        return torch.stack(all_pairs_a), torch.stack(all_pairs_b)


    def get_random_unique_receptive_field_pairs(self):
        """Generate random unique index pairs within the receptive field for each kernel.
        No self-connections or duplicate pairs.
        """
        c, h_k, w_k = self.channels, self.receptive_field_size, self.receptive_field_size
        sample_size = 2**self.tree_depth

        # Pre-compute all RF positions
        h_rf = torch.arange(0, h_k, device=self.device)
        w_rf = torch.arange(0, w_k, device=self.device)
        c_rf = torch.arange(0, c, device=self.device)

        h_rf_grid, w_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)

        num_positions = h_k * w_k * c
        max_unique_pairs = num_positions * (num_positions - 1) // 2

        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")

        # Use torch.randperm for efficient unique sampling
        # Create all possible pair indices
        triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=self.device)
        total_pairs = triu_indices.shape[1]

        all_pairs_a = []
        all_pairs_b = []

        # Generate different unique pairs for each kernel
        for _ in range(self.num_kernels):
            # Randomly select sample_size pairs
            selected_pair_indices = torch.randperm(total_pairs, device=self.device)[:sample_size]
            selected_i = triu_indices[0, selected_pair_indices]
            selected_j = triu_indices[1, selected_pair_indices]

            pairs_a = all_positions[selected_i]
            pairs_b = all_positions[selected_j]

            all_pairs_a.append(pairs_a)
            all_pairs_b.append(pairs_b)

        # Stack all kernel pairs: shape (num_kernels, sample_size, 3)
        return torch.stack(all_pairs_a), torch.stack(all_pairs_b)


    def apply_sliding_window(self, pairs_tuple):
        """Apply sliding window to the receptive field pairs across all kernel positions."""
        pairs_a, pairs_b = pairs_tuple  # Shape: (num_kernels, sample_size, 3)
        h, w = self.in_dim[0], self.in_dim[1]
        h_k, w_k = self.receptive_field_size, self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding

        assert h_k <= h_padded and w_k <= w_padded, (
            f"Receptive field size ({h_k}, {w_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid = torch.meshgrid(h_starts, w_starts, indexing="ij")

        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel with its unique pairs
        for kernel_idx in range(self.num_kernels):
            stacked_as = []
            stacked_bs = []

            # Get the pairs for this specific kernel
            kernel_pairs_a = pairs_a[kernel_idx]  # Shape: (sample_size, 3)
            kernel_pairs_b = pairs_b[kernel_idx]  # Shape: (sample_size, 3)

            # Slide the kernel over the image (across all positions)
            for h_start, w_start in zip(h_grid.flatten(), w_grid.flatten()):
                # Apply sliding window offset
                indices_a = torch.stack([
                    kernel_pairs_a[:, 0] + h_start,
                    kernel_pairs_a[:, 1] + w_start,
                    kernel_pairs_a[:, 2]
                ], dim=-1)

                indices_b = torch.stack([
                    kernel_pairs_b[:, 0] + h_start,
                    kernel_pairs_b[:, 1] + w_start,
                    kernel_pairs_b[:, 2]
                ], dim=-1)

                stacked_as.append(indices_a)
                stacked_bs.append(indices_b)

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)

    
    def get_indices_from_kernel_pairs(self, pairs_tuple):
        indices = [
            self.apply_sliding_window(pairs_tuple)
        ]
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=self.device)
            right_indices = torch.arange(1, size, 2, device=self.device)
            indices.append((left_indices, right_indices))
        return indices

    # ------------------------------------------------------------------ #
    #  CUDA acceleration (Approach A: flatten-and-reuse dense kernel)     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_reverse_map(flat_a, flat_b, in_dim):
        """Build CSR-like reverse mapping: for each input index, which output neurons use it.

        This is required by the dense CUDA backward_x kernel.
        Returns (given_x_indices_of_y_start, given_x_indices_of_y).
        """
        given_x_indices_of_y = [[] for _ in range(in_dim)]
        a_np = flat_a.cpu().numpy()
        b_np = flat_b.cpu().numpy()
        out_dim = len(a_np)
        for y in range(out_dim):
            given_x_indices_of_y[int(a_np[y])].append(y)
            given_x_indices_of_y[int(b_np[y])].append(y)
        device = flat_a.device
        rev_start = torch.tensor(
            np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(),
            device=device, dtype=torch.int64,
        )
        rev_map = torch.tensor(
            [item for sublist in given_x_indices_of_y for item in sublist],
            dtype=torch.int64, device=device,
        )
        return rev_start, rev_map

    def _prepare_cuda_indices(self):
        """Precompute flat 1-D indices and reverse maps for every tree level.

        Level 0 flattens the multi-dimensional sliding-window indices into
        linear offsets into a ``(C*H_p*W_p, batch)`` tensor so the existing
        dense CUDA kernel can be re-used.  Levels 1+ pair consecutive
        elements in the sample dimension.
        """
        H_p = self.in_dim[0] + 2 * self.padding
        W_p = self.in_dim[1] + 2 * self.padding

        self._cuda_level_data = []

        # --- Level 0: sliding-window indices → flat 1-D ---------------
        left_indices, right_indices = self.indices[0]
        # shape: (K, S, D, 3)  where 3 = (h, w, c)
        K, S, D = left_indices.shape[0], left_indices.shape[1], left_indices.shape[2]

        flat_a = (
            left_indices[..., 2] * H_p * W_p
            + left_indices[..., 0] * W_p
            + left_indices[..., 1]
        ).reshape(-1).to(torch.int64).contiguous()

        flat_b = (
            right_indices[..., 2] * H_p * W_p
            + right_indices[..., 0] * W_p
            + right_indices[..., 1]
        ).reshape(-1).to(torch.int64).contiguous()

        in_dim_flat = self.channels * H_p * W_p
        rev_start, rev_map = self._build_reverse_map(flat_a, flat_b, in_dim_flat)

        self._cuda_level_data.append(dict(
            a=flat_a, b=flat_b,
            rev_start=rev_start, rev_map=rev_map,
            K=K, S=S, D=D,
        ))

        # --- Levels 1+ : pair-wise tree reduction ---------------------
        D_prev = D
        for level in range(1, self.tree_depth + 1):
            D_new = D_prev // 2

            k_idx = torch.arange(K, device=self.device)
            s_idx = torch.arange(S, device=self.device)
            d_idx = torch.arange(D_new, device=self.device)
            kg, sg, dg = torch.meshgrid(k_idx, s_idx, d_idx, indexing="ij")

            flat_a = (kg * S * D_prev + sg * D_prev + 2 * dg).reshape(-1).to(torch.int64).contiguous()
            flat_b = (kg * S * D_prev + sg * D_prev + 2 * dg + 1).reshape(-1).to(torch.int64).contiguous()

            in_dim_flat = K * S * D_prev
            rev_start, rev_map = self._build_reverse_map(flat_a, flat_b, in_dim_flat)

            self._cuda_level_data.append(dict(
                a=flat_a, b=flat_b,
                rev_start=rev_start, rev_map=rev_map,
                K=K, S=S, D=D_new,
            ))
            D_prev = D_new

    def _get_weighting_func(self):
        """Return the weight-transformation function for the current forward_sampling mode."""
        return {
            "soft": soft_raw,
            "hard": hard_raw,
            "gumbel_soft": lambda w: gumbel_softmax(w, tau=self.temperature, hard=False),
            "gumbel_hard": lambda w: gumbel_softmax(w, tau=self.temperature, hard=True),
        }[self.forward_sampling]

    def forward_cuda(self, x):
        """CUDA-accelerated forward pass.

        Flattens the 4-D input and the multi-dimensional sliding-window
        indices into the 2-D layout expected by the dense CUDA kernel,
        replicates shared weights across spatial positions, and chains
        one kernel launch per tree level.

        For ``parametrization='raw'`` the native CUDA kernel is used.
        For ``parametrization='walsh'`` the flat index structure is reused
        with standard PyTorch ops (still runs on GPU).
        """
        assert x.device.type == "cuda", f"CUDA forward requires a CUDA tensor, got {x.device}"
        assert x.ndim == 4, f"Expected 4-D input (batch, C, H, W), got {x.ndim}-D"
        batch = x.shape[0]

        # Pad input if needed
        if self.padding > 0:
            x = torch.nn.functional.pad(
                x,
                (self.padding, self.padding, self.padding, self.padding),
                mode="constant",
                value=0,
            )

        # Flatten 4-D → 2-D  (C*H_p*W_p, batch)
        x_flat = x.reshape(batch, -1).transpose(0, 1).contiguous()

        if self.parametrization == "raw":
            return self._forward_cuda_raw(x_flat, batch)
        elif self.parametrization == "walsh":
            return self._forward_cuda_walsh(x_flat, batch)
        else:
            raise ValueError(f"Unknown parametrization: {self.parametrization}")

    def _forward_cuda_raw(self, x_flat, batch):
        """CUDA forward path for the 'raw' (16-gate softmax) parametrization.

        Uses the native ``LogicDenseCudaFunction`` kernel.
        """
        weighting_func = self._get_weighting_func()

        current_level = None
        for level_idx in range(self.tree_depth + 1):
            ld = self._cuda_level_data[level_idx]
            K, S, D = ld["K"], ld["S"], ld["D"]

            # --- prepare input ----------------------------------------
            if level_idx == 0:
                x_in = x_flat
            else:
                x_in = current_level.reshape(batch, -1).transpose(0, 1).contiguous()

            # --- prepare weights (shared across spatial positions) -----
            if self.training:
                stacked_w = torch.stack(
                    [weighting_func(w) for w in self.tree_weights[level_idx]]
                )  # (D, K, 16)
            else:
                stacked_w = torch.stack([
                    torch.nn.functional.one_hot(w.argmax(-1), 16).to(torch.float32)
                    for w in self.tree_weights[level_idx]
                ])  # (D, K, 16)

            # Expand: (D, K, 16) → (K, S, D, 16) → (K*S*D, 16)
            w_flat = (
                stacked_w
                .permute(1, 0, 2)
                .unsqueeze(1)
                .expand(-1, S, -1, -1)
                .reshape(-1, 16)
                .to(x_in.dtype)
            )

            # --- CUDA kernel launch -----------------------------------
            if self.training:
                y_flat = LogicDenseCudaFunction.apply(
                    x_in, ld["a"], ld["b"], w_flat,
                    ld["rev_start"], ld["rev_map"],
                )
            else:
                with torch.no_grad():
                    y_flat = LogicDenseCudaFunction.apply(
                        x_in, ld["a"], ld["b"], w_flat,
                        ld["rev_start"], ld["rev_map"],
                    )

            current_level = y_flat.transpose(0, 1).reshape(batch, K, S, D)

        # Final reshape: (batch, K, S, 1) → (batch, K, H_out, W_out)
        reshape_h = (self.in_dim[0] + 2 * self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2 * self.padding - self.receptive_field_size) // self.stride + 1
        return current_level.squeeze(-1).view(batch, K, reshape_h, reshape_w)

    def _forward_cuda_walsh(self, x_flat, batch):
        """CUDA forward path for the 'walsh' parametrization.

        Uses the precomputed flat indices for efficient 1-D gather, then
        applies the Walsh basis expansion and activation via PyTorch ops
        (running on the CUDA device).
        """
        current_level = None
        for level_idx in range(self.tree_depth + 1):
            ld = self._cuda_level_data[level_idx]
            K, S, D = ld["K"], ld["S"], ld["D"]

            # --- prepare input ----------------------------------------
            if level_idx == 0:
                x_in = x_flat  # (in_flat, batch)
            else:
                x_in = current_level.reshape(batch, -1).transpose(0, 1).contiguous()

            # --- gather via flat 1-D indices --------------------------
            a = x_in[ld["a"]]  # (K*S*D, batch)
            b = x_in[ld["b"]]  # (K*S*D, batch)

            # --- Walsh basis: {0,1} → {-1,+1} ------------------------
            A = 2 * a - 1
            B = 2 * b - 1
            basis = torch.stack([torch.ones_like(A), A, B, A * B], dim=-1)  # (K*S*D, batch, 4)

            # --- weights: (D_nodes, K, 4) → (K*S*D, 1, 4) ------------
            stacked_w = torch.stack(
                [w for w in self.tree_weights[level_idx]]
            )  # (D_nodes, K, 4)
            w_expanded = (
                stacked_w
                .permute(1, 0, 2)       # (K, D_nodes, 4)
                .unsqueeze(1)           # (K, 1, D_nodes, 4)
                .expand(-1, S, -1, -1)  # (K, S, D_nodes, 4)
                .reshape(-1, 1, 4)      # (K*S*D, 1, 4)
            )

            # --- dot product: (K*S*D, batch, 4) · (K*S*D, 1, 4) → sum → (K*S*D, batch)
            logits = (basis * w_expanded).sum(dim=-1)  # (K*S*D, batch)

            # --- activation -------------------------------------------
            if self.training:
                if level_idx == 0:
                    # Level 0: use the user-selected sampling strategy
                    if self.forward_sampling == "soft":
                        output = soft_walsh(logits, tau=self.temperature)
                    elif self.forward_sampling == "hard":
                        output = hard_walsh(logits, tau=self.temperature)
                    elif self.forward_sampling == "gumbel_soft":
                        output = gumbel_sigmoid(logits, tau=self.temperature, hard=False)
                    elif self.forward_sampling == "gumbel_hard":
                        output = gumbel_sigmoid(logits, tau=self.temperature, hard=True)
                else:
                    # Tree reduction levels: plain sigmoid (matches Python path)
                    output = torch.sigmoid(logits / self.temperature)
            else:
                output = (logits > 0).to(torch.float32)

            current_level = output.transpose(0, 1).reshape(batch, K, S, D)

        # Final reshape: (batch, K, S, 1) → (batch, K, H_out, W_out)
        reshape_h = (self.in_dim[0] + 2 * self.padding - self.receptive_field_size) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2 * self.padding - self.receptive_field_size) // self.stride + 1
        return current_level.squeeze(-1).view(batch, K, reshape_h, reshape_w)


class LogicConv3d(nn.Module):
    """3d convolutional layer with differentiable logic operations.

    This layer implements a 3d convolution with differentiable logic operations.
    It uses a binary tree structure to combine input features using logical
    operations.
    """

    def __init__(
        self,
        in_dim: _size_3_t,
        device: str = "cuda",
        grad_factor: float = 1.0,
        channels: int = 1,
        num_kernels: int = 16,
        tree_depth: int = None,
        receptive_field_size: _size_3_t = None,
        implementation: str = None,
        connections: str = "random",  # or 'random-unique'
        stride: int = 1,
        padding: int = None,
    ):
        """Initialize the 3d logic convolutional layer.

        Args:
            in_dim: Input dimensions (height, width, depth)
            device: Device to run the layer on
            grad_factor: Gradient factor for the logic operations
            channels: Number of input channels
            num_kernels: Number of output kernels
            tree_depth: Depth of the binary tree
            receptive_field_size: Size of the receptive field
            implementation: Implementation type ("python" or "cuda")
            connections: Connection type: "random" or "unique". The latter will overwrite
                the tree_depth parameter and use a full binary tree of all possible connections
                within the receptive field.
            stride: Stride of the convolution
            padding: Padding of the convolution
        """
        super().__init__()

        self.receptive_field_size = _triple(receptive_field_size)
        assert (
            (stride <= self.receptive_field_size[0]) and
            (stride <= self.receptive_field_size[1]) and
            (stride <= self.receptive_field_size[2])), (
                f"Stride ({stride}) cannot be larger than receptive field size "
                f"({receptive_field_size})"
            )

        self.tree_weights = torch.nn.ModuleList()
        for i in reversed(range(tree_depth + 1)):  # Iterate over tree levels
            level_weights = torch.nn.ParameterList()
            for _ in range(2**i):  # Iterate over nodes at this level
                weights = torch.zeros(
                    num_kernels, 16, device=device
                )  # Initialize with zeros
                weights[:, 3] = 5  # Set the fourth element (index 3) to 5
                # Wrap as a trainable parameter
                level_weights.append(torch.nn.Parameter(weights))
            self.tree_weights.append(level_weights)
        self.in_dim = _triple(in_dim)
        self.device = device
        self.grad_factor = grad_factor
        self.num_kernels = num_kernels
        self.tree_depth = tree_depth
        self.channels = channels
        self.stride = stride
        self.padding = padding
        self.connections = connections
        if connections == "random":
            self.kernel_pairs = self.get_random_receptive_field_pairs()
        elif connections == "random-unique":
            self.kernel_pairs = self.get_random_unique_receptive_field_pairs()
        else:
            raise ValueError(f"Unknown connections type: {connections}")
        self.indices = self.get_indices_from_kernel_pairs(self.kernel_pairs)


    def forward(self, x):
        """Implement the binary tree using the pre-selected indices."""
        current_level = x
        # apply zero padding
        left_indices, right_indices = self.indices[0]
        a_h, a_w, a_d, a_c = (
            left_indices[..., 0],
            left_indices[..., 1],
            left_indices[..., 2],
            left_indices[..., 3]
        )
        b_h, b_w, b_d, b_c = (
            right_indices[..., 0],
            right_indices[..., 1],
            right_indices[..., 2],
            right_indices[..., 3]
        )
        a = current_level[:, a_c, a_h, a_w, a_d]
        b = current_level[:, b_c, b_h, b_w, b_d]

        # Process first level
        level_weights = torch.stack(
            [torch.nn.functional.softmax(w, dim=-1) for w in self.tree_weights[0]],
            dim=0,
        )
        if not self.training:
            level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                torch.float32
            )

        current_level = bin_op_cnn(a, b, level_weights)

        # Process remaining levels
        for level in range(1, self.tree_depth + 1):
            left_indices, right_indices = self.indices[level]
            a = current_level[..., left_indices]
            b = current_level[..., right_indices]
            level_weights = torch.stack(
                [
                    torch.nn.functional.softmax(w, dim=-1)
                    for w in self.tree_weights[level]
                ],
                dim=0,
            )

            if not self.training:
                level_weights = torch.nn.functional.one_hot(level_weights.argmax(-1), 16).to(
                    torch.float32
                )

            current_level = bin_op_cnn(a, b, level_weights)

        # Reshape flattened output
        reshape_h = (self.in_dim[0] + 2*self.padding - self.receptive_field_size[0]) // self.stride + 1
        reshape_w = (self.in_dim[1] + 2*self.padding - self.receptive_field_size[1]) // self.stride + 1
        reshape_d = (self.in_dim[2] + 2*self.padding - self.receptive_field_size[2]) // self.stride + 1

        current_level = current_level.view(
            current_level.shape[0],
            current_level.shape[1],
            reshape_h,
            reshape_w,
            reshape_d
        )

        return current_level


    def get_random_receptive_field_pairs(self):
        """Generate random index pairs within the receptive field for each kernel.
        May contain self connections and duplicate connections.
        """
        c, h_k, w_k, d_k = (
            self.channels,
            self.receptive_field_size[0],
            self.receptive_field_size[1],
            self.receptive_field_size[2]
        )
        sample_size = 2**self.tree_depth

        all_pairs_a = []
        all_pairs_b = []

        # Generate different random pairs for each kernel
        for _ in range(self.num_kernels):
            # Randomly select positions within the receptive field
            h_indices = torch.randint(0, h_k, (2 * sample_size,), device=self.device)
            w_indices = torch.randint(0, w_k, (2 * sample_size,), device=self.device)
            d_indices = torch.randint(0, d_k, (2 * sample_size,), device=self.device)
            c_indices = torch.randint(0, c, (2 * sample_size,), device=self.device)

            # Stack the indices
            indices = torch.stack([h_indices, w_indices, d_indices, c_indices], dim=-1)

            # Split for binary tree (split the random connections)
            pairs_a = indices[:sample_size]
            pairs_b = indices[sample_size:]

            all_pairs_a.append(pairs_a)
            all_pairs_b.append(pairs_b)

        # Stack all kernel pairs: shape (num_kernels, sample_size, 4)
        return torch.stack(all_pairs_a), torch.stack(all_pairs_b)


    def get_random_unique_receptive_field_pairs(self):
        """Generate random unique index pairs within the receptive field for each kernel.
        No self-connections or duplicate pairs.
        """
        c, h_k, w_k, d_k = (
            self.channels,
            self.receptive_field_size[0],
            self.receptive_field_size[1],
            self.receptive_field_size[2]
        )
        sample_size = 2**self.tree_depth

        # Pre-compute all RF positions
        h_rf = torch.arange(0, h_k, device=self.device)
        w_rf = torch.arange(0, w_k, device=self.device)
        d_rf = torch.arange(0, d_k, device=self.device)
        c_rf = torch.arange(0, c, device=self.device)

        h_rf_grid, w_rf_grid, d_rf_grid, c_rf_grid = torch.meshgrid(h_rf, w_rf, d_rf, c_rf, indexing="ij")
        all_positions = torch.stack([
            h_rf_grid.flatten(),
            w_rf_grid.flatten(),
            d_rf_grid.flatten(),
            c_rf_grid.flatten()
        ], dim=1)

        num_positions = h_k * w_k * d_k * c
        max_unique_pairs = num_positions * (num_positions - 1) // 2

        if sample_size > max_unique_pairs:
            raise ValueError(f"Not enough unique pairs: need {sample_size}, have {max_unique_pairs}")

        # Use torch.randperm for efficient unique sampling
        # Create all possible pair indices
        triu_indices = torch.triu_indices(num_positions, num_positions, offset=1, device=self.device)
        total_pairs = triu_indices.shape[1]

        all_pairs_a = []
        all_pairs_b = []

        # Generate different unique pairs for each kernel
        for _ in range(self.num_kernels):
            # Randomly select sample_size pairs
            selected_pair_indices = torch.randperm(total_pairs, device=self.device)[:sample_size]
            selected_i = triu_indices[0, selected_pair_indices]
            selected_j = triu_indices[1, selected_pair_indices]

            pairs_a = all_positions[selected_i]
            pairs_b = all_positions[selected_j]

            all_pairs_a.append(pairs_a)
            all_pairs_b.append(pairs_b)

        # Stack all kernel pairs: shape (num_kernels, sample_size, 4)
        return torch.stack(all_pairs_a), torch.stack(all_pairs_b)


    def apply_sliding_window(self, pairs_tuple):
        """Apply sliding window to the receptive field pairs across all kernel positions."""
        pairs_a, pairs_b = pairs_tuple  # Shape: (num_kernels, sample_size, 4)
        h, w, d = self.in_dim[0], self.in_dim[1], self.in_dim[2]
        h_k, w_k, d_k = self.receptive_field_size

        # Account for padding
        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        d_padded = d + 2 * self.padding

        assert (h_k <= h_padded and w_k <= w_padded) and d_k <= d_padded, (
            f"Receptive field size ({h_k}, {w_k}, {d_k}) must fit within input dimensions "
            f"({h_padded}, {w_padded}, {d_padded}) after padding."
        )

        # Generate all possible positions the kernel can slide to
        h_starts = torch.arange(0, h_padded - h_k + 1, self.stride, device=self.device)
        w_starts = torch.arange(0, w_padded - w_k + 1, self.stride, device=self.device)
        d_starts = torch.arange(0, d_padded - d_k + 1, self.stride, device=self.device)

        # Generate meshgrid for all possible starting points of the receptive field
        h_grid, w_grid, d_grid = torch.meshgrid(h_starts, w_starts, d_starts, indexing="ij")

        all_stacked_as = []
        all_stacked_bs = []

        # Process for each kernel with its unique pairs
        for kernel_idx in range(self.num_kernels):
            stacked_as = []
            stacked_bs = []

            # Get the pairs for this specific kernel
            kernel_pairs_a = pairs_a[kernel_idx]  # Shape: (sample_size, 4)
            kernel_pairs_b = pairs_b[kernel_idx]  # Shape: (sample_size, 4)

            # Slide the kernel over the image (across all positions)
            for h_start, w_start, d_start in zip(h_grid.flatten(), w_grid.flatten(), d_grid.flatten()):
                # Apply sliding window offset
                indices_a = torch.stack([
                    kernel_pairs_a[:, 0] + h_start,
                    kernel_pairs_a[:, 1] + w_start,
                    kernel_pairs_a[:, 2] + d_start,
                    kernel_pairs_a[:, 3]
                ], dim=-1)

                indices_b = torch.stack([
                    kernel_pairs_b[:, 0] + h_start,
                    kernel_pairs_b[:, 1] + w_start,
                    kernel_pairs_b[:, 2] + d_start,
                    kernel_pairs_b[:, 3]
                ], dim=-1)

                stacked_as.append(indices_a)
                stacked_bs.append(indices_b)

            # After sliding over the whole image, store the result for this kernel
            all_stacked_as.append(torch.stack(stacked_as, dim=0))
            all_stacked_bs.append(torch.stack(stacked_bs, dim=0))

        # Stack the results for all kernels
        return torch.stack(all_stacked_as), torch.stack(all_stacked_bs)


    def get_indices_from_kernel_pairs(self, pairs_tuple):
        indices = [
            self.apply_sliding_window(pairs_tuple)
        ]
        for level in range(self.tree_depth):
            size = 2 ** (self.tree_depth - level)
            left_indices = torch.arange(0, size, 2, device=self.device)
            right_indices = torch.arange(1, size, 2, device=self.device)
            indices.append((left_indices, right_indices))
        return indices


class OrPooling(torch.nn.Module):
    """Logic gate based pooling layer."""

    # create layer that selects max in the kernel

    def __init__(self, kernel_size, stride, padding=0):
        super(OrPooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool the max value in the kernel."""
        if (x.dim() == 4):
            x = torch.nn.functional.max_pool2d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        elif (x.dim() == 5):
            x = torch.nn.functional.max_pool3d(
                x,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
            )
        else:
            raise NotImplementedError(
                "OrPooling only implemented for input tensor with rank 4 or 5"
            )
        return x
