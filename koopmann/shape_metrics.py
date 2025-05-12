# Adapted from: https://github.com/wrongu/repsim
from collections import OrderedDict

import torch
from torch import is_tensor

from koopmann.log import logger
from tqdm import tqdm


def build_acts_dict(data_train_loader, model, only_first_last, device):
    model.eval()

    original_act_dict = OrderedDict()
    for inputs, _ in data_train_loader:
        inputs = inputs.to(device)

        # Forward
        with torch.no_grad():
            _ = model(inputs)

        # Get acts
        batch_act_dict = model.get_forward_activations()
        batch_act_dict.popitem()  # Pop the last one

        # Build dictionary
        if only_first_last:
            keys = list(batch_act_dict.keys())
            first, last = keys[0], keys[-1]
            temp = set(keys) - set([first, last])

            # Pop everything in the middle
            for key in temp:
                batch_act_dict.pop(key)

        # Move to CPU and concatenate
        for key, acts in batch_act_dict.items():
            acts_cpu = acts.cpu()
            if key in original_act_dict:
                original_act_dict[key] = torch.cat([original_act_dict[key], acts_cpu], dim=0)
            else:
                original_act_dict[key] = acts_cpu

    # Clear CUDA cache to ensure GPU memory is freed
    torch.cuda.empty_cache()

    # We are using up a lot of memory.
    # Does this help out the GPU?
    del batch_act_dict

    return original_act_dict


def undo_preprocessing_acts(input_tensor, preproc_dict, index, device):
    cloned_tensor = input_tensor.clone()

    ### UNDO NORMALIZING
    norm_key = f"norm_{index}"
    if norm_key in preproc_dict:
        cloned_tensor = cloned_tensor * preproc_dict[norm_key].to(device)

    ### UNDO WHITENING
    wh_eigvals_key = f"wh_eigvals_{index}"
    wh_eigvecs_key = f"wh_eigvecs_{index}"
    wh_alpha_key = f"wh_alpha_{index}"
    if wh_eigvals_key in preproc_dict:
        cloned_tensor = Processor._unwhiten(
            cloned_tensor,
            preproc_dict[wh_eigvals_key],
            preproc_dict[wh_eigvecs_key],
            preproc_dict[wh_alpha_key],
        )

    ### UNDO PROJECTION
    directions_key = f"directions_{index}"
    if directions_key in preproc_dict:
        cloned_tensor = Processor._dim_restore_svd(
            cloned_tensor, preproc_dict[directions_key].to(device)
        )

    ### UNDO SHIFT
    mean_key = f"means_{index}"
    if mean_key in preproc_dict:
        cloned_tensor = cloned_tensor + preproc_dict[mean_key].to(device)

    return cloned_tensor


def preprocess_acts(
    original_act_dict, svd_dim, whiten_alpha, device, preprocess_dict={}, skip_svd=False
):
    # Init empty dict
    processed_act_dict = OrderedDict()

    for key, curr_act in tqdm(original_act_dict.items(), desc="Processing activations"):
        ### FLATTEN
        processed_act = torch.flatten(curr_act.clone().to(device), start_dim=1)

        ### MEAN CENTERING
        mean_key = f"means_{key}"
        means = preprocess_dict.setdefault(
            mean_key, torch.mean(processed_act, dim=0, keepdim=True).contiguous()
        )
        processed_act = processed_act - means.to(device)

        ### DIMENSIONALITY REDUCTION
        if not skip_svd:
            directions_key = f"directions_{key}"
            if directions_key not in preprocess_dict:
                processed_act, directions = Processor._dim_reduce_svd(processed_act, svd_dim)
                preprocess_dict[directions_key] = directions.contiguous()
            else:
                processed_act = processed_act @ preprocess_dict[f"directions_{key}"].T.to(device)

        ### PARAMETERIZED WHITENING
        wh_eigvals_key = f"wh_eigvals_{key}"
        wh_eigvecs_key = f"wh_eigvecs_{key}"
        wh_alpha_key = f"wh_alpha_{key}"
        if wh_eigvals_key not in preprocess_dict:
            processed_act, wh_eigvals, wh_eigvecs = Processor._whiten(
                processed_act, alpha=whiten_alpha
            )
            preprocess_dict[wh_eigvals_key] = (
                wh_eigvals.contiguous() if is_tensor(wh_eigvals) else torch.empty(1)
            )
            preprocess_dict[wh_eigvecs_key] = (
                wh_eigvecs.contiguous() if is_tensor(wh_eigvals) else torch.empty(1)
            )
            preprocess_dict[wh_alpha_key] = torch.tensor(whiten_alpha)
        else:
            processed_act = Processor._whiten_with_params(
                processed_act,
                alpha=whiten_alpha,
                eigenvalues=preprocess_dict[wh_eigvals_key],
                eigenvectors=preprocess_dict[wh_eigvecs_key],
            )

        ### NORMALIZING
        norm_key = f"norm_{key}"
        norms = preprocess_dict.setdefault(
            norm_key, (torch.linalg.norm(processed_act, ord="fro") / 1_00).contiguous()
        )
        processed_act = processed_act / norms.to(device)

        ### STORE
        processed_act_dict[key] = processed_act

    ### ALIGN TO FINAL ACT
    align_idx = list(processed_act_dict.keys())[-1]
    anchor_act = processed_act_dict[align_idx]
    for key, curr_act in processed_act_dict.items():
        if key != align_idx:
            align_key = f"align_{key}"
            if align_key not in preprocess_dict:
                _, aligned_act, _, rot_matrix = Processor._orthogonal_procrustes(
                    anchor_act, curr_act, anchor="a"
                )
                preprocess_dict[align_key] = rot_matrix
            else:
                rot_matrix = preprocess_dict[align_key]
                aligned_act = curr_act @ rot_matrix
            processed_act_dict[key] = aligned_act

    return processed_act_dict


def prepare_acts(
    data_train_loader,
    model,
    device,
    svd_dim,
    whiten_alpha=1,
    preprocess=True,
    preprocess_dict=None,
    only_first_last=True,
):
    # Hook model
    model.eval().hook_model().to(device)

    # Collect all activations
    original_act_dict = build_acts_dict(data_train_loader, model, only_first_last, device)

    # Initialize preprocessing dict
    preprocess_dict = preprocess_dict or OrderedDict()

    # Carry out preprocessing
    if preprocess:
        # Preprocess and populate `preprocess_dict`
        processed_act_dict = preprocess_acts(
            original_act_dict=original_act_dict,
            svd_dim=svd_dim,
            whiten_alpha=whiten_alpha,
            preprocess_dict=preprocess_dict,
            device=device,
        )
    else:
        # Init empty
        processed_act_dict = OrderedDict()

        # Move to GPU
        for key, act in original_act_dict.items():
            original_act_dict[key] = act.to(device)

    return (original_act_dict, processed_act_dict, preprocess_dict)


class Processor:
    @staticmethod
    def _whiten(x, alpha, clip_eigs=1e-9):
        """Compute (partial) whitening transform of x. When alpha=0 it is classic ZCA whitening and
        columns of x are totally decorrelated. When alpha=1, nothing happens.

        Assumes x is already centered.
        """

        # This is a shortcut: when alpha == 1, z is identity
        if alpha == 1:
            return x, None, None

        eigenvalues, eigenvectors = torch.linalg.eigh(x.T @ x / len(x))
        eigenvalues = torch.clip(eigenvalues, min=clip_eigs, max=None)
        d = alpha + (1 - alpha) * (eigenvalues**-0.5)
        # From right to left, the transformation (1) projects x onto v, (2) divides by stdev in each
        # direction, and (3) rotates back to align with original directions in x-space (ZCA)
        z = eigenvectors @ torch.diag(d) @ eigenvectors.T
        # Think of this as (z @ x.T).T, but note z==z.T
        return x @ z, eigenvalues, eigenvectors

    @staticmethod
    def _whiten_with_params(x, alpha, eigenvalues, eigenvectors):
        """Compute (partial) whitening transform of x, given eigenvectors and eigenvalues.
        When alpha=0 it is classic ZCA whitening and columns of x are totally decorrelated.
        When alpha=1, nothing happens.

        Assumes x is already centered.
        """
        # This is a shortcut: when alpha == 1, z is identity
        if alpha == 1:
            return x

        d = alpha + (1 - alpha) * (eigenvalues**-0.5)
        z = eigenvectors @ torch.diag(d) @ eigenvectors.T
        return x @ z

    @staticmethod
    def _unwhiten(x_whitened, eigvals, eigvecs, alpha, clip_eigs=1e-9):
        """Undo the whitening transformation."""

        # This is a shortcut: when alpha == 1, z is identity
        if alpha == 1:
            return x_whitened

        # Clip eigenvalues exactly as in the whitening function
        e = torch.clip(eigvals, min=clip_eigs, max=None)

        # More numerically stable computation of the inverse scaling
        if alpha == 0:  # Handle pure whitening case separately
            d_inv = e**0.5  # Direct inverse of e**-0.5
        else:
            # Compute inverse scaling factors more precisely
            d_inv = 1.0 / (alpha + (1 - alpha) * (e**-0.5))

        # Create the inverse transformation matrix
        z_inv = eigvecs @ torch.diag(d_inv) @ eigvecs.T

        # Force symmetry to reduce numerical errors
        z_inv = (z_inv + z_inv.T) / 2.0

        # Apply the inverse transformation
        return x_whitened @ z_inv

    @staticmethod
    def _dim_reduce_svd(x, dim):
        if dim:
            # PCA to truncate -- project onto top p principal axes (no rescaling)
            # NOTE: For large matrices, standard SVD is not stable.
            with torch.no_grad():
                # _, S, Vh = torch.linalg.svd(x, full_matrices=False)
                U, S, V = torch.svd_lowrank(x, q=dim, niter=5)
                Vh = V.T

            subset = x[:1000, :]
            total_var = torch.norm(subset, p="fro") ** 2
            error_squared = torch.norm(subset - (U[:1000, :] @ torch.diag(S) @ Vh), p="fro") ** 2
            fvu = error_squared / total_var
            # logger.info(f"Dim: {dim}")
            # logger.info(f"Variance explained: {100*(1-fvu):.2f}%")

            # Fix sign ambiguity - make the component with largest magnitude positive
            for i in range(min(dim, Vh.shape[0])):
                max_idx = torch.argmax(torch.abs(Vh[i]))
                if Vh[i, max_idx] < 0:
                    Vh[i] *= -1

            # Permute
            Vh_trunc = Vh[:dim, :]
        else:
            num_features = x.shape[-1]
            Vh_trunc = torch.eye(num_features).to(x.device)

        # import numpy as np; import matplotlib.pyplot as plt; signal = (subset[0] @ Vh[:dim, :].T).cpu(); side = int(np.sqrt(signal.shape[0])); plt.figure(figsize=(8, 8)); plt.imshow(signal[:side*side].reshape(side, side)); plt.colorbar(); plt.savefig('pca_signal.png'); plt.close()
        # import numpy as np; import matplotlib.pyplot as plt; signal = (subset[0] @ Vh_permuted.T).cpu(); side = int(np.sqrt(signal.shape[0])); plt.figure(figsize=(8, 8)); plt.imshow(signal[:side*side].reshape(side, side)); plt.colorbar(); plt.savefig('pca_signal.png'); plt.close()

        # svd returns v.T, so the principal axes are in the *rows*. The following einsum is
        # equivalent to x @ vT.T[:, :p] but a bit faster because the transpose is not actually
        # performed.
        return torch.einsum("mn,pn->mp", x, Vh_trunc), Vh_trunc

    @staticmethod
    def _dim_restore_svd(x_reduced, Vh_trunc):
        """
        Undo dimensionality reduction performed by _dim_reduce_svd.
        """
        # The original dim reduction was: x_reduced = x @ Vh_trunc.T
        # To undo, we multiply by Vh_trunc (pseudo-inverse of Vh_trunc.T)
        # This is equivalent to: x_reconstructed = x_reduced @ Vh_trunc
        return torch.einsum("mp,pn->mn", x_reduced, Vh_trunc)

    @staticmethod
    def _orthogonal_procrustes(a, b, anchor="middle"):
        """Provided a and b, each matrix of size (m, p) that are already centered and scaled, solve the
        orthogonal procrustest problem (rotate a and b into a common frame that minimizes distances).

        If anchor="middle" (default) then both a and b If anchor="a", then a is left unchanged and b is
        rotated towards it If anchor="b", then b is left unchanged and a is rotated towards it

        See
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        :return: new_a, new_b the rotated versions of a and b, minimizing element-wise squared
            differences
        """
        r_a, r_b = Processor._orthogonal_procrustes_rotation(a, b, anchor)
        # Apply rotation to a if needed
        if r_a is not None:
            a = a @ r_a
        # Apply rotation to b if needed
        if r_b is not None:
            b = b @ r_b
        return a, b, r_a, r_b

    @staticmethod
    def _orthogonal_procrustes_rotation(a, b, anchor="middle"):
        """Provided a and b, each matrix of size (m, p) that are already centered and scaled, solve the
        orthogonal procrustest problem (rotate a and b into a common frame that minimizes distances).

        If anchor="middle" (default) then both a and b If anchor="a", then a is left unchanged and b is
        rotated towards it If anchor="b", then b is left unchanged and a is rotated towards it

        See
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        :return: r_a and r_b, which, when right-multiplied with a and b, gives the aligned coordinates,
            or None for each if no transform is required
        """
        with torch.no_grad():
            u, _, v = torch.linalg.svd(a.T @ b)
        # Helpful trick to see how these are related: u is the inverse of u.T, and likewise v is
        # inverse of v.T. We get to the anchor=a and anchor=b solutions by right-multiplying both
        # return values by u.T or right-multiplying both return values by v, respectively (if both
        # return values are rotated in the same way, it preserves the shape).
        if anchor == "middle":
            return u, v.T
        elif anchor == "a":
            return None, v.T @ u.T
        elif anchor == "b":
            return u @ v, None
        else:
            raise ValueError(f"Invalid 'anchor' argument: {anchor} (must be 'middle', 'a', or 'b')")
