# Adapted from: https://github.com/wrongu/repsim


from collections import OrderedDict
from copy import deepcopy

import torch


def undo_processing(input_tensor, preproc_dict, index):
    cloned_tensor = input_tensor.clone()
    if f"norms_{index}" in preproc_dict:
        cloned_tensor *= preproc_dict[f"norms_{index}"]
    if f"directions_{index}" in preproc_dict:
        cloned_tensor = cloned_tensor @ preproc_dict[f"directions_{index}"]
    if f"means_{index}" in preproc_dict:
        cloned_tensor += preproc_dict[f"means_{index}"]
    return cloned_tensor


def prepare_acts(
    data_train_loader,
    model,
    device,
    new_dim,
    whiten_alpha=1,
    preprocess=True,
    preprocess_dict=None,
    only_first_last=True,
):
    # Hook model
    model.eval().hook_model()

    # Collect all activations
    original_act_dict = OrderedDict()
    for inputs, _ in data_train_loader:
        inputs = inputs.to(device)

        # Forward
        with torch.no_grad():
            _ = model(inputs)

        # Concatenate
        batch_act_dict = model.get_forward_activations()
        batch_act_dict.popitem()  # Pop the last one

        if only_first_last:
            keys = list(batch_act_dict.keys())
            for key in keys[1:-1]:
                batch_act_dict.pop(key)

        for key, acts in batch_act_dict.items():
            if key in original_act_dict:
                original_act_dict[key] = torch.cat([original_act_dict[key], acts], dim=0)
            else:
                original_act_dict[key] = acts

    processed_act_dict = deepcopy(original_act_dict)

    # Initialize preprocessing dict if needed
    preprocess_dict = preprocess_dict or OrderedDict()

    if preprocess:
        for key, curr_act in original_act_dict.items():
            processed_act = torch.flatten(curr_act.clone(), start_dim=1)

            # Mean center
            means = preprocess_dict.get(
                f"means_{key}", torch.mean(processed_act, dim=0, keepdim=True)
            )
            if f"means_{key}" not in preprocess_dict:
                preprocess_dict[f"means_{key}"] = means.contiguous()
            processed_act -= means

            # Dim reduce
            if f"directions_{key}" not in preprocess_dict:
                processed_act, directions = Processor._dim_reduce(processed_act, new_dim)
                preprocess_dict[f"directions_{key}"] = directions.contiguous()
            else:
                processed_act = processed_act @ preprocess_dict[f"directions_{key}"].T

            # Parameterized whitening
            processed_act = Processor._whiten(processed_act, alpha=whiten_alpha)

            # Normalize
            norms = preprocess_dict.get(
                f"norms_{key}", torch.linalg.norm(processed_act, ord="fro") / 100
            )
            if f"norms_{key}" not in preprocess_dict:
                preprocess_dict[f"norms_{key}"] = norms.contiguous()
            processed_act /= norms

            processed_act_dict[key] = processed_act

        # Align each act to final one
        align_idx = list(processed_act_dict.keys())[-1]
        anchor_act = processed_act_dict[align_idx]
        for key, curr_act in processed_act_dict.items():
            if key != align_idx:
                _, aligned_act = Processor._orthogonal_procrustes(anchor_act, curr_act, anchor="a")
                processed_act_dict[key] = aligned_act

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
            return x

        e, v = torch.linalg.eigh(x.T @ x / len(x))
        e = torch.clip(e, min=clip_eigs, max=None)
        d = alpha + (1 - alpha) * (e**-0.5)
        # From right to left, the transformation (1) projects x onto v, (2) divides by stdev in each
        # direction, and (3) rotates back to align with original directions in x-space (ZCA)
        z = v @ torch.diag(d) @ v.T
        # Think of this as (z @ x.T).T, but note z==z.T
        return x @ z

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
        return a @ r_a if r_a is not None else a, b @ r_b if r_b is not None else b

    @staticmethod
    def _pad_zeros(x, p):
        m, d = x.size()
        num_pad = p - d
        return torch.hstack([x.view(m, d), x.new_zeros(m, num_pad)])

    @staticmethod
    def _dim_reduce(x, p):
        # PCA to truncate -- project onto top p principal axes (no rescaling)
        with torch.no_grad():
            _, S, vT = torch.linalg.svd(x, full_matrices=False)

        # Fix sign ambiguity - make the component with largest magnitude positive
        for i in range(min(p, vT.shape[0])):
            max_idx = torch.argmax(torch.abs(vT[i]))
            if vT[i, max_idx] < 0:
                vT[i] *= -1

        # svd returns v.T, so the principal axes are in the *rows*. The following einsum is
        # equivalent to x @ vT.T[:, :p] but a bit faster because the transpose is not actually
        # performed.
        return torch.einsum("mn,pn->mp", x, vT[:p, :]), vT[:p, :]
