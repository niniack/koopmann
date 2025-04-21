import torch


class Processor:
    @staticmethod
    def _whiten(x, alpha, clip_eigs=1e-9):
        """Compute (partial) whitening transform of x. When alpha=0 it is classic ZCA whitening and
        columns of x are totally decorrelated. When alpha=1, nothing happens.

        Assumes x is already centered.
        """
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
            _, _, vT = torch.linalg.svd(x, full_matrices=False)
        # svd returns v.T, so the principal axes are in the *rows*. The following einsum is
        # equivalent to x @ vT.T[:, :p] but a bit faster because the transpose is not actually
        # performed.
        return torch.einsum("mn,pn->mp", x, vT[:p, :])
