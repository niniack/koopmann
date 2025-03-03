import torch
from torcheval.metrics import MulticlassAccuracy

from koopmann.models import (
    MLP,
    Autoencoder,
    ExponentialKoopmanAutencoder,
    LowRankKoopmanAutoencoder,
)


@torch.no_grad()
def analyze_misclassification(model, autoencoder, x_misclassified, x_correct):
    # Get Koopman matrix and its eigendecomposition
    K = autoencoder.koopman_matrix.linear_layer.weight.T.detach()
    eigenvalues, eigenvectors = torch.linalg.eig(K)

    # Encode both inputs
    phi_mis = autoencoder._encode(x_misclassified)
    phi_cor = autoencoder._encode(x_correct)

    # Project difference onto eigenbasis
    diff = phi_mis - phi_cor
    projections = torch.abs(torch.einsum("i,ji->j", diff, eigenvectors))

    print(projections)

    # # Identify top modes contributing to misclassification
    # top_indices = torch.argsort(projections, descending=True)[:5]

    # for idx in top_indices:
    #     eigenvalue = eigenvalues[idx]
    #     contribution = projections[idx].item()

    #     print(f"Mode {idx}: contribution={contribution:.4f}, eigenvalue={eigenvalue}")

    # # Demonstrate impact by selectively removing this mode
    # corrected = modify_encoding(
    #     phi_mis, eigenvectors[:, idx], phi_cor.dot(eigenvectors[:, idx])
    # )
    # output = model(autoencoder._decode(corrected))
    # print(f"  Predicted class after correction: {output.argmax().item()}")


@torch.no_grad()
def compare_model_autoencoder_acc(
    model, autoencoder, k, num_classes, mlp_inputs, ae_inputs, labels
):
    # Feed to MLP
    mlp_metric = MulticlassAccuracy(average=None, num_classes=num_classes)
    mlp_output = model(mlp_inputs)
    mlp_metric.update(mlp_output, labels.squeeze())

    # Feed to autoencoder
    koopman_metric = MulticlassAccuracy(average=None, num_classes=num_classes)
    pred_act = autoencoder(ae_inputs, k=k).predictions[-1]
    koopman_output = model.modules[-2:](pred_act)
    koopman_metric.update(koopman_output, labels.squeeze())

    return mlp_metric.compute(), koopman_metric.compute()


@torch.no_grad()
def shrink_eigenvalues(matrix, shrink_factor=0.95, indices=None):
    # Ensure the matrix is square
    assert matrix.shape[0] == matrix.shape[1], "Input matrix must be square"

    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(matrix)
    n_eigenvalues = len(eigenvalues)

    # Create a copy of eigenvalues to modify
    shrunk_eigenvalues = eigenvalues.clone()

    if indices is not None:
        # Convert list to tensor if needed
        if isinstance(indices, list):
            indices = torch.tensor(indices)

        # Validate indices
        assert torch.all(
            (indices >= 0) & (indices < n_eigenvalues)
        ), f"All indices must be between 0 and {n_eigenvalues-1}"

        # Find the top eigenvalues by magnitude
        eig_magnitudes = torch.abs(eigenvalues)
        _, top_magnitude_indices = torch.topk(eig_magnitudes, n_eigenvalues)

        # Select the specified indices from the sorted top eigenvalues
        selected_indices = top_magnitude_indices[indices]

        print(indices)
        print("real", selected_indices)
        print(eigenvalues[selected_indices])

        # Shrink only the selected top eigenvalues
        shrunk_eigenvalues[selected_indices] = eigenvalues[selected_indices] * shrink_factor

    else:
        # Shrink all eigenvalues
        shrunk_eigenvalues = eigenvalues * shrink_factor

    # Reconstruct the matrix with shrunk eigenvalues
    # For a matrix A = PDP^(-1), where D is diagonal matrix of eigenvalues
    # and P is matrix of eigenvectors
    eigenvectors_inv = torch.linalg.inv(eigenvectors)
    diagonal_matrix = torch.diag(shrunk_eigenvalues)
    modified_matrix = eigenvectors @ diagonal_matrix @ eigenvectors_inv

    # Handle numerical issues - if the output should be real, remove small imaginary parts
    if torch.is_complex(matrix):
        return modified_matrix
    else:
        # Check if imaginary parts are negligible
        if torch.max(torch.abs(modified_matrix.imag)) < 1e-10:
            return modified_matrix.real, (
                eigenvalues[selected_indices],
                eigenvectors[selected_indices],
            )
        else:
            # If not negligible, there might be an issue
            print("Warning: Reconstructed matrix has non-negligible imaginary parts")
            return modified_matrix.real, (
                eigenvalues[selected_indices],
                eigenvectors[selected_indices],
            )
