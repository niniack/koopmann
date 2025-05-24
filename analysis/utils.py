import plotly.express as px

from koopmann.models import (
    MLP,
    ConvResNet,
    ExponentialKoopmanAutencoder,
    KoopmanAutoencoder,
    LowRankKoopmanAutoencoder,
    ResMLP,
)


def load_autoencoder(file_dir: str, ae_name: str):
    # Autoenoder path in work dir
    ae_file_path = f"{file_dir}/{ae_name}.safetensors"

    # Choose model based on flag
    if "standard" in ae_name:
        AutoencoderClass = KoopmanAutoencoder
    elif "lowrank" in ae_name:
        AutoencoderClass = LowRankKoopmanAutoencoder
    elif "exponential" in ae_name:
        AutoencoderClass = ExponentialKoopmanAutencoder

    autoencoder, ae_metadata = AutoencoderClass.load_model(file_path=ae_file_path)
    _ = autoencoder.eval()

    return autoencoder, ae_metadata


def load_model(file_dir: str, model_name: str) -> tuple:
    """Hooked and in eval mode."""
    # Original model path
    model_file_path = f"{file_dir}/{model_name}.safetensors"

    lower_model_name = model_name.lower()

    if "probed" in lower_model_name:
        model, model_metadata = MLP.load_model(file_path=model_file_path)
        model.modules[-2].remove_nonlinearity()
        model.modules[-3].remove_nonlinearity()
        # model.modules[-3].update_nonlinearity("leaky_relu")
    elif "resnet" in lower_model_name:
        model, model_metadata = ConvResNet.load_model(file_path=model_file_path)
    else:
        if "res" in lower_model_name:
            model, model_metadata = ResMLP.load_model(file_path=model_file_path)
        else:
            model, model_metadata = MLP.load_model(file_path=model_file_path)

    model.eval()

    return model, model_metadata


def imshow(x):
    fig = px.imshow(x.detach(), color_continuous_scale="balance_r", color_continuous_midpoint=0.0)
    fig.update_layout(coloraxis_showscale=False)
    fig.show()


def scatter(x, y, labels, z=None, colormap=None):
    palette = px.colors.qualitative.Plotly
    # colormap = {
    #     "0": palette[0],
    #     "1": palette[1],
    #     "2": palette[2],
    #     "3": palette[3],
    # }
    if z is not None:
        fig = px.scatter_3d(
            x=x,
            y=y,
            z=z,
            color=[str(label) for label in labels],
            color_discrete_map=colormap,
        )
    else:
        fig = px.scatter(
            x=x,
            y=y,
            color=[str(label) for label in labels],
            color_discrete_map=colormap,
        )
    fig.update_layout(showlegend=True, width=800)
    fig.update_traces(marker=dict(size=3, line=dict(width=0.0001, color="DarkSlateGrey")))
    fig.show()
