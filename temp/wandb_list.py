"""List the project's runs with a few config keys and summary numbers (read only)."""
import wandb
from dotenv import load_dotenv

load_dotenv("/home/nsa325/work/koopmann/.env")
runs = wandb.Api().runs("nishantaswani/tokae", order="-created_at", per_page=200)
keys = ["dataset", "ae_dim", "top_k", "ae_hidden", "ae_layers", "jordan", "shared_modes", "tokens", "num_epochs", "weight_decay"]
for count, run in enumerate(runs):
    if count >= 120:
        break
    config = [str(run.config.get(key)) for key in keys]
    summary = run.summary
    numbers = [summary.get("best/val_accuracy"), summary.get("val/clip_agreement"), summary.get("val/one_step_fvu"), summary.get("val/reconstruction_fvu")]
    print(run.id, run.name[:28].ljust(28), run.state[:8].ljust(8), " ".join(config), "|", " ".join("-" if n is None else f"{n:.3f}" for n in numbers))
