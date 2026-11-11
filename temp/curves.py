import sys, wandb
from dotenv import load_dotenv
load_dotenv(".env")
api = wandb.Api()
for rid in sys.argv[1:]:
    run = api.run(f"nishantaswani/tokae/{rid}")
    keys = ["val/zero_shot_accuracy", "train/zero_shot_accuracy", "loss/kl", "loss/embedding", "loss/state", "val/kl_loss", "val/embedding_loss", "val/one_step_fvu", "train/one_step_fvu", "_runtime"]
    h = run.history(keys=keys, pandas=True, samples=1000)
    print(run.name, run.state, run.config.get("lr"), run.config.get("adam_lr"), run.config.get("num_epochs"), run.config.get("views"))
    print(h.iloc[[0,1,2,4,6,9,14,19,29,39,49,59] if len(h) > 59 else list(range(0, len(h), max(1, len(h)//12)))].round(4).to_string())
