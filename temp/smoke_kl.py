import sys

sys.path.insert(0, "/home/nsa325/work/koopmann/scripts/tokae")
import train

original_init, original_finish = train.wandb.init, train.wandb.finish


def init(*args, **kwargs):
    run = original_init(*args, **kwargs)
    train.wandb.log = lambda logs: print({k: round(v, 4) for k, v in logs.items() if k.startswith(("loss/", "val/"))}, flush=True)
    return run


def finish(*args, **kwargs):
    print("summary", {k: round(v, 4) for k, v in dict(train.wandb.run.summary).items() if isinstance(v, (int, float))}, flush=True)
    return original_finish(*args, **kwargs)


train.wandb.init, train.wandb.finish = init, finish
train.main(
    dataset="imagenet", views=1, num_epochs=1, ae_dim=257, ae_hidden=256, jordan=True,
    lambda_variance=0, lambda_covariance=0, lambda_embedding=1, lambda_kl=1, name="smoke",
)
