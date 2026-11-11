import fire
def main(lambda_kl: float = 0.0, weight_decay: float = 0.0):
    print("kl", lambda_kl, "wd", weight_decay)
fire.Fire(main)
