import numpy as np
import pandas as pd
t = pd.read_csv("phase_size_all.csv")
ALWAYS = {861, 338, 144, 560, 117, 988, 871, 746, 652, 533, 331, 77, 68, 653, 877, 465}
t["group"] = np.where(t["mode"].isin(ALWAYS), "always on", "feature")
for name in ["mean phase", "mean size", "removed"]:
    t[f"drop, {name}"] = 100 * (t["accuracy, base"] - t[f"accuracy, {name}"])
pd.set_option("display.width", 250)
print(t[t.group == "always on"][["mode", "|λ|", "angle°", "phase spread°", "size spread", "drop, mean phase", "drop, mean size", "drop, removed"]].round(2).to_string(index=False))
f = t[t.group == "feature"]
print(len(f), "feature modes kept on at least 250 images")
cols = ["drop, mean phase", "drop, mean size", "drop, removed", "changed, mean phase", "changed, mean size", "changed, removed", "kl, mean phase", "kl, mean size", "kl, removed"]
print(f[cols].describe().loc[["mean", "25%", "50%", "75%"]].round(3).to_string())
print("usage-weighted mean drops:", {c: round(np.average(f[c], weights=f["usage"]), 3) for c in cols[:3]})
print("modes where averaging the phase changes more answers than averaging the size:", int((f["changed, mean phase"] > f["changed, mean size"]).sum()), "of", len(f))
print("modes where averaging the phase moves the answer (KL) more than averaging the size:", int((f["kl, mean phase"] > f["kl, mean size"]).sum()))
ratio = f["kl, mean phase"] / f["kl, removed"]
print("KL of mean phase as a share of removal's: quartiles", ratio.quantile([.25, .5, .75]).round(3).tolist(), "| mean size:", (f["kl, mean size"] / f["kl, removed"]).quantile([.25, .5, .75]).round(3).tolist())
print("correlation of phase spread with the phase share:", round(np.corrcoef(f["phase spread°"], ratio)[0, 1], 3), "| of |λ| angle:", round(np.corrcoef(f["angle°"].abs(), ratio)[0, 1], 3), "| of |λ|:", round(np.corrcoef(f["|λ|"], ratio)[0, 1], 3))
for lo, hi in [(0, 20), (20, 40), (40, 60), (60, 200)]:
    g = f[(f["phase spread°"] >= lo) & (f["phase spread°"] < hi)]
    print(f"phase spread {lo}-{hi}°: {len(g)} modes, drop mean phase {g['drop, mean phase'].mean():.2f}, mean size {g['drop, mean size'].mean():.2f}, removed {g['drop, removed'].mean():.2f}; changed {g['changed, mean phase'].mean():.3f} / {g['changed, mean size'].mean():.3f} / {g['changed, removed'].mean():.3f}")
big = f[f["|λ|"] >= 5]
print("modes with |λ| ≥ 5:", len(big), "changed", big["changed, mean phase"].mean().round(3), big["changed, mean size"].mean().round(3), big["changed, removed"].mean().round(3), "| drops", big["drop, mean phase"].mean().round(2), big["drop, mean size"].mean().round(2), big["drop, removed"].mean().round(2))
print(f.sort_values("changed, mean phase", ascending=False).head(12)[["mode", "usage", "|λ|", "angle°", "phase spread°", "size spread", "changed, mean phase", "changed, mean size", "changed, removed", "drop, mean phase", "drop, mean size", "drop, removed"]].round(3).to_string(index=False))
