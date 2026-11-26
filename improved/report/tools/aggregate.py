import csv, collections, numpy as np, json
HOLD = {"jpeg", "motion_blur"}; SEEN = {"gaussian_noise", "gaussian_blur", "contrast"}
R = list(csv.DictReader(open("per_condition.csv")))
runs = collections.defaultdict(dict)
for r in R: runs[(r["model"], int(r["seed"]))][(r["corruption"], int(r["severity"]))] = float(r["accuracy"])
summ = collections.defaultdict(list); percorr = collections.defaultdict(lambda: collections.defaultdict(list))
for (m, s), d in sorted(runs.items()):
    clean = d[("clean", 0)]
    seen = np.mean([v for (c, sv), v in d.items() if c in SEEN]); hold = np.mean([v for (c, sv), v in d.items() if c in HOLD])
    summ[m].append((clean, seen, hold))
    for c in list(SEEN) + list(HOLD):
        percorr[m][c].append(np.mean([v for (cc, sv), v in d.items() if cc == c]))
out = {}
print(f"{'model':<18}{'clean':>14}{'seen':>14}{'holdout':>14}")
for m, v in summ.items():
    a = np.array(v) * 100
    print(f"{m:<18}" + "".join(f"{a[:,i].mean():8.1f} ±{a[:,i].std():4.1f}" for i in range(3)))
    out[m] = {"n": len(v), "clean": [a[:,0].mean(), a[:,0].std()], "seen": [a[:,1].mean(), a[:,1].std()], "holdout": [a[:,2].mean(), a[:,2].std()],
              "per_corruption": {c: [float(np.mean(x) * 100), float(np.std(x) * 100)] for c, x in percorr[m].items()}}
print()
print(f"{'model':<18}" + "".join(f"{c:>16}" for c in ["gaussian_noise", "gaussian_blur", "contrast", "jpeg", "motion_blur"]))
for m in summ:
    print(f"{m:<18}" + "".join(f"{out[m]['per_corruption'][c][0]:10.1f} ±{out[m]['per_corruption'][c][1]:4.1f}" for c in ["gaussian_noise", "gaussian_blur", "contrast", "jpeg", "motion_blur"]))
json.dump(out, open("agg.json", "w"), indent=1)
