"""
analyze_threshold.py — "열화 시 붕괴 = 정보 손실인가, 보정(임계값) 붕괴인가" 를 분리하는 분석

배경 (1차 실험): 열화로 정확도가 62.5% 까지 떨어진 조건에서도 AUC 는 0.85~0.95 로 남아 있었고,
NORMAL 재현율만 0 으로 갔다. 즉 순위는 유지되고 예측 확률이 PNEUMONIA 쪽으로 밀렸다는 뜻.
그렇다면 임계값만 다시 잡아도 상당 부분 회복되어야 한다. 이 스크립트가 그걸 수치로 확인한다.

입력: eval_robustness.py / train_lora_vit.py 가 저장한 results/<tag>_probs.npz
      (test_<cond>, val_<cond>, y_test, y_val)

조건별로 세 가지 정확도를 비교:
  acc@0.5        : 기본 임계값 (보고된 값)
  acc@val-thr    : 같은 열화를 가한 val 에서 균형정확도 최대 임계값을 고른 뒤 test 에 적용 (현실적으로 가능한 복구)
  acc@oracle-thr : test 에서 직접 고른 최적 임계값 (상한, 참고용)

출력: results/threshold_<tag>.csv, results/threshold_<tag>.png (조건별 세 정확도 + 확률 히스토그램)

사용:
  python analyze_threshold.py --probs results/base_s0_probs.npz --tag base_s0
  python analyze_threshold.py --probs results/base_s0_probs.npz results/both_s0_probs.npz --tag cmp
"""
import argparse, csv, os
import numpy as np
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score

from degrade import CORRUPTIONS, SEVERITIES, HOLDOUT_CORRUPTIONS

THRS = np.linspace(0.02, 0.98, 97)


def best_thr(y, p):
    # 후보 = 고정 격자 + 예측 확률 자체의 분위수.
    # 열화가 심하면 확률이 0.98 이상에 몰리므로(noise s4: NORMAL 평균 0.98) 고정 격자만으로는 상한에 걸린다.
    cand = np.unique(np.concatenate([THRS, np.quantile(p, np.linspace(0.005, 0.995, 199))]))
    scores = [balanced_accuracy_score(y, p >= t) for t in cand]
    return float(cand[int(np.argmax(scores))])


def analyze(npz_path):
    d = np.load(npz_path)
    y, vy = d["y_test"], d["y_val"]
    has_val = "val_clean_s0" in d
    rows = []
    for name in ["clean"] + list(CORRUPTIONS):
        for s in ([0] if name == "clean" else SEVERITIES):
            key = f"{name}_s{s}"
            p = d[f"test_{key}"]
            r = {"corruption": name, "severity": s, "holdout": name in HOLDOUT_CORRUPTIONS,
                 "auc": roc_auc_score(y, p), "acc_05": accuracy_score(y, p >= 0.5),
                 "mean_prob": float(p.mean()), "mean_prob_normal": float(p[y == 0].mean()),
                 "rn_05": float(((p >= 0.5) == 0)[y == 0].mean())}
            t_or = best_thr(y, p)
            r["thr_oracle"] = t_or; r["acc_oracle"] = accuracy_score(y, p >= t_or)
            if has_val:
                t_v = best_thr(vy, d[f"val_{key}"])
                r["thr_val"] = t_v; r["acc_valthr"] = accuracy_score(y, p >= t_v)
                r["rn_valthr"] = float(((p >= t_v) == 0)[y == 0].mean())
            rows.append(r)
    return rows, d, y


def write_csv(rows, path):
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def plot(all_rows, npzs, out_png):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(CORRUPTIONS)
    n = len(all_rows)
    fig, axes = plt.subplots(n, len(names) + 1, figsize=(3.1 * (len(names) + 1), 3.3 * n), squeeze=False)
    for i, (tag, rows) in enumerate(all_rows.items()):
        clean = [r for r in rows if r["corruption"] == "clean"][0]
        for j, name in enumerate(names):
            ax = axes[i][j]
            rs = sorted([r for r in rows if r["corruption"] == name], key=lambda r: r["severity"])
            xs = [0] + [r["severity"] for r in rs]
            ax.plot(xs, [clean["acc_05"]] + [r["acc_05"] for r in rs], "o-", color="0.55", ms=3, label="acc @0.5")
            if "acc_valthr" in rs[0]:
                ax.plot(xs, [clean["acc_valthr"]] + [r["acc_valthr"] for r in rs], "o-", color="C0", ms=3, label="acc @val-thr")
            ax.plot(xs, [clean["acc_oracle"]] + [r["acc_oracle"] for r in rs], "--", color="0.3", lw=0.8, label="acc @oracle-thr")
            ax.plot(xs, [clean["auc"]] + [r["auc"] for r in rs], ":", color="C3", lw=1, label="AUC")
            ax.set_ylim(0.4, 1.0); ax.set_xticks(range(6)); ax.grid(alpha=0.3)
            ax.set_title(f"{tag} · {name}" + (" (holdout)" if name in HOLDOUT_CORRUPTIONS else ""), fontsize=9)
            if j == 0: ax.set_ylabel("accuracy / AUC")
            if i == 0 and j == 0: ax.legend(fontsize=7)
        # 확률 히스토그램: 깨끗한 영상 vs 대비 저하 강도 3 (NORMAL 만)
        d = npzs[tag]; y = d["y_test"]
        ax = axes[i][len(names)]
        ax.hist(d["test_clean_s0"][y == 0], bins=25, range=(0, 1), alpha=0.6, color="0.5", label="clean · NORMAL")
        ax.hist(d["test_contrast_s3"][y == 0], bins=25, range=(0, 1), alpha=0.6, color="C0", label="contrast s3 · NORMAL")
        ax.axvline(0.5, color="k", lw=0.8, ls="--")
        ax.set_title(f"{tag} · P(pneumonia) of NORMAL images", fontsize=9); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(out_png, dpi=140); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", nargs="+", required=True)
    ap.add_argument("--tag", default="threshold")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    all_rows, npzs = {}, {}
    for p in a.probs:
        if not os.path.exists(p):
            print(f"[건너뜀] {p} 없음 — eval_robustness.py 를 먼저 실행해야 함"); continue
        tag = os.path.basename(p).replace("_probs.npz", "")
        rows, d, y = analyze(p)
        all_rows[tag], npzs[tag] = rows, d
        write_csv(rows, os.path.join(a.out_dir, f"threshold_{tag}.csv"))
        seen = [r for r in rows if r["severity"] > 0 and not r["holdout"]]
        hold = [r for r in rows if r["severity"] > 0 and r["holdout"]]
        def m(rs, k): return float(np.mean([r[k] for r in rs])) if k in rs[0] else float("nan")
        print(f"\n=== {tag} ===")
        print(f"{'':<22}{'acc@0.5':>9}{'acc@val-thr':>13}{'acc@oracle':>12}{'AUC':>8}")
        for lab, rs in [("clean", [rows[0]]), ("seen corruptions", seen), ("holdout corruptions", hold)]:
            print(f"{lab:<22}{m(rs,'acc_05'):>9.4f}{m(rs,'acc_valthr'):>13.4f}{m(rs,'acc_oracle'):>12.4f}{m(rs,'auc'):>8.4f}")
        worst = min(rows, key=lambda r: r["acc_05"])
        print(f"최악 조건 {worst['corruption']} s{worst['severity']}: acc@0.5 {worst['acc_05']:.3f} → "
              f"val-thr {worst.get('acc_valthr', float('nan')):.3f} (thr {worst.get('thr_val', float('nan')):.4f}), "
              f"oracle {worst['acc_oracle']:.3f} | NORMAL 평균 P(pneu) {worst['mean_prob_normal']:.2f} (clean {rows[0]['mean_prob_normal']:.2f})")
    if not all_rows:
        raise SystemExit("분석할 파일이 없습니다.")
    plot(all_rows, npzs, os.path.join(a.out_dir, f"threshold_{a.tag}.png"))
    print("saved →", os.path.join(a.out_dir, f"threshold_{a.tag}.png"))
