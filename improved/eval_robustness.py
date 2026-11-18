"""
eval_robustness.py — 저장된 모델을 깨끗한 test + 모든 열화×강도에서 평가

단일 모델:
    python eval_robustness.py --model runs/base_s0/best.keras --data data/input --tag base_s0
    (--clahe 는 학습 때 썼으면 평가 때도 반드시 켜야 함. config.json 이 옆에 있으면 자동으로 읽음)

여러 결과 비교 그림:
    python eval_robustness.py --compare results/base_s0.csv results/both_s0.csv --out results/compare.png

출력: results/<tag>.csv  (corruption, severity, accuracy, f1, auc, recall_normal, recall_pneumonia)
      results/<tag>.png  (강도별 정확도 곡선)
"""
import argparse, csv, json, os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from degrade import CORRUPTIONS, SEVERITIES, HOLDOUT_CORRUPTIONS, degrade, apply_clahe

CLASSES = ["NORMAL", "PNEUMONIA"]


def list_files(split_dir):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        d = os.path.join(split_dir, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(d, f)); labels.append(i)
    return paths, np.array(labels)


def load_gray(paths, img_size):
    out = np.empty((len(paths), img_size, img_size), dtype=np.uint8)
    for i, p in enumerate(paths):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        out[i] = cv2.resize(g, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return out


def to_batch(gray_imgs, clahe):
    if clahe:
        gray_imgs = np.stack([apply_clahe(g) for g in gray_imgs])
    rgb = np.repeat(gray_imgs[..., None], 3, axis=-1).astype(np.float32) / 255.0
    return rgb


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
        "recall_normal": cm[0, 0] / max(cm[0].sum(), 1),
        "recall_pneumonia": cm[1, 1] / max(cm[1].sum(), 1),
    }


def run_eval(a):
    cfg_path = os.path.join(os.path.dirname(a.model), "config.json")
    clahe, img_size = a.clahe, a.img_size
    if os.path.exists(cfg_path) and not a.ignore_config:
        cfg = json.load(open(cfg_path))
        clahe = cfg.get("clahe", clahe); img_size = cfg.get("img_size", img_size)
        print(f"config.json 사용: clahe={clahe}, img_size={img_size}")

    model = keras.models.load_model(a.model)
    paths, y = list_files(os.path.join(a.data, "test"))
    base = load_gray(paths, img_size)
    print(f"test {len(y)}장 로드")
    # val 도 같은 조건으로 평가 → analyze_threshold.py 가 val 에서 임계값을 고르고 test 에 적용
    vpaths, vy = list_files(os.path.join(a.data, "val"))
    vbase = load_gray(vpaths, img_size) if (len(vy) and not a.no_val) else None

    rows, probs = [], {"y_test": y, "y_val": vy}
    def record(name, s, prob, vprob=None):
        m = metrics(y, prob); m.update({"corruption": name, "severity": s,
                                        "holdout": name in HOLDOUT_CORRUPTIONS})
        rows.append(m)
        key = f"{name}_s{s}"
        probs[f"test_{key}"] = prob.astype(np.float32)
        if vprob is not None:
            probs[f"val_{key}"] = vprob.astype(np.float32)
        print(f"{name:<16} s={s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  "
              f"rN={m['recall_normal']:.3f} rP={m['recall_pneumonia']:.3f}")

    def run(imgs):
        return model.predict(to_batch(imgs, clahe), batch_size=a.batch, verbose=0)[:, 1]

    record("clean", 0, run(base), run(vbase) if vbase is not None else None)

    for name in CORRUPTIONS:
        for s in SEVERITIES:
            np.random.seed(a.seed + s)   # 노이즈/모션 각도 재현성
            deg = np.stack([degrade(g, name, s) for g in base])
            vdeg = np.stack([degrade(g, name, s) for g in vbase]) if vbase is not None else None
            record(name, s, run(deg), run(vdeg) if vdeg is not None else None)

    os.makedirs(a.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(a.out_dir, f"{a.tag}_probs.npz"), **probs)
    csv_path = os.path.join(a.out_dir, f"{a.tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "severity", "holdout", "accuracy", "f1", "auc",
                                          "recall_normal", "recall_pneumonia"])
        w.writeheader(); w.writerows(rows)

    # 요약: 평균 열화 정확도 (seen / holdout 분리)
    seen = [r["accuracy"] for r in rows if r["severity"] > 0 and not r["holdout"]]
    hold = [r["accuracy"] for r in rows if r["severity"] > 0 and r["holdout"]]
    summary = {"clean_acc": rows[0]["accuracy"], "mean_acc_seen_corruptions": float(np.mean(seen)),
               "mean_acc_holdout_corruptions": float(np.mean(hold)),
               "mean_acc_all_corruptions": float(np.mean(seen + hold))}
    json.dump(summary, open(os.path.join(a.out_dir, f"{a.tag}_summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:<30}{v:.4f}")

    plot_single(rows, os.path.join(a.out_dir, f"{a.tag}.png"), a.tag)
    print("saved →", csv_path)


def plot_single(rows, out_png, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    clean = [r for r in rows if r["severity"] == 0][0]["accuracy"]
    plt.figure(figsize=(7, 4.5))
    for name in CORRUPTIONS:
        xs = [r["severity"] for r in rows if r["corruption"] == name]
        ys = [r["accuracy"] for r in rows if r["corruption"] == name]
        ls = "--" if name in HOLDOUT_CORRUPTIONS else "-"
        plt.plot([0] + xs, [clean] + ys, ls, marker="o", label=name + (" (holdout)" if ls == "--" else ""))
    plt.axhline(clean, color="gray", lw=0.8, ls=":")
    plt.xlabel("severity"); plt.ylabel("accuracy"); plt.ylim(0.4, 1.0); plt.xticks(range(6))
    plt.title(title); plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()


def run_compare(csvs, out_png):
    """여러 모델의 CSV를 읽어 열화 종류별 subplot 으로 비교."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {}
    for p in csvs:
        tag = os.path.splitext(os.path.basename(p))[0]
        with open(p) as f:
            data[tag] = list(csv.DictReader(f))
    names = list(CORRUPTIONS)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), sharey=True)
    for ax, name in zip(axes, names):
        for tag, rows in data.items():
            clean = float([r for r in rows if r["corruption"] == "clean"][0]["accuracy"])
            pts = sorted(((int(r["severity"]), float(r["accuracy"])) for r in rows if r["corruption"] == name))
            ax.plot([0] + [s for s, _ in pts], [clean] + [acc for _, acc in pts], marker="o", ms=3, label=tag)
        ax.set_title(name + ("\n(holdout)" if name in HOLDOUT_CORRUPTIONS else ""), fontsize=10)
        ax.set_xlabel("severity"); ax.set_xticks(range(6)); ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy"); axes[0].set_ylim(0.4, 1.0); axes[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print("saved →", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data", default="data/input")
    ap.add_argument("--tag", default="model")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--img-size", type=int, default=150)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--ignore-config", action="store_true")
    ap.add_argument("--no-val", action="store_true", help="val 평가 생략 (임계값 분석 불가)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--compare", nargs="+", help="비교할 결과 CSV 들")
    ap.add_argument("--out", default="results/compare.png")
    a = ap.parse_args()
    if a.compare:
        run_compare(a.compare, a.out)
    else:
        if not a.model:
            ap.error("--model 또는 --compare 중 하나는 필요")
        run_eval(a)
