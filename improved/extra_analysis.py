"""추가 분석 (재학습 없음) — 이미 학습된 모델과 저장된 예측 확률만 사용한다.

  extra   : 학습에 쓰지 않은 열화 5종(스페클·샷 노이즈, 초점 흐림, 해상도 저하, 밝기)을 추가로 평가
            → results/extra_<run>.csv, extra_<run>_probs.npz, extra_summary.txt
  stats   : 베이스라인 vs 최종 모델의 정확도 차이에 대한 통계 검정 (같은 seed 끼리 짝지음)
            - 영상 단위 paired bootstrap 95% 신뢰구간 (조건 여러 개를 묶어도 같은 영상끼리의 상관을 보존)
            - 깨끗한 test 에 대한 McNemar 정확 검정
            → results/stats_summary.txt
  gradcam : 같은 영상의 깨끗한/열화 버전에서 두 모델이 보는 부위 (Grad-CAM) → results/gradcam.png

사용:
  python extra_analysis.py --data data/input --runs runs --results results            # 전부
  python extra_analysis.py --data data/input --runs runs --results results --parts stats
끝난 extra 평가는 건너뛰므로 중간에 끊겨도 다시 실행하면 이어서 돈다.
"""
import argparse, csv, json, os, time
from math import comb
import numpy as np

from degrade import (CORRUPTIONS, EXTRA_CORRUPTIONS, HOLDOUT_CORRUPTIONS, SEVERITIES, TRAIN_CORRUPTIONS,
                     degrade, degrade_extra)

TAGS = ["base_v2", "clahe_v2", "aug_v2", "both_v2"]
NAMES = {"base_v2": "베이스라인", "clahe_v2": "CLAHE만", "aug_v2": "열화 증강만", "both_v2": "CLAHE+열화 증강"}


def out(lines, path):
    text = "\n".join(lines)
    print(text); open(path, "w").write(text + "\n")
    print(f"저장 → {path}")


def find_runs(runs_dir, tags):
    found = []
    for t in tags:
        for s in range(10):
            d = os.path.join(runs_dir, f"{t}_s{s}")
            if os.path.exists(os.path.join(d, "best.keras")):
                found.append((t, s, d))
    return found


# ── extra ────────────────────────────────────────────────────────────────────
def part_extra(a):
    from tensorflow import keras
    from eval_robustness import list_files, load_gray, to_batch
    runs = find_runs(a.runs, TAGS)
    print(f"[extra] 대상 모델 {len(runs)}개, 추가 열화 {len(EXTRA_CORRUPTIONS)}종 × 강도 {len(SEVERITIES)}단계")
    paths, y = list_files(os.path.join(a.data, "test"))
    cache = {}
    for i, (tag, seed, d) in enumerate(runs):
        run = f"{tag}_s{seed}"
        csv_path = os.path.join(a.results, f"extra_{run}.csv")
        npz_path = os.path.join(a.results, f"extra_{run}_probs.npz")
        if os.path.exists(csv_path) and os.path.exists(npz_path):
            print(f"  [{i + 1}/{len(runs)}] {run} — 이미 완료, 건너뜀"); continue
        t0 = time.time()
        cfg = json.load(open(os.path.join(d, "config.json")))
        clahe, size = bool(cfg.get("clahe", False)), int(cfg.get("img_size", 150))
        if size not in cache:
            cache[size] = load_gray(paths, size)
        base = cache[size]
        model = keras.models.load_model(os.path.join(d, "best.keras"))
        rows, probs = [], {"y_test": y}
        for name in EXTRA_CORRUPTIONS:
            for s in SEVERITIES:
                np.random.seed(s)                     # 모든 모델이 같은 열화 영상을 보도록
                deg = np.stack([degrade_extra(g, name, s) for g in base])
                p = model.predict(to_batch(deg, clahe), batch_size=64, verbose=0)[:, 1]
                probs[f"test_{name}_s{s}"] = p.astype(np.float32)
                rows.append({"corruption": name, "severity": s, "accuracy": float(((p >= 0.5).astype(int) == y).mean())})
        np.savez_compressed(npz_path, **probs)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "severity", "accuracy"]); w.writeheader(); w.writerows(rows)
        print(f"  [{i + 1}/{len(runs)}] {run} 완료 ({time.time() - t0:.0f}초)  평균 정확도 {np.mean([r['accuracy'] for r in rows]) * 100:.1f}%", flush=True)

    # 요약
    acc = {}
    for tag, seed, _ in runs:
        p = os.path.join(a.results, f"extra_{tag}_s{seed}.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                acc.setdefault((tag, r["corruption"], int(r["severity"])), []).append(float(r["accuracy"]) * 100)
    names = list(EXTRA_CORRUPTIONS)
    L = ["", "=== 추가 열화(학습에 쓰지 않음) 정확도 %, seed 평균 ± 표준편차, 강도 1~5 평균 ===",
         f"{'모델':<18}{'seeds':>6}" + "".join(f"{n:>18}" for n in names) + f"{'5종 평균':>16}"]
    for tag in TAGS:
        n_seed = len(acc.get((tag, names[0], 1), []))
        if not n_seed:
            continue
        cells, per_seed_all = [], np.zeros(n_seed)
        for n in names:
            per_seed = np.mean([acc[(tag, n, s)] for s in SEVERITIES], axis=0)   # seed 별 강도 평균
            per_seed_all += per_seed / len(names)
            cells.append(f"{per_seed.mean():>11.1f} ± {per_seed.std():>4.1f}")
        L.append(f"{NAMES[tag]:<18}{n_seed:>6}" + "".join(cells) + f"{per_seed_all.mean():>9.1f} ± {per_seed_all.std():>4.1f}")
    L += ["", "=== 강도별 정확도 % (seed 평균) ==="]
    for n in names:
        L.append(f"[{n}]")
        for tag in TAGS:
            if (tag, n, 1) in acc:
                L.append(f"  {NAMES[tag]:<18}" + "".join(f"  s{s}: {np.mean(acc[(tag, n, s)]):5.1f}" for s in SEVERITIES))
    out(L, os.path.join(a.results, "extra_summary.txt"))


# ── stats ────────────────────────────────────────────────────────────────────
def mcnemar_exact(b, c):
    """b: A만 맞힘, c: B만 맞힘. 양측 정확 이항 검정 p값."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def part_stats(a):
    groups = {"깨끗한 test": [("clean", 0)],
              "학습에 쓴 열화(15조건)": [(n, s) for n in TRAIN_CORRUPTIONS for s in SEVERITIES],
              "학습에 안 쓴 열화: JPEG·모션블러(10조건)": [(n, s) for n in HOLDOUT_CORRUPTIONS for s in SEVERITIES],
              "추가 열화 5종(25조건)": [(n, s) for n in EXTRA_CORRUPTIONS for s in SEVERITIES]}
    rng = np.random.default_rng(0)
    L = ["=== 베이스라인 → CLAHE+열화 증강: 정확도 차이 (%p) ===",
         "같은 seed 의 두 모델을 같은 test 영상으로 비교. 신뢰구간은 영상을 복원추출한 paired bootstrap (10,000회).",
         f"{'조건 묶음':<40}{'seed':>5}{'베이스':>9}{'최종':>9}{'차이':>8}   95% 신뢰구간"]
    pooled = {g: [] for g in groups}
    for seed in range(10):
        def load(tag):
            p1 = os.path.join(a.results, f"{tag}_s{seed}_probs.npz")
            if not os.path.exists(p1):
                return None
            d = dict(np.load(p1))
            p2 = os.path.join(a.results, f"extra_{tag}_s{seed}_probs.npz")
            if os.path.exists(p2):
                d.update({k: v for k, v in np.load(p2).items() if k != "y_test"})
            return d
        A, B = load("base_v2"), load("both_v2")
        if A is None or B is None:
            continue
        y = A["y_test"]
        for g, conds in groups.items():
            keys = [f"test_{n}_s{s}" for n, s in conds]
            if not all(k in A and k in B for k in keys):
                continue
            ca = np.stack([(A[k] >= 0.5).astype(int) == y for k in keys]).mean(0)   # 영상별 정답률
            cb = np.stack([(B[k] >= 0.5).astype(int) == y for k in keys]).mean(0)
            d = cb - ca
            idx = rng.integers(0, len(y), size=(10000, len(y)))
            boot = d[idx].mean(1) * 100
            lo, hi = np.percentile(boot, [2.5, 97.5])
            pooled[g].append(d)
            extra = ""
            if g == "깨끗한 test":
                b_, c_ = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
                extra = f"   McNemar: 베이스만 정답 {b_}, 최종만 정답 {c_}, p={mcnemar_exact(b_, c_):.3f}"
            L.append(f"{g:<40}{seed:>5}{ca.mean() * 100:>9.1f}{cb.mean() * 100:>9.1f}{d.mean() * 100:>+8.1f}   [{lo:+.1f}, {hi:+.1f}]"
                     + ("  ← 0 포함(차이 불확실)" if lo <= 0 <= hi else "") + extra)
    L += ["", "=== seed 를 합친 결과 (영상별 차이를 seed 평균한 뒤 bootstrap) ==="]
    for g, ds in pooled.items():
        if ds:
            d = np.mean(ds, axis=0)
            idx = rng.integers(0, len(d), size=(10000, len(d)))
            lo, hi = np.percentile(d[idx].mean(1) * 100, [2.5, 97.5])
            L.append(f"{g:<40} seed {len(ds)}개  차이 {d.mean() * 100:+.1f}%p   95% CI [{lo:+.1f}, {hi:+.1f}]"
                     + ("  ← 0 포함" if lo <= 0 <= hi else ""))
    L += ["", "주의: bootstrap 은 '이 624장이 아닌 다른 영상을 뽑았다면' 의 불확실성만 반영한다.",
          "      학습을 다시 했을 때의 변동(seed 간 차이)은 seed 별 줄을 비교해서 본다."]
    out(L, os.path.join(a.results, "stats_summary.txt"))


# ── gradcam ──────────────────────────────────────────────────────────────────
def part_gradcam(a):
    import cv2, tensorflow as tf
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorflow import keras
    from eval_robustness import CLASSES, list_files, load_gray, to_batch
    conds = [("clean", 0, "clean"), ("gaussian_noise", 4, "noise s4"), ("contrast", 3, "contrast s3"),
             ("motion_blur", 4, "motion blur s4")]
    models = []
    for tag, short in [("base_v2", "baseline"), ("both_v2", "ours")]:
        d = os.path.join(a.runs, f"{tag}_s{a.seed}")
        cfg = json.load(open(os.path.join(d, "config.json")))
        m = keras.models.load_model(os.path.join(d, "best.keras"))
        conv = [l for l in m.layers if isinstance(l, keras.layers.Conv2D)][-1]
        size_ = int(cfg.get("img_size", 150))
        x = inp = keras.Input((size_, size_, 3)); fmap = None      # Sequential 모델을 층 단위로 다시 연결 (가중치 공유)
        for l in m.layers:
            x = l(x)
            if l is conv:
                fmap = x
        g = keras.Model(inp, [fmap, x])
        models.append((short, g, bool(cfg.get("clahe", False)), int(cfg.get("img_size", 150))))
    size = models[0][3]
    paths, y = list_files(os.path.join(a.data, "test"))
    rng = np.random.default_rng(a.pick_seed)
    pick = np.concatenate([rng.choice(np.where(y == 0)[0], 2, replace=False), rng.choice(np.where(y == 1)[0], 2, replace=False)])
    imgs = load_gray([paths[i] for i in pick], size)

    def cam(gm, x):
        x = tf.convert_to_tensor(x)
        with tf.GradientTape() as tape:
            fmap, pred = gm(x, training=False)
            cls = tf.argmax(pred, axis=1)
            score = tf.gather(pred, cls, axis=1, batch_dims=1)
        grads = tape.gradient(score, fmap)
        w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        h = tf.nn.relu(tf.reduce_sum(w * fmap, axis=-1)).numpy()
        h = h / (h.max(axis=(1, 2), keepdims=True) + 1e-8)
        return h, pred.numpy()[:, 1]

    fig, ax = plt.subplots(len(pick) * 2, len(conds), figsize=(3.2 * len(conds), 3.3 * len(pick) * 2))
    for c, (name, s, eng) in enumerate(conds):
        np.random.seed(s)
        deg = imgs if s == 0 else np.stack([degrade(g, name, s) for g in imgs])
        for mi, (short, gm, clahe, _) in enumerate(models):
            h, p = cam(gm, to_batch(deg, clahe))
            for r in range(len(pick)):
                A = ax[r * 2 + mi, c]; truth = int(y[pick[r]]); pred = int(p[r] >= 0.5)
                A.imshow(deg[r], cmap="gray", vmin=0, vmax=255)
                A.imshow(cv2.resize(h[r], (size, size)), cmap="jet", alpha=0.4, vmin=0, vmax=1); A.axis("off")
                A.set_title((eng + "\n" if r == 0 and mi == 0 else "") +
                            f"{short}: {CLASSES[pred]} (P={p[r]:.2f}) {'OK' if pred == truth else 'WRONG'} | truth {CLASSES[truth]}",
                            fontsize=8, loc="left")
    fig.tight_layout(); path = os.path.join(a.results, "gradcam.png"); fig.savefig(path, dpi=110); plt.close(fig)
    print(f"[gradcam] 저장 → {path}  (영상 {len(pick)}장 × 조건 {len(conds)}개 × 모델 2개, seed {a.seed} 모델)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input"); ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--parts", default="extra,stats,gradcam")
    ap.add_argument("--seed", type=int, default=0, help="gradcam 에 쓸 모델 seed")
    ap.add_argument("--pick-seed", type=int, default=7, help="gradcam 예시 영상을 고르는 난수 시드")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    for part in a.parts.split(","):
        print("=" * 72 + f"\n[{part}]", flush=True)
        try:
            {"extra": part_extra, "stats": part_stats, "gradcam": part_gradcam}[part](a)
        except Exception as e:            # 한 단계가 실패해도 나머지는 계속
            import traceback; traceback.print_exc()
            print(f"[{part}] 실패: {e} — 다음 단계로 넘어감")


if __name__ == "__main__":
    main()
