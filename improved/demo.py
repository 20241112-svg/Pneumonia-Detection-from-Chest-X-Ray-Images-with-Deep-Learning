"""시연용 스크립트 — 학습된 두 모델(베이스라인 / CLAHE+열화 증강)을 불러와 실제로 동작하는지 보여준다.

  1) 열화 예시 그림 (X-ray 한 장 × 열화 5종 × 강도 0~5)
  2) 같은 영상의 깨끗한 버전·열화 버전에 대한 두 모델의 예측 비교
  3) test 전체(624장)에 몇 개 조건을 적용해 두 모델의 정확도를 계산
  4) 전체 실험 요약표 출력 (results/summary_table.txt 가 있으면)

사용:
  python demo.py --data data/input --runs runs --results results --out demo_out
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from degrade import CORRUPTIONS, degrade
from eval_robustness import CLASSES, list_files, load_gray, to_batch

# (열화 이름, 강도, 그림에 쓸 영어 라벨)
DEMO_CONDITIONS = [("clean", 0, "clean"), ("gaussian_noise", 4, "noise s4"),
                   ("contrast", 3, "contrast s3"), ("motion_blur", 4, "motion blur s4 (unseen)")]


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    model = keras.models.load_model(os.path.join(run_dir, "best.keras"))
    return model, bool(cfg.get("clahe", False)), int(cfg.get("img_size", 150))


def apply(imgs, name, s, seed=0):
    """eval_robustness.py 와 같은 시드 규칙 → 같은 조건이면 같은 열화 영상."""
    if s == 0:
        return imgs
    np.random.seed(seed + s)
    return np.stack([degrade(g, name, s) for g in imgs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="test/NORMAL, test/PNEUMONIA 가 들어 있는 폴더")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--results", default="results")
    ap.add_argument("--base", default="base_v2_s0", help="베이스라인 실행 이름")
    ap.add_argument("--ours", default="both_v2_s0", help="최종 모델 실행 이름")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7, help="예시 영상을 고르는 난수 시드")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    print("=" * 70 + "\n[0] 학습된 모델 불러오기")
    models = {}
    for label, tag in [("베이스라인", a.base), ("CLAHE+열화 증강", a.ours)]:
        m, clahe, size = load(os.path.join(a.runs, tag))
        models[label] = (m, clahe, size)
        print(f"  {label:<14} ← {a.runs}/{tag}/best.keras  (파라미터 {m.count_params():,}개, CLAHE={clahe}, 입력 {size}px)")
    size = next(iter(models.values()))[2]

    paths, y = list_files(os.path.join(a.data, "test"))
    imgs = load_gray(paths, size)
    print(f"  test 영상 {len(y)}장 (NORMAL {int((y == 0).sum())}, PNEUMONIA {int((y == 1).sum())})")

    def predict(label, batch):
        m, clahe, _ = models[label]
        return m.predict(to_batch(batch, clahe), batch_size=64, verbose=0)[:, 1]

    # ── 1) 열화 예시 ─────────────────────────────────────────────────────────
    print("=" * 70 + "\n[1] 열화 예시 그림")
    g = imgs[int(np.where(y == 1)[0][0])]
    names = list(CORRUPTIONS)
    fig, ax = plt.subplots(len(names), 6, figsize=(12, 2 * len(names)))
    for r, name in enumerate(names):
        for s in range(6):
            np.random.seed(s)
            ax[r, s].imshow(degrade(g, name, s), cmap="gray", vmin=0, vmax=255); ax[r, s].axis("off")
            if r == 0:
                ax[r, s].set_title("original" if s == 0 else f"severity {s}", fontsize=10)
        ax[r, 0].text(-0.08, 0.5, name, transform=ax[r, 0].transAxes, ha="right", va="center", fontsize=10)
    fig.tight_layout(); p1 = os.path.join(a.out, "demo_1_degradations.png"); fig.savefig(p1, dpi=110); plt.close(fig)
    print(f"  저장 → {p1}")

    # ── 2) 예측 비교 ─────────────────────────────────────────────────────────
    print("=" * 70 + "\n[2] 같은 영상에 대한 두 모델의 예측 (P = 모델이 낸 폐렴 확률, 0.5 이상이면 폐렴)")
    rng = np.random.default_rng(a.seed)
    half = a.n_examples // 2
    pick = np.concatenate([rng.choice(np.where(y == 0)[0], half, replace=False),
                           rng.choice(np.where(y == 1)[0], a.n_examples - half, replace=False)])
    print(f"  예시 영상은 시드 {a.seed} 로 무작위 선택 (NORMAL {half}장, PNEUMONIA {a.n_examples - half}장)")
    fig, ax = plt.subplots(len(pick), len(DEMO_CONDITIONS), figsize=(3.4 * len(DEMO_CONDITIONS), 3.5 * len(pick)))
    right = {k: 0 for k in models}; total = 0
    for c, (name, s, eng) in enumerate(DEMO_CONDITIONS):
        deg = apply(imgs[pick], name, s)
        pr = {k: predict(k, deg) for k in models}
        for r, idx in enumerate(pick):
            truth = int(y[idx]); total += 1
            lines = [f"truth: {CLASSES[truth]}"]
            row = f"  {os.path.basename(paths[idx])[:28]:<28} {eng:<24} 정답 {CLASSES[truth]:<9}"
            for k, short in zip(models, ["baseline", "ours"]):
                pred = int(pr[k][r] >= 0.5); ok = pred == truth; right[k] += ok
                lines.append(f"{short}: {CLASSES[pred]} (P={pr[k][r]:.2f}) {'OK' if ok else 'WRONG'}")
                row += f" | {k} → {CLASSES[pred]:<9} P={pr[k][r]:.2f} {'O' if ok else 'X'}"
            print(row)
            ax[r, c].imshow(deg[r], cmap="gray", vmin=0, vmax=255); ax[r, c].axis("off")
            ax[r, c].set_title((eng + "\n" if r == 0 else "") + "\n".join(lines), fontsize=8.5, loc="left")
    fig.tight_layout(); p2 = os.path.join(a.out, "demo_2_predictions.png"); fig.savefig(p2, dpi=110); plt.close(fig)
    print("  이 예시에서 맞힌 수: " + ", ".join(f"{k} {v}/{total}" for k, v in right.items()))
    print(f"  저장 → {p2}")

    # ── 3) test 전체 정확도 ──────────────────────────────────────────────────
    print("=" * 70 + f"\n[3] test {len(y)}장 전체에 열화를 적용해 정확도 계산")
    print(f"  {'조건':<26}" + "".join(f"{k:>18}" for k in models))
    for name, s, eng in DEMO_CONDITIONS:
        t0 = time.time(); deg = apply(imgs, name, s)
        accs = [float(((predict(k, deg) >= 0.5).astype(int) == y).mean()) for k in models]
        print(f"  {eng:<26}" + "".join(f"{v * 100:>17.1f}%" for v in accs) + f"   ({time.time() - t0:.0f}초)")
    print(f"  참고: 전부 PNEUMONIA 로 답하면 {(y == 1).mean() * 100:.1f}%")

    # ── 4) 전체 실험 요약 ────────────────────────────────────────────────────
    table = os.path.join(a.results, "summary_table.txt")
    if os.path.exists(table):
        print("=" * 70 + "\n[4] 전체 실험 요약 (seed 평균 ± 표준편차)")
        print(open(table).read())
    print("=" * 70 + "\n시연 끝.")


if __name__ == "__main__":
    main()
