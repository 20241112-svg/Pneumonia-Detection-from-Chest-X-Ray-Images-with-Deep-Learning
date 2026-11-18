"""
data_stats.py — 열화 강도표의 근거: 데이터셋 안에 자연적으로 존재하는 노이즈·대비 범위 측정

질문: "노이즈 σ=50, 대비 0.18 같은 강도 값은 자의적이지 않은가?"
답: Kaggle 영상 자체의 노이즈 수준과 대비 분포를 재서, 각 강도가 그 분포의 어디에 놓이는지 보인다.
   강도 1~2 가 데이터 내 하위 5~1% 영상 수준이면 "현실에 있는 열화", 3~5 는 "그보다 심한 조건" 으로 정당화.

측정 지표 (영상별, 224px 리사이즈 후)
  noise_sigma : Immerkær (1996) 빠른 노이즈 분산 추정 — 라플라시안 커널 응답의 평균 절대값 기반
  rms_contrast: 픽셀 표준편차 / 255
  mean_bright : 평균 밝기 / 255
  sharpness   : 라플라시안 분산 (블러 지표, 클수록 선명)

사용:
  python data_stats.py --data data/input --out results/data_stats
"""
import argparse, csv, os
import numpy as np
import cv2

from degrade import degrade, SEVERITIES

_LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def noise_sigma(g):
    h, w = g.shape
    r = cv2.filter2D(g.astype(np.float32), -1, _LAP, borderType=cv2.BORDER_REFLECT)
    return float(np.sqrt(np.pi / 2) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def stats(g):
    return {"noise_sigma": noise_sigma(g), "rms_contrast": float(g.std() / 255.0),
            "mean_bright": float(g.mean() / 255.0),
            "sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input")
    ap.add_argument("--out", default="results/data_stats")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-per-class", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    rows = []
    for split in ["train", "test"]:
        for c in ["NORMAL", "PNEUMONIA"]:
            d = os.path.join(a.data, split, c)
            files = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpeg", ".jpg", ".png")))[: a.max_per_class]
            for f in files:
                g = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
                g = cv2.resize(g, (a.img_size, a.img_size), interpolation=cv2.INTER_AREA)
                r = stats(g); r.update({"split": split, "cls": c, "file": f}); rows.append(r)
    with open(a.out + "_per_image.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    keys = ["noise_sigma", "rms_contrast", "mean_bright", "sharpness"]
    arr = {k: np.array([r[k] for r in rows]) for k in keys}
    pct = [1, 5, 25, 50, 75, 95, 99]
    print(f"영상 {len(rows)}장 ({a.img_size}px)\n{'metric':<14}" + "".join(f"{'p'+str(p):>9}" for p in pct))
    summary = {}
    for k in keys:
        v = np.percentile(arr[k], pct); summary[k] = dict(zip(pct, v))
        print(f"{k:<14}" + "".join(f"{x:>9.3f}" for x in v))

    # 중앙값 영상(대비 기준)에 각 열화를 가했을 때 지표가 어디로 가는지
    med_idx = int(np.argsort(arr["rms_contrast"])[len(rows) // 2])
    mr = rows[med_idx]
    g = cv2.imread(os.path.join(a.data, mr["split"], mr["cls"], mr["file"]), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (a.img_size, a.img_size), interpolation=cv2.INTER_AREA)
    print(f"\n중앙값 영상에 열화 적용 시 지표 (원본: σ={noise_sigma(g):.2f}, 대비={g.std()/255:.3f}, 선명도={cv2.Laplacian(g, cv2.CV_32F).var():.0f})")
    print(f"{'corruption':<16}{'sev':>4}{'noise_sigma':>12}{'rms_contrast':>13}{'sharpness':>11}   데이터 내 백분위")
    sev_rows = []
    def pct_of(k, x, lower_is_worse):
        q = (arr[k] < x).mean() * 100
        return q
    for name, k, lower_worse in [("gaussian_noise", "noise_sigma", False), ("contrast", "rms_contrast", True),
                                  ("gaussian_blur", "sharpness", True), ("motion_blur", "sharpness", True)]:
        for s in SEVERITIES:
            np.random.seed(7 + s)
            st = stats(degrade(g, name, s))
            q = pct_of(k, st[k], lower_worse)
            note = (f"{k} 가 데이터의 {q:.1f}% 지점" if lower_worse else f"{k} 가 데이터의 {q:.1f}% 지점(상위 {100-q:.1f}%)")
            print(f"{name:<16}{s:>4}{st['noise_sigma']:>12.2f}{st['rms_contrast']:>13.3f}{st['sharpness']:>11.0f}   {note}")
            sev_rows.append({"corruption": name, "severity": s, **st, "metric": k, "percentile_in_data": q})
    with open(a.out + "_severity_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sev_rows[0].keys())); w.writeheader(); w.writerows(sev_rows)

    # 그림: 분포 히스토그램 + 강도 위치
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (k, name, lab) in zip(axes, [("noise_sigma", "gaussian_noise", "noise σ (Immerkær)"),
                                         ("rms_contrast", "contrast", "RMS contrast"),
                                         ("sharpness", "gaussian_blur", "sharpness (Laplacian var)")]):
        ax.hist(arr[k], bins=50, color="0.6")
        for r in [r for r in sev_rows if r["corruption"] == name]:
            ax.axvline(r[k], color="C0", lw=1); ax.text(r[k], ax.get_ylim()[1] * 0.9, f"s{r['severity']}", color="C0", fontsize=8, ha="center")
        ax.set_title(f"{lab} — dataset vs. severity", fontsize=9); ax.set_xlabel(k)
        if k == "sharpness": ax.set_xscale("log")
    plt.tight_layout(); plt.savefig(a.out + ".png", dpi=140); plt.close()
    print("saved →", a.out + "_per_image.csv,", a.out + "_severity_map.csv,", a.out + ".png")


if __name__ == "__main__":
    main()
