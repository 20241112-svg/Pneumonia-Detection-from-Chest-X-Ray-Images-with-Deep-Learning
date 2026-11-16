"""
make_split.py — Kaggle chest_xray 원본을 train / val / test 로 재분할

문제: Kaggle 원본 val 폴더는 16장(클래스당 8장)뿐이라 early stopping/모델 선택 기준으로 쓸 수 없음.
해결: train 에서 클래스별 val_ratio 만큼 떼어 val 을 만들고, test(624장)는 원본 그대로 유지.

사용:
    python make_split.py --src data/raw/chest_xray --dst data/input --val-ratio 0.1 --seed 42
"""
import argparse, os, shutil, random

CLASSES = ["NORMAL", "PNEUMONIA"]


def copy_list(files, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for f in files:
        shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Kaggle 압축 해제 폴더 (train/val/test 포함)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-orig-val", action="store_true",
                    help="원본 val 16장을 train 풀에 합치지 않음 (기본: 합쳐서 재분할)")
    a = ap.parse_args()
    a.merge_orig_val = not a.keep_orig_val
    rng = random.Random(a.seed)

    if os.path.exists(a.dst):
        raise SystemExit(f"{a.dst} 가 이미 있습니다. 지우고 다시 실행하세요.")

    summary = {}
    for c in CLASSES:
        pool = [os.path.join(a.src, "train", c, f) for f in os.listdir(os.path.join(a.src, "train", c))]
        if a.merge_orig_val and os.path.isdir(os.path.join(a.src, "val", c)):
            pool += [os.path.join(a.src, "val", c, f) for f in os.listdir(os.path.join(a.src, "val", c))]
        pool = sorted(p for p in pool if p.lower().endswith((".jpeg", ".jpg", ".png")))
        rng.shuffle(pool)
        n_val = int(round(len(pool) * a.val_ratio))
        val, train = pool[:n_val], pool[n_val:]
        copy_list(train, os.path.join(a.dst, "train", c))
        copy_list(val,   os.path.join(a.dst, "val", c))

        test = [os.path.join(a.src, "test", c, f) for f in os.listdir(os.path.join(a.src, "test", c))]
        copy_list(test, os.path.join(a.dst, "test", c))
        summary[c] = (len(train), len(val), len(test))

    print(f"{'class':<10}{'train':>8}{'val':>8}{'test':>8}")
    for c, (tr, va, te) in summary.items():
        print(f"{c:<10}{tr:>8}{va:>8}{te:>8}")
    with open(os.path.join(a.dst, "SPLIT_INFO.txt"), "w") as f:
        f.write(f"seed={a.seed} val_ratio={a.val_ratio} test=original\n{summary}\n")


if __name__ == "__main__":
    main()
