"""
train_lora_vit.py — 사전학습 ViT 에 LoRA 어댑터만 학습해서 폐렴 분류 + 열화 강건성 평가 (PyTorch)

"구조 변경" 실험 축: 원본 CNN(55만 파라미터, 처음부터 학습) 을
ImageNet 사전학습 ViT 로 교체하되, 전체 가중치는 고정하고 LoRA(저순위 행렬 A,B) 만 학습.

모드:
  --mode lora       : LoRA (qkv/proj/fc1/fc2 에 r=8) + 분류 head          ← 주 실험
  --mode head       : 분류 head 만 학습 (linear probe)                     ← 하한 비교군
  --mode full       : 전체 미세조정                                        ← 상한 비교군

같은 데이터 분할(data/input), 같은 열화(degrade.py), 같은 CSV 형식(results/<tag>.csv) 을 쓰므로
eval_robustness.py --compare 로 CNN 결과와 바로 겹쳐 그릴 수 있음.

예:
  python train_lora_vit.py --data data/input --tag vit_lora --seed 0 --mode lora
  python train_lora_vit.py --data data/input --tag vit_lora_both --seed 0 --mode lora --clahe --degrade-aug
  python eval_robustness.py --compare results/base_s0.csv results/vit_lora_s0.csv --out results/cmp_vit.png
"""
import argparse, csv, json, os, random, time
import numpy as np
import cv2
import torch, torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from degrade import CORRUPTIONS, SEVERITIES, HOLDOUT_CORRUPTIONS, degrade, random_train_degrade, apply_clahe

CLASSES = ["NORMAL", "PNEUMONIA"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# 데이터 (train_baseline.py 와 동일한 전처리 순서: resize → [degrade] → [clahe] → [affine aug] → 정규화)
# ---------------------------------------------------------------------------
def list_files(split_dir):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        d = os.path.join(split_dir, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(d, f)); labels.append(i)
    return paths, np.array(labels, dtype=np.int64)


def affine_aug(img, rng, shear=0.2, zoom=0.2, hflip=True):
    h, w = img.shape[:2]
    sh = rng.uniform(-shear, shear)
    zx = rng.uniform(1 - zoom, 1 + zoom); zy = rng.uniform(1 - zoom, 1 + zoom)
    M = np.array([[zx, -np.sin(sh) * zx, 0.0], [0.0, np.cos(sh) * zy, 0.0]], dtype=np.float32)
    cx, cy = w / 2, h / 2
    M[:, 2] = [cx - M[0, 0] * cx - M[0, 1] * cy, cy - M[1, 0] * cx - M[1, 1] * cy]
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if hflip and rng.random() < 0.5:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


def to_tensor(gray_u8):
    rgb = np.repeat(gray_u8[..., None], 3, axis=-1).astype(np.float32) / 255.0
    rgb = (rgb - MEAN) / STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class XrayDS(Dataset):
    def __init__(self, split_dir, img_size, training, clahe, degrade_aug, hflip, seed):
        self.paths, self.labels = list_files(split_dir)
        self.img_size, self.training, self.clahe, self.degrade_aug, self.hflip = img_size, training, clahe, degrade_aug, hflip
        self.seed, self.epoch = seed, 0

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i, self.epoch]) if self.training else np.random.default_rng(0)
        img = cv2.imread(self.paths[i], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.training and self.degrade_aug:
            img = random_train_degrade(img, rng=rng)
        if self.clahe:
            img = apply_clahe(img)
        if self.training:
            img = affine_aug(img, rng, hflip=self.hflip)
        return to_tensor(img), int(self.labels[i])


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def build_model(arch, mode, lora_r, lora_alpha, lora_dropout, pretrained=True):
    model = timm.create_model(arch, pretrained=pretrained, num_classes=2)
    if mode == "full":
        pass
    elif mode == "head":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("head")
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
                         target_modules=["qkv", "proj", "fc1", "fc2"],  # timm ViT 블록 내부 Linear 이름
                         modules_to_save=["head"])
        model = get_peft_model(model, cfg)
    else:
        raise ValueError(mode)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[{arch} / {mode}] trainable {n_tr:,} / total {n_all:,} ({100*n_tr/n_all:.2f}%)")
    return model


# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, imgs_u8, clahe, batch, device):
    model.eval(); out = []
    for s in range(0, len(imgs_u8), batch):
        chunk = imgs_u8[s:s + batch]
        if clahe:
            chunk = [apply_clahe(g) for g in chunk]
        x = torch.stack([to_tensor(g) for g in chunk]).to(device)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(x)
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float("nan"),
            "recall_normal": cm[0, 0] / max(cm[0].sum(), 1), "recall_pneumonia": cm[1, 1] / max(cm[1].sum(), 1),
            "confusion_matrix": cm.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input")
    ap.add_argument("--out", default="runs"); ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="vit_lora"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", default="vit_small_patch16_224.augreg_in21k_ft_in1k",
                    help="timm 모델명 (vit_small_* ≈ 22M, vit_base_* ≈ 86M)")
    ap.add_argument("--mode", default="lora", choices=["lora", "head", "full"])
    ap.add_argument("--lora-r", type=int, default=8); ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--img-size", type=int, default=224); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="기본: lora 5e-4, head 1e-3, full 3e-5")
    ap.add_argument("--clahe", action="store_true"); ap.add_argument("--degrade-aug", action="store_true")
    ap.add_argument("--no-flip", action="store_true"); ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="(테스트용) 사전학습 가중치 없이")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(a.out, f"{a.tag}_s{a.seed}"); os.makedirs(run_dir, exist_ok=True)
    os.makedirs(a.results, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    hflip = not a.no_flip
    tr = XrayDS(os.path.join(a.data, "train"), a.img_size, True, a.clahe, a.degrade_aug, hflip, a.seed)
    va = XrayDS(os.path.join(a.data, "val"), a.img_size, False, a.clahe, False, hflip, a.seed)
    tr_dl = DataLoader(tr, batch_size=a.batch, shuffle=True, num_workers=a.workers, drop_last=True, pin_memory=True)
    va_dl = DataLoader(va, batch_size=a.batch * 2, shuffle=False, num_workers=a.workers)
    print(f"train {len(tr)} (NORMAL {(tr.labels==0).sum()} / PNEU {(tr.labels==1).sum()}) | val {len(va)} | device {device}")

    model = build_model(a.arch, a.mode, a.lora_r, a.lora_alpha, a.lora_dropout, pretrained=not a.no_pretrained).to(device)
    lr = a.lr or {"lora": 5e-4, "head": 1e-3, "full": 3e-5}[a.mode]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(tr_dl))
    if a.no_class_weight:
        cw = None
    else:
        n0, n1 = (tr.labels == 0).sum(), (tr.labels == 1).sum()
        cw = torch.tensor([len(tr) / (2 * n0), len(tr) / (2 * n1)], dtype=torch.float32, device=device)
    print("class_weight:", None if cw is None else cw.tolist())
    crit = nn.CrossEntropyLoss(weight=cw)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best, best_ep, bad, hist = float("inf"), -1, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        tr.epoch = ep                      # 샘플별 증강 시드 = (seed, index, epoch)
        model.train(); tl, tc, tn = 0.0, 0, 0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(x); loss = crit(logits.float(), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); tn += len(y)
        # val
        model.eval(); vl, vp, vy = 0.0, [], []
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits = model(x)
                vl += crit(logits.float(), y).item() * len(y)
                vp.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()); vy.append(y.cpu().numpy())
        vp, vy = np.concatenate(vp), np.concatenate(vy); vl /= len(va)
        vacc = accuracy_score(vy, vp >= 0.5); vauc = roc_auc_score(vy, vp)
        hist.append({"epoch": ep, "loss": tl / tn, "accuracy": tc / tn, "val_loss": vl, "val_accuracy": vacc, "val_auc": vauc})
        print(f"ep {ep+1:02d}/{a.epochs}  loss {tl/tn:.4f} acc {tc/tn:.4f} | val_loss {vl:.4f} val_acc {vacc:.4f} val_auc {vauc:.4f}  ({time.time()-t0:.0f}s)")
        if vl < best:
            best, best_ep, bad = vl, ep, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stopping (best epoch {best_ep+1})"); break
    json.dump(hist, open(os.path.join(run_dir, "history.json"), "w"), indent=1)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))

    # ---- 테스트: 깨끗한 + 열화 25조건 (eval_robustness.py 와 동일한 CSV 형식) ----
    te_paths, te_y = list_files(os.path.join(a.data, "test"))
    load = lambda ps: [cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE), (a.img_size, a.img_size), interpolation=cv2.INTER_AREA) for p in ps]
    base = load(te_paths)
    va_paths, va_y = list_files(os.path.join(a.data, "val"))
    vbase = load(va_paths)                     # analyze_threshold.py 용 val 확률도 저장
    rows, probs = [], {"y_test": te_y, "y_val": va_y}
    def record(name, s, prob, vprob):
        m = metrics(te_y, prob); m.update({"corruption": name, "severity": s, "holdout": name in HOLDOUT_CORRUPTIONS}); rows.append(m)
        probs[f"test_{name}_s{s}"] = prob.astype(np.float32); probs[f"val_{name}_s{s}"] = vprob.astype(np.float32)
        print(f"{name:<16} s={s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}  rN={m['recall_normal']:.3f} rP={m['recall_pneumonia']:.3f}")
        return m
    P = lambda imgs: predict_probs(model, imgs, a.clahe, a.batch * 2, device)
    clean = record("clean", 0, P(base), P(vbase))
    for name in CORRUPTIONS:
        for s in SEVERITIES:
            np.random.seed(123 + s)
            dte = [degrade(g, name, s) for g in base]
            dva = [degrade(g, name, s) for g in vbase]
            record(name, s, P(dte), P(dva))

    tag = f"{a.tag}_s{a.seed}"
    np.savez_compressed(os.path.join(a.results, f"{tag}_probs.npz"), **probs)
    with open(os.path.join(a.results, f"{tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corruption", "severity", "holdout", "accuracy", "f1", "auc", "recall_normal", "recall_pneumonia"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    seen = [r["accuracy"] for r in rows if r["severity"] > 0 and not r["holdout"]]
    hold = [r["accuracy"] for r in rows if r["severity"] > 0 and r["holdout"]]
    summary = {"clean_acc": clean["accuracy"], "mean_acc_seen_corruptions": float(np.mean(seen)),
               "mean_acc_holdout_corruptions": float(np.mean(hold)), "mean_acc_all_corruptions": float(np.mean(seen + hold)),
               "best_epoch": best_ep + 1, "train_time_sec": time.time() - t0,
               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    json.dump(summary, open(os.path.join(a.results, f"{tag}_summary.json"), "w"), indent=2)
    json.dump({**clean, "confusion_matrix": clean["confusion_matrix"]}, open(os.path.join(run_dir, "test_metrics.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:<30}{v:.4f}" if isinstance(v, float) else f"{k:<30}{v}")
    print("saved →", os.path.join(a.results, f"{tag}.csv"))


if __name__ == "__main__":
    main()
