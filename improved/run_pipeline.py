"""
run_pipeline.py — 전체 실험을 순서대로 돌리는 드라이버 (끊겨도 이어서 실행)

- 이미 끝난 단계(결과 파일이 있는 단계)는 건너뛴다 → 런타임이 끊기면 0·1·2·3번 셀 후 같은 명령을 다시 실행.
- 각 단계가 끝날 때마다 runs/ results/ 를 Drive 로 복사한다.
- 한 단계가 실패해도 다음 단계로 넘어가고, 마지막에 실패 목록을 출력한다.
- v2 CNN 은 tag 를 base_v2 / both_v2 로 써서 1차(v1) 모델(base_s0 등)을 덮어쓰지 않는다.

단계 (--stages 로 일부만 선택 가능, 기본은 전부):
  v1    : 1차 모델 4개 재평가(확률 저장) → 임계값 분석 → 데이터 통계      (학습 없음)
  vit   : ViT-S/16 + LoRA (기본 / CLAHE+증강), seed 0
  cnn   : v2 CNN base_v2 / both_v2, seed 0 + 임계값 분석
  seeds : base_v2 / both_v2, seed 1·2
  orig  : 원본 CNN 그대로(BN 없음) — 학습 실패 재현
  vitseeds : vit_lora_both seed 1·2   (--stages 에 명시했을 때만)

사용 (Colab):
  !python run_pipeline.py
  !python run_pipeline.py --stages v1,vit
"""
import argparse, glob, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOG = []          # (단계, 상태, 초)


def sh(cmd, env=None):
    print("\n$ " + " ".join(cmd), flush=True)
    e = dict(os.environ); e.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    if env: e.update(env)
    return subprocess.run(cmd, env=e).returncode


def script(name):
    return os.path.join(HERE, name)


def save_to_drive(drive):
    if drive and os.path.isdir(drive):
        subprocess.run(f'cp -ru runs results "{drive}/" 2>/dev/null', shell=True)
        print(f"[drive] 저장 → {drive}", flush=True)


def step(name, done_files, fn, drive):
    """done_files 가 모두 있으면 건너뜀. fn() 은 returncode(0=성공) 반환."""
    if done_files and all(os.path.exists(f) for f in done_files):
        print(f"\n===== [건너뜀] {name} (이미 완료) =====", flush=True)
        LOG.append((name, "skip", 0)); return True
    print(f"\n===== [시작] {name} =====", flush=True)
    t0 = time.time()
    try:
        rc = fn()
    except Exception as ex:           # 드라이버 자체 오류도 다음 단계로
        print("오류:", ex); rc = 1
    dt = time.time() - t0
    ok = (rc == 0) and all(os.path.exists(f) for f in done_files)
    LOG.append((name, "ok" if ok else "FAIL", dt))
    print(f"===== [{'완료' if ok else '실패'}] {name} ({dt/60:.1f}분) =====", flush=True)
    save_to_drive(drive)
    return ok


def summary_json(tag, seed):
    return f"results/{tag}_s{seed}_summary.json"


def train_eval_cnn(data, tag, seed, flags, quick):
    extra = ["--epochs", "1", "--patience", "1"] if quick else []
    base = [PY, script("train_baseline.py"), "--data", data, "--tag", tag, "--seed", str(seed)] + flags + extra
    rc = sh(base)
    if rc != 0:
        print("[재시도] GPU determinism 을 끄고 다시 학습 (--no-deterministic)", flush=True)
        rc = sh(base + ["--no-deterministic"])
        if rc != 0:
            return rc
    return sh([PY, script("eval_robustness.py"), "--model", f"runs/{tag}_s{seed}/best.keras",
               "--data", data, "--tag", f"{tag}_s{seed}"])


def train_vit(data, tag, seed, flags, quick):
    extra = ["--epochs", "1", "--no-pretrained", "--workers", "0", "--batch", "8"] if quick else []
    return sh([PY, script("train_lora_vit.py"), "--data", data, "--tag", tag, "--seed", str(seed),
               "--mode", "lora"] + flags + extra)


def analyze(tags, out_tag):
    probs = [f"results/{t}_probs.npz" for t in tags if os.path.exists(f"results/{t}_probs.npz")]
    if not probs:
        print("분석할 확률 파일 없음"); return 1
    return sh([PY, script("analyze_threshold.py"), "--probs"] + probs + ["--tag", out_tag])


def aggregate():
    """results/*_summary.json 을 tag 별로 모아 seed 평균 ± 표준편차 표를 출력·저장."""
    groups = {}
    for f in sorted(glob.glob("results/*_s[0-9]_summary.json")):
        name = os.path.basename(f)[: -len("_summary.json")]
        tag, seed = name.rsplit("_s", 1)
        groups.setdefault(tag, []).append((int(seed), json.load(open(f))))
    keys = [("clean_acc", "깨끗한"), ("mean_acc_seen_corruptions", "학습열화"),
            ("mean_acc_holdout_corruptions", "홀드아웃"), ("mean_acc_all_corruptions", "전체열화")]
    lines = [f"{'model':<18}{'seeds':>6}" + "".join(f"{k[1]:>16}" for k in keys)]
    rows = []
    for tag, runs in groups.items():
        row = {"model": tag, "n_seeds": len(runs), "seeds": ",".join(str(s) for s, _ in runs)}
        cells = []
        for k, _ in keys:
            v = np.array([r[k] for _, r in runs]) * 100
            row[k + "_mean"] = float(v.mean()); row[k + "_std"] = float(v.std(ddof=0))
            cells.append(f"{v.mean():6.1f} ± {v.std(ddof=0):4.1f}" if len(v) > 1 else f"{v.mean():6.1f}       ")
        lines.append(f"{tag:<18}{len(runs):>6}" + "".join(f"{c:>16}" for c in cells)); rows.append(row)
    # BN 재보정 전후 (v2 CNN)
    recal = []
    for f in sorted(glob.glob("runs/*/test_metrics.json")):
        m = json.load(open(f))
        if m.get("bn_recal_batches"):
            recal.append(f"{os.path.basename(os.path.dirname(f)):<18} acc {m['clean_acc_before_bn_recal']*100:5.1f} → {m['accuracy']*100:5.1f}   "
                         f"NORMAL 재현율 {m['recall_normal_before_bn_recal']:.3f} → {m['recall_normal']:.3f}")
    text = "\n".join(lines) + ("\n\nBN 재보정 전 → 후 (깨끗한 test)\n" + "\n".join(recal) if recal else "")
    print("\n================ 결과 요약 (정확도 %, seed 평균 ± 표준편차) ================\n" + text, flush=True)
    os.makedirs("results", exist_ok=True)
    open("results/summary_table.txt", "w").write(text + "\n")
    json.dump(rows, open("results/summary_table.json", "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input")
    ap.add_argument("--drive", default="/content/drive/MyDrive/pneumonia_robustness")
    ap.add_argument("--stages", default="v1,vit,cnn,seeds,orig")
    ap.add_argument("--quick", action="store_true", help="(개발용) 1 epoch, ViT 사전학습 없이 — 파이프라인 동작 확인")
    a = ap.parse_args()
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    os.makedirs("runs", exist_ok=True); os.makedirs("results", exist_ok=True)
    D, Q = a.data, a.quick
    t_all = time.time()

    if "v1" in stages:
        for t in ["base", "clahe", "aug", "both"]:
            model = f"runs/{t}_s0/best.keras"
            if not os.path.exists(model):
                print(f"\n[건너뜀] 1차 모델 없음: {model}"); continue
            step(f"1차 모델 재평가: {t}", [f"results/{t}_v1_s0_probs.npz", f"results/{t}_v1_s0_summary.json"],
                 lambda t=t, model=model: sh([PY, script("eval_robustness.py"), "--model", model, "--data", D, "--tag", f"{t}_v1_s0"]),
                 a.drive)
        step("임계값 분석 (1차 모델)", [], lambda: analyze([f"{t}_v1_s0" for t in ["base", "clahe", "aug", "both"]], "cnn_v1"), a.drive)
        step("데이터 통계 (열화 강도 근거)", ["results/data_stats.png"],
             lambda: sh([PY, script("data_stats.py"), "--data", D, "--out", "results/data_stats"]), a.drive)

    if "vit" in stages:
        subprocess.run([PY, "-m", "pip", "uninstall", "-y", "-q", "torchao"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        step("ViT+LoRA (seed 0)", [summary_json("vit_lora", 0)], lambda: train_vit(D, "vit_lora", 0, [], Q), a.drive)
        step("ViT+LoRA + CLAHE + 증강 (seed 0)", [summary_json("vit_lora_both", 0)],
             lambda: train_vit(D, "vit_lora_both", 0, ["--clahe", "--degrade-aug"], Q), a.drive)

    CNN = [("base_v2", ["--bn"]), ("both_v2", ["--bn", "--clahe", "--degrade-aug"])]
    if "cnn" in stages:
        for tag, flags in CNN:
            step(f"v2 CNN {tag} (seed 0)", [summary_json(tag, 0), f"results/{tag}_s0_probs.npz"],
                 lambda tag=tag, flags=flags: train_eval_cnn(D, tag, 0, flags, Q), a.drive)
        step("임계값 분석 (v2 + ViT)", [],
             lambda: analyze(["base_v2_s0", "both_v2_s0", "vit_lora_s0", "vit_lora_both_s0"], "v2"), a.drive)

    if "seeds" in stages:
        for seed in [1, 2]:
            for tag, flags in CNN:
                step(f"v2 CNN {tag} (seed {seed})", [summary_json(tag, seed), f"results/{tag}_s{seed}_probs.npz"],
                     lambda tag=tag, flags=flags, seed=seed: train_eval_cnn(D, tag, seed, flags, Q), a.drive)

    if "orig" in stages:
        extra = ["--epochs", "1"] if Q else []
        step("원본 CNN 그대로 (BN 없음, 학습 실패 재현)", ["runs/base_orig_s0/test_metrics.json"],
             lambda: sh([PY, script("train_baseline.py"), "--data", D, "--tag", "base_orig", "--seed", "0"] + extra), a.drive)

    if "vitseeds" in stages:
        for seed in [1, 2]:
            step(f"ViT+LoRA + CLAHE + 증강 (seed {seed})", [summary_json("vit_lora_both", seed)],
                 lambda seed=seed: train_vit(D, "vit_lora_both", seed, ["--clahe", "--degrade-aug"], Q), a.drive)

    aggregate()
    save_to_drive(a.drive)
    print("\n================ 단계별 상태 ================")
    for name, st, dt in LOG:
        print(f"{st:<5} {dt/60:6.1f}분  {name}")
    fails = [n for n, st, _ in LOG if st == "FAIL"]
    print(f"\n총 {(time.time()-t_all)/60:.0f}분. " + (f"실패 {len(fails)}개: {fails}" if fails else "실패 없음."))


if __name__ == "__main__":
    main()
