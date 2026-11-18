"""
train_baseline.py — anjanatiha 베이스 CNN의 TF2/Keras3 포팅 + 개선 플래그

베이스라인 재현 (원본 설정과 동일: 150x150, rescale 1/255, shear 0.2, zoom 0.2, h-flip, Adam, CE, class_weight balanced, early stop):
    python train_baseline.py --data data/input --seed 0 --tag base

개선판:
    --clahe          CLAHE 전처리 (학습·평가 동일 적용)
    --degrade-aug    학습 시 TRAIN_CORRUPTIONS 열화 증강 (HOLDOUT 열화는 사용하지 않음)
    --backbone       efficientnetb0 | resnet50  (ImageNet 전이학습, 기본은 베이스 CNN)
    --no-flip        X-ray 좌우 플립 증강 끄기 (실험 항목)

출력: runs/<tag>_s<seed>/ {best.keras, history.json, test_metrics.json, config.json}
"""
import argparse, json, os, random, time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import class_weight as skcw
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

from degrade import random_train_degrade, apply_clahe

CLASSES = ["NORMAL", "PNEUMONIA"]   # index 0 = NORMAL, 1 = PNEUMONIA (flow_from_directory 알파벳 순과 동일)


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def list_files(split_dir):
    paths, labels = [], []
    for i, c in enumerate(CLASSES):
        d = os.path.join(split_dir, c)
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpeg", ".jpg", ".png")):
                paths.append(os.path.join(d, f)); labels.append(i)
    return np.array(paths), np.array(labels, dtype=np.int32)


def _affine_aug(img, rng, shear=0.2, zoom=0.2, hflip=True):
    """원본 ImageDataGenerator(shear_range, zoom_range, horizontal_flip) 근사."""
    h, w = img.shape[:2]
    sh = rng.uniform(-shear, shear)                    # ImageDataGenerator shear는 라디안
    zx = rng.uniform(1 - zoom, 1 + zoom); zy = rng.uniform(1 - zoom, 1 + zoom)
    M = np.array([[zx, -np.sin(sh) * zx, 0.0],
                  [0.0, np.cos(sh) * zy, 0.0]], dtype=np.float32)
    # 중심 기준 변환
    cx, cy = w / 2, h / 2
    M[:, 2] = [cx - M[0, 0] * cx - M[0, 1] * cy, cy - M[1, 0] * cx - M[1, 1] * cy]
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    if hflip and rng.random() < 0.5:
        img = img[:, ::-1]
    return np.ascontiguousarray(img)


# 에포크 카운터: 증강 난수를 (seed, 샘플 인덱스, 에포크) 로 결정하기 위해 콜백이 갱신
EPOCH = {"n": 0}


class EpochCounter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        EPOCH["n"] = epoch


def make_loader(img_size, training, clahe, degrade_aug, hflip, seed):
    def _load(path, label, idx):
        p = path.decode() if isinstance(path, bytes) else str(path)
        # 재현성: 병렬 스레드가 난수 생성기를 공유하지 않도록 샘플마다 독립 시드
        rng = np.random.default_rng([int(seed), int(idx), int(EPOCH["n"])])
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
        if training and degrade_aug:
            img = random_train_degrade(img, rng=rng)
        if clahe:
            img = apply_clahe(img)
        if training:
            img = _affine_aug(img, rng, hflip=hflip)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        return img, np.int32(label)

    def tf_load(path, label, idx):
        img, lab = tf.numpy_function(_load, [path, label, idx], [tf.float32, tf.int32])
        img.set_shape((img_size, img_size, 3)); lab.set_shape(())
        return img, tf.one_hot(lab, 2)
    return tf_load


def make_dataset(split_dir, img_size, batch, training, clahe, degrade_aug, hflip, seed, shuffle=None):
    paths, labels = list_files(split_dir)
    idx = np.arange(len(paths), dtype=np.int64)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels, idx))
    if training if shuffle is None else shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(make_loader(img_size, training, clahe, degrade_aug, hflip, seed),
                num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE), labels


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def baseline_cnn(img_size, bn=False, init="glorot_uniform"):
    """
    원본 get_conv_model() 을 channels_last 로 그대로 포팅.

    bn=False : 원본과 동일 (Conv-ReLU ×10, BN 없음).
               → TF2 재현에서 활성값이 층을 지날수록 소실(입력 std 0.13 → 마지막 conv 0.006)되어
                 loss 가 ln2(0.693) 에서 전혀 내려가지 않고 전부 한 클래스로 예측함 (3 seed 재현).
    bn=True  : 각 Conv 뒤에 BatchNormalization 만 추가 (채널 수·커널·풀링·Dense 전부 동일).
               최소 수정으로 학습 가능하게 만든 "수정 베이스라인".
    """
    m = keras.Sequential(name="baseline_cnn" + ("_bn" if bn else ""))
    m.add(layers.Input((img_size, img_size, 3)))

    def conv(f, **kw):
        if bn:
            m.add(layers.Conv2D(f, 3, kernel_initializer=init, **kw))
            m.add(layers.BatchNormalization())
            m.add(layers.Activation("relu"))
        else:
            m.add(layers.Conv2D(f, 3, activation="relu", kernel_initializer=init, **kw))

    conv(16, padding="same"); conv(16, padding="same"); m.add(layers.MaxPooling2D(2))
    conv(32, padding="same"); conv(32, padding="same"); m.add(layers.MaxPooling2D(2))
    conv(64, padding="same"); conv(64, padding="same"); m.add(layers.MaxPooling2D(2))
    conv(96, dilation_rate=2, padding="same"); conv(96, padding="valid"); m.add(layers.MaxPooling2D(2))
    conv(128, dilation_rate=2, padding="same"); conv(128, padding="valid"); m.add(layers.MaxPooling2D(2))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu", kernel_initializer=init))
    m.add(layers.Dropout(0.4))
    m.add(layers.Dense(2, activation="softmax"))
    return m


def backbone_model(name, img_size, weights="imagenet"):
    inp = layers.Input((img_size, img_size, 3))
    x = layers.Rescaling(255.0)(inp)                # 입력은 0~1 → 0~255 로 복원
    if name == "efficientnetb0":
        base = keras.applications.EfficientNetB0(include_top=False, weights=weights, input_tensor=x)
    elif name == "resnet50":
        x = keras.applications.resnet50.preprocess_input(x)
        base = keras.applications.ResNet50(include_top=False, weights=weights, input_tensor=x)
    else:
        raise ValueError(name)
    y = layers.GlobalAveragePooling2D()(base.output)  # 원본 InceptionV3 코드에 빠져 있던 부분
    y = layers.Dropout(0.3)(y)
    out = layers.Dense(2, activation="softmax")(y)
    return keras.Model(inp, out, name=name)


# ---------------------------------------------------------------------------
def set_seed(s, deterministic=True):
    random.seed(s); np.random.seed(s); tf.random.set_seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    if deterministic:
        # GPU 커널의 비결정적 누적 연산까지 고정 (속도 약간 느려짐). 같은 seed → 같은 결과
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception as e:  # 일부 환경에서 미지원
            print("op determinism 미적용:", e)


def recalibrate_bn(model, clean_train_ds, max_batches=60):
    """
    BatchNorm running mean/var 를 '깨끗한' 학습 영상으로 다시 계산한다 (가중치는 불변).
    열화 증강으로 학습하면 BN 통계가 열화 영상이 섞인 분포로 잡혀 깨끗한 입력과 어긋나는데
    (aug 단독 모델: 깨끗한 정확도 79.5%, NORMAL 재현율 0.47), 이 단계가 그 편향을 제거한다.
    momentum 을 (k-1)/k 로 바꿔가며 호출하면 running 값 = 배치 통계의 정확한 누적 평균.
    """
    bns = [l for l in model.layers if isinstance(l, layers.BatchNormalization)]
    if not bns:
        return 0
    saved = [(l.momentum) for l in bns]
    for l in bns:
        l.moving_mean.assign(tf.zeros_like(l.moving_mean))
        l.moving_variance.assign(tf.ones_like(l.moving_variance))
    k = 0
    for x, _ in clean_train_ds:
        k += 1
        for l in bns:
            l.momentum = (k - 1) / k
        model(x, training=True)
        if k >= max_batches:
            break
    for l, m in zip(bns, saved):
        l.momentum = m
    return k


def evaluate(model, ds, y_true):
    prob = model.predict(ds, verbose=0)[:, 1]
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "recall_normal": float(cm[0, 0] / max(cm[0].sum(), 1)),
        "recall_pneumonia": float(cm[1, 1] / max(cm[1].sum(), 1)),
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/input")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img-size", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--degrade-aug", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    ap.add_argument("--backbone", default=None, choices=[None, "efficientnetb0", "resnet50"])
    ap.add_argument("--weights", default="imagenet", help="backbone 가중치 (imagenet | none)")
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--bn", action="store_true", help="베이스 CNN 각 Conv 뒤에 BatchNorm 추가 (원본은 BN 없음 → 학습 실패)")
    ap.add_argument("--init", default="glorot_uniform", help="커널 초기화 (glorot_uniform=원본 | he_normal)")
    ap.add_argument("--no-bn-recal", action="store_true", help="학습 후 BN 통계 재보정 끄기 (기본: 켜짐)")
    ap.add_argument("--no-deterministic", action="store_true", help="TF op determinism 끄기")
    a = ap.parse_args()

    set_seed(a.seed, deterministic=not a.no_deterministic)
    run_dir = os.path.join(a.out, f"{a.tag}_s{a.seed}")
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    hflip = not a.no_flip
    train_ds, ytr = make_dataset(os.path.join(a.data, "train"), a.img_size, a.batch, True,  a.clahe, a.degrade_aug, hflip, a.seed)
    val_ds,   yva = make_dataset(os.path.join(a.data, "val"),   a.img_size, a.batch, False, a.clahe, False, hflip, a.seed)
    test_ds,  yte = make_dataset(os.path.join(a.data, "test"),  a.img_size, a.batch, False, a.clahe, False, hflip, a.seed)
    print(f"train {len(ytr)} (NORMAL {int((ytr==0).sum())} / PNEU {int((ytr==1).sum())}) | val {len(yva)} | test {len(yte)}")

    if a.backbone:
        model = backbone_model(a.backbone, a.img_size, None if a.weights == "none" else a.weights)
    else:
        model = baseline_cnn(a.img_size, bn=a.bn, init=a.init)
    model.summary(line_length=100)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=a.lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])

    cw = None
    if not a.no_class_weight:
        w = skcw.compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=ytr)
        cw = {0: float(w[0]), 1: float(w[1])}
    print("class_weight:", cw)

    best_path = os.path.join(run_dir, "best.keras")
    cbs = [
        keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss", save_best_only=True, verbose=0),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(os.path.join(run_dir, "log.csv")),
        EpochCounter(),
    ]

    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, callbacks=cbs, class_weight=cw, verbose=2)
    train_time = time.time() - t0
    json.dump({k: [float(v) for v in vs] for k, vs in hist.history.items()},
              open(os.path.join(run_dir, "history.json"), "w"), indent=1)

    model = keras.models.load_model(best_path)
    res_norecal = evaluate(model, test_ds, yte)
    n_recal = 0
    if not a.no_bn_recal and (a.bn or a.backbone):
        # 재보정용: 깨끗한 train (증강 없음, CLAHE 는 추론과 동일하게 적용), 큰 배치
        clean_tr, _ = make_dataset(os.path.join(a.data, "train"), a.img_size, 64, False, a.clahe, False, hflip, a.seed, shuffle=True)
        n_recal = recalibrate_bn(model, clean_tr)
        os.replace(best_path, os.path.join(run_dir, "best_norecal.keras"))  # 재보정 전 모델 보관
        model.save(best_path)   # best.keras = 재보정된 모델 (eval_robustness 가 쓰는 파일)
    res = evaluate(model, test_ds, yte)
    res.update({"val_best_loss": float(min(hist.history["val_loss"])),
                "epochs_run": len(hist.history["loss"]), "train_time_sec": train_time,
                "bn_recal_batches": n_recal,
                "clean_acc_before_bn_recal": res_norecal["accuracy"],
                "recall_normal_before_bn_recal": res_norecal["recall_normal"]})
    json.dump(res, open(os.path.join(run_dir, "test_metrics.json"), "w"), indent=2)

    print("\n=== TEST (clean) ===")
    for k in ["accuracy", "precision", "recall", "f1", "auc", "recall_normal", "recall_pneumonia"]:
        print(f"{k:<18}{res[k]:.4f}")
    print("confusion_matrix [[TN FP][FN TP]] =", res["confusion_matrix"])
    if n_recal:
        print(f"BN 재보정 전 → 후: acc {res_norecal['accuracy']:.4f} → {res['accuracy']:.4f}, "
              f"recall_normal {res_norecal['recall_normal']:.3f} → {res['recall_normal']:.3f}  ({n_recal} batches)")
    print("saved →", run_dir)


if __name__ == "__main__":
    main()
