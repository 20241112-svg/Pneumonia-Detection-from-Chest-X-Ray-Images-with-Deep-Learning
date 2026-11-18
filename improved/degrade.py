"""
degrade.py — 흉부 X-ray 열화(corruption) 함수 모음 + 벤치마크 정의

모든 함수 입력/출력: uint8 grayscale 또는 RGB numpy 배열 (H, W) 또는 (H, W, 3), 값 범위 0~255.
severity: 1(약) ~ 5(강). 0이면 원본 그대로 반환.

사용:
    from degrade import degrade, CORRUPTIONS, TRAIN_CORRUPTIONS, HOLDOUT_CORRUPTIONS
    img_bad = degrade(img, "gaussian_noise", severity=3)

설계 원칙
- TRAIN_CORRUPTIONS: 학습 증강에 써도 되는 열화
- HOLDOUT_CORRUPTIONS: 학습에 절대 쓰지 않고 평가에만 쓰는 열화 ("본 적 없는 열화" 일반화 측정)
"""
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# 강도 테이블 (1~5). 값은 X-ray 150~224px 입력 기준으로 보정. 필요하면 조정.
# ---------------------------------------------------------------------------
_NOISE_SIGMA   = [0, 8, 16, 24, 36, 50]        # 가우시안 노이즈 표준편차 (0~255 스케일)
_BLUR_SIGMA    = [0, 0.6, 1.0, 1.5, 2.2, 3.0]  # 가우시안 블러 sigma (px)
_CONTRAST_F    = [1.0, 0.75, 0.55, 0.40, 0.28, 0.18]  # 대비 배율 (1 = 원본)
_GAMMA         = [1.0, 1.4, 1.8, 2.3, 2.9, 3.6]       # 감마 (>1 → 어두워짐, 저노출 흉내)
_JPEG_Q        = [100, 40, 25, 15, 10, 6]      # JPEG 품질
_MOTION_K      = [1, 3, 5, 9, 13, 17]          # 모션 블러 커널 길이 (px)


def _to_float(img):
    return img.astype(np.float32)


def _to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _rng(rng):
    # rng 미지정 시 전역 np.random 상태에서 파생 → np.random.seed() 로 평가 재현 가능
    return rng if rng is not None else np.random.default_rng(np.random.randint(2**31 - 1))


def gaussian_noise(img, s, rng=None):
    """저선량 촬영 시 양자 노이즈 근사."""
    rng = _rng(rng)
    noise = rng.normal(0.0, _NOISE_SIGMA[s], size=img.shape)
    return _to_uint8(_to_float(img) + noise)


def gaussian_blur(img, s, rng=None):
    """초점/미세 움직임에 의한 흐림."""
    sigma = _BLUR_SIGMA[s]
    if sigma <= 0:
        return img.copy()
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma)


def contrast(img, s, rng=None):
    """대비 저하 + 감마로 노출 차이 흉내 (장비·설정 편차)."""
    x = _to_float(img) / 255.0
    mean = x.mean()
    x = (x - mean) * _CONTRAST_F[s] + mean          # 대비 축소
    x = np.clip(x, 0, 1) ** _GAMMA[s]               # 감마 (어둡게)
    return _to_uint8(x * 255.0)


def jpeg(img, s, rng=None):
    """전송/저장 압축 아티팩트. (홀드아웃)"""
    q = _JPEG_Q[s]
    if q >= 100:
        return img.copy()
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    assert ok
    flag = cv2.IMREAD_COLOR if img.ndim == 3 else cv2.IMREAD_GRAYSCALE
    out = cv2.imdecode(buf, flag)
    if img.ndim == 3 and out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return out


def motion_blur(img, s, rng=None):
    """방향성 모션 블러 (호흡/움직임). (홀드아웃)"""
    k = _MOTION_K[s]
    if k <= 1:
        return img.copy()
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0                          # 가로 방향
    angle = _rng(rng).uniform(-30, 30)               # 약간 랜덤 각도
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, M, (k, k))
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


# ---------------------------------------------------------------------------
# 레지스트리
# ---------------------------------------------------------------------------
CORRUPTIONS = {
    "gaussian_noise": gaussian_noise,
    "gaussian_blur":  gaussian_blur,
    "contrast":       contrast,
    "jpeg":           jpeg,
    "motion_blur":    motion_blur,
}

TRAIN_CORRUPTIONS   = ["gaussian_noise", "gaussian_blur", "contrast"]
HOLDOUT_CORRUPTIONS = ["jpeg", "motion_blur"]
SEVERITIES          = [1, 2, 3, 4, 5]

assert set(TRAIN_CORRUPTIONS) | set(HOLDOUT_CORRUPTIONS) == set(CORRUPTIONS)
assert not set(TRAIN_CORRUPTIONS) & set(HOLDOUT_CORRUPTIONS)


def degrade(img, name, severity, rng=None):
    """단일 열화 적용. severity 0 → 원본 복사. rng 를 주면 노이즈·모션 각도가 결정적."""
    if severity == 0:
        return img.copy()
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption: {name}")
    if not 1 <= severity <= 5:
        raise ValueError("severity must be 0..5")
    return CORRUPTIONS[name](img, severity, rng=rng)


_SEV_WEIGHTS = np.array([0.35, 0.30, 0.20, 0.15])   # severity 1,2,3,4 샘플링 비율


def random_train_degrade(img, p=0.5, max_severity=4, p_compose=0.5, rng=None):
    """
    학습 증강용. 확률 p 로 TRAIN_CORRUPTIONS 열화를 적용. HOLDOUT 열화는 절대 사용하지 않음.

    v2 (2026-11-18) — 1차 실험 결과에 따른 변경:
      - 강도 1~3 → 1~4 (가중치 .35/.30/.20/.15). 1차에서 블러 강도 4~5 는 증강 모델도 개선이 없었음
        (both: blur s4 0.71 vs base 0.75) → 학습 때 본 최대 강도가 평가 강도보다 낮았던 것이 원인.
      - 확률 p_compose 로 서로 다른 열화 2종을 순차 합성. 단일 열화만 학습하면 "그 열화 패턴" 을 외우고
        홀드아웃 열화로 일반화되지 않음 (both: motion_blur·jpeg 가 base 이하). 열화를 섞어 다양성을 높이면
        보지 않은 열화에 대한 강건성이 오른다는 것이 AugMix (Hendrycks et al., ICLR 2020) 의 핵심 결과.
    """
    rng = rng or np.random.default_rng()
    if rng.random() >= p:
        return img
    k = 2 if rng.random() < p_compose else 1
    names = rng.choice(TRAIN_CORRUPTIONS, size=k, replace=False)
    w = _SEV_WEIGHTS[:max_severity] / _SEV_WEIGHTS[:max_severity].sum()
    for i, name in enumerate(names):
        s = int(rng.choice(np.arange(1, max_severity + 1), p=w))
        if k == 2:
            s = max(1, s - 1)       # 합성 시에는 각 열화를 한 단계 약하게 (과도한 파괴 방지)
        img = degrade(img, str(name), s, rng=rng)
    return img


def apply_clahe(img, clip=2.0, tile=8):
    """
    CLAHE 전처리. 입력 uint8 (H,W) 또는 (H,W,3). RGB면 grayscale로 바꿔 적용 후 3채널 복제.
    학습과 평가에 동일하게 적용해야 함.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    if img.ndim == 3:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = clahe.apply(g)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    return clahe.apply(img)


if __name__ == "__main__":
    # 간단 자가 테스트 + 샘플 그리드 저장
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="테스트용 X-ray 한 장 경로 (없으면 합성 이미지)")
    ap.add_argument("--out", default="degrade_grid.png")
    a = ap.parse_args()

    if a.image and os.path.exists(a.image):
        base = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
        base = cv2.resize(base, (224, 224))
    else:
        yy, xx = np.mgrid[0:224, 0:224]
        base = _to_uint8(128 + 60 * np.sin(xx / 12.0) * np.cos(yy / 15.0))

    rows = []
    for name in CORRUPTIONS:
        row = [degrade(base, name, s) for s in [0] + SEVERITIES]
        rows.append(np.hstack(row))
    grid = np.vstack(rows)
    cv2.imwrite(a.out, grid)
    print("rows =", list(CORRUPTIONS), "| cols = severity 0..5 | saved", a.out)
