# 베이스 레포 분석 & 현대 TF 포팅 노트

베이스: `anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning` (MIT, ★166)
실제 코드: `code/Detection of Pneumonia from Chest X-Ray Images 1.0.0.3.ipynb` 단일 노트북 (37개 코드 셀). `code/obsolete/`는 과거 버전 모음이라 무시.
환경: Python 3.6 / Keras 2.2.4 / TensorFlow 1.12 (2018-12). **현재 TF 2.x + Keras 3에서는 그대로 실행 불가.**

---

## 1. 노트북 구조 요약

| 셀 | 내용 |
|---|---|
| 0 | import (standalone `keras`, TF1 API) |
| 1–8 | 디렉터리/플롯 유틸 함수 (학습과 무관) |
| 10 | 경로 설정: `data/input/{train,val,test}` |
| 13 | `ImageDataGenerator` — rescale 1/255, shear 0.2, zoom 0.2, h-flip, **target 150×150, batch 163** |
| 14–15 | `class_weight` (balanced) |
| 18 | 콜백: ModelCheckpoint(val_acc), EarlyStopping(patience 5), ReduceLROnPlateau, TensorBoard |
| 19 | **`get_conv_model()` — 실제 사용된 커스텀 CNN** (README 수치 89.53%는 이 모델) |
| 20 | `get_model()` — InceptionV3 전이학습 (주석 처리, 미사용) |
| 22 | `fit_generator`, Adam 기본 lr, categorical CE, 100 epoch (early stop) |
| 28–30 | 테스트 평가: accuracy / precision / recall / F1 / confusion matrix |

### 커스텀 CNN 구조 (셀 19)
```
Conv16-Conv16-Pool → Conv32-Conv32-Pool → Conv64-Conv64-Pool
→ Conv96(dil2)-Conv96(valid)-Pool → Conv128(dil2)-Conv128(valid)-Pool
→ Flatten → Dense64 → Dropout0.4 → Dense2(softmax)
```
3×3 커널, ReLU, same 패딩(일부 valid). 파라미터 수 소형. 이 구조를 **그대로** 포팅해야 "베이스라인 재현"이 성립함.

---

## 2. 고쳐야 할 부분 (그대로 두면 에러 또는 잘못된 결과)

### A. 실행 자체가 안 되는 것
| # | 원본 | 문제 | 수정 |
|---|---|---|---|
| 1 | `import keras` (standalone 2.2.4) | 현재 Keras 3와 API 불일치 | `from tensorflow import keras` 또는 `import keras` (Keras 3) 중 하나로 통일 |
| 2 | `K.set_image_dim_ordering('th')` | 함수 제거됨. 채널 순서를 channels_first로 강제 | 삭제. channels_last(기본) 사용, 모델 `input_shape=(150,150,3)` |
| 3 | `Conv2D(..., input_shape=(3,150,150))` | 위 2번과 세트. channels_last에선 shape 불일치 | `(150,150,3)` |
| 4 | `from keras.utils import np_utils` | 제거됨 | 미사용이므로 삭제 |
| 5 | `from keras.wrappers.scikit_learn import KerasClassifier` | 제거됨 | 미사용이므로 삭제 |
| 6 | `tf.reset_default_graph()` | TF1 전용 | `keras.backend.clear_session()`만 남김 |
| 7 | `model.fit_generator / evaluate_generator / predict_generator` | 제거됨 | `fit / evaluate / predict` (generator 직접 전달 가능) |
| 8 | `TensorBoard(batch_size=...)` | 인자 제거됨 | 인자 삭제 |
| 9 | `compute_class_weight('balanced', np.unique(y), y)` | sklearn ≥1.2 키워드 전용 | `compute_class_weight(class_weight='balanced', classes=..., y=...)` |
| 10 | `class_weight=<ndarray>` → `fit()` | dict 필요 | `{0: w0, 1: w1}` |
| 11 | `monitor='val_acc'`, `history.history['acc']` | 키 이름 변경 | `val_accuracy` / `accuracy` |
| 12 | 체크포인트 `*.hdf5` | Keras 3는 `.keras` 권장, `.h5`도 가능 | `.keras` |
| 13 | `testing_dir + "\\" + filename` | Windows 경로 하드코딩 | `os.path.join` |
| 14 | 셀 35: `cols`, `rows` 미정의 | NameError | 변수 정의 또는 셀 삭제 |
| 15 | `mlxtend.plot_confusion_matrix` | 추가 의존성 | sklearn `ConfusionMatrixDisplay`로 대체 |

### B. 돌아가긴 하지만 잘못된 것 (발표에서 "베이스 레포의 결함"으로 언급 가능)
| # | 내용 | 영향 |
|---|---|---|
| 16 | `get_model()`(InceptionV3)에 GlobalAveragePooling 없이 `Dense(2)`를 바로 연결 | 출력이 (B, 3, 3, 2) 4D 텐서 → 라벨 shape 불일치. 이 모델이 "미사용"인 이유로 추정 |
| 17 | 데이터 분할 320/320은 저자가 임의로 만든 것, 스크립트 없음 | **재현 불가**. Kaggle 원본은 val 16 / test 624. → 우리 쪽에서 train의 10%를 떼어 val을 만들고, test 624는 그대로 고정 (분할 스크립트 제공) |
| 18 | `batch_size=163`, validation batch = 전체 val 수 | GPU 메모리에 따라 OOM. 32로 통일 |
| 19 | `horizontal_flip=True` | X-ray는 좌우 비대칭(심장 위치)이라 플립 증강은 논란 있음. 베이스라인 재현 땐 유지, 개선판에선 끄는 것도 실험 항목 가능 |
| 20 | `shuffle=True` 학습 generator + `seed` 미지정 | 시드 고정 안 됨 → 재현성 없음. `tf.random.set_seed`, `np.random.seed`, generator `seed=` 추가 |
| 21 | README 지표 "Accuracy (F-1) Score 89.53%" | 정확도인지 F1인지 불명확. 우리 재현 결과를 기준값으로 삼고 README 수치는 참고만 |
| 22 | **원본 CNN 구조 자체가 TF2에서 학습되지 않음** (2026-11-16 Colab T4 실측) | loss 가 ln2=0.693 에서 12 epoch 내내 고정, 테스트 전부 PNEUMONIA 예측(acc 0.625 = 폐렴 비율, AUC 0.5). 원인: Conv-ReLU 10층에 BN 이 없어 활성값이 층을 지날수록 소실(초기화 직후 입력 std 0.13 → 마지막 conv std 0.006 → 로짓 차이 0.003). lr 1e-4, He 초기화로도 복구 안 됨. **각 Conv 뒤 BatchNorm 추가(`--bn`)로 해결** — 채널·커널·풀링·Dense 는 원본과 동일 |

---

## 3. 데이터 준비 (Kaggle)

```bash
# kaggle API (토큰 필요) 또는 웹에서 직접 다운로드
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw
# → data/raw/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
python make_split.py --src data/raw/chest_xray --dst data/input --val-ratio 0.1 --seed 42
# → data/input/{train,val,test}  (val 16장 문제 해결, test 624 고정)
```

---

## 4. 제공 스크립트

| 파일 | 역할 |
|---|---|
| `make_split.py` | Kaggle 원본 → train/val(10%)/test 재분할 (test는 원본 유지) |
| `degrade.py` | 열화 함수 5종 × 강도 5단계 + 홀드아웃 구분. 다른 스크립트가 import |
| `train_baseline.py` | 베이스 CNN을 TF2로 포팅. `--clahe`, `--degrade-aug`, `--backbone` 플래그로 개선판도 같은 스크립트에서 학습 |
| `eval_robustness.py` | 저장된 모델을 깨끗한 test + 모든 열화×강도에서 평가 → CSV + 곡선 그림 |

### 실험 매트릭스 (2주차)
```
# 원본 그대로 (학습 실패 재현용, 보고서에 "원본은 재현 불가" 근거로 1~2 seed 만)
python train_baseline.py --seed 0 --tag base_orig
# 수정 베이스라인 = 원본 + BN (이후 모든 개선 실험의 기준)
python train_baseline.py --seed 0 --tag base --bn
# 전처리만
python train_baseline.py --seed 0 --bn --clahe --tag clahe
# 증강만 (홀드아웃 열화 제외)
python train_baseline.py --seed 0 --bn --degrade-aug --tag aug
# 둘 다
python train_baseline.py --seed 0 --bn --clahe --degrade-aug --tag both
# (선택) 백본 교체
python train_baseline.py --seed 0 --backbone efficientnetb0 --clahe --degrade-aug --tag effb0_both

# 평가 (각 모델)
python eval_robustness.py --model runs/base_s0/best.keras --tag base_s0
```
`--clahe`는 학습·평가 양쪽에 똑같이 적용되어야 하므로 `eval_robustness.py`에도 같은 플래그가 있음.

### 열화 종류와 홀드아웃
| 열화 | 현실 근거 | 학습 증강 사용 |
|---|---|---|
| gaussian_noise | 저선량 촬영 시 양자 노이즈 | ○ |
| gaussian_blur | 환자 움직임, 초점 | ○ |
| contrast (감마/대비 저하) | 장비·노출 설정 차이 | ○ |
| jpeg | 전송·저장 압축 | **× (홀드아웃)** |
| motion_blur (방향성) | 호흡·움직임 | **× (홀드아웃)** |

홀드아웃 2종은 학습에 절대 쓰지 않고 평가에만 사용 → "본 적 없는 열화에도 강건한가"를 답함.
