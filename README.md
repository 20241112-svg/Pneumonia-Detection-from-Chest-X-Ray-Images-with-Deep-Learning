# 흉부 X-ray 폐렴 분류 모델의 열화(저화질) 강건성 개선

GitHub 공개 폐렴 분류 모델을 가져와, 영상 화질이 나빠졌을 때 성능이 무너지는 문제를 측정하고 줄인 수업 과제입니다.

![열화 강도별 정확도](improved/report/robustness_curves.png)

## 참고 코드 (원본)
- 레포: [anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning](https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning) (MIT 라이선스) — [zip 다운로드](https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning/archive/refs/heads/master.zip)
- 데이터: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- 원본 내용: 커스텀 CNN(Conv 10층)으로 흉부 X-ray를 정상/폐렴으로 분류, 보고 정확도 89.53%
- 이 레포는 원본의 포크입니다. 원본 코드는 `code/`에, 원본 소개는 [`README_original.md`](README_original.md)에 그대로 있습니다.

## 내 코드
모두 [`improved/`](improved/) 폴더에 있습니다. 자세한 설명과 전체 결과는 [`improved/README.md`](improved/README.md).

| 파일 | 내용 |
|---|---|
| `train_baseline.py` | 원본 노트북을 TF 2 / Keras 3로 옮긴 학습 스크립트 + 개선 옵션(`--bn --clahe --degrade-aug`) |
| `degrade.py` | 열화 5종(노이즈·블러·대비 저하·JPEG 압축·모션블러) × 강도 5단계 |
| `eval_robustness.py` | 깨끗한 test + 열화 25개 조건 평가 |
| `analyze_threshold.py` | 붕괴 원인 분석(임계값 분석) |
| `train_lora_vit.py` | 구조 변경 비교군: 사전학습 ViT-S/16 + LoRA |
| `run_pipeline.py`, `colab_run.ipynb` | 전체 실험 실행 |
| `demo.py` | 시연: 학습된 모델로 예측·정확도 확인 |
| `report/` | 최종 결과 그림, 조건별 결과 CSV |

## 개선한 점
1. **원본 재현과 학습 실패 수정.** 원본 구조는 현재 환경에서 학습되지 않습니다(전부 폐렴 예측, 62.5%). 원인(활성값 소실)을 진단하고 각 Conv 뒤에 BatchNorm을 추가해 88.8%를 재현했습니다.
2. **열화 평가 기준 추가.** 원본은 깨끗한 영상만 평가합니다. test 624장에 열화 5종 × 강도 5단계 = 25개 조건을 적용해 평가하고, 그중 2종(JPEG·모션블러)은 학습에 쓰지 않았습니다.
3. **열화 증강 + CLAHE 전처리로 강건성 개선.** 학습 영상 수는 그대로 두고, 학습 중 영상을 무작위로 열화시켜 보여줍니다.

| test 624장 정확도 (seed 3개 평균) | 깨끗한 영상 | 학습에 쓴 열화 | 학습에 쓰지 않은 열화 |
|---|---|---|---|
| 베이스라인 (원본 + BN) | 88.8% | 71.2% | 81.3% |
| **개선 모델 (CLAHE + 열화 증강)** | **89.7%** | **85.8%** | **86.5%** |

4. **원인 분석.** 붕괴는 정보 손실이 아니라 예측 확률이 한쪽으로 쏠리는 현상임을 임계값 분석으로 확인했습니다. BatchNorm 통계 재보정의 효과도 따로 측정했습니다.
5. **구조 변경 비교.** 사전학습 ViT + LoRA는 깨끗한 영상에서는 비슷하거나 높지만 열화에서는 개선 모델보다 낮았습니다.

한계: 데이터셋 하나, 합성 열화, 학습에 쓰지 않은 열화 중 개선이 확인된 것은 모션블러입니다. 자세한 내용은 [`improved/README.md`](improved/README.md)의 "한계".

## 실행
```bash
cd improved
pip install -r requirements.txt
python make_split.py --src data/raw/chest_xray --dst data/input --val-ratio 0.1 --seed 42
python train_baseline.py --data data/input --tag base_v2 --seed 0 --bn
python train_baseline.py --data data/input --tag both_v2 --seed 0 --bn --clahe --degrade-aug
python eval_robustness.py --model runs/both_v2_s0/best.keras --tag both_v2_s0
python demo.py --data data/input --runs runs --results results     # 시연
```
Colab에서는 `improved/colab_run.ipynb`를 사용합니다.

코드 작성에 AI 도구(Claude)의 도움을 받았습니다.
