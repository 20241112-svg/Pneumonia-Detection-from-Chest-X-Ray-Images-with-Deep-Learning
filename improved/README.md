# improved/ — 열화(저화질) 영상에 대한 폐렴 분류 모델 강건성 개선

원본 레포: [anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning](https://github.com/anjanatiha/Pneumonia-Detection-from-Chest-X-Ray-Images-with-Deep-Learning) (MIT)
원본 코드는 `../code/` 노트북 그대로 두고, 이 폴더에 포팅·개선 코드를 추가했습니다.

## 문제 정의
원본 모델은 깨끗한 Kaggle 테스트 영상에서는 약 89% 정확도를 내지만, 실제 현장 조건(저선량 노이즈, 움직임 블러, 노출·대비 편차, 압축)에서는 성능이 어떻게 변하는지 평가되지 않았습니다.
이 작업은 **(1) 열화 벤치마크를 정의해 원본 모델이 무너지는 지점을 측정하고, (2) 전처리·증강·백본 교체로 강건성을 올리면서 깨끗한 영상 성능은 유지**하는 것을 목표로 합니다.

## 원본 대비 변경점
| 구분 | 내용 |
|---|---|
| 포팅 | Keras 2.2.4 / TF 1.12 노트북 → TF 2.x / Keras 3 스크립트 (`train_baseline.py`). 동일 구조·동일 증강·동일 손실. 고친 항목은 `PORTING_NOTES.md` |
| 학습 실패 수정 | 원본 CNN(BN 없는 Conv-ReLU 10층)은 TF2에서 **활성값 소실로 학습이 안 됨**(loss 0.693 고정, 전부 폐렴 예측, acc 0.625). 각 Conv 뒤 BatchNorm 만 추가한 `--bn` 을 수정 베이스라인으로 사용 (채널·커널·풀링·Dense 동일) |
| 데이터 분할 | 원본의 재현 불가능한 320/320 분할 → Kaggle test 624장 고정 + train 10% 를 val 로 (`make_split.py`, seed 고정) |
| 열화 벤치마크 | 노이즈·블러·대비·JPEG·모션블러 × 강도 5단계 (`degrade.py`). JPEG·모션블러는 **홀드아웃**(학습에 미사용) |
| 개선 플래그 | `--clahe` 전처리, `--degrade-aug` 열화 증강, `--backbone efficientnetb0/resnet50` 전이학습 |
| 구조 변경 (LoRA ViT) | `train_lora_vit.py` — ImageNet 사전학습 ViT-S/16 을 고정하고 **LoRA 어댑터(약 60만 파라미터, 2.7%)** 만 학습. `--mode head`(linear probe) / `full`(전체 미세조정) 비교군 포함. 같은 분할·열화·CSV 형식 사용 |
| 평가 | 깨끗한 test + 25개 열화 조건에서 accuracy/F1/AUC/클래스별 recall → CSV + 곡선 (`eval_robustness.py`) |

## 실행
Colab: `colab_run.ipynb` 를 열어 순서대로 실행 (GPU 런타임).

로컬:
```bash
pip install -r requirements.txt
python make_split.py --src data/raw/chest_xray --dst data/input --val-ratio 0.1 --seed 42
python train_baseline.py --data data/input --tag base --seed 0 --bn       # 수정 베이스라인
python eval_robustness.py --model runs/base_s0/best.keras --tag base_s0  # 열화 평가
python train_baseline.py --data data/input --tag both --seed 0 --bn --clahe --degrade-aug
python eval_robustness.py --model runs/both_s0/best.keras --tag both_s0
python eval_robustness.py --compare results/base_s0.csv results/both_s0.csv --out results/compare.png

# 구조 변경: 사전학습 ViT + LoRA (학습·열화 평가 한 번에, 결과는 같은 results/ 형식)
python train_lora_vit.py --data data/input --tag vit_lora --seed 0 --mode lora
python train_lora_vit.py --data data/input --tag vit_lora_both --seed 0 --mode lora --clahe --degrade-aug
python eval_robustness.py --compare results/base_s0.csv results/both_s0.csv results/vit_lora_s0.csv results/vit_lora_both_s0.csv --out results/compare_vit.png
```

## 실험 매트릭스
| tag | bn | clahe | degrade-aug | backbone |
|---|---|---|---|---|
| base_orig | – | – | – | 원본 CNN (학습 실패 재현) |
| base | ○ | – | – | 원본 CNN + BN |
| clahe | ○ | ○ | – | 원본 CNN + BN |
| aug | ○ | – | ○ | 원본 CNN + BN |
| both | ○ | ○ | ○ | 원본 CNN + BN |
| effb0_both | – | ○ | ○ | EfficientNet-B0 |
| vit_lora | – | – | – | ViT-S/16 + LoRA (train_lora_vit.py) |
| vit_lora_both | – | ○ | ○ | ViT-S/16 + LoRA |
| vit_head / vit_full | – | – | – | ViT-S/16 linear probe / 전체 미세조정 (비교군) |

핵심 비교는 seed 0/1/2 평균 ± 표준편차로 보고.

## 결과
(실험 후 채움)
