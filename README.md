# **의료 데이터 반출 제한 환경에서의 연합 평가 및 합성 데이터 기반 의료 AI 평가 프레임워크**


이화여자대학교 졸업프로젝트에서 수행한  
**MBP 팀의 의료 AI 벤치마크 연구 및 구현 코드**입니다.


## 프로젝트 소개

의료 AI 모델이 실제 임상 환경에서 활용되기 위해서는 단일 공개 데이터셋에서의 성능뿐만 아니라, 다양한 병원 환경에서의 일반화 성능과 안정성을 검증해야 합니다. 그러나 실제 의료 데이터는 개인정보 보호, 병원 보안 규정, IRB 및 법적 제한 등으로 인해 외부 반출이 어렵습니다.

이로 인해 기존 의료 AI 평가는 공개 데이터셋이나 단일 기관 데이터에 의존하는 경우가 많으며, 실제 병원 환경에서 발생하는 데이터 분포 차이, 촬영 조건 차이, 환자 구성 차이, 라벨 기준 차이를 충분히 반영하기 어렵다는 한계가 있습니다.

본 프로젝트는 이러한 문제를 해결하기 위해 병원 내부 real evaluation과 병원 외부 proxy evaluation을 결합한 평가 프레임워크를 제안합니다.


## 핵심 아이디어 

**“원본 데이터는 병원 내부에 두고, 평가에 필요한 feature-level 정보만 활용한다.”**

본 프로젝트의 핵심 아이디어는 원본 의료 데이터를 직접 공유하지 않고도 모델 평가가 가능하도록, 모델의 중간 표현 또는 통계 정보를 활용하는 것입니다.

- 원본 의료 데이터는 병원 내부 보안 환경에서만 사용
- 병원 내부에서 원본 데이터 기반 real evaluation 수행
- 모델 encoder를 통해 feature vector 추출
- 필요 시 feature vector 대신 class-wise mean, covariance, sample count 등 통계치 생성
- 병원 외부에서 proxy feature set 구성
- 동일한 classifier head를 이용해 proxy evaluation 수행
- real AUROC와 proxy AUROC 비교를 통해 평가 정합성 분석

이를 통해 의료 데이터 반출 제한 환경에서도 모델 성능을 간접적으로 검증할 수 있는 평가 구조를 제안합니다.


## 프로젝트 개요

본 연구는 **의료 데이터 비공개 환경에서 모델을 어떻게 평가할 것인가**에 초점을 둡니다.

기존의 의료 AI 벤치마크는 대부분 공개 데이터셋을 기반으로 여러 모델의 성능을 비교하는 방식으로 구성됩니다. 그러나 실제 병원 데이터는 외부 공개가 어렵고, 병원마다 데이터 분포가 다르기 때문에 공개 데이터셋 성능만으로는 실제 임상 적용 가능성을 판단하기 어렵습니다.

이에 본 연구는 다음과 같은 평가 흐름을 설계하였습니다.

1. 병원 내부에서 원본 데이터 기반 real evaluation 수행
2. 모델 encoder를 이용해 feature vector 추출
3. feature vector 또는 class-wise statistics 기반 proxy set 구성
4. 병원 외부에서 proxy evaluation 수행
4. real AUROC와 proxy AUROC 비교
5. 모델별, 질병별 평가 결과의 일관성 분석


## 프로젝트 목표

- 의료 데이터 반출 제한 환경에서 활용 가능한 AI 평가 프레임워크 제안
- 원본 데이터 없이도 외부에서 모델 성능을 근사적으로 평가할 수 있는 proxy evaluation 구조 설계
- real evaluation과 proxy evaluation의 AUROC 비교를 통한 평가 정합성 검증
- 모델별, 질병별 성능 경향이 proxy evaluation에서도 유지되는지 분석
- 의료 AI 모델의 실제 임상 도입 전 사전 검증 가능성 탐색

## 연구 기간

- 2025년 2학기 ~ 2026년 1학기


## 연구 범위

본 프로젝트에서는 의료 AI 평가 프레임워크의 가능성을 확인하기 위해 다음과 같은 task를 중심으로 설계 및 실험을 진행하였습니다.

**1. 치과 파노라마 X-ray 기반 Multi-label Classification**

치과 파노라마 X-ray 이미지를 대상으로 여러 치과 소견의 존재 여부를 예측하는 multi-label classification task를 설정하였습니다.

- 치아 우식
- 치근단 병소
- 잔존치근
- 치주골 소실
- 매복치

각 모델에 대해 원본 이미지 기반 real AUROC와 proxy feature 기반 proxy AUROC를 계산하고, 두 결과의 차이를 비교합니다.

**2. ECG 기반 Multi-label Classification**

심전도 데이터의 경우 signal-level multi-label classification task로 확장 가능하도록 설계하였습니다.

- AF
- 1° AVB
- RBBB
- PVC
- LBBB


## Data
**1. Hospital Data**

본 연구에서는 다음 병원 데이터를 사용하였습니다.

- 이대목동병원 치과 파노라마 X-ray 데이터
- 이대서울병원 치과 파노라마 X-ray 데이터
- 이대목동병원 ECG 데이터
- 이대서울병원 ECG 데이터

병원 데이터는 개인정보 보호 및 의료 데이터 반출 제한으로 인해 repository에 포함하지 않습니다.

**2. 학습용 공개데이터**
병원 데이터 기반 평가를 수행하기 전, 모델 학습 및 실험 구조 검증을 위해 공개 의료 데이터를 활용하였습니다.

- [DENTEX Challenge 2023](https://dentex.grand-challenge.org/data/)
- [Dataset for Automating Dental Condition Detection on Panoramic Radiographs](https://zenodo.org/records/15487430)


## 실험결과
<img width="1218" height="404" alt="image" src="https://github.com/user-attachments/assets/212973e1-ea78-4930-b1ad-379fdb203df2" />
- 왼쪽 그래프는 모델의 tier별 AUROC 비교 결과를 나타낸다. 각 점은 서로 다른 backbone과 성능 tier 조합을 의미하며, 색상은 low, middle, high tier를 나타낸다. 전반적으로 high tier 모델은 real AUROC와 proxy AUROC가 모두 높은 영역에 위치하였고, low tier 모델은 상대적으로 낮은 AUROC 영역에 위치하였다. 이는 proxy evaluation이 모델의 절대적인 성능 수준뿐만 아니라 학습 정도에 따른 성능 차이도 어느 정도 반영하고 있음을 보여준다. 특히 다수의 점이 대각선 근처에 분포하므로, proxy AUROC가 real AUROC의 전반적인 경향을 비교적 잘 따라가는 것을 확인할 수 있다.
- 오른쪽 그래프는 질병 라벨별 real AUROC와 proxy AUROC의 비교 결과를 나타낸다. 색상은 치아 우식, 치근단 병소, 잔존치근, 치주골 소실, 매복치와 같은 질병 라벨을 의미하며, 마커 형태는 사용된 모델 backbone을 나타낸다. 대부분의 점들이 대각선 주변에 분포하고 있어, proxy evaluation이 질병별 real evaluation 결과를 전반적으로 잘 근사하는 것으로 볼 수 있다. 특히 real AUROC가 높은 라벨에서는 proxy AUROC도 함께 높은 값을 보이는 경향이 나타났으며, 이는 proxy feature가 원본 데이터에서 모델이 학습한 판별 정보를 일정 수준 보존하고 있음을 의미한다.


## 팀원 소개 (MBP Team)

- 김채영  
- 정지현  


## 문서 구조


- `GroundRule.md` : 팀 
- `README.md` : 프로젝트 개요 및 진행 현황
- `Ideation.md` : 프로젝트 문제정의, 핵심 아이디어, 설계 사상 정리
- `docs/` : 보고서 및 발표 자료
- `src/` : 실험 및 평가 코드


## Notes
본 repository에는 개인정보 보호 및 병원 데이터 반출 제한으로 인해 원본 의료 데이터가 포함되어 있지 않습니다.
실험 재현을 위해서는 병원 내부 승인된 환경에서 데이터를 준비하고고, 위에 명시한 공개 데이터셋을 별도로 다운로드해야 합니다.
