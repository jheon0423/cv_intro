# CIFAR-10 KNN Assignment

CIFAR-10 이미지 분류를 위한 KNN 직접 구현 및 성능 평가 코드입니다. KNN 분류에는 `sklearn.neighbors.KNeighborsClassifier`를 사용하지 않습니다.

## 실행 방법

Python 3.10 이상 환경에서 아래 명령어를 실행합니다.

```bash
python -m pip install -r requirements.txt
python main.py
```

데이터가 없으면 토론토 대학교의 공식 CIFAR-10 Python 버전(약 163MB)을 `data/` 폴더로 다운로드합니다. 인터넷 사용이 어려우면 [공식 CIFAR-10 사이트](https://www.cs.toronto.edu/~kriz/cifar.html)에서 `cifar-10-python.tar.gz`를 받아 `data/` 폴더에 넣거나 다음과 같이 기존 파일 경로를 전달합니다.

```bash
python main.py --data /path/to/cifar-10-python.tar.gz
```

압축을 풀었다면 `--data`에 `cifar-10-batches-py` 폴더를 지정해도 됩니다.

## 실험 설정

기본 설정은 공식 훈련 데이터에서 계층적 무작위 추출한 5,000장과 공식 테스트 데이터 1,000장을 사용합니다. 전체 데이터를 사용하려면 `--train-size 0 --test-size 0`을 지정합니다. 전체 데이터는 KNN의 연산량 때문에 처리 시간이 상당히 길어질 수 있습니다.

- k: 1, 3, 5, 7, 9
- 거리 척도: Euclidean, Manhattan, Cosine
- 검증: Stratified 5-fold 교차검증
- 특성: 32×32×3 RGB 픽셀을 3,072차원으로 평탄화하고 255로 나눠 정규화
- 평가: Accuracy, macro Precision, macro Recall, macro F1-score
- 최적 조합: 교차검증 평균 Accuracy가 가장 높은 조합을 선정하고 동률이면 F1-score로 비교
- 최종 평가: 선정한 조합을 훈련 데이터 전체에 학습한 뒤 독립 테스트 세트에서 1회 평가
- 투표 동률: 클래스 번호가 가장 작은 클래스 선택

다른 설정의 예시는 다음과 같습니다.

```bash
python main.py --train-size 10000 --test-size 2000 --k-values 1 3 5 7 11 --metrics euclidean cosine --batch-size 64 --seed 42
python main.py --train-size 0 --test-size 0 --output-dir results_full
```

## 생성되는 결과

`results/` 폴더에 다음 파일이 자동 생성됩니다.

| 파일 | 내용 |
|---|---|
| `cv_folds.csv` | 각 k, 거리 척도, fold별 4개 평가 지표 |
| `cv_summary.csv` | 각 설정의 5-fold 평균 및 표준편차 |
| `test_metrics.csv` | 최적 설정의 독립 테스트 성능 |
| `test_per_class.csv` | 10개 클래스별 Precision, Recall, F1-score |
| `test_confusion_matrix.csv` | 테스트 혼동 행렬 |
| `experiment_report.md` | 결과 비교표 및 간단한 분석 보고서 |

보고서에 기록되는 값은 실제 실행 결과로 자동 채워지며, 미리 만들어 둔 가상 결과값은 포함하지 않습니다. 전체 데이터로 실험하려면 메모리와 실행 시간을 충분히 확보하세요.
