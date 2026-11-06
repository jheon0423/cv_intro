# CIFAR-10 KNN Experiment Report

## Experimental setup

- Training samples: 5000
- Test samples: 1000
- Features: 32×32×3 pixels, flattened to 3072 dimensions; pixel intensities divided by 255
- Validation: 5-fold stratified cross-validation on training data only
- Metrics: Accuracy, macro Precision, macro Recall, macro F1-score
- Tie handling: smallest class label is selected when votes are tied
- Model selection: highest mean cross-validation Accuracy, then macro F1-score

## Cross-validation results (mean ± sample standard deviation)

| Distance | k | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| manhattan | 9 | 0.3066 ± 0.0118 | 0.3810 ± 0.0213 | 0.3066 ± 0.0118 | 0.2921 ± 0.0174 |
| manhattan | 5 | 0.3032 ± 0.0114 | 0.3668 ± 0.0182 | 0.3032 ± 0.0114 | 0.2895 ± 0.0126 |
| manhattan | 7 | 0.2962 ± 0.0171 | 0.3681 ± 0.0357 | 0.2962 ± 0.0171 | 0.2829 ± 0.0241 |
| cosine | 1 | 0.2934 ± 0.0220 | 0.3251 ± 0.0292 | 0.2934 ± 0.0220 | 0.2822 ± 0.0269 |
| manhattan | 1 | 0.2924 ± 0.0095 | 0.3350 ± 0.0189 | 0.2924 ± 0.0095 | 0.2874 ± 0.0111 |
| manhattan | 3 | 0.2884 ± 0.0206 | 0.3677 ± 0.0266 | 0.2884 ± 0.0206 | 0.2760 ± 0.0212 |
| cosine | 7 | 0.2874 ± 0.0051 | 0.3519 ± 0.0166 | 0.2874 ± 0.0051 | 0.2682 ± 0.0101 |
| cosine | 5 | 0.2870 ± 0.0087 | 0.3488 ± 0.0196 | 0.2870 ± 0.0087 | 0.2681 ± 0.0136 |
| cosine | 9 | 0.2866 ± 0.0076 | 0.3534 ± 0.0182 | 0.2866 ± 0.0076 | 0.2648 ± 0.0096 |
| cosine | 3 | 0.2834 ± 0.0179 | 0.3533 ± 0.0334 | 0.2834 ± 0.0179 | 0.2652 ± 0.0248 |
| euclidean | 9 | 0.2778 ± 0.0140 | 0.3750 ± 0.0178 | 0.2778 ± 0.0140 | 0.2528 ± 0.0149 |
| euclidean | 7 | 0.2762 ± 0.0085 | 0.3643 ± 0.0107 | 0.2762 ± 0.0085 | 0.2526 ± 0.0098 |
| euclidean | 1 | 0.2746 ± 0.0114 | 0.3221 ± 0.0070 | 0.2746 ± 0.0114 | 0.2641 ± 0.0069 |
| euclidean | 5 | 0.2694 ± 0.0128 | 0.3361 ± 0.0213 | 0.2694 ± 0.0128 | 0.2480 ± 0.0172 |
| euclidean | 3 | 0.2686 ± 0.0074 | 0.3484 ± 0.0156 | 0.2686 ± 0.0074 | 0.2477 ± 0.0078 |

## Selected configuration and held-out test results

- Distance metric: **manhattan**
- k: **9**
- Test Accuracy: **0.3180**
- Test macro Precision: **0.3919**
- Test macro Recall: **0.3180**
- Test macro F1-score: **0.3026**

## Brief comparison and explanation

- Best k for each distance metric: cosine: k=1 (Accuracy 0.2934), euclidean: k=9 (Accuracy 0.2778), manhattan: k=9 (Accuracy 0.3066).
- cosine: k=1 → k=9에서 평균 Accuracy 변화 -0.0068. 해당 거리 척도의 최적 k는 1.
- euclidean: k=1 → k=9에서 평균 Accuracy 변화 +0.0032. 해당 거리 척도의 최적 k는 9.
- manhattan: k=1 → k=9에서 평균 Accuracy 변화 +0.0142. 해당 거리 척도의 최적 k는 9.
- Euclidean distance measures straight-line pixel differences, Manhattan distance sums absolute pixel differences, and cosine distance compares the direction of pixel vectors.
- A small k can be sensitive to individual images or noise; a large k smooths predictions but can include neighbors from other classes.
- Raw-pixel KNN ignores spatial structure such as edges and object positions, which can limit image-classification performance.
- The test set was not used to tune k or the distance metric.

## Output files

- cv_folds.csv: metrics for each configuration and fold
- cv_summary.csv: cross-validation means and standard deviations
- test_metrics.csv: held-out test metrics for the selected configuration
- test_per_class.csv: per-class Precision, Recall and F1-score
- test_confusion_matrix.csv: true-class rows and predicted-class columns
