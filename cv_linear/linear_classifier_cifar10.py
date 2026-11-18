"""CIFAR-10 random linear classifier assignment (NO training).

Requirements: numpy, matplotlib
Run: python linear_classifier_cifar10.py
Or:  python linear_classifier_cifar10.py --dataset ./cifar-10-python.tar.gz

On execution, this script updates RESULTS.md with *actual* CIFAR-10 results
and creates two visualization images under generated_results/.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import tarfile
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Works in terminal / headless environments.
import matplotlib.pyplot as plt
import numpy as np

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
SEED = 42
WEIGHT_STD = 0.01
MARGIN = 1.0
SCRIPT_DIR = Path(__file__).resolve().parent


def load_batch_from_bytes(raw: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Read original CIFAR-10 Python-format test_batch data."""
    import io

    obj = pickle.load(io.BytesIO(raw), encoding="bytes")
    images = obj.get(b"data", obj.get("data"))
    labels = obj.get(b"labels", obj.get("labels"))
    if images is None or labels is None:
        raise ValueError("test_batch 형식이 올바르지 않습니다 (data/labels 누락).")
    images = np.asarray(images, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    if images.ndim != 2 or images.shape[1] != 3072 or images.shape[0] != labels.size:
        raise ValueError("CIFAR-10 배열 형태가 올바르지 않습니다 (N x 3072).")
    if np.any((labels < 0) | (labels >= 10)):
        raise ValueError("CIFAR-10의 클래스 ID 범위는 0~9입니다.")
    return images, labels


def batch_from_tar(archive: Path) -> tuple[np.ndarray, np.ndarray]:
    # We read the test batch directly, rather than extracting the entire archive.
    with tarfile.open(archive, mode="r:gz") as tar:
        member = next(
            (m for m in tar.getmembers() if m.isfile() and Path(m.name).name == "test_batch"),
            None,
        )
        if member is None:
            raise FileNotFoundError("압축파일에서 CIFAR-10 test_batch를 찾지 못했습니다.")
        stream = tar.extractfile(member)
        if stream is None:
            raise ValueError("압축파일의 test_batch를 읽을 수 없습니다.")
        return load_batch_from_bytes(stream.read())


def get_batch(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    path = path.expanduser().resolve()
    if path.is_file() and path.name == "test_batch":
        return (*load_batch_from_bytes(path.read_bytes()), str(path))
    if path.is_file() and (path.name.endswith(".tar.gz") or path.suffix == ".tgz"):
        return (*batch_from_tar(path), str(path))
    if path.is_dir():
        candidates = (
            path / "test_batch",
            path / "cifar-10-batches-py" / "test_batch",
            path / "cifar-10-python.tar.gz",
            path / "data" / "cifar-10-batches-py" / "test_batch",
            path / "data" / "cifar-10-python.tar.gz",
        )
        for candidate in candidates:
            if candidate.exists():
                return get_batch(candidate)
    raise FileNotFoundError(f"사용 가능한 CIFAR-10 test_batch 또는 압축파일 없음: {path}")


def auto_get_dataset(given_path: str | None, no_download: bool):
    if given_path:
        return get_batch(Path(given_path))
    for folder in (SCRIPT_DIR, Path.cwd(), SCRIPT_DIR.parent):
        try:
            return get_batch(folder)
        except FileNotFoundError:
            continue

    if no_download:
        raise FileNotFoundError("CIFAR-10 데이터가 발견되지 않았으며 --no-download 옵션이 설정되었습니다.")

    target = SCRIPT_DIR / "cifar-10-python.tar.gz"
    partial = SCRIPT_DIR / "cifar-10-python.tar.gz.part"
    print("CIFAR-10 파일이 없어 공식 배포처에서 자동 다운로드를 시도합니다 (약 163MB).")
    print("출처:", CIFAR_URL)
    try:
        urllib.request.urlretrieve(CIFAR_URL, partial)
        h = hashlib.md5()  # Official dataset checksum, not cryptographic security.
        with partial.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != CIFAR_MD5:
            raise ValueError("다운로드 파일 MD5 체크섬이 공식 배포처 정보와 일치하지 않습니다.")
        partial.replace(target)
        return get_batch(target)
    except Exception as exc:
        raise RuntimeError(
            "CIFAR-10 데이터 다운로드에 실패했습니다.\n"
            "공식 사이트에서 'CIFAR-10 python version'을 내려받고,\n"
            "cifar-10-python.tar.gz를 .py 파일 옆에 놓거나\n"
            "--dataset /path/to/cifar-10-python.tar.gz 옵션으로 실행하세요.\n"
            f"원인: {exc}"
        ) from exc
    finally:
        partial.unlink(missing_ok=True)


def initialize_parameters(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=WEIGHT_STD, size=(10, 32 * 32 * 3))
    bias = np.zeros(10, dtype=np.float64)
    return weights, bias


def linear_scores(x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """f(x) = Wx + b. x: (N,3072), W: (10,3072), b: (10,)."""
    return x @ weights.T + bias[None, :]


def multiclass_svm_hinge_loss(scores: np.ndarray, labels: np.ndarray, delta: float = MARGIN):
    """Unregularized sum-over-incorrect-classes multiclass SVM hinge loss."""
    n = scores.shape[0]
    correct = scores[np.arange(n), labels][:, None]
    margins = np.maximum(0.0, scores - correct + delta)
    margins[np.arange(n), labels] = 0.0
    per_image = margins.sum(axis=1)
    return per_image, float(per_image.mean())


def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def select_examples(predictions: np.ndarray, labels: np.ndarray, correct_count: int = 2, incorrect_count: int = 4) -> np.ndarray:
    """Choose illustrative examples, not an unbiased accuracy sample."""
    correct = np.flatnonzero(predictions == labels)[:correct_count]
    incorrect = np.flatnonzero(predictions != labels)[:incorrect_count]
    if correct.size == 0:
        raise RuntimeError("검사한 이미지 중 정답 예시가 없습니다. --scan-count를 늘려주세요.")
    if incorrect.size == 0:
        raise RuntimeError("검사한 이미지 중 오답 예시가 없습니다. --scan-count를 늘려주세요.")
    return np.sort(np.concatenate((correct, incorrect)))


def to_hwc(flat: np.ndarray) -> np.ndarray:
    """CIFAR-10 uses channel-first flattened storage: R(1024), G(1024), B(1024)."""
    return flat.reshape(3, 32, 32).transpose(1, 2, 0)


def save_examples_figure(raw_images: np.ndarray, labels: np.ndarray, predictions: np.ndarray, indices: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, index in zip(axes.flat, indices):
        ax.imshow(to_hwc(raw_images[int(index)]))
        status = "CORRECT" if labels[index] == predictions[index] else "WRONG"
        ax.set_title(
            f"test #{index} | {status}\ntrue: {CLASSES[labels[index]]} | pred: {CLASSES[predictions[index]]}",
            fontsize=10,
        )
        ax.axis("off")
    for ax in axes.flat[len(indices):]:
        ax.axis("off")
    fig.suptitle("Random linear classifier: CIFAR-10 test examples (no training)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_weights_figure(weights: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i, ax in enumerate(axes.flat):
        image = to_hwc(weights[i])
        # Display normalization only. Raw parameters are not modified.
        low, high = image.min(), image.max()
        display = (image - low) / (high - low + 1e-12)
        ax.imshow(display)
        ax.set_title(f"{i}: {CLASSES[i]}", fontsize=11)
        ax.axis("off")
    fig.suptitle("Untrained class weights (32 x 32 x 3, normalized for display)", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_results(
    indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    probabilities: np.ndarray,
    per_image_loss: np.ndarray,
    avg_loss: float,
    weights: np.ndarray,
    dataset_source: str,
    output_dir: Path,
) -> Path:
    descriptions = []
    for row, idx in enumerate(indices):
        truth = int(labels[row])
        pred = int(predictions[row])
        status = "정답" if truth == pred else "오답"
        descriptions.append([
            f"{int(idx)}", CLASSES[truth], CLASSES[pred], status,
            f"{per_image_loss[row]:.6f}",
            f"{probabilities[row].sum():.12f}",
        ])

    text = [
        "# CIFAR-10 무작위 선형 분류기 — 실행 결과 및 해설",
        "",
        "> 이 보고서는 `linear_classifier_cifar10.py` 실행 결과를 바탕으로 자동 생성되었습니다. **모델 학습/최적화는 전혀 수행하지 않았습니다.**",
        "",
        "## 1. 환경 및 실험 설정",
        "",
        f"- 데이터: 공식 CIFAR-10 **test_batch** ({dataset_source}), RGB 32×32×3, 클래스 10개.",
        "- 입력: `uint8` 픽셀을 `[0, 1]`로 나눈 뒤, CIFAR 공식 채널 순서 그대로 3,072차원 벡터로 사용.",
        f"- 시드: **{SEED}**, 가중치: `W ~ N(0, {WEIGHT_STD}²)` / `W.shape=(10,3072)`, 편향: `b=0`.",
        "- 점수 계산: `f(x)=Wx+b`; 최고 점수 `argmax`가 예측 클래스.",
        "- 예시 선정: 앞쪽 검사 이미지 중 **처음 찾은 정답 최대 2개 + 오답 최대 4개**. 분류 결과를 보여주기 위한 의도적인 선정이므로 대표 표본이나 정확도 추정용이 아님.",
        f"- 선택된 이미지 수: **{len(indices)}개**. 정확도·정밀도·재현율·F1은 계산하지 않음.",
        "",
        "## 2. 예시 이미지와 예측 결과",
        "",
        "![선택한 CIFAR-10 원본 이미지 및 예측](generated_results/selected_examples.png)",
        "",
        markdown_table(["test 인덱스", "실제 클래스", "예측 클래스", "결과", "hinge loss", "확률 합"], descriptions),
        "",
        "## 3. 모든 클래스 점수와 Softmax 확률",
        "",
        "아래 표에서 `score`는 제한 없는 실수 점수이고, `softmax`는 0~1 사이 확률 형태의 값입니다. 가장 큰 score와 가장 큰 softmax는 같은 클래스를 가리킵니다.",
        "",
    ]
    for row, idx in enumerate(indices):
        truth = int(labels[row])
        pred = int(predictions[row])
        text += [
            f"### test #{int(idx)} — 실제 {CLASSES[truth]}, 예측 {CLASSES[pred]}",
            "",
            markdown_table(
                ["클래스(ID)", "원점수 score", "Softmax 확률"],
                [[f"{CLASSES[j]} ({j})", f"{scores[row,j]:+.6f}", f"{probabilities[row,j]:.6f}"] for j in range(10)],
            ),
            "",
            f"- 최고 점수 클래스: **{CLASSES[pred]}**; 확률 합: **{probabilities[row].sum():.12f}**; 해당 이미지 hinge loss: **{per_image_loss[row]:.6f}**",
            "",
        ]
    text += [
        "## 4. 다중 클래스 SVM hinge loss",
        "",
        "각 이미지에서 정답 클래스를 `y`라 할 때, `L_i = Σ_{j≠y} max(0, s_j - s_y + 1)`로 계산했습니다. 정답 클래스 자체는 손실 합산에서 제외했으며, 정규화항도 넣지 않았습니다.",
        "",
        f"- 예시 {len(indices)}개 평균 손실: **{avg_loss:.6f}**",
        "- 정답을 예측했더라도 2등 이하의 점수들과 margin(1.0)을 충분히 벌리지 못하면 hinge loss는 0보다 클 수 있습니다.",
        "- 위 평균은 의도적으로 선정한 몇 개 예시의 평균이지 CIFAR-10 전체 테스트셋 평균은 아닙니다.",
        "",
        "## 5. Softmax 원점수와 확률의 차이",
        "",
        "`softmax(s_j) = exp(s_j - m) / Σ_k exp(s_k - m)` (`m=max_k s_k`)로 안정적으로 구현했습니다. 입력값마다 최대 점수를 빼면 overflow를 예방하면서 동일한 확률을 얻습니다.",
        "",
        f"- 모든 예시에서 확률의 합이 1인지 검사: **{'통과' if np.allclose(probabilities.sum(axis=1),1.0,rtol=0,atol=1e-12) else '실패'}** (`np.allclose`, atol=1e-12).",
        "- Softmax는 점수 순서를 바꾸지 않으므로 예측 클래스를 바꾸지 않습니다. 단, 학습되지 않은 모델의 softmax 값을 *신뢰할 수 있는 예측 확률*이라고 해석할 수는 없습니다.",
        "",
        "## 6. 정답/오답 사례 해석",
        "",
    ]
    good = next((r for r in range(len(indices)) if labels[r] == predictions[r]), None)
    bad = next((r for r in range(len(indices)) if labels[r] != predictions[r]), None)
    if good is not None:
        text += [f"- **정답 사례:** test #{int(indices[good])}의 실제 클래스는 `{CLASSES[labels[good]]}`이고, 무작위 점수의 최대값도 `{CLASSES[predictions[good]]}`에 해당합니다. 다만 올바른 특징을 학습한 결과라고 볼 수는 없습니다."]
    if bad is not None:
        text += [f"- **오답 사례:** test #{int(indices[bad])}의 실제 클래스는 `{CLASSES[labels[bad]]}`이지만 점수 최댓값은 `{CLASSES[predictions[bad]]}`이므로 잘못 분류되었습니다."]
    text += [
        "- 가중치가 무작위 상태에서 고정되어 있기 때문에 이미지 속 사물의 모양이나 의미에 맞는 판단 기준이 만들어지지 않았습니다. 우연히 맞는 이미지가 있더라도 학습 효과는 아닙니다.",
        "",
        "## 7. 클래스별 32×32×3 가중치 시각화",
        "",
        "![무작위 클래스 가중치 시각화](generated_results/weights_visualization.png)",
        "",
        f"- `W`의 원소 평균: **{weights.mean():+.8f}**, 표준편차: **{weights.std():.8f}**, 범위: **[{weights.min():+.6f}, {weights.max():+.6f}]**.",
        "- 행렬 `W`의 각 행(클래스별 3,072차원 가중치)을 CIFAR의 RGB 채널 순서에 맞춰 `(3,32,32) → (32,32,3)`으로 변형했습니다.",
        "- 이미지 출력 시에만 클래스별 min-max 정규화를 적용하여 눈에 보이도록 했고, 실제 계산에 사용한 `W`는 수정하지 않았습니다.",
        "- 그림에 불규칙한 색 잡음이 나타나는 이유는 각 픽셀 가중치가 무작위로 정해졌고 학습·업데이트가 전혀 없기 때문입니다. 물체의 윤곽, 색 또는 질감을 특정 클래스와 연결하는 구조를 학습하지 못했습니다.",
        "",
        "## 8. 실행 및 재현 방법",
        "",
        "```bash",
        "pip install numpy matplotlib",
        "python linear_classifier_cifar10.py",
        "# 로컬 압축파일을 사용할 경우:",
        "python linear_classifier_cifar10.py --dataset ./cifar-10-python.tar.gz",
        "```",
        "",
        "실행 후 이 `RESULTS.md`가 실제 계산값으로 다시 생성되고 `generated_results/`에 2개의 PNG가 만들어집니다. 원본 CIFAR-10 163MB 데이터는 제출 ZIP에 포함하지 않습니다.",
        "",
        "데이터 출처: [CIFAR-10 공식 배포 페이지](https://cave.cs.toronto.edu/kriz/cifar.html).",
        "",
    ]
    target = SCRIPT_DIR / "RESULTS.md"
    target.write_text("\n".join(text), encoding="utf-8")
    return target


def basic_self_check() -> None:
    w, b = initialize_parameters()
    assert w.shape == (10, 3072) and b.shape == (10,)
    assert np.all(b == 0)
    assert np.array_equal(initialize_parameters()[0], w)
    zero_x = np.zeros((1, 3072), dtype=np.float64)
    zero_scores = linear_scores(zero_x, w, b)
    assert np.all(zero_scores == 0)
    zero_prob = softmax(zero_scores)
    assert np.allclose(zero_prob, 0.1)
    losses, average = multiclass_svm_hinge_loss(zero_scores, np.array([3]))
    assert losses[0] == 9.0 and average == 9.0
    assert np.allclose(softmax(np.array([[1200.0, 1201.0]] )).sum(), 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random (untrained) CIFAR-10 linear classifier")
    parser.add_argument("--dataset", default=None, help="CIFAR-10 archive / test_batch / extracted directory")
    parser.add_argument("--no-download", action="store_true", help="Disable automatic dataset download")
    parser.add_argument("--scan-count", type=int, default=1000, help="Test images scanned to find illustrative correct/wrong cases")
    args = parser.parse_args()

    basic_self_check()
    raw_images, all_labels, source = auto_get_dataset(args.dataset, args.no_download)
    n = min(max(args.scan_count, 1), len(all_labels))
    # Official CIFAR flattened representation is already CHW. No reordering here.
    x = raw_images[:n].astype(np.float64) / 255.0
    y = all_labels[:n]

    weights, bias = initialize_parameters(SEED)
    all_scores = linear_scores(x, weights, bias)
    all_predictions = np.argmax(all_scores, axis=1)
    indices = select_examples(all_predictions, y)
    scores = all_scores[indices]
    labels = y[indices]
    predictions = all_predictions[indices]
    probabilities = softmax(scores)
    per_image_loss, avg_loss = multiclass_svm_hinge_loss(scores, labels)
    assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.array_equal(np.argmax(probabilities, axis=1), predictions)

    output_dir = SCRIPT_DIR / "generated_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_examples_figure(raw_images, y, all_predictions, indices, output_dir / "selected_examples.png")
    save_weights_figure(weights, output_dir / "weights_visualization.png")
    report = write_results(
        indices, labels, predictions, scores, probabilities,
        per_image_loss, avg_loss, weights, source, output_dir,
    )

    print("=" * 72)
    print("CIFAR-10 RANDOM LINEAR CLASSIFIER (NO TRAINING)")
    print("Data:", source)
    print(f"W shape={weights.shape}, b shape={bias.shape}, seed={SEED}")
    print(f"Chosen indices: {indices.tolist()}")
    print("=" * 72)
    for i, idx in enumerate(indices):
        correct = labels[i] == predictions[i]
        print(f"\nTEST #{idx}: true={CLASSES[labels[i]]}, pred={CLASSES[predictions[i]]}, {'CORRECT' if correct else 'WRONG'}")
        print("Class              Score          Softmax")
        for c in range(10):
            print(f"{CLASSES[c]:<12}  {scores[i,c]:+12.6f}     {probabilities[i,c]:.6f}")
        print(f"P-sum={probabilities[i].sum():.12f}, hinge loss={per_image_loss[i]:.6f}")
    print("\nMean multiclass SVM hinge loss (selected examples):", f"{avg_loss:.6f}")
    print("Markdown report:", report)
    print("Figures:", output_dir / "selected_examples.png", "and", output_dir / "weights_visualization.png")


if __name__ == "__main__":
    main()
