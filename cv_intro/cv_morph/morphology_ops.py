import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_greyscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def _slide(image: np.ndarray, se: np.ndarray, pad: int, stride: int, reduce_fn) -> np.ndarray:
    """dilation/erosion 공통 로직. reduce_fn 에 np.max 또는 np.min 을 넣어서 사용."""
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)

    h, w = image.shape
    se_h, se_w = se.shape
    se_mask = se.astype(bool)

    out_h = (h - 1) // stride + 1
    out_w = (w - 1) // stride + 1
    output = np.zeros((out_h, out_w), dtype=np.uint8)

    r_off = pad - se_h // 2
    c_off = pad - se_w // 2

    for out_i, i in enumerate(range(0, h, stride)):
        for out_j, j in enumerate(range(0, w, stride)):
            r0, c0 = i + r_off, j + c_off
            window = padded[r0:r0 + se_h, c0:c0 + se_w]
            output[out_i, out_j] = reduce_fn(window[se_mask])

    return output


def dilate(image: np.ndarray, se: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
    """Dilation: 이웃 중 최댓값."""
    return _slide(image, se, pad, stride, np.max)


def erode(image: np.ndarray, se: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
    """Erosion: 이웃 중 최솟값."""
    return _slide(image, se, pad, stride, np.min)


def opening(image: np.ndarray, se: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
    """Opening = Erosion 한 번 -> Dilation 한 번 (작은 돌기/노이즈 제거)."""
    eroded = erode(image, se, pad, stride)
    return dilate(eroded, se, pad, stride)


def closing(image: np.ndarray, se: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
    """Closing = Dilation 한 번 -> Erosion 한 번 (작은 구멍/틈 메움)."""
    dilated = dilate(image, se, pad, stride)
    return erode(dilated, se, pad, stride)


# 이번 데모에서 쓸 structuring element: Square 3x3 (가장 기본적인 SE)
SE = np.ones((3, 3), dtype=np.uint8)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    img = load_greyscale(path)

    results = {
        "Original": img,
        "Erosion": erode(img, SE),
        "Opening": opening(img, SE),
        "Closing": closing(img, SE),
    }

    for name, result in results.items():
        if name == "Original":
            continue
        out_name = f"{name.lower()}.png"
        Image.fromarray(result).save(out_name)
        print(f"Saved: {out_name}")

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    for ax, (name, result) in zip(axes, results.items()):
        ax.imshow(result, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("morphology_comparison.png", dpi=150)
    print("Saved: morphology_comparison.png")


if __name__ == "__main__":
    main()
