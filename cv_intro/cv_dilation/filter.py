import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_greyscale(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.uint8)


def pad_image(image: np.ndarray, pad: int = 1) -> np.ndarray:
    return np.pad(image, ((pad, pad), (pad, pad)),
                  mode="constant", constant_values=0)


def dilate(image: np.ndarray, padded: np.ndarray, se: np.ndarray, pad: int = 1, stride: int = 1) -> np.ndarray:
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
            output[out_i, out_j] = window[se_mask].max()

    return output


STRUCTURING_ELEMENTS = {
    "Vertical 3x1": np.array([[1],
                               [1],
                               [1]]),
    "Horizontal 1x3": np.array([[1, 1, 1]]),
    "Square 3x3": np.ones((3, 3), dtype=np.uint8),
    "Circle/Disk 3x3": np.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]]),
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "input.png"
    img = load_greyscale(path)

    padded = pad_image(img, pad=1)
    print(f"Padded image size: {padded.shape}")

    results = {"Original": img}
    for name, se in STRUCTURING_ELEMENTS.items():
        results[name] = dilate(img, padded, se, pad=1, stride=1)
        out_name = f"dilated_{name.replace('/', '_').replace(' ', '_')}.png"
        Image.fromarray(results[name]).save(out_name)
        print(f"Saved: {out_name}")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, (name, out) in zip(axes, results.items()):
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("dilation_comparison.png", dpi=150)
    print("Saved: dilation_comparison.png")


if __name__ == "__main__":
    main()

