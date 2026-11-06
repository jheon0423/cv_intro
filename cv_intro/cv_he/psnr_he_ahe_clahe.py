import cv2
import numpy as np
from PIL import Image


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10((255.0 ** 2) / mse) if mse != 0 else float('inf')



def apply_he(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = cv2.equalizeHist(ycrcb[:,:,0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_ahe(img):
    clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8,8))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    ycrcb[:,:,0] = clahe.apply(ycrcb[:,:,0])

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)




orig = np.array(Image.open("original.png").convert("RGB"))


noise = np.random.normal(0, 25, orig.shape)
noisy = np.clip(orig.astype(np.float64) + noise, 0, 255).astype(np.uint8)


Image.fromarray(noisy).save("noisy.png")


he_img = apply_he(noisy)
ahe_img = apply_ahe(noisy)
clahe_img = apply_clahe(noisy)



Image.fromarray(he_img).save("he.png")
Image.fromarray(ahe_img).save("ahe.png")
Image.fromarray(clahe_img).save("clahe.png")


print("PSNR original vs noisy :", psnr(orig, noisy))
print("PSNR original vs HE    :", psnr(orig, he_img))
print("PSNR original vs AHE   :", psnr(orig, ahe_img))
print("PSNR original vs CLAHE :", psnr(orig, clahe_img))