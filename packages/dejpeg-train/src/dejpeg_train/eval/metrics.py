"""IQA metrics for the FBCNN baseline and later gates.

PSNR / PSNR-B / SSIM are ported verbatim from FBCNN's utils_image.py so the
baseline table is directly comparable to published FBCNN numbers. LPIPS-Alex
uses the ``lpips`` package (net='alex') -- the evaluation-gate perceptual metric
(NOT trained on; LPIPS-VGG is the training metric, metric-leakage guard §3.4).

All image args are uint8 [0, 255], HWC RGB. LPIPS converts to [-1, 1] internally.
pyiqa-only metrics (MUSIQ/MANIQA/CLIPIQA/DISTS) are DEFERRED to Real-web-500.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

_BORDER = 0  # FBCNN evaluates full image (no border crop)


def rgb2ycbcr(img: np.ndarray, only_y: bool = True) -> np.ndarray:
    """MATLAB-equivalent rgb2ycbcr (matches FBCNN utils_image)."""
    in_type = img.dtype
    img = img.astype(np.float32)
    if in_type != np.uint8:
        img *= 255.0
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    return rlt.astype(in_type)


def compute_bef(img: np.ndarray) -> float:
    """Blocking effect factor (FBCNN utils_image, verbatim)."""
    block = 8
    height, width = img.shape[:2]
    H = list(range(width - 1))
    H_B = list(range(block - 1, width - 1, block))
    H_BC = list(set(H) - set(H_B))
    V = list(range(height - 1))
    V_B = list(range(block - 1, height - 1, block))
    V_BC = list(set(V) - set(V_B))
    D_B = 0.0
    D_BC = 0.0
    for i in H_B:
        diff = img[:, i] - img[:, i + 1]
        D_B += np.sum(diff ** 2)
    for i in H_BC:
        diff = img[:, i] - img[:, i + 1]
        D_BC += np.sum(diff ** 2)
    for j in V_B:
        diff = img[j, :] - img[j + 1, :]
        D_B += np.sum(diff ** 2)
    for j in V_BC:
        diff = img[j, :] - img[j + 1, :]
        D_BC += np.sum(diff ** 2)
    N_HB = height * (width / block - 1)
    N_HBC = height * (width - 1) - N_HB
    N_VB = width * (height / block - 1)
    N_VBC = width * (height - 1) - N_VB
    D_B /= (N_HB + N_VB)
    D_BC /= (N_HBC + N_VBC)
    eta = math.log2(block) / math.log2(min(height, width)) if D_B > D_BC else 0
    return eta * (D_B - D_BC)


def psnr(img1: np.ndarray, img2: np.ndarray, border: int = _BORDER) -> float:
    if img1.shape != img2.shape:
        raise ValueError("images must have the same dimensions")
    h, w = img1.shape[:2]
    img1 = img1[border : h - border, border : w - border].astype(np.float64)
    img2 = img2[border : h - border, border : w - border].astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))


def psnr_b(gt: np.ndarray, compressed: np.ndarray, border: int = _BORDER) -> float:
    """PSNR-B. gt = ground truth, compressed = restored/input image."""
    if gt.shape != compressed.shape:
        raise ValueError("images must have the same dimensions")
    h, w = gt.shape[:2]
    gt = gt[border : h - border, border : w - border].astype(np.float64)
    comp = compressed[border : h - border, border : w - border].astype(np.float64)
    if comp.shape[-1] == 3:
        bef = compute_bef(rgb2ycbcr(comp).astype(np.float64))
    else:
        bef = compute_bef(comp)
    mse = np.mean((gt - comp) ** 2)
    mse_b = mse + bef
    if mse_b == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse_b))


def _ssim_single(img1: np.ndarray, img2: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean())


def ssim(img1: np.ndarray, img2: np.ndarray, border: int = _BORDER) -> float:
    """MATLAB-equivalent SSIM (matches FBCNN utils_image). RGB: mean over channels."""
    if img1.shape != img2.shape:
        raise ValueError("images must have the same dimensions")
    h, w = img1.shape[:2]
    img1 = img1[border : h - border, border : w - border]
    img2 = img2[border : h - border, border : w - border]
    if img1.ndim == 2:
        return _ssim_single(img1, img2)
    if img1.ndim == 3 and img1.shape[2] == 3:
        return float(np.mean([_ssim_single(img1[:, :, i], img2[:, :, i]) for i in range(3)]))
    raise ValueError("wrong input image dimensions")


class LPIPSAlex:
    """LPIPS-Alex (eval-gate perceptual metric). Lazily loads weights on first use."""

    def __init__(self, device: str = "cpu"):
        import lpips

        self.device = device
        self.model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        import torch

        def to_tensor(x):
            t = torch.from_numpy(x.astype(np.float32) / 255.0 * 2 - 1).permute(2, 0, 1).unsqueeze(0)
            return t.to(self.device)
        with torch.no_grad():
            return float(self.model(to_tensor(pred), to_tensor(target)).mean().item())


def compute_all(gt: np.ndarray, pred: np.ndarray, lpips_metric=None, border: int = _BORDER) -> dict:
    """Compute PSNR / PSNR-B / SSIM (always) and LPIPS-Alex (if metric provided)."""
    out = {"psnr": psnr(gt, pred, border), "psnr_b": psnr_b(gt, pred, border), "ssim": ssim(gt, pred, border)}
    if lpips_metric is not None:
        out["lpips_alex"] = lpips_metric(pred, gt)
    return out
