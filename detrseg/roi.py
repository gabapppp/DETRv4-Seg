"""Shared pixel geometry for label creation, training, and inference."""

import math

import cv2
import numpy as np


def roi_bounds(bbox, width, height, padding=0.2):
    x, y, w, h = map(float, bbox)
    if not all(math.isfinite(v) for v in (x, y, w, h, padding)) or w <= 0 or h <= 0 or padding < 0:
        raise ValueError("Invalid bbox or padding")
    if x + w <= 0 or y + h <= 0 or x >= width or y >= height:
        raise ValueError("BBox does not intersect image")
    return (
        max(0, math.floor(x - padding * w)),
        max(0, math.floor(y - padding * h)),
        min(width, math.ceil(x + (1 + padding) * w)),
        min(height, math.ceil(y + (1 + padding) * h)),
    )


def bbox_prior(bbox, bounds):
    x0, y0, x1, y1 = bounds
    x, y, w, h = bbox
    yy, xx = np.mgrid[y0:y1, x0:x1]
    # Pixel centers, consistently used at training and inference.
    return ((xx + 0.5 >= x) & (xx + 0.5 < x + w) & (yy + 0.5 >= y) & (yy + 0.5 < y + h)).astype(
        np.uint8
    )


def roi_content_bounds(bounds, size, resize_mode="resize"):
    if resize_mode == "resize":
        return 0, 0, size, size
    if resize_mode != "native":
        raise ValueError("ROI resize mode must be resize or native")
    x0, y0, x1, y1 = bounds
    width, height = x1 - x0, y1 - y0
    if width > size or height > size:
        raise ValueError(
            f"Native ROI {width}x{height} exceeds {size}x{size} canvas. "
            f"Increase --size to at least {max(width, height)}; native mode never downsamples."
        )
    left, top = (size - width) // 2, (size - height) // 2
    return left, top, left + width, top + height


def roi_canvas(array, size, resize_mode="resize", interpolation=cv2.INTER_LINEAR):
    """Resize legacy ROIs, or copy native pixels into a centered zero-padded canvas."""
    height, width = array.shape[:2]
    left, top, right, bottom = roi_content_bounds((0, 0, width, height), size, resize_mode)
    if resize_mode == "resize":
        return cv2.resize(array, (size, size), interpolation=interpolation)
    canvas = np.zeros((size, size, *array.shape[2:]), dtype=array.dtype)
    canvas[top:bottom, left:right] = array
    return canvas


def restore_roi(probability, bounds, resize_mode="resize"):
    """Undo ROI input geometry before thresholding at native image coordinates."""
    x0, y0, x1, y1 = bounds
    if resize_mode == "resize":
        return cv2.resize(probability, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    if probability.ndim != 2 or probability.shape[0] != probability.shape[1]:
        raise ValueError("Native ROI prediction must have the square input canvas shape")
    left, top, right, bottom = roi_content_bounds(bounds, probability.shape[0], resize_mode)
    return probability[top:bottom, left:right]


def roi_inputs(image, bbox, size, padding, resize_mode="resize"):
    height, width = image.shape[:2]
    bounds = roi_bounds(bbox, width, height, padding)
    x0, y0, x1, y1 = bounds
    rgb = roi_canvas(image[y0:y1, x0:x1], size, resize_mode)
    prior = roi_canvas(bbox_prior(bbox, bounds), size, resize_mode, cv2.INTER_NEAREST)
    return rgb, prior, bounds


def mask_support(bbox, bounds, overflow=0.0):
    """Expand each bbox side by overflow times the original width/height."""
    if not math.isfinite(overflow) or overflow < 0:
        raise ValueError("mask overflow must be finite and non-negative")
    x, y, w, h = bbox
    return bbox_prior(
        [x - w * overflow, y - h * overflow, w * (1 + 2 * overflow), h * (1 + 2 * overflow)], bounds
    )


def clean_component(
    mask, prior, min_fraction=0.1, max_fraction=0.95, fill_mode="holes", support=None
):
    """Select the main object; optionally reconstruct a near-convex particle silhouette.

    Filling closed contours alone cannot recover a bright core when a dark rim is
    open (a C/U shape). Convex mode closes that opening, using a particle-specific
    shape assumption. It must not be enabled for arbitrary concave objects.
    """
    if fill_mode not in ("holes", "convex"):
        raise ValueError("fill_mode must be holes or convex")
    support = prior if support is None else support
    mask = ((mask > 0) & (support > 0)).astype(np.uint8)
    count, labels, _stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None, "no_foreground"
    py, px = np.nonzero(prior)
    center = np.array([px.mean(), py.mean()])
    span = max(float(np.ptp(px)), float(np.ptp(py)), 1.0)
    # Rank by overlap with the original bbox so exterior neighbors cannot win
    # merely because the new permitted output region is larger.
    scores = [
        float(((labels == i) & (prior > 0)).sum())
        / (1 + 4 * np.linalg.norm(centroids[i] - center) / span)
        for i in range(1, count)
    ]
    if max(scores) == 0:
        return None, "no_bbox_overlap"
    selected = (labels == 1 + int(np.argmax(scores))).astype(np.uint8)
    contours, _ = cv2.findContours(selected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if fill_mode == "convex":
        contours = [cv2.convexHull(np.concatenate(contours))]
    cv2.drawContours(selected, contours, -1, 1, thickness=cv2.FILLED)
    selected &= support
    fraction = float(selected.sum() / max(1, prior.sum()))
    if not min_fraction <= fraction <= max_fraction:
        return None, "area_fraction"
    return selected, None


def pseudo_mask(image, bbox, padding=0.2, grabcut=False, seed=42, fill_mode="holes"):
    height, width = image.shape[:2]
    bounds = roi_bounds(bbox, width, height, padding)
    x0, y0, x1, y1 = bounds
    crop = image[y0:y1, x0:x1]
    prior = bbox_prior(bbox, bounds)
    support = prior
    if min(prior.sum(), (prior == 0).sum()) < 5 or min(crop.shape[:2]) < 3:
        return None, {"reason": "insufficient_context"}
    # No forced foreground on uniform images: spatial features alone are not evidence.
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    if float(lab.std(axis=(0, 1)).max()) < 2:
        return None, {"reason": "low_contrast"}
    ch, cw = crop.shape[:2]
    yy, xx = np.mgrid[:ch, :cw]
    features = np.concatenate(
        (lab.reshape(-1, 3), (xx / cw * 3).reshape(-1, 1), (yy / ch * 3).reshape(-1, 1)), axis=1
    ).astype(np.float32)
    cv2.setRNGSeed(seed)
    _, labels, _ = cv2.kmeans(
        features,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    labels = labels.reshape(ch, cw)
    scores = [
        float((labels[prior > 0] == i).mean() - (labels[prior == 0] == i).mean()) for i in range(2)
    ]
    foreground = int(np.argmax(scores))
    if scores[foreground] < 0.08:
        return None, {"reason": "weak_cluster_separation", "separation": scores[foreground]}
    max_fraction = 0.95 * float(support.sum() / max(1, prior.sum()))
    mask, reason = clean_component(
        labels == foreground, prior, max_fraction=max_fraction, fill_mode=fill_mode, support=support
    )
    if mask is None:
        return None, {"reason": reason}
    if grabcut:
        gc = np.full((ch, cw), cv2.GC_BGD, np.uint8)
        gc[support > 0] = cv2.GC_PR_BGD
        gc[mask > 0] = cv2.GC_PR_FGD
        sure = cv2.erode(mask, np.ones((3, 3), np.uint8))
        if sure.any():
            gc[sure > 0] = cv2.GC_FGD
        try:
            cv2.grabCut(
                crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK
            )
            mask, reason = clean_component(
                (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD),
                prior,
                max_fraction=max_fraction,
                fill_mode=fill_mode,
                support=support,
            )
        except cv2.error:
            return None, {"reason": "grabcut_failed"}
        if mask is None:
            return None, {"reason": reason}
    full = np.zeros((height, width), np.uint8)
    full[y0:y1, x0:x1] = mask
    return full, {
        "source": ("pseudo_grabcut" if grabcut else "pseudo_kmeans")
        + ("_convex" if fill_mode == "convex" else ""),
        "fill_mode": fill_mode,
        "label_overflow": 0.0,
        "separation": scores[foreground],
        "area_fraction": float(mask.sum() / prior.sum()),
    }
