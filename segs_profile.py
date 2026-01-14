from typing import Any, List, Optional, Tuple


def _extract_seg_list(segs: Any) -> List[Any]:
    """
    Impact SEGS обычно выглядит как: [(W,H), [SEG(...), SEG(...)]]
    """
    if segs is None:
        return []
    if isinstance(segs, (list, tuple)) and len(segs) >= 2 and isinstance(segs[1], list):
        return segs[1]
    if isinstance(segs, list):
        return segs
    return []


def _get_label(seg: Any) -> str:
    if hasattr(seg, "label"):
        return str(seg.label or "")
    if isinstance(seg, dict):
        return str(seg.get("label", "") or "")
    return ""


def _get_bbox(seg: Any) -> Optional[Tuple[float, float, float, float]]:
    if hasattr(seg, "bbox"):
        b = seg.bbox
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    if isinstance(seg, dict):
        b = seg.get("bbox")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return None


def _bbox_area(bbox: Tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _pick_largest(segs: List[Any], target_label: str) -> Optional[Any]:
    best = None
    best_area = -1.0
    for s in segs:
        if _get_label(s) != target_label:
            continue
        bbox = _get_bbox(s)
        if bbox is None:
            continue
        a = _bbox_area(bbox)
        if a > best_area:
            best_area = a
            best = s
    return best


class SEGSIsProfile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segs": ("SEGS",),
                "threshold": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING")
    RETURN_NAMES = ("is_profile", "eye_ratio", "debug")
    FUNCTION = "run"
    CATEGORY = "utils/segs"

    def run(self, segs, threshold: float):
        seg_list = _extract_seg_list(segs)

        left = _pick_largest(seg_list, "left_eye")
        right = _pick_largest(seg_list, "right_eye")

        # Если одного глаза нет — считаем профилем
        if left is None or right is None:
            debug = (
                f"eye_ratio=0.0 "
                f"(missing_eye: left={left is not None}, right={right is not None})"
            )
            return True, 0.0, debug

        a_left = _bbox_area(_get_bbox(left))
        a_right = _bbox_area(_get_bbox(right))

        eps = 1e-6
        eye_ratio = float(max(a_left, a_right) / (min(a_left, a_right) + eps))
        is_profile = eye_ratio > float(threshold)

        debug = (
            f"eye_ratio={eye_ratio:.3f} "
            f"(AL={a_left:.1f}, AR={a_right:.1f}, thr={threshold})"
        )
        return is_profile, eye_ratio, debug

