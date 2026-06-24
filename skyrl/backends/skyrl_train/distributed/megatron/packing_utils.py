import math


def get_packed_seq_align_size(tp_size: int, cp_size: int) -> int:
    """Return global per-subsequence padding needed for TP/CP layout and FP8."""
    if cp_size > 1:
        layout_align = tp_size * cp_size * 2
    else:
        layout_align = tp_size
    return math.lcm(layout_align, 16 * cp_size)
