"""Anchor-based location of SASS code inside a binary file.

cuobjdump reports SASS offsets relative to the start of the code section;
patchers need absolute file offsets. This module finds the mapping by
searching the binary for the raw bytes of the first few instructions of a
kernel and cross-validating with a second anchor (strongest variant of the
historical implementations; scans all match positions).
"""

from typing import List, Optional, Tuple

from .cuobjdump import parse_sass_instructions


def find_code_base(binary_data: bytes, sass: str, kernel_name: str,
                   max_anchors: int = 5) -> Optional[int]:
    """Find the file offset where the SASS code section starts.

    Returns ``code_base`` such that ``file_offset = code_base + sass_offset``,
    or ``None`` if no consistent placement was found.
    """
    instrs = parse_sass_instructions(sass, kernel_name)
    anchors: List[Tuple[int, bytes]] = [
        (ins.offset, ins.encoding_bytes_le) for ins in instrs[:max_anchors]
    ]
    if not anchors:
        return None

    for sass_off, target in anchors:
        pos = 0
        while True:
            pos = binary_data.find(target, pos)
            if pos == -1:
                break
            candidate_base = pos - sass_off
            pos += 1
            if candidate_base < 0:
                continue
            if len(anchors) == 1:
                return candidate_base
            # Cross-validate with a different anchor
            other_off, other_target = (
                anchors[-1] if anchors[-1] != (sass_off, target) else anchors[0])
            check_pos = candidate_base + other_off
            if 0 <= check_pos <= len(binary_data) - len(other_target):
                if binary_data[check_pos:check_pos + len(other_target)] == other_target:
                    return candidate_base
    return None
