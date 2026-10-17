"""Resource register-map extraction from a compiled harness.

A patched instruction reads whatever registers are live at the MEASURE slot.
The generator therefore has to target the *specific* registers in which the
harness parks each resource — a global memory descriptor, a texture-object
handle, a surface-object handle, valid address/coordinate registers — rather
than hardcoding UR4/R2/etc.

This module reads those registers straight out of the harness's own SASS: the
harness performs a real ``LDG``/``TEX``/``SULD``/``STS`` on each resource, so
their operands *are* the live register map. The map is advisory — whether a
register is actually live at the measure slot (vs merely used elsewhere) is
confirmed by validation — but it captures the harness's convention exactly.

See :class:`RegisterMap` for the fields the generator consumes.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

from ..binary.cuobjdump import find_kernel, get_sass_dump, parse_sass_instructions

# desc[UR8][R6.64+0x8]  ->  descriptor UR + 64-bit address base
_DESC = re.compile(r"desc\[UR(\d+)\]\[R(\d+)\.64")
# Hopper/Blackwell bindless: TEX RZ, R12, R12, UR4  ->  handle in a uniform reg
_TEX = re.compile(r"^(?:TEX|TLD|TLD4|TXD|TXQ|TMML|TXA)\S*\s+"
                  r"(\w+),\s*R(\d+),\s*R(\d+),\s*UR(\d+)")
# Pre-Hopper bindless (Volta..Ada): TEX.SCR RZ, R12, R10, R13  ->  the texture
# handle is a *GPR* (R13), coords in the preceding GPR (R10). Distinguished
# from the Hopper form by a GPR (not UR) in the handle position.
_TEX_GPR = re.compile(r"^(?:TEX|TLD|TLD4|TXD|TXQ|TMML|TXA)\S*\s+"
                      r"\w+,\s*R\d+,\s*R(\d+),\s*R(\d+)")
# SULD.D... R0, [R16], UR10   /  SUST.D... [R16], R0, UR10
_SURF = re.compile(r"^(?:SULD|SUST|SURED|SUATOM|SUQUERY)\S*\s+"
                   r".*?\[R(\d+)\].*?\bUR(\d+)\b")
# STS [R6], R5  /  LDS R5, [R6+0x8]  /  LDS R5, [UR6+0x8]
_SHARED = re.compile(r"^(?:STS|LDS)\S*\s+.*?\[(U?R)(\d+)")
# UTMALDG.1D [UR40], [UR38]  ->  [shared dest], [TMA descriptor]
_TMA = re.compile(r"^UTMALDG\S*\s+\[UR(\d+)\],\s*\[UR(\d+)\]")
# The harness's aligned atomic scratch: RED.E.ADD [R4.64] (pre-Hopper) or
# RED.E.ADD desc[UR8][R4.64] (Hopper+). The base GPR holds a >=256B-aligned
# address suitable for 8/16-byte atomics (ATOMG/RED/CAS).
_ATOM = re.compile(r"^(?:RED|REDG|ATOMG|ATOM)\S*\s+(?:desc\[UR\d+\])?\[R(\d+)\.64")


@dataclass
class RegisterMap:
    """Registers in which a harness parks each patchable resource."""
    global_desc_ur: Optional[int] = None    # UR of desc[URx] in LDG/STG
    address_regs: List[int] = field(default_factory=list)   # 64-bit bases
    tex_handle_ur: Optional[int] = None     # URc of TEX (Hopper/Blackwell)
    tex_coord_regs: List[int] = field(default_factory=list)  # Ra,Rb of TEX
    tex_handle_reg: Optional[int] = None    # GPR handle of TEX.SCR (pre-Hopper)
    tex_coord_reg: Optional[int] = None     # GPR coord of TEX.SCR (pre-Hopper)
    surf_handle_ur: Optional[int] = None    # URc of SULD/SUST
    surf_coord_regs: List[int] = field(default_factory=list)
    shared_regs: List[int] = field(default_factory=list)     # R/UR base of STS/LDS
    tma_desc_ur: Optional[int] = None       # descriptor UR of UTMALDG
    tma_shared_ur: Optional[int] = None      # shared-dest UR of UTMALDG
    atom_addr_reg: Optional[int] = None      # aligned atomic base GPR (RED/ATOMG)

    def to_json(self) -> dict:
        return {
            "global_desc_ur": self.global_desc_ur,
            "address_regs": list(self.address_regs),
            "tex_handle_ur": self.tex_handle_ur,
            "tex_coord_regs": list(self.tex_coord_regs),
            "surf_handle_ur": self.surf_handle_ur,
            "surf_coord_regs": list(self.surf_coord_regs),
            "shared_regs": list(self.shared_regs),
            "tma_desc_ur": self.tma_desc_ur,
            "tma_shared_ur": self.tma_shared_ur,
            "tex_handle_reg": self.tex_handle_reg,
            "tex_coord_reg": self.tex_coord_reg,
            "atom_addr_reg": self.atom_addr_reg,
        }

    @classmethod
    def from_json(cls, data: dict) -> "RegisterMap":
        return cls(
            global_desc_ur=data.get("global_desc_ur"),
            address_regs=list(data.get("address_regs", [])),
            tex_handle_ur=data.get("tex_handle_ur"),
            tex_coord_regs=list(data.get("tex_coord_regs", [])),
            surf_handle_ur=data.get("surf_handle_ur"),
            surf_coord_regs=list(data.get("surf_coord_regs", [])),
            shared_regs=list(data.get("shared_regs", [])),
            tma_desc_ur=data.get("tma_desc_ur"),
            tma_shared_ur=data.get("tma_shared_ur"),
            tex_handle_reg=data.get("tex_handle_reg"),
            tex_coord_reg=data.get("tex_coord_reg"),
            atom_addr_reg=data.get("atom_addr_reg"),
        )


def extract_register_map(binary_path, kernel_hint: Optional[str] = None,
                         sass: Optional[str] = None) -> RegisterMap:
    """Parse a harness binary's SASS into a :class:`RegisterMap`.

    Args:
        binary_path: compiled harness binary.
        kernel_hint: substring selecting the measurement kernel.
        sass: pre-fetched SASS dump (skips cuobjdump when given).
    """
    if sass is None:
        sass = get_sass_dump(str(binary_path))
    kernel = find_kernel(sass, kernel_hint)
    instrs = parse_sass_instructions(sass, kernel)

    desc_ur: Counter = Counter()
    addr_regs: Counter = Counter()
    tex_handle: Counter = Counter()
    tex_coords: Counter = Counter()
    surf_handle: Counter = Counter()
    surf_coords: Counter = Counter()
    shared: Counter = Counter()
    tma_desc: Counter = Counter()
    tma_shared: Counter = Counter()
    tex_handle_reg: Counter = Counter()
    tex_coord_reg: Counter = Counter()
    atom_reg: Counter = Counter()

    for ins in instrs:
        text = ins.text
        for m in _DESC.finditer(text):
            desc_ur[int(m.group(1))] += 1
            addr_regs[int(m.group(2))] += 1
        m = _TEX.match(text)
        if m:
            tex_handle[int(m.group(4))] += 1
            tex_coords[int(m.group(2))] += 1
            tex_coords[int(m.group(3))] += 1
        elif _TEX_GPR.match(text):  # pre-Hopper: GPR handle (only if not UR form)
            g = _TEX_GPR.match(text)
            tex_coord_reg[int(g.group(1))] += 1
            tex_handle_reg[int(g.group(2))] += 1
        m = _SURF.match(text)
        if m:
            surf_coords[int(m.group(1))] += 1
            surf_handle[int(m.group(2))] += 1
        m = _SHARED.match(text)
        if m:
            shared[int(m.group(2))] += 1
        m = _TMA.match(text)
        if m:
            tma_shared[int(m.group(1))] += 1
            tma_desc[int(m.group(2))] += 1
        m = _ATOM.match(text)
        if m:
            atom_reg[int(m.group(1))] += 1

    def top(counter: Counter, k: int = 4) -> List[int]:
        return [v for v, _ in counter.most_common(k)]

    def one(counter: Counter) -> Optional[int]:
        return counter.most_common(1)[0][0] if counter else None

    return RegisterMap(
        global_desc_ur=one(desc_ur),
        address_regs=top(addr_regs),
        tex_handle_ur=one(tex_handle),
        tex_coord_regs=top(tex_coords, 2),
        surf_handle_ur=one(surf_handle),
        surf_coord_regs=top(surf_coords, 2),
        shared_regs=top(shared),
        tma_desc_ur=one(tma_desc),
        tma_shared_ur=one(tma_shared),
        tex_handle_reg=one(tex_handle_reg),
        tex_coord_reg=one(tex_coord_reg),
        atom_addr_reg=one(atom_reg),
    )
