"""EoSS — full-screen curses UI for the EoSS binary modifier.

Implements design "3a" from the Hex Editor UI Explorations canvas:
box-drawn panes, a numbered tab bar, a function-key toolbar, and a
three-column Instructions workbench (STRUCTURE | instructions+hex | FIELDS /
SCHEDULING / CHECKS), with vim-style movement and F-key actions.

Tabs: 1 Instructions · 2 Hex · 3 ELF Header · 4 .nv.info · 5 Notes ·
      6 ISA (spec browser) · 7 Disasm (cuobjdump cross-check) ·
      8 Measure (GPU timing bench / campaigns)
Keys: Tab cycles panes · hjkl/arrows move · Enter edit/activate ·
      F1/? help · F3/g goto · F4// search · F5/s save · F8/u undo ·
      F10/q quit · v variant · o opcode · n generate generic
"""

import curses
import struct
from typing import List, Optional, Tuple

from .. import paths
from ..enc.control import INSTRUCTION_SIZE
from .decode import DecodedInstruction
from .elfview import CUDA_FLAG_BITS, ELF64_HEADER_FIELDS, ELF_TYPE_NAMES
from .measure import MODES, PATTERNS, SLOT_CHOICES, MeasureSession
from .model import EditError, EditorModel, TextSection

TABS = ["Instructions", "Hex", "ELF Header", ".nv.info", "Notes", "ISA",
        "Disasm", "Measure"]
(TAB_INSTR, TAB_HEX, TAB_ELF, TAB_NVINFO, TAB_NOTES, TAB_ISA,
 TAB_DISASM, TAB_MEASURE) = range(8)

PANE_STRUCTURE, PANE_INSTR, PANE_FIELDS = range(3)

CONTROL_FIELDS = [("stall", 15), ("yield_", 1), ("wr_bar", 7),
                  ("rd_bar", 7), ("wt_bar", 63)]

# color pair ids
C_BORDER, C_ACCENT, C_CHIP, C_MNEMONIC, C_ERR, C_OK, C_TEXT, C_DIM, C_SEL, \
    C_EDIT = range(1, 11)
#: one pair per scoreboard barrier B0-B5
C_BAR = list(range(11, 17))
#: mnemonic modifier suffixes (.E, .64, .CONSTANT ...) — distinct from base
C_MNEM_MOD = 17

#: color schemes, matching the design canvas directions (256-color values):
#: dark = designs 1a/3a · phosphor = 2b terminal · light = 2c lab instrument
THEME_ORDER = ["dark", "phosphor", "light"]
THEMES = {
    "dark": {
        "bg": -1,
        "pairs": {C_BORDER: 242, C_ACCENT: 179, C_MNEMONIC: 110,
                  C_MNEM_MOD: 176, C_ERR: 167, C_OK: 108, C_TEXT: 252,
                  C_DIM: 242, C_EDIT: 179},
        "chip": (16, 179), "sel": (16, 179),
        "bars": [179, 110, 108, 167, 176, 116],
    },
    "phosphor": {
        "bg": -1,
        "pairs": {C_BORDER: 65, C_ACCENT: 114, C_MNEMONIC: 151,
                  C_MNEM_MOD: 72, C_ERR: 174, C_OK: 108, C_TEXT: 108,
                  C_DIM: 65, C_EDIT: 144},
        "chip": (16, 114), "sel": (16, 114),
        "bars": [144, 114, 72, 174, 139, 109],
    },
    "light": {
        "bg": 255,
        "pairs": {C_BORDER: 250, C_ACCENT: 130, C_MNEMONIC: 25,
                  C_MNEM_MOD: 91, C_ERR: 124, C_OK: 28, C_TEXT: 236,
                  C_DIM: 245, C_EDIT: 130},
        "chip": (231, 130), "sel": (16, 178),
        "bars": [130, 25, 28, 124, 91, 30],
    },
}

HELP_LINES = [
    "EoSS — SASS binary modifier",
    "",
    "  1-8          switch tab            Tab    next pane",
    "  h j k l      move (arrows too)     Enter  edit / activate",
    "  g / F3       goto offset           /  F4  search instruction",
    "  o            opcode picker         v      cycle opcode variant",
    "  n            generate a generic, valid encoding of the opcode",
    "  e            edit selected field   u / F8 undo last edit",
    "  s / F5       save modified binary  q / F10 quit",
    "  PgUp/PgDn    slide instruction up/down (^U/^D page)",
    "  t            left pane: scoreboard tracker <-> ELF structure",
    "               (Tab slides the tracker lanes when they overflow)",
    "  c            cycle color scheme (dark / phosphor / light)",
    "",
    "ISA tab: browse every instruction class of the loaded spec;",
    "/ filters, Enter/p stamps a generic encoding at the cursor.",
    "Disasm tab: cuobjdump -sass of the edited buffer; r re-runs,",
    "Enter jumps to the instruction, ✗ marks encoding mismatches.",
    "Measure tab: GPU timing bench. b builds a slot harness (nvcc);",
    "d builds a validated instruction DB (runs each op, keeps the",
    "clean ones — sweeps then skip faulting encodings). Enter",
    "measures the cursor instruction (or sweeps the ISA) as TRIGGER,",
    "hot vs cold; m mode · r src · p pattern · x cancel · e export.",
    "",
    "Scoreboard: ■ sets barrier (wr) · ◆ sets barrier (rd) · □ waits;",
    "each barrier B0-B5 has its own color, lines join set to wait.",
    "Fields accept enum names (R6, PT, EN), Bucket@Name or numbers",
    "(0x1F, 0b101, 12). Hex tab: Enter rewrites bytes at the cursor.",
    "",
    "press any key to close",
]


class ListState:
    """Cursor + scroll offset for one scrollable list."""

    def __init__(self):
        self.cursor = 0
        self.top = 0

    def clamp(self, n: int, page: int) -> None:
        self.cursor = max(0, min(self.cursor, n - 1)) if n else 0
        if page <= 0:
            self.top = 0
            return
        if self.cursor < self.top:
            self.top = self.cursor
        if self.cursor >= self.top + page:
            self.top = self.cursor - page + 1
        self.top = max(0, min(self.top, max(0, n - page)))

    def move(self, delta: int, n: int, page: int) -> None:
        self.cursor += delta
        self.clamp(n, page)


class App:
    def __init__(self, screen, model: EditorModel):
        self.screen = screen
        self.model = model
        self.tab = TAB_INSTR
        self.pane = PANE_INSTR
        self.status: Tuple[str, int] = ("", C_DIM)

        self.structure = ListState()
        self.instr = ListState()
        self.fields = ListState()
        self.hexview = ListState()  # cursor = byte offset in file
        self.elf = ListState()
        self.elf_image = 0
        self.nvinfo = ListState()
        self.notes = ListState()
        self.isa = ListState()
        self.isa_query = ""
        self._isa_rows_cache = None
        self.disasm = ListState()
        self._disasm_cache = None  # (rows, error) — rebuilt after edits
        self.measure = MeasureSession(model)
        self.results = ListState()

        self.section_idx = 0 if model.sections else -1
        self._structure_rows: List[Tuple[str, str, int]] = []
        self._nvinfo_cache = None
        self._notes_cache = None

        self.left_mode = "score"  # "score" (barrier tracker) | "structure"
        self.score_xoff = 0  # first visible barrier lane
        self._score_fit = 6  # lanes that fit (set during draw)
        self.theme = "dark"

    def apply_background(self) -> None:
        """Paint the whole screen in the theme's base colors (light themes
        need an explicit background; -1 themes keep the terminal default)."""
        if curses.has_colors():
            self.screen.bkgd(" ", curses.color_pair(C_TEXT))

    def action_theme(self) -> None:
        idx = THEME_ORDER.index(self.theme) if self.theme in THEME_ORDER else 0
        self.theme = THEME_ORDER[(idx + 1) % len(THEME_ORDER)]
        if not apply_theme(self.theme):
            self.set_status("theme switching needs a 256-color terminal", C_ERR)
            return
        self.apply_background()
        self.set_status(f"color scheme: {self.theme}", C_ACCENT)

    # -- data helpers --------------------------------------------------------

    @property
    def section(self) -> Optional[TextSection]:
        if 0 <= self.section_idx < len(self.model.sections):
            return self.model.sections[self.section_idx]
        return None

    def rel(self, abs_offset: int) -> int:
        """File offset -> image-relative display offset (like nvdisasm)."""
        sec = self.section
        image = self.model.image_for_section(sec) if sec else None
        return abs_offset - (image.base if image else 0)

    def current_offset(self) -> Optional[int]:
        sec = self.section
        if sec is None or sec.n_instructions == 0:
            return None
        return sec.abs_offset + self.instr.cursor * INSTRUCTION_SIZE

    def current_decoded(self) -> Optional[DecodedInstruction]:
        off = self.current_offset()
        return self.model.decode_at(off) if off is not None else None

    def field_rows(self, decoded: DecodedInstruction) -> List[Tuple[str, str]]:
        """(kind, name) rows the FIELDS-pane cursor can sit on."""
        rows = [("field", f.name) for f in decoded.fields]
        rows += [("control", name) for name, _mx in CONTROL_FIELDS]
        if decoded.defn is not None:
            rows.append(("action", "generate"))
        return rows

    def invalidate(self) -> None:
        self._nvinfo_cache = None
        self._notes_cache = None
        self._disasm_cache = None

    def nvinfo_records(self):
        if self._nvinfo_cache is None:
            self._nvinfo_cache = self.model.nvinfo_records()
        return self._nvinfo_cache

    def note_entries(self):
        if self._notes_cache is None:
            self._notes_cache = self.model.notes()
        return self._notes_cache

    def isa_all_rows(self) -> List[Tuple[str, str, int, str]]:
        """(mnemonic, class name, opcode, pipeline) for every spec class."""
        if self._isa_rows_cache is None:
            rows = []
            for name, ic in self.model.spec.instructions.items():
                mnemonic = self.model.decoder.mnemonic_for(ic, ic.opcode)
                rows.append((mnemonic, name, ic.opcode, ic.pipeline or ""))
            rows.sort()
            self._isa_rows_cache = rows
        return self._isa_rows_cache

    def isa_rows(self) -> List[Tuple[str, str, int, str]]:
        q = self.isa_query.lower()
        rows = self.isa_all_rows()
        if q:
            rows = [r for r in rows
                    if q in r[0].lower() or q in r[1].lower()]
        return rows

    def disasm_rows(self):
        """cuobjdump view rows: ("func", name, None) headers and
        ("instr", SassInstruction, (abs_offset | None, enc_matches)) lines.
        Returns (rows, error) — error is set when cuobjdump fails."""
        if self._disasm_cache is None:
            try:
                dump = self.model.disassemble()
            except Exception as exc:
                self._disasm_cache = ([], str(exc))
                return self._disasm_cache
            rows = []
            for func, instrs in dump:
                rows.append(("func", func, None))
                sec = self.model.section_for_function(func)
                for ins in instrs:
                    abs_off = match = None
                    if sec is not None and ins.offset < sec.size:
                        abs_off = sec.abs_offset + ins.offset
                        match = (bytes(self.model.data[abs_off:abs_off + 8])
                                 == ins.encoding_bytes_le)
                    rows.append(("instr", ins, (abs_off, match)))
            self._disasm_cache = (rows, None)
        return self._disasm_cache

    # -- curses helpers ------------------------------------------------------

    def color(self, pair: int, bold: bool = False) -> int:
        attr = curses.color_pair(pair) if curses.has_colors() else 0
        return attr | (curses.A_BOLD if bold else 0)

    def put(self, y: int, x: int, text: str, attr: int = 0) -> int:
        h, w = self.screen.getmaxyx()
        if y < 0 or y >= h or x >= w:
            return x
        text = text[:max(0, w - x)]
        try:
            self.screen.addstr(y, x, text, attr)
        except curses.error:
            pass
        return x + len(text)

    def hline(self, y: int, x0: int, x1: int, ch: str = "─",
              attr: int = 0) -> None:
        if x1 > x0:
            self.put(y, x0, ch * (x1 - x0), attr)

    def set_status(self, message: str, pair: int = C_DIM) -> None:
        self.status = (message, pair)

    # -- prompt --------------------------------------------------------------

    def prompt(self, label: str, initial: str = "") -> Optional[str]:
        """Modal single-line input on the bottom row. Esc cancels."""
        h, w = self.screen.getmaxyx()
        buf = list(initial)
        curses.curs_set(1)
        try:
            while True:
                self.hline(h - 1, 0, w, " ")
                x = self.put(h - 1, 0, ":" + label + " ",
                             self.color(C_ACCENT, bold=True))
                x = self.put(h - 1, x, "".join(buf), self.color(C_TEXT))
                self.put(h - 1, min(x, w - 1), "", 0)
                try:
                    self.screen.move(h - 1, min(x, w - 1))
                except curses.error:
                    pass
                self.screen.refresh()
                ch = self.screen.getch()
                if ch in (10, 13, curses.KEY_ENTER):
                    return "".join(buf)
                if ch == 27:
                    return None
                if ch in (curses.KEY_BACKSPACE, 127, 8):
                    if buf:
                        buf.pop()
                elif ch == 21:  # ^U
                    buf.clear()
                elif 32 <= ch < 127:
                    buf.append(chr(ch))
        finally:
            curses.curs_set(0)

    def picker(self, title: str, items: List[Tuple[str, str]]) -> Optional[str]:
        """Modal filterable list (the TUI's dropdown). Returns the selected
        payload, or None on Esc. Type to filter, ↑/↓ move, Enter selects."""
        h, w = self.screen.getmaxyx()
        box_w = min(w - 6, 72)
        box_h = min(h - 4, 22)
        y0, x0 = (h - box_h) // 2, (w - box_w) // 2
        page = box_h - 4
        query = ""
        cursor = top = 0
        while True:
            q = query.lower()
            shown = [it for it in items if q in it[0].lower()]
            cursor = max(0, min(cursor, len(shown) - 1)) if shown else 0
            if cursor < top:
                top = cursor
            if cursor >= top + page:
                top = cursor - page + 1
            top = max(0, min(top, max(0, len(shown) - page)))

            border = self.color(C_ACCENT)
            self.put(y0, x0, "┌" + "─" * (box_w - 2) + "┐", border)
            self.put(y0, x0 + 2, f" {title} ", self.color(C_CHIP, bold=True))
            self.put(y0 + 1, x0, "│" + " " * (box_w - 2) + "│", border)
            self.put(y0 + 1, x0 + 2, "> " + query + "█",
                     self.color(C_TEXT, bold=True))
            count = f" {len(shown)}/{len(items)} "
            self.put(y0 + 1, x0 + box_w - len(count) - 1, count,
                     self.color(C_DIM))
            for r in range(page):
                y = y0 + 2 + r
                self.put(y, x0, "│" + " " * (box_w - 2) + "│", border)
                i = top + r
                if i < len(shown):
                    attr = self.color(C_SEL, bold=True) if i == cursor \
                        else self.color(C_TEXT)
                    self.put(y, x0 + 2, shown[i][0][:box_w - 4].ljust(box_w - 4),
                             attr)
            self.put(y0 + box_h - 1, x0, "└" + "─" * (box_w - 2) + "┘", border)
            self.put(y0 + box_h - 1, x0 + 2,
                     " Enter select · Esc cancel ", self.color(C_DIM))
            self.screen.refresh()

            ch = self.screen.getch()
            if ch == 27:
                return None
            if ch in (10, 13, curses.KEY_ENTER):
                return shown[cursor][1] if shown else None
            if ch in (curses.KEY_DOWN,):
                cursor += 1
            elif ch in (curses.KEY_UP,):
                cursor -= 1
            elif ch == curses.KEY_NPAGE:
                cursor += page
            elif ch == curses.KEY_PPAGE:
                cursor -= page
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                query = query[:-1]
            elif ch == 21:  # ^U
                query = ""
            elif 32 <= ch < 127:
                query += chr(ch)
                cursor = 0

    # =========================================================================
    # drawing
    # =========================================================================

    def draw(self) -> None:
        self.screen.erase()
        h, w = self.screen.getmaxyx()
        if h < 12 or w < 60:
            self.put(0, 0, "terminal too small (need at least 60x12)",
                     self.color(C_ERR, bold=True))
            self.screen.refresh()
            return
        body_top, body_bottom = 3, h - 3  # rows body_top..body_bottom-1
        self.draw_frame(h, w, body_top, body_bottom)
        if self.tab == TAB_INSTR:
            self.draw_instructions_tab(body_top, body_bottom, w)
        elif self.tab == TAB_HEX:
            self.draw_hex_tab(body_top, body_bottom, w)
        elif self.tab == TAB_ELF:
            self.draw_elf_tab(body_top, body_bottom, w)
        elif self.tab == TAB_NVINFO:
            self.draw_nvinfo_tab(body_top, body_bottom, w)
        elif self.tab == TAB_ISA:
            self.draw_isa_tab(body_top, body_bottom, w)
        elif self.tab == TAB_DISASM:
            self.draw_disasm_tab(body_top, body_bottom, w)
        elif self.tab == TAB_MEASURE:
            self.draw_measure_tab(body_top, body_bottom, w)
        else:
            self.draw_notes_tab(body_top, body_bottom, w)
        self.draw_footer(h, w)
        self.screen.refresh()

    def draw_frame(self, h: int, w: int, body_top: int, body_bottom: int) -> None:
        border = self.color(C_BORDER)
        # top border with chips
        self.hline(0, 0, w, "─", border)
        self.put(0, 0, "┌", border)
        self.put(0, w - 1, "┐", border)
        x = self.put(0, 1, "─", border)
        x = self.put(0, x, " EoSS ", self.color(C_CHIP, bold=True))
        x = self.put(0, x, "─┤", border)
        x = self.put(0, x, f" {self.model.path.name} ", self.color(C_TEXT))
        x = self.put(0, x, "├─┤", border)
        image = next(iter(self.model.cuda_images), None)
        arch = f" {image.sm_name} · ELF64 · LE " if image else " no CUDA image "
        x = self.put(0, x, arch, self.color(C_DIM))
        x = self.put(0, x, "├", border)
        n_mod = self.model.n_modified
        chip = f" ● {n_mod} byte{'s' if n_mod != 1 else ''} modified "
        cx = w - 1 - len(chip) - 1
        if cx > x:
            self.put(0, cx - 1, "┤", border)
            self.put(0, cx, chip,
                     self.color(C_ACCENT, bold=True) if n_mod else self.color(C_DIM))
            self.put(0, cx + len(chip), "├", border)
        # tab bar
        self.put(1, 0, "│", border)
        self.put(1, w - 1, "│", border)
        x = 2
        for i, name in enumerate(TABS):
            label = f" {i + 1} {name} "
            attr = self.color(C_SEL, bold=True) if i == self.tab \
                else self.color(C_DIM)
            x = self.put(1, x, label, attr)
            x += 1
        # bottom of body
        self.hline(body_bottom, 0, w, "─", border)
        self.put(body_bottom, 0, "└", border)
        self.put(body_bottom, w - 1, "┘", border)

    def draw_footer(self, h: int, w: int) -> None:
        pairs = [("F1", "Help"), ("F3", "Goto"), ("F4", "Search"),
                 ("F5", "Save"), ("F8", "Undo"), ("F10", "Quit")]
        x = 1
        for key, label in pairs:
            x = self.put(h - 2, x, key, self.color(C_CHIP))
            x = self.put(h - 2, x, label + " ", self.color(C_DIM))
        hint = "Tab⇥ next pane · hjkl move · Enter edit"
        if w - len(hint) - 2 > x:
            self.put(h - 2, w - len(hint) - 2, hint, self.color(C_BORDER))
        message, pair = self.status
        self.hline(h - 1, 0, w, " ")
        if message:
            self.put(h - 1, 1, message[:w - 2], self.color(pair))

    # -- instructions tab ----------------------------------------------------

    def draw_instructions_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        left_w = 22
        right_w = 33 if w >= 100 else max(24, w // 4)
        mid_x0, mid_x1 = left_w + 1, w - right_w - 2
        # column separators
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        self.put(2, left_w, "┬", border)
        self.put(2, mid_x1, "┬", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, left_w, "│", border)
            self.put(y, mid_x1, "│", border)
            self.put(y, w - 1, "│", border)
        self.put(bottom, left_w, "┴", border)
        self.put(bottom, mid_x1, "┴", border)

        if self.left_mode == "score":
            self.draw_scoreboard(top, bottom, 1, left_w)
        else:
            self.draw_structure(top, bottom, 1, left_w)
        self.draw_center(top, bottom, mid_x0, mid_x1)
        self.draw_inspector(top, bottom, mid_x1 + 1, w - 1)

    def structure_rows(self) -> List[Tuple[str, str, int]]:
        """(kind, label, payload) rows: kind in header|section|text."""
        rows: List[Tuple[str, str, int]] = []
        for i, image in enumerate(self.model.cuda_images):
            rows.append(("header", f"ELF Header @ {image.base:04X}", i))
            for sec in image.sections:
                if not sec.name:
                    continue
                if sec.name.startswith(".text"):
                    for j, tsec in enumerate(self.model.sections):
                        if tsec.abs_offset == sec.abs_offset:
                            rows.append(("text", sec.name, j))
                            break
                elif sec.name != ".shstrtab":
                    rows.append(("section", sec.name, sec.abs_offset))
        if not rows:
            for j, tsec in enumerate(self.model.sections):
                rows.append(("text", tsec.name, j))
        self._structure_rows = rows
        return rows

    def draw_structure(self, top: int, bottom: int, x0: int, x1: int) -> None:
        width = x1 - x0
        rows = self.structure_rows()
        range_rows = 5
        page = max(1, bottom - top - 1 - range_rows)
        self.structure.clamp(len(rows), page)
        y = top
        header = " STRUCTURE · t track"
        self.put(y, x0, (header + " " * width)[:width], self.color(C_DIM))
        y += 1
        for i in range(self.structure.top,
                       min(len(rows), self.structure.top + page)):
            kind, label, payload = rows[i]
            selected = (i == self.structure.cursor
                        and self.pane == PANE_STRUCTURE)
            active = kind == "text" and payload == self.section_idx
            marker = "▸ " if active else "  "
            text = (marker + label)[:width - 1]
            attr = self.color(C_TEXT)
            if kind == "header":
                attr = self.color(C_DIM)
            if active:
                attr = self.color(C_ACCENT, bold=True)
            if selected:
                attr = self.color(C_SEL, bold=True)
            self.put(y, x0, " " + text.ljust(width - 1), attr)
            y += 1
        # RANGE block pinned at the bottom of the pane
        y = bottom - range_rows
        self.hline(y, x0 + 1, x1 - 1, "─", self.color(C_BORDER))
        y += 1
        self.put(y, x0, " RANGE", self.color(C_DIM))
        y += 1
        sec = self.section
        if sec:
            self.put(y, x0, f"  off  0x{self.rel(sec.abs_offset):04X}",
                     self.color(C_TEXT))
            self.put(y + 1, x0, f"  size 0x{sec.size:04X}", self.color(C_TEXT))
            self.put(y + 2, x0, f"  {sec.n_instructions} instr",
                     self.color(C_DIM))

    def draw_scoreboard(self, top: int, bottom: int, x0: int, x1: int) -> None:
        """Barrier tracker: one colored lane per scoreboard B0-B5, rows
        aligned 1:1 with the instruction list; ■/◆ set (wr/rd), □ wait,
        │ joins each set to the wait that consumes it."""
        width = x1 - x0
        sec = self.section
        legend_rows = 4
        focused = self.pane == PANE_STRUCTURE
        if sec is None:
            self.put(top, x0, " SCOREBOARD".ljust(width), self.color(C_DIM))
            return
        lane_w = 3
        self._score_fit = max(1, min(6, (width - 2) // lane_w))
        max_xoff = 6 - self._score_fit
        self.score_xoff = max(0, min(self.score_xoff, max_xoff))
        hex_rows = 6
        list_page = max(1, bottom - top - hex_rows - 3)
        n = sec.n_instructions
        self.instr.clamp(n, list_page)
        spans = self.model.barrier_spans(sec)
        events = self.model.barrier_events(sec)

        def lane_x(b: int) -> int:
            return x0 + 1 + (b - self.score_xoff) * lane_w

        # header row holds only the lane labels so rows align with the list
        self.put(top, x0, ("▸" if focused else " ").ljust(width),
                 self.color(C_ACCENT, bold=True))
        for b in range(self.score_xoff, self.score_xoff + self._score_fit):
            self.put(top, lane_x(b), f"B{b}", self.color(C_BAR[b], bold=True))
        if max_xoff and self.score_xoff < max_xoff:
            self.put(top, x1 - 1, "→", self.color(C_ACCENT))
        if self.score_xoff:
            self.put(top, x0, "←", self.color(C_ACCENT))

        first = self.instr.top
        for r in range(min(n - first, list_page)):
            i = first + r
            y = top + 1 + r
            selected = i == self.instr.cursor
            base = self.color(C_SEL) if selected else 0
            self.put(y, x0, " " * width, base)
            wr, rd, wt = events[i]
            for b in range(self.score_xoff, self.score_xoff + self._score_fit):
                x = lane_x(b)
                attr = base | self.color(C_BAR[b], bold=True)
                mark = None
                if wr == b:
                    mark = "■"
                elif rd == b:
                    mark = "◆"
                elif wt & (1 << b):
                    mark = "□"
                else:
                    for start, end, _kind in spans[b]:
                        if end is not None and start < i < end:
                            mark = "│"
                            break
                        if end is None and start < i:
                            mark = "╎"  # set but never awaited
                            break
                if mark:
                    self.put(y, x, mark, attr)

        # legend pinned at the bottom of the pane
        y = bottom - legend_rows
        self.hline(y, x0 + 1, x1 - 1, "─", self.color(C_BORDER))
        self.put(y + 1, x0, " ■ set wr  ◆ set rd", self.color(C_DIM))
        self.put(y + 2, x0, " □ wait   │ pending", self.color(C_DIM))
        self.put(y + 3, x0, " t structure view", self.color(C_BORDER))

    def draw_center(self, top: int, bottom: int, x0: int, x1: int) -> None:
        width = x1 - x0
        sec = self.section
        hex_rows = 6
        list_page = max(1, bottom - top - hex_rows - 3)
        if sec is None:
            self.put(top + 1, x0 + 1, "no CUDA .text sections found",
                     self.color(C_ERR))
            return
        n = sec.n_instructions
        self.instr.clamp(n, list_page)
        title = f" {sec.name} · {n} instr"
        self.put(top, x0, title[:width].ljust(width), self.color(C_DIM))
        y = top + 1
        for i in range(self.instr.top, min(n, self.instr.top + list_page)):
            off = sec.abs_offset + i * INSTRUCTION_SIZE
            decoded = self.model.decode_at(off)
            selected = i == self.instr.cursor
            self.draw_instr_line(y, x0, x1, decoded, selected)
            y += 1
        # hex divider + synced hex
        y = bottom - hex_rows - 1
        self.hline(y, x0, x1, "─", self.color(C_BORDER))
        self.put(y + 1, x0, " HEX (synced)".ljust(width), self.color(C_DIM))
        self.draw_synced_hex(y + 2, bottom, x0, x1)

    def draw_instr_line(self, y: int, x0: int, x1: int,
                        decoded: DecodedInstruction, selected: bool) -> None:
        width = x1 - x0
        base_attr = self.color(C_SEL) if selected else 0
        pad = " " * width
        self.put(y, x0, pad, base_attr)
        x = self.put(y, x0, f" {self.rel(decoded.offset):04X} ",
                     base_attr | (self.color(C_ACCENT, bold=True) if selected
                                  else self.color(C_DIM)))
        bad = decoded.n_failed > 0 or decoded.defn is None
        mnemonic = decoded.mnemonic
        guard = ""
        if mnemonic.startswith("@") and " " in mnemonic:
            guard, _, mnemonic = mnemonic.partition(" ")
            guard += " "
        base, dot, mods = mnemonic.partition(".")
        if bad:
            base_mn = mod_mn = self.color(C_ERR, bold=True)
        elif selected:
            base_mn = mod_mn = self.color(C_SEL, bold=True)
        else:
            base_mn = self.color(C_MNEMONIC, bold=True)
            mod_mn = self.color(C_MNEM_MOD, bold=True)
        if guard:
            x = self.put(y, x, guard,
                         base_attr | (0 if selected else self.color(C_ACCENT)))
        x = self.put(y, x, base, base_mn)
        if dot:
            x = self.put(y, x, "." + mods, mod_mn)
        pad = max(0, 7 - len(guard) - len(mnemonic))
        x = self.put(y, x, " " * pad, base_attr)
        x = self.put(y, x, f" {decoded.operands}",
                     base_attr or self.color(C_TEXT))
        badges = []
        c = decoded.control
        if c.get("wr_bar", 7) != 7:
            badges.append((f"[B{c['wr_bar']}w]", C_ERR))
        if c.get("rd_bar", 7) != 7:
            badges.append((f"[B{c['rd_bar']}r]", C_OK))
        if c.get("wt_bar", 0):
            badges.append((f"[wt{c['wt_bar']:02X}]", C_ACCENT))
        if decoded.n_failed:
            badges.append((f"!! {decoded.n_failed}", C_ERR))
        reserve = 2 if selected else 0
        bx = x1 - reserve - sum(len(b) + 1 for b, _ in badges)
        for text, pair in badges:
            if bx > x:
                bx = self.put(y, bx, " " + text, base_attr | self.color(pair))
        if selected and x1 - 2 >= x:
            self.put(y, x1 - 2, "◂", self.color(C_SEL, bold=True))

    def _field_byte_range(self) -> Tuple[int, int]:
        """Bytes (within the instruction) covered by the selected field."""
        decoded = self.current_decoded()
        if not decoded or self.pane != PANE_FIELDS:
            return (-1, -1)
        rows = self.field_rows(decoded)
        if not rows or self.fields.cursor >= len(rows):
            return (-1, -1)
        kind, name = rows[self.fields.cursor]
        if kind == "control":
            return (13, 15)  # control bytes
        if kind == "action":
            return (0, 15)  # generate replaces the whole instruction
        view = next((f for f in decoded.fields if f.name == name), None)
        if view is None:
            return (-1, -1)
        from ..enc.bits import parse_bit_spec
        segs = parse_bit_spec(view.bits)
        if not segs:
            return (-1, -1)
        lo = min(s[0] for s in segs) // 8
        hi = max(s[1] for s in segs) // 8
        return (lo, hi)

    def draw_synced_hex(self, top: int, bottom: int, x0: int, x1: int) -> None:
        sec = self.section
        if sec is None:
            return
        rows = bottom - top
        sel = self.instr.cursor
        first = max(0, min(sel - rows // 2,
                           sec.n_instructions - rows))
        fb_lo, fb_hi = self._field_byte_range()
        for r in range(rows):
            i = first + r
            if i >= sec.n_instructions:
                break
            off = sec.abs_offset + i * INSTRUCTION_SIZE
            selected = i == sel
            y = top + r
            base_attr = self.color(C_SEL) if selected else 0
            self.put(y, x0, " " * (x1 - x0), base_attr)
            x = self.put(y, x0, f" {self.rel(off):04X}  ",
                         base_attr | (0 if selected else self.color(C_DIM)))
            for b in range(INSTRUCTION_SIZE):
                pos = off + b
                byte = self.model.data[pos]
                attr = base_attr or self.color(C_TEXT)
                if pos in self.model.modified:
                    attr = self.color(C_EDIT, bold=True)
                elif selected and fb_lo <= b <= fb_hi:
                    attr = self.color(C_ACCENT, bold=True)
                x = self.put(y, x, f"{byte:02X}", attr)
                if b % 2 == 1 and b != INSTRUCTION_SIZE - 1:
                    x = self.put(y, x, " ", base_attr)

    def draw_inspector(self, top: int, bottom: int, x0: int, x1: int) -> None:
        width = x1 - x0
        decoded = self.current_decoded()
        if decoded is None:
            self.put(top, x0, " FIELDS", self.color(C_DIM))
            return
        rows = self.field_rows(decoded)
        head = f" FIELDS · {decoded.mnemonic} @ {self.rel(decoded.offset):X}"
        if len(decoded.candidates) > 1:
            head += f" ·v{len(decoded.candidates)}"
        self.put(top, x0, head[:width].ljust(width), self.color(C_DIM))
        checks_rows = 4
        sched_rows = len(CONTROL_FIELDS) + 1
        page = max(1, bottom - top - 1 - sched_rows - checks_rows - 1)
        n_fields = len(decoded.fields)
        self.fields.cursor = max(0, min(self.fields.cursor, len(rows) - 1))
        if self.fields.cursor < n_fields:
            if self.fields.cursor < self.fields.top:
                self.fields.top = self.fields.cursor
            if self.fields.cursor >= self.fields.top + page:
                self.fields.top = self.fields.cursor - page + 1
        self.fields.top = max(0, min(self.fields.top, max(0, n_fields - page)))

        y = top + 1
        start = self.fields.top
        for i in range(start, min(n_fields, start + page)):
            f = decoded.fields[i]
            selected = (self.pane == PANE_FIELDS and self.fields.cursor == i)
            attr = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            line = f"  {f.name[:9]:<9}[{f.display[:9]:<9}] {f.bits_text}"
            self.put(y, x0, line[:width].ljust(width), attr)
            y += 1
        if n_fields > start + page:
            self.put(y - 1, x1 - 3, " ↓", self.color(C_DIM))

        self.put(y, x0, " SCHEDULING".ljust(width), self.color(C_DIM))
        y += 1
        c = decoded.control
        for j, (name, _mx) in enumerate(CONTROL_FIELDS):
            idx = n_fields + j
            selected = (self.pane == PANE_FIELDS and self.fields.cursor == idx)
            attr = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            label = {"stall": "stall", "yield_": "yield", "wr_bar": "wr bar",
                     "rd_bar": "rd bar", "wt_bar": "wt mask"}[name]
            value = c.get(name, 0)
            shown = f"{value:06b}" if name == "wt_bar" else str(value)
            self.put(y, x0, f"  {label:<8}[{shown:<6}]"[:width].ljust(width), attr)
            y += 1

        self.put(y, x0, " CHECKS".ljust(width), self.color(C_DIM))
        y += 1
        ordered = ([c for c in decoded.checks if c.status == "fail"]
                   + [c for c in decoded.checks if c.status == "unknown"]
                   + [c for c in decoded.checks if c.status == "pass"])
        n_pass = sum(1 for c in decoded.checks if c.status == "pass")
        shown_any = False
        for check in ordered:
            if y >= bottom - 1:
                break
            if check.status == "pass":
                if not shown_any:
                    self.put(y, x0, f"  ✓ {n_pass} conditions pass",
                             self.color(C_OK))
                    y += 1
                    shown_any = True
                continue
            mark, pair = ("✗", C_ERR) if check.status == "fail" else ("~", C_ACCENT)
            text = f"  {mark} {check.name}"
            if check.message:
                text += f": {check.message}"
            self.put(y, x0, text[:width], self.color(pair))
            y += 1
        if not decoded.checks and y < bottom:
            self.put(y, x0, "  (no conditions)", self.color(C_DIM))

        if decoded.defn is not None:
            selected = (self.pane == PANE_FIELDS
                        and self.fields.cursor == len(rows) - 1)
            attr = self.color(C_SEL, bold=True) if selected \
                else self.color(C_ACCENT)
            self.put(bottom - 1, x0, " [ n generate generic ]"[:width]
                     .ljust(width), attr)

    # -- hex tab -------------------------------------------------------------

    def draw_hex_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        data = self.model.data
        page = bottom - top - 1
        n_rows = (len(data) + 15) // 16
        cursor = self.hexview.cursor
        row = cursor // 16
        top_row = min(max(0, row - page // 2), max(0, n_rows - page))
        head = ("  OFFSET    00 01 02 03  04 05 06 07  "
                "08 09 0A 0B  0C 0D 0E 0F   ASCII")
        self.put(top, 1, head[:w - 2], self.color(C_DIM))
        for r in range(page):
            ri = top_row + r
            if ri >= n_rows:
                break
            y = top + 1 + r
            base = ri * 16
            x = self.put(y, 1, f"  {base:08X}  ", self.color(C_DIM))
            ascii_repr = []
            for b in range(16):
                pos = base + b
                if pos >= len(data):
                    x = self.put(y, x, "   ")
                    ascii_repr.append(" ")
                    continue
                byte = data[pos]
                attr = self.color(C_TEXT)
                if pos in self.model.modified:
                    attr = self.color(C_EDIT, bold=True)
                if pos == cursor:
                    attr = self.color(C_SEL, bold=True)
                x = self.put(y, x, f"{byte:02X}", attr)
                x = self.put(y, x, "  " if b % 4 == 3 else " ")
                ch = chr(byte) if 32 <= byte < 127 else "·"
                ascii_repr.append(ch)
            self.put(y, x + 1, "".join(ascii_repr),
                     self.color(C_DIM))
        info = f" cursor 0x{cursor:X} · Enter rewrite bytes"
        self.put(bottom - 1, 1, info, self.color(C_ACCENT))

    # -- ELF tab -------------------------------------------------------------

    def draw_elf_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        images = self.model.images
        if not images:
            self.put(top + 1, 2, "no ELF images found", self.color(C_ERR))
            return
        self.elf_image = max(0, min(self.elf_image, len(images) - 1))
        image = images[self.elf_image]
        head = (f" image {self.elf_image + 1}/{len(images)} @ 0x{image.base:X}"
                f" · {image.machine_name}"
                + (f" · {image.sm_name}" if image.is_cuda else "")
                + "   ([ ] switch image)")
        self.put(top, 2, head[:w - 4], self.color(C_DIM))
        n = len(ELF64_HEADER_FIELDS)
        self.elf.clamp(n, bottom - top - 3)
        y = top + 2
        for i, (key, label, off, size) in enumerate(ELF64_HEADER_FIELDS):
            if y >= bottom - 1:
                break
            fmt = {2: "<H", 4: "<I", 8: "<Q"}[size]
            value = struct.unpack_from(fmt, self.model.data, image.base + off)[0]
            selected = i == self.elf.cursor
            note = ""
            if key == "e_type":
                note = ELF_TYPE_NAMES.get(value, "")
            elif key == "e_machine":
                note = image.machine_name if value == image.machine else ""
            elif key == "e_flags" and image.is_cuda:
                bits = [name for mask, name in CUDA_FLAG_BITS if value & mask]
                note = image.sm_name + (" · " + "|".join(bits) if bits else "")
            attr = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            edited = any(image.base + off + b in self.model.modified
                         for b in range(size))
            if edited and not selected:
                attr = self.color(C_EDIT, bold=True)
            line = f"  {label:<28} 0x{value:0{size * 2}X}"
            x = self.put(y, 2, line[:w - 4], attr)
            if note:
                self.put(y, min(x + 2, w - 3), note[:w - x - 4],
                         self.color(C_DIM))
            y += 1

    # -- nv.info / notes tabs ------------------------------------------------

    def draw_nvinfo_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        records = self.nvinfo_records()
        page = bottom - top - 2
        self.nvinfo.clamp(len(records), page)
        self.put(top, 2, f" {len(records)} .nv.info records"
                 " (Enter edits scalar values)", self.color(C_DIM))
        y = top + 1
        last_section = None
        i = self.nvinfo.top
        shown = 0
        while i < len(records) and shown < page:
            rec = records[i]
            if rec.section != last_section:
                last_section = rec.section
                self.put(y, 2, f" {rec.section}"[:w - 4],
                         self.color(C_MNEMONIC, bold=True))
                y += 1
                shown += 1
                if shown >= page:
                    break
            selected = i == self.nvinfo.cursor
            attr = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            editable = "✎ " if rec.scalar else "  "
            line = (f"   {editable}{rec.attr_name:<36} {rec.format_name:<10} "
                    f"{rec.summary}")
            self.put(y, 2, line[:w - 4], attr)
            y += 1
            shown += 1
            i += 1

    def draw_notes_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        notes = self.note_entries()
        page = bottom - top - 2
        self.notes.clamp(len(notes), page)
        self.put(top, 2, f" {len(notes)} note entries", self.color(C_DIM))
        y = top + 1
        for i in range(self.notes.top, min(len(notes), self.notes.top + page)):
            note = notes[i]
            selected = i == self.notes.cursor
            attr = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            desc = note.desc[:16].hex(" ") + (" …" if len(note.desc) > 16 else "")
            line = (f"  {note.section:<20} type {note.n_type:<3} "
                    f"{note.name:<12} {desc}")
            self.put(y, 2, line[:w - 4], attr)
            y += 1

    # -- ISA / disasm tabs ---------------------------------------------------

    def draw_isa_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        left_w = min(46, w // 2)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        self.put(2, left_w, "┬", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, left_w, "│", border)
            self.put(y, w - 1, "│", border)
        self.put(bottom, left_w, "┴", border)

        rows = self.isa_rows()
        total = len(self.isa_all_rows())
        page = bottom - top - 1
        self.isa.clamp(len(rows), page)
        sm = self.model.spec.sm or "?"
        head = f" SM{sm} · {len(rows)}/{total} classes"
        if self.isa_query:
            head += f" · /{self.isa_query}"
        self.put(top, 1, head[:left_w - 1].ljust(left_w - 1),
                 self.color(C_DIM))
        y = top + 1
        for i in range(self.isa.top, min(len(rows), self.isa.top + page)):
            mnemonic, name, _opcode, _pipe = rows[i]
            selected = i == self.isa.cursor
            attr = self.color(C_SEL, bold=True) if selected \
                else self.color(C_TEXT)
            line = f" {mnemonic:<12} {name}"
            self.put(y, 1, line[:left_w - 1].ljust(left_w - 1), attr)
            y += 1
        if not rows:
            self.put(top + 1, 2, "no class matches the filter",
                     self.color(C_DIM))
        self.draw_isa_detail(top, bottom, left_w + 1, w - 1,
                             rows[self.isa.cursor] if rows else None)

    def draw_isa_detail(self, top: int, bottom: int, x0: int, x1: int,
                        row) -> None:
        width = x1 - x0
        self.put(top, x0, " CLASS".ljust(width), self.color(C_DIM))
        if row is None:
            return
        _mnemonic, name, opcode, pipeline = row
        ic = self.model.spec.get(name)
        if ic is None:
            return
        lines: List[Tuple[str, int]] = []
        lines.append((f" {name}", self.color(C_ACCENT, bold=True)))
        lines.append((f"  opcode 0x{opcode:04X}"
                      + (f" · {pipeline}" if pipeline else ""),
                      self.color(C_TEXT)))
        aliases = ", ".join(f"{a} 0x{v:04X}" for a, v in ic.opcodes.items())
        if aliases:
            lines.append((f"  aliases: {aliases}", self.color(C_DIM)))
        info = ic.latency_info
        latencies = (info.get("latencies") or {}).get("TABLE_TRUE") or []
        lat_bits = []
        if latencies:
            lat_bits.append(f"latency {latencies[0]}")
        if info.get("gpr_self_latency") is not None:
            lat_bits.append(f"self {info['gpr_self_latency']}")
        occupancy = (info.get("pipeline_resource") or {}).get("occupancy")
        if occupancy is not None:
            lat_bits.append(f"occupancy {occupancy}")
        if lat_bits:
            lines.append(("  " + " · ".join(lat_bits), self.color(C_TEXT)))
        props = ic.properties
        prop_bits = [str(props[k]) for k in
                     ("INSTRUCTION_TYPE", "MEM_SCBD_TYPE") if props.get(k)]
        if prop_bits:
            lines.append(("  " + " · ".join(prop_bits), self.color(C_DIM)))
        n_cond = sum(1 for c in ic.conditions if isinstance(c, dict)
                     and not str(c.get("name", "")).startswith("FORMAT "))
        lines.append((f"  {len(ic.arguments)} arguments · "
                      f"{n_cond} conditions", self.color(C_TEXT)))
        lines.append(("", 0))
        lines.append((" SYNTAX", self.color(C_DIM)))
        text = ic.data.get("TEXT") or ic.data.get("FORMAT", {}).get("TEXT", "")
        for chunk_start in range(0, len(text), max(1, width - 3)):
            lines.append(("  " + text[chunk_start:chunk_start + width - 3],
                          self.color(C_MNEMONIC)))
            if len(lines) > bottom - top - 3:
                break
        y = top + 1
        for line, attr in lines:
            if y >= bottom - 1:
                break
            self.put(y, x0, line[:width].ljust(width), attr)
            y += 1
        self.put(bottom - 1, x0,
                 " [ Enter stamp generic at cursor ]"[:width].ljust(width),
                 self.color(C_ACCENT))

    def draw_disasm_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        rows, error = self.disasm_rows()
        head = " cuobjdump -sass (edited buffer) · r re-run · Enter jump"
        self.put(top, 1, head[:w - 2].ljust(w - 2), self.color(C_DIM))
        if error:
            self.put(top + 2, 2, "cuobjdump failed:", self.color(C_ERR, bold=True))
            for i in range(0, len(error), w - 6):
                if top + 3 + i // (w - 6) >= bottom - 1:
                    break
                self.put(top + 3 + i // (w - 6), 3, error[i:i + w - 6],
                         self.color(C_ERR))
            return
        page = bottom - top - 1
        self.disasm.clamp(len(rows), page)
        y = top + 1
        for i in range(self.disasm.top, min(len(rows), self.disasm.top + page)):
            kind, payload, extra = rows[i]
            selected = i == self.disasm.cursor
            if kind == "func":
                attr = self.color(C_SEL, bold=True) if selected \
                    else self.color(C_MNEMONIC, bold=True)
                self.put(y, 1, f" {payload}"[:w - 2].ljust(w - 2), attr)
                y += 1
                continue
            abs_off, match = extra
            edited = abs_off is not None and any(
                abs_off + b in self.model.modified
                for b in range(INSTRUCTION_SIZE))
            attr = self.color(C_SEL, bold=True) if selected \
                else self.color(C_EDIT, bold=True) if edited \
                else self.color(C_TEXT)
            marker = "✗" if match is False else "●" if edited else " "
            line = f"  {payload.offset:04X}  {marker} {payload.text}"
            self.put(y, 1, line[:w - 2].ljust(w - 2), attr)
            if match is False and not selected:
                self.put(y, 9, "✗", self.color(C_ERR, bold=True))
            y += 1
        if not rows:
            self.put(top + 2, 2, "no SASS instructions found",
                     self.color(C_DIM))

    # -- measure tab ---------------------------------------------------------

    def _cursor_trigger(self) -> Tuple[Optional[bytes], str]:
        """The instruction under the Instructions-tab cursor, as a TRIGGER."""
        off = self.current_offset()
        if off is None:
            return None, "(no instruction)"
        decoded = self.model.decode_at(off)
        return self.model.instruction_bytes(off), \
            f"{decoded.text}  @{self.rel(off):04X}"

    def sorted_results(self) -> List:
        return sorted(self.measure.rows,
                      key=lambda r: (r.error is not None, -r.magnitude))

    def draw_measure_tab(self, top: int, bottom: int, w: int) -> None:
        border = self.color(C_BORDER)
        self.put(2, 0, "├", border)
        self.hline(2, 1, w - 1, "─", border)
        self.put(2, w - 1, "┤", border)
        for y in range(top, bottom):
            self.put(y, 0, "│", border)
            self.put(y, w - 1, "│", border)
        if not self.measure._refreshed:
            self.measure.refresh()
        m = self.measure
        x0 = 2
        y = top

        # SETUP -------------------------------------------------------------
        gpu = f"{m.gpu_label}" + (f" · sm_{m.sm}" if m.sm else "")
        self.put(y, x0, " SETUP", self.color(C_DIM))
        self.put(y, max(x0, w - len(gpu) - 2), gpu, self.color(C_DIM))
        y += 1

        def check(ok, label, detail, hint=""):
            nonlocal y
            mark, pair = ("✓", C_OK) if ok else ("✗", C_ERR)
            self.put(y, x0, f"   {mark} ", self.color(pair, bold=True))
            self.put(y, x0 + 5, f"{label:<10}", self.color(C_TEXT))
            self.put(y, x0 + 16, detail, self.color(C_TEXT if ok else C_DIM))
            if hint and not ok:
                self.put(y, x0 + 16 + len(detail) + 2, hint,
                         self.color(C_ACCENT))
            y += 1

        check(m.gpu_ok, "GPU", m.gpu_label if m.gpu_ok else "not detected")
        _extra = ("".join(x for x, p in [(" +super", m.super_path),
                                         (" +tma", m.tma_path),
                                         (" +cga", m.cga_path)] if p))
        check(bool(m.harness_path), "Harness",
              f"harness_sm{m.sm}{_extra} ({m.slots} slots)" if m.harness_path
              else "not built",
              "[ b build (nvcc) ]" if m.can_build else "nvcc missing")
        check(bool(m.offsets_path), "Offsets",
              m.offsets_path.name if m.offsets_path else "not extracted")
        if m.db_path:
            check(True, "Instr DB", f"{m.db_path.name} (validated)")
        else:
            # optional, not an error: sweeps fall back to raw generic encodings
            self.put(y, x0, "   ○ ", self.color(C_DIM, bold=True))
            self.put(y, x0 + 5, f"{'Instr DB':<10}", self.color(C_TEXT))
            self.put(y, x0 + 16, "generic (unfiltered — many ops fault)",
                     self.color(C_DIM))
            if m.ready:
                self.put(y, x0 + 16 + 38, "[ d build validated DB ]",
                         self.color(C_ACCENT))
            y += 1
        y += 1

        # CONFIG ------------------------------------------------------------
        cfg = (f" CONFIG   mode [{m.mode}]   trials [{m.trials}]   "
               f"build slots [{m.slots}]")
        self.put(y, x0, cfg, self.color(C_TEXT))
        y += 1
        cur_bytes, cur_label = self._cursor_trigger()
        src_cursor = "●" if m.source == "cursor" else "○"
        src_sweep = "●" if m.source == "sweep" else "○"
        self.put(y, x0, f" SOURCE   {src_cursor} cursor: ",
                 self.color(C_TEXT))
        self.put(y, x0 + 19, cur_label[:w - x0 - 22],
                 self.color(C_ACCENT if m.source == "cursor" else C_DIM))
        y += 1
        src = "validated DB" if m.db_path else "generic — d to filter"
        self.put(y, x0, f"          {src_sweep} sweep:  "
                 f"{m.pattern} pattern · ≤{m.sweep_limit} triggers · {src}",
                 self.color(C_ACCENT if m.source == "sweep" else C_DIM))
        y += 1

        # RUN ---------------------------------------------------------------
        self.hline(y, x0, w - 1, "─", border)
        self.put(y, x0, " RUN ", self.color(C_DIM))
        y += 1
        job = m.job
        if job is not None and not job.finished:
            if job.total:
                frac = job.done / job.total
                barw = min(30, w - x0 - 24)
                fill = int(frac * barw)
                self.put(y, x0, "  " + "█" * fill + "░" * (barw - fill),
                         self.color(C_ACCENT))
                self.put(y, x0 + 3 + barw,
                         f" {job.done}/{job.total} {job.label[:20]}",
                         self.color(C_TEXT))
            else:
                self.put(y, x0, f"  {job.label}", self.color(C_ACCENT))
        elif job is not None and job.error:
            self.put(y, x0, f"  ✗ {job.error[:w - x0 - 4]}",
                     self.color(C_ERR))
        elif m.last_status:
            self.put(y, x0, f"  {m.last_status[:w - x0 - 4]}",
                     self.color(C_OK))
        else:
            hint = ("Enter measure · b build harness · d build DB · "
                    "m mode · r source · p pattern · x cancel · e export")
            self.put(y, x0, "  " + hint[:w - x0 - 4], self.color(C_DIM))
        y += 2

        # RESULTS -----------------------------------------------------------
        rows = self.sorted_results()
        self.put(y, x0, f" RESULTS · {len(rows)} (↓|diff|)".ljust(w - 3),
                 self.color(C_DIM))
        y += 1
        hdr = f"   {'trigger':<26}{'mode':<11}{'hot':>7}{'cold':>7}{'diff':>8}{'n':>7}"
        self.put(y, x0, hdr[:w - 3], self.color(C_DIM))
        y += 1
        page = max(1, bottom - y)
        self.results.clamp(len(rows), page)
        for i in range(self.results.top, min(len(rows), self.results.top + page)):
            r = rows[i]
            selected = i == self.results.cursor
            base = self.color(C_SEL, bold=True) if selected else self.color(C_TEXT)
            if r.error:
                line = f"   {r.trigger[:26]:<26}{r.mode:<11}  {r.error}"
                self.put(y, x0, line[:w - 3].ljust(w - 3),
                         base if selected else self.color(C_ERR))
            else:
                diff = f"{r.diff_mean:+.1f}"
                line = (f"   {r.trigger[:26]:<26}{r.mode:<11}"
                        f"{r.hot_mean:>7.1f}{r.cold_mean:>7.1f}"
                        f"{diff:>8}{r.n:>7}")
                self.put(y, x0, line[:w - 3].ljust(w - 3), base)
                if not selected and abs(r.diff_mean) >= 1.0:
                    dx = x0 + 3 + 26 + 11 + 7 + 7
                    self.put(y, dx, f"{diff:>8}", self.color(
                        C_ERR if r.diff_mean > 0 else C_OK, bold=True))
            y += 1
        if not rows:
            self.put(y, x0, "   (no measurements yet — press Enter)",
                     self.color(C_DIM))

    # -- help overlay --------------------------------------------------------

    def show_help(self) -> None:
        h, w = self.screen.getmaxyx()
        box_w = min(w - 4, max(len(line) for line in HELP_LINES) + 6)
        box_h = min(h - 2, len(HELP_LINES) + 2)
        y0 = (h - box_h) // 2
        x0 = (w - box_w) // 2
        border = self.color(C_ACCENT)
        self.put(y0, x0, "┌" + "─" * (box_w - 2) + "┐", border)
        for i in range(1, box_h - 1):
            self.put(y0 + i, x0, "│" + " " * (box_w - 2) + "│", border)
            line = HELP_LINES[i - 1] if i - 1 < len(HELP_LINES) else ""
            self.put(y0 + i, x0 + 3, line[:box_w - 6], self.color(C_TEXT))
        self.put(y0 + box_h - 1, x0, "└" + "─" * (box_w - 2) + "┘", border)
        self.screen.refresh()
        self.screen.getch()

    # =========================================================================
    # actions
    # =========================================================================

    def action_goto(self) -> None:
        text = self.prompt("goto offset (hex)")
        if not text:
            return
        try:
            value = int(text.strip().replace("0x", ""), 16)
        except ValueError:
            self.set_status(f"bad offset: {text}", C_ERR)
            return
        if self.tab == TAB_HEX:
            self.hexview.cursor = max(0, min(value, len(self.model.data) - 1))
            self.set_status(f"cursor at 0x{self.hexview.cursor:X}")
            return
        sec = self.section
        image = self.model.image_for_section(sec) if sec else None
        base = image.base if image else 0
        for j, tsec in enumerate(self.model.sections):
            rel0 = tsec.abs_offset - base
            if rel0 <= value < rel0 + tsec.size:
                self.section_idx = j
                self.instr.cursor = (value - rel0) // INSTRUCTION_SIZE
                self.set_status(f"at {tsec.name}+0x{value - rel0:X}")
                return
        self.set_status(f"offset 0x{value:X} not in a .text section", C_ERR)

    def action_search(self) -> None:
        sec = self.section
        if sec is None:
            return
        text = self.prompt("search")
        if not text:
            return
        found = self.model.search_instruction(sec, text,
                                              self.instr.cursor + 1)
        if found is None:
            self.set_status(f"no match for {text!r}", C_ERR)
        else:
            self.instr.cursor = found
            self.set_status(f"match at instruction {found}")

    def action_save(self) -> None:
        text = self.prompt("save to", str(self.model.default_save_path()))
        if not text:
            return
        try:
            out = self.model.save(text)
        except OSError as exc:
            self.set_status(f"save failed: {exc}", C_ERR)
            return
        self.set_status(f"saved {self.model.n_modified} modified bytes"
                        f" to {out}", C_OK)

    def action_undo(self) -> None:
        label = self.model.undo()
        self.invalidate()
        if label is None:
            self.set_status("nothing to undo")
        else:
            self.set_status(f"undid: {label}", C_ACCENT)

    def action_edit_field(self) -> None:
        decoded = self.current_decoded()
        if decoded is None or decoded.defn is None:
            return
        rows = self.field_rows(decoded)
        if not rows:
            return
        self.fields.cursor = min(self.fields.cursor, len(rows) - 1)
        kind, name = rows[self.fields.cursor]
        offset = decoded.offset
        if kind == "action":
            self.action_generate()
            return
        if kind == "control":
            current = decoded.control.get(name, 0)
            text = self.prompt(f"set {name} (now {current})")
            if text is None or not text.strip():
                return
            try:
                self.model.set_control(offset, name, int(text.strip(), 0))
                self.invalidate()
                self.set_status(f"{name} set", C_OK)
            except (EditError, ValueError) as exc:
                self.set_status(str(exc), C_ERR)
            return
        view = next(f for f in decoded.fields if f.name == name)
        text = None
        choices = self._field_choices(view)
        if choices:
            items = [("(custom value…)", "\x00custom")]
            for member, value in choices:
                marker = "◂ current" if value == view.value else ""
                items.append((f"{member:<24} {value:>5}  0x{value:X}  {marker}",
                              member))
            chosen = self.picker(f"{name} · {view.bucket or view.source}",
                                 items)
            if chosen is None:
                return
            if chosen != "\x00custom":
                text = chosen
        if text is None:
            text = self.prompt(f"set {name} (now {view.display})")
            if text is None or not text.strip():
                return
        try:
            value = self.model.set_field(offset, name, text)
            self.invalidate()
            self.set_status(f"{name} = {text.strip()} (0x{value:X})", C_OK)
        except EditError as exc:
            self.set_status(str(exc), C_ERR)

    def _field_choices(self, view) -> List[Tuple[str, int]]:
        """(member name, value) options for an enum-typed field — the
        DEFAULTS bucket members, or the special-register table."""
        limit = (1 << view.width) if view.width else None
        out = []
        if view.bucket:
            for member, value in (self.model.decoder.bucket(view.bucket)
                                  or {}).items():
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
                if value >= 0 and (limit is None or value < limit):
                    out.append((member, value))
        elif view.source == "SpecialRegister":
            for value, member in sorted(
                    self.model.spec.special_registers.items()):
                if value >= 0 and (limit is None or value < limit):
                    out.append((member, value))
        out.sort(key=lambda kv: (kv[1], kv[0]))
        return out

    def action_opcode(self) -> None:
        decoded = self.current_decoded()
        if decoded is None:
            return
        items = []
        for name, ic in self.model.spec.instructions.items():
            mnemonic = self.model.decoder.mnemonic_for(ic, ic.opcode)
            items.append((f"{mnemonic:<14} {name:<38} 0x{ic.opcode:04X}",
                          name))
        items.sort(key=lambda it: it[0])
        chosen = self.picker("opcode", items)
        if chosen is None:
            return
        defn = self.model.spec.get(chosen)
        if defn is None:
            return
        self.model.set_opcode(decoded.offset, defn.opcode)
        # pin the picked class so multi-variant opcodes decode as chosen
        self.model.variant_overrides[decoded.offset] = chosen
        self.invalidate()
        new = self.model.decode_at(decoded.offset)
        self.set_status(f"opcode 0x{defn.opcode:04X} → {new.mnemonic} "
                        f"({chosen})", C_OK)

    def action_slide(self, delta: int) -> None:
        """Slide the selected instruction one slot up (-1) or down (+1)."""
        sec = self.section
        off = self.current_offset()
        if sec is None or off is None:
            return
        target = self.instr.cursor + delta
        if not 0 <= target < sec.n_instructions:
            self.set_status("already at the section edge")
            return
        try:
            self.model.swap_instructions(
                off, sec.abs_offset + target * INSTRUCTION_SIZE)
        except EditError as exc:
            self.set_status(str(exc), C_ERR)
            return
        self.instr.cursor = target
        self.invalidate()
        decoded = self.current_decoded()
        name = decoded.mnemonic if decoded else "instruction"
        self.set_status(f"{name} moved {'down' if delta > 0 else 'up'}"
                        f" to {self.rel(self.current_offset() or 0):04X}",
                        C_ACCENT)

    def action_variant(self) -> None:
        off = self.current_offset()
        if off is None:
            return
        name = self.model.cycle_variant(off)
        if name is None:
            self.set_status("single-variant opcode")
        else:
            self.set_status(f"variant: {name}", C_ACCENT)

    def action_generate(self) -> None:
        """Replace the selected instruction with a generic valid encoding
        of its current opcode, built by the framework generator."""
        decoded = self.current_decoded()
        if decoded is None:
            return
        try:
            name = self.model.generate_generic(decoded.offset)
        except EditError as exc:
            self.set_status(str(exc), C_ERR)
            return
        self.invalidate()
        new = self.model.decode_at(decoded.offset)
        if new.n_failed:
            self.set_status(f"generated {new.text} — {new.n_failed}"
                            " conditions still fail", C_ACCENT)
        else:
            self.set_status(f"generated {new.text} ({name})", C_OK)

    def action_stamp_isa(self) -> None:
        """Stamp a generic encoding of the ISA-browser selection over the
        instruction at the Instructions-tab cursor."""
        rows = self.isa_rows()
        if not rows:
            return
        _mnemonic, name, _opcode, _pipe = rows[self.isa.cursor]
        off = self.current_offset()
        if off is None:
            self.set_status("no instruction selected (tab 1)", C_ERR)
            return
        try:
            self.model.write_generic(off, name)
        except EditError as exc:
            self.set_status(str(exc), C_ERR)
            return
        self.invalidate()
        new = self.model.decode_at(off)
        self.tab = TAB_INSTR
        self.set_status(f"stamped {new.text} ({name})"
                        f" at {self.rel(off):04X}", C_OK)

    def action_measure_run(self) -> None:
        if self.measure.busy():
            self.set_status("a measurement job is already running", C_ACCENT)
            return
        if not self.measure.ready:
            if self.measure.can_build:
                self.set_status("not ready — press b to build a harness", C_ERR)
            else:
                self.set_status("not ready — need a GPU and nvcc", C_ERR)
            return
        cur_bytes, cur_label = self._cursor_trigger()
        err = self.measure.start_measure(cur_bytes, cur_label)
        if err:
            self.set_status(err, C_ERR)
        else:
            what = ("cursor instruction" if self.measure.source == "cursor"
                    else f"{self.measure.pattern} sweep")
            self.set_status(f"measuring {what}…", C_ACCENT)

    def action_measure_build(self) -> None:
        if self.measure.busy():
            self.set_status("a job is already running", C_ACCENT)
            return
        err = self.measure.start_build()
        if err:
            self.set_status(err, C_ERR)
        else:
            self.set_status(f"building {self.measure.slots}-slot harness…",
                            C_ACCENT)

    def action_measure_build_db(self) -> None:
        if self.measure.busy():
            self.set_status("a job is already running", C_ACCENT)
            return
        err = self.measure.start_build_db()
        if err:
            self.set_status(err, C_ERR)
        else:
            self.set_status("building validated instruction DB "
                            "(runs each op on the GPU; x cancels)…", C_ACCENT)

    def action_measure_export(self) -> None:
        if not self.measure.rows:
            self.set_status("no results to export", C_ERR)
            return
        default = str(paths.generated_dir(self.measure.sm or "0")
                      / "measure_results.csv") if self.measure.sm else \
            "measure_results.csv"
        text = self.prompt("export CSV to", default)
        if not text:
            return
        try:
            n = self.measure.export_csv(text.strip())
        except OSError as exc:
            self.set_status(f"export failed: {exc}", C_ERR)
            return
        self.set_status(f"exported {n} rows to {text.strip()}", C_OK)

    def action_isa_filter(self) -> None:
        text = self.prompt("filter classes", self.isa_query)
        if text is None:
            return
        self.isa_query = text.strip()
        self.isa.cursor = self.isa.top = 0

    def action_disasm_jump(self) -> None:
        """Jump from a cuobjdump line to the same instruction in tab 1."""
        rows, _error = self.disasm_rows()
        if not rows:
            return
        kind, _payload, extra = rows[self.disasm.cursor]
        if kind != "instr" or extra[0] is None:
            self.set_status("no matching .text section for this line")
            return
        abs_off = extra[0]
        for j, sec in enumerate(self.model.sections):
            if sec.abs_offset <= abs_off < sec.abs_offset + sec.size:
                self.section_idx = j
                self.instr.cursor = (abs_off - sec.abs_offset) \
                    // INSTRUCTION_SIZE
                self.tab = TAB_INSTR
                self.pane = PANE_INSTR
                self.set_status(
                    f"at {sec.name}+0x{abs_off - sec.abs_offset:X}")
                return

    def activate_structure(self) -> None:
        rows = self._structure_rows
        if not rows:
            return
        kind, _label, payload = rows[self.structure.cursor]
        if kind == "text":
            self.section_idx = payload
            self.instr.cursor = self.instr.top = 0
            self.pane = PANE_INSTR
        elif kind == "header":
            self.elf_image = payload
            self.tab = TAB_ELF
        else:
            self.hexview.cursor = payload
            self.tab = TAB_HEX

    def action_edit_elf(self) -> None:
        images = self.model.images
        if not images:
            return
        image = images[min(self.elf_image, len(images) - 1)]
        key, label, off, size = ELF64_HEADER_FIELDS[self.elf.cursor]
        fmt = {2: "<H", 4: "<I", 8: "<Q"}[size]
        current = struct.unpack_from(fmt, self.model.data, image.base + off)[0]
        text = self.prompt(f"set {key} (now 0x{current:X})")
        if text is None or not text.strip():
            return
        try:
            value = int(text.strip(), 0)
            if value < 0 or value >= (1 << (size * 8)):
                raise ValueError("out of range")
            self.model.set_bytes(image.base + off,
                                 value.to_bytes(size, "little"),
                                 f"{key}=0x{value:X}")
            self.invalidate()
            self.set_status(f"{label} = 0x{value:X}", C_OK)
        except (ValueError, EditError) as exc:
            self.set_status(f"bad value: {exc}", C_ERR)

    def action_edit_nvinfo(self) -> None:
        records = self.nvinfo_records()
        if not records:
            return
        rec = records[min(self.nvinfo.cursor, len(records) - 1)]
        if rec.scalar is None:
            self.set_status("record has no editable scalar value")
            return
        label, abs_off, size = rec.scalar
        current = int.from_bytes(self.model.data[abs_off:abs_off + size],
                                 "little")
        text = self.prompt(f"set {rec.attr_name} {label} (now {current})")
        if text is None or not text.strip():
            return
        try:
            value = int(text.strip(), 0)
            if value < 0 or value >= (1 << (size * 8)):
                raise ValueError("out of range")
            self.model.set_bytes(abs_off, value.to_bytes(size, "little"),
                                 f"{rec.attr_name}={value}")
            self.invalidate()
            self.set_status(f"{rec.attr_name} {label} = {value}", C_OK)
        except (ValueError, EditError) as exc:
            self.set_status(f"bad value: {exc}", C_ERR)

    # =========================================================================
    # key dispatch
    # =========================================================================

    def handle_key(self, ch: int) -> bool:
        """Returns False to quit."""
        self.set_status("")
        if ch == curses.KEY_RESIZE:
            return True
        if ch in (curses.KEY_F10, ord("q")):
            if self.model.n_modified:
                answer = self.prompt(
                    f"{self.model.n_modified} bytes modified — quit without"
                    " saving? (y/n)")
                if not answer or answer.strip().lower() not in ("y", "yes"):
                    return True
            return False
        if ch in (curses.KEY_F1, ord("?")):
            self.show_help()
            return True
        if ch in (curses.KEY_F5, ord("s")):
            self.action_save()
            return True
        if ch in (curses.KEY_F8, ord("u")):
            self.action_undo()
            return True
        if ch in (curses.KEY_F3, ord("g")):
            self.action_goto()
            return True
        if ch in (curses.KEY_F4, ord("/")):
            if self.tab == TAB_ISA:
                self.action_isa_filter()
            else:
                self.action_search()
            return True
        if ch == ord("c"):
            self.action_theme()
            return True
        if ord("1") <= ch <= ord("7"):
            self.tab = ch - ord("1")
            return True
        if self.tab == TAB_INSTR:
            self.handle_instructions_key(ch)
        elif self.tab == TAB_HEX:
            self.handle_hex_key(ch)
        elif self.tab == TAB_ELF:
            self.handle_elf_key(ch)
        elif self.tab == TAB_NVINFO:
            self.handle_list_key(ch, self.nvinfo,
                                 len(self.nvinfo_records()),
                                 self.action_edit_nvinfo)
        elif self.tab == TAB_ISA:
            self.handle_isa_key(ch)
        elif self.tab == TAB_DISASM:
            self.handle_disasm_key(ch)
        elif self.tab == TAB_MEASURE:
            self.handle_measure_key(ch)
        else:
            self.handle_list_key(ch, self.notes, len(self.note_entries()), None)
        return True

    def handle_instructions_key(self, ch: int) -> None:
        sec = self.section
        # the scoreboard pane navigates instructions, like the center pane
        score_pane = (self.pane == PANE_STRUCTURE and self.left_mode == "score")
        instr_like = self.pane == PANE_INSTR or score_pane
        if ch == 9:  # Tab
            if score_pane and self._score_fit < 6:
                self.score_xoff = (self.score_xoff + 1) % (6 - self._score_fit + 1)
            else:
                self.pane = (self.pane + 1) % 3
            return
        if ch in (curses.KEY_BTAB,):
            self.pane = (self.pane - 1) % 3
            return
        if ch == ord("t"):
            self.left_mode = "structure" if self.left_mode == "score" else "score"
            return
        if ch in (ord("l"), curses.KEY_RIGHT):
            self.pane = min(self.pane + 1, PANE_FIELDS)
            return
        if ch in (ord("h"), curses.KEY_LEFT):
            self.pane = max(self.pane - 1, PANE_STRUCTURE)
            return
        if ch in (curses.KEY_NPAGE, curses.KEY_PPAGE) and instr_like:
            self.action_slide(1 if ch == curses.KEY_NPAGE else -1)
            return
        down = ch in (ord("j"), curses.KEY_DOWN)
        up = ch in (ord("k"), curses.KEY_UP)
        page_down = ch == 4  # ^D
        page_up = ch == 21  # ^U
        delta = (1 if down else -1 if up else
                 16 if page_down else -16 if page_up else 0)
        if delta:
            if instr_like and sec:
                self.instr.move(delta, sec.n_instructions, 10 ** 6)
                self.fields.cursor = 0
                self.fields.top = 0
            elif self.pane == PANE_STRUCTURE:
                self.structure.move(delta, len(self._structure_rows) or 1, 10 ** 6)
            elif self.pane == PANE_FIELDS:
                decoded = self.current_decoded()
                n = len(self.field_rows(decoded)) if decoded else 0
                self.fields.move(delta, n, 10 ** 6)
            return
        if ch in (10, 13, curses.KEY_ENTER, ord("e")):
            if instr_like:
                self.pane = PANE_FIELDS
            elif self.pane == PANE_STRUCTURE:
                self.activate_structure()
            else:
                self.action_edit_field()
            return
        if ch == ord("o"):
            self.action_opcode()
        elif ch == ord("v"):
            self.action_variant()
        elif ch == ord("n"):
            self.action_generate()

    def handle_hex_key(self, ch: int) -> None:
        n = len(self.model.data)
        cur = self.hexview.cursor
        moves = {ord("h"): -1, curses.KEY_LEFT: -1, ord("l"): 1,
                 curses.KEY_RIGHT: 1, ord("j"): 16, curses.KEY_DOWN: 16,
                 ord("k"): -16, curses.KEY_UP: -16,
                 curses.KEY_NPAGE: 16 * 16, curses.KEY_PPAGE: -16 * 16}
        if ch in moves:
            self.hexview.cursor = max(0, min(n - 1, cur + moves[ch]))
            return
        if ch in (10, 13, curses.KEY_ENTER, ord("e")):
            current = bytes(self.model.data[cur:cur + 2]).hex()
            text = self.prompt(f"bytes @0x{cur:X} (hex)", current)
            if text is None or not text.strip():
                return
            cleaned = text.strip().replace(" ", "").replace("_", "")
            try:
                new = bytes.fromhex(cleaned)
                if not new:
                    raise ValueError("empty")
                self.model.set_bytes(cur, new)
                self.invalidate()
                self.set_status(f"wrote {len(new)} bytes at 0x{cur:X}", C_OK)
            except (ValueError, EditError) as exc:
                self.set_status(f"bad hex: {exc}", C_ERR)

    def handle_elf_key(self, ch: int) -> None:
        n = len(ELF64_HEADER_FIELDS)
        if ch in (ord("j"), curses.KEY_DOWN):
            self.elf.move(1, n, 10 ** 6)
        elif ch in (ord("k"), curses.KEY_UP):
            self.elf.move(-1, n, 10 ** 6)
        elif ch == ord("]"):
            self.elf_image = min(self.elf_image + 1,
                                 max(0, len(self.model.images) - 1))
        elif ch == ord("["):
            self.elf_image = max(0, self.elf_image - 1)
        elif ch in (10, 13, curses.KEY_ENTER, ord("e")):
            self.action_edit_elf()

    def handle_isa_key(self, ch: int) -> None:
        if ch == ord("p"):
            self.action_stamp_isa()
            return
        self.handle_list_key(ch, self.isa, len(self.isa_rows()),
                             self.action_stamp_isa)

    def handle_disasm_key(self, ch: int) -> None:
        if ch == ord("r"):
            self._disasm_cache = None
            self.set_status("re-ran cuobjdump", C_ACCENT)
            return
        rows, _error = self.disasm_rows()
        self.handle_list_key(ch, self.disasm, len(rows),
                             self.action_disasm_jump)

    def handle_measure_key(self, ch: int) -> None:
        m = self.measure
        if ch in (10, 13, curses.KEY_ENTER):
            self.action_measure_run()
            return
        if ch == ord("b"):
            self.action_measure_build()
            return
        if ch == ord("d"):
            self.action_measure_build_db()
            return
        if ch == ord("x"):
            m.cancel()
            return
        if ch == ord("e"):
            self.action_measure_export()
            return
        if m.busy():
            return  # config is locked while a job runs
        if ch == ord("m"):
            m.mode = MODES[(MODES.index(m.mode) + 1) % len(MODES)]
        elif ch == ord("r"):
            m.source = "sweep" if m.source == "cursor" else "cursor"
        elif ch == ord("p"):
            m.pattern = PATTERNS[(PATTERNS.index(m.pattern) + 1) % len(PATTERNS)]
        elif ch == ord("["):
            m.trials = max(1, m.trials - 1)
        elif ch == ord("]"):
            m.trials = min(50, m.trials + 1)
        elif ch == ord("{"):
            m.slots = SLOT_CHOICES[(SLOT_CHOICES.index(m.slots) - 1)
                                   % len(SLOT_CHOICES)]
        elif ch == ord("}"):
            m.slots = SLOT_CHOICES[(SLOT_CHOICES.index(m.slots) + 1)
                                   % len(SLOT_CHOICES)]
        elif ch == ord("<"):
            m.sweep_limit = max(8, m.sweep_limit - 16)
        elif ch == ord(">"):
            m.sweep_limit = min(4096, m.sweep_limit + 16)
        elif ch in (ord("j"), curses.KEY_DOWN):
            self.results.move(1, len(self.measure.rows), 10 ** 6)
        elif ch in (ord("k"), curses.KEY_UP):
            self.results.move(-1, len(self.measure.rows), 10 ** 6)

    def handle_list_key(self, ch: int, state: ListState, n: int,
                        on_enter) -> None:
        if ch in (ord("j"), curses.KEY_DOWN):
            state.move(1, n, 10 ** 6)
        elif ch in (ord("k"), curses.KEY_UP):
            state.move(-1, n, 10 ** 6)
        elif ch == curses.KEY_NPAGE:
            state.move(16, n, 10 ** 6)
        elif ch == curses.KEY_PPAGE:
            state.move(-16, n, 10 ** 6)
        elif ch in (10, 13, curses.KEY_ENTER, ord("e")) and on_enter:
            on_enter()

    # -- main loop -----------------------------------------------------------

    def run(self) -> None:
        curses.curs_set(0)
        self.screen.keypad(True)
        while True:
            self.measure.pump()
            self.draw()
            # while a background job runs, poll so progress stays live;
            # otherwise block on input (no busy loop, cache stays warm)
            self.screen.timeout(150 if self.measure.busy() else -1)
            ch = self.screen.getch()
            if ch == -1:  # timeout tick: just redraw with fresh progress
                continue
            if not self.handle_key(ch):
                break


def apply_theme(name: str) -> bool:
    """(Re)initialize all color pairs for a theme. Returns False when the
    terminal cannot support theme switching (fewer than 256 colors)."""
    if not curses.has_colors():
        return False
    curses.start_color()
    curses.use_default_colors()
    if curses.COLORS < 256:
        gray, gold = curses.COLOR_BLUE, curses.COLOR_YELLOW
        curses.init_pair(C_BORDER, gray, -1)
        curses.init_pair(C_ACCENT, gold, -1)
        curses.init_pair(C_CHIP, curses.COLOR_BLACK, gold)
        curses.init_pair(C_MNEMONIC, curses.COLOR_CYAN, -1)
        curses.init_pair(C_MNEM_MOD, curses.COLOR_MAGENTA, -1)
        curses.init_pair(C_ERR, curses.COLOR_RED, -1)
        curses.init_pair(C_OK, curses.COLOR_GREEN, -1)
        curses.init_pair(C_TEXT, curses.COLOR_WHITE, -1)
        curses.init_pair(C_DIM, gray, -1)
        curses.init_pair(C_SEL, curses.COLOR_BLACK, gold)
        curses.init_pair(C_EDIT, gold, -1)
        for pair, color in zip(C_BAR, [curses.COLOR_YELLOW, curses.COLOR_CYAN,
                                       curses.COLOR_GREEN, curses.COLOR_RED,
                                       curses.COLOR_MAGENTA,
                                       curses.COLOR_BLUE]):
            curses.init_pair(pair, color, -1)
        return False
    theme = THEMES.get(name, THEMES["dark"])
    bg = theme["bg"]
    for pair, fg in theme["pairs"].items():
        curses.init_pair(pair, fg, bg)
    curses.init_pair(C_CHIP, *theme["chip"])
    curses.init_pair(C_SEL, *theme["sel"])
    for pair, color in zip(C_BAR, theme["bars"]):
        curses.init_pair(pair, color, bg)
    return True


def run(model: EditorModel, theme: str = "dark") -> None:
    """Launch the TUI (blocks until quit)."""

    def main(screen):
        apply_theme(theme)
        app = App(screen, model)
        app.theme = theme if theme in THEMES else "dark"
        app.apply_background()
        app.run()

    curses.wrapper(main)
