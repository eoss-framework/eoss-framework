"""Full ELF image parsing for the TUI: section tables, symbols, ELF header
fields, ``.nv.info`` records and ``.note`` sections.

Port of the corresponding JavaScript in ``VisualModifier/modifier.html``
(scanElfHeaders / parseElfImage / parseNvInfoRecords / notes decoding),
restricted to the little-endian ELF64 CUDA images the framework targets.
"""

import struct
from dataclasses import dataclass, field
from typing import List, Optional

ELF_MAGIC = b"\x7fELF"

ELF_TYPE_NAMES = {0: "ET_NONE", 1: "ET_REL", 2: "ET_EXEC", 3: "ET_DYN", 4: "ET_CORE"}
ELF_MACHINE_NAMES = {3: "EM_386", 40: "EM_ARM", 62: "EM_X86_64", 183: "EM_AARCH64",
                     190: "EM_CUDA", 243: "EM_RISCV"}

CUDA_SM_NAMES = {
    0x34: "SM52", 0x3c: "SM60", 0x3d: "SM61", 0x46: "SM70", 0x48: "SM72",
    0x4b: "SM75", 0x50: "SM80", 0x56: "SM86", 0x57: "SM87", 0x59: "SM89",
    0x5a: "SM90/SM90A", 0x64: "SM100", 0x78: "SM120", 0x79: "SM121",
}

CUDA_FLAG_BITS = [
    (0x100, "EF_CUDA_TEXMODE_UNIFIED"),
    (0x200, "EF_CUDA_TEXMODE_INDEPENDANT"),
    (0x400, "EF_CUDA_64BIT_ADDRESS"),
    (0x800, "EF_CUDA_ACCELERATORS"),
    (0x1000, "EF_CUDA_SW_FLAG_V2"),
]

NVINFO_FORMAT_NAMES = {0x01: "EIFMT_FLAG", 0x03: "EIFMT_HVAL", 0x04: "EIFMT_SVAL"}

NVINFO_ATTR_NAMES = {
    0x04: "EIATTR_CTAIDZ_USED",
    0x05: "EIATTR_MAX_THREADS",
    0x0a: "EIATTR_PARAM_CBANK",
    0x11: "EIATTR_FRAME_SIZE",
    0x12: "EIATTR_MIN_STACK_SIZE",
    0x17: "EIATTR_KPARAM_INFO",
    0x19: "EIATTR_CBANK_PARAM_SIZE",
    0x1b: "EIATTR_MAXREG_COUNT",
    0x1c: "EIATTR_EXIT_INSTR_OFFSETS",
    0x1e: "EIATTR_CRS_STACK_SIZE",
    0x28: "EIATTR_COOP_GROUP_INSTR_OFFSETS",
    0x29: "EIATTR_COOP_GROUP_MASK_REGIDS",
    0x2f: "EIATTR_REGCOUNT",
    0x31: "EIATTR_INT_WARP_WIDE_INSTR_OFFSETS",
    0x34: "EIATTR_INDIRECT_BRANCH_TARGETS",
    0x35: "EIATTR_SW2861232_WAR",
    0x37: "EIATTR_CUDA_API_VERSION",
    0x44: "EIATTR_UNUSED_LOAD_BYTE_OFFSET",
}

#: attrs whose SVAL payload begins with a 4-byte symbol table index
NVINFO_SYMBOL_ATTRS = {0x0a, 0x11, 0x12, 0x2f}

#: (key, label, offset, size) for the ELF64 header table (little-endian)
ELF64_HEADER_FIELDS = [
    ("e_type", "Type", 0x10, 2),
    ("e_machine", "Machine", 0x12, 2),
    ("e_version", "Version", 0x14, 4),
    ("e_entry", "Entry Point", 0x18, 8),
    ("e_phoff", "Program Header Offset", 0x20, 8),
    ("e_shoff", "Section Header Offset", 0x28, 8),
    ("e_flags", "Flags", 0x30, 4),
    ("e_ehsize", "Header Size", 0x34, 2),
    ("e_phentsize", "Program Header Entry Size", 0x36, 2),
    ("e_phnum", "Program Header Count", 0x38, 2),
    ("e_shentsize", "Section Header Entry Size", 0x3A, 2),
    ("e_shnum", "Section Header Count", 0x3C, 2),
    ("e_shstrndx", "String Table Index", 0x3E, 2),
]


def read_cstring(data: bytes, offset: int) -> str:
    end = data.find(b"\x00", offset)
    if end == -1:
        end = len(data)
    return data[offset:end].decode("ascii", errors="replace")


@dataclass
class Section:
    index: int
    name: str
    sh_type: int
    flags: int
    offset: int  # relative to image base
    size: int
    link: int
    info: int
    entsize: int
    base: int  # image base (absolute file offset of the ELF header)

    @property
    def abs_offset(self) -> int:
        return self.base + self.offset

    @property
    def executable(self) -> bool:
        return bool(self.flags & 0x4)


@dataclass
class Symbol:
    index: int
    name: str
    info: int
    shndx: int
    value: int
    size: int


@dataclass
class NvInfoRecord:
    section: str
    abs_offset: int  # absolute file offset of the record's format byte
    format: int
    attr: int
    length: int
    data_offset: int  # absolute offset of the value payload (after any symbol idx)
    symbol: Optional[str] = None
    summary: str = ""
    #: (label, abs_offset, size) of an editable scalar value, if any
    scalar: Optional[tuple] = None

    @property
    def attr_name(self) -> str:
        return NVINFO_ATTR_NAMES.get(self.attr, f"EIATTR_0x{self.attr:02X}")

    @property
    def format_name(self) -> str:
        return NVINFO_FORMAT_NAMES.get(self.format, f"EIFMT_0x{self.format:02X}")


@dataclass
class NoteEntry:
    section: str
    abs_offset: int
    n_type: int
    name: str
    desc: bytes


@dataclass
class ElfImage:
    base: int
    machine: int
    flags: int
    header_values: dict = field(default_factory=dict)
    sections: List[Section] = field(default_factory=list)
    symbols: List[Symbol] = field(default_factory=list)

    @property
    def sm_name(self) -> str:
        return CUDA_SM_NAMES.get(self.flags & 0xFF, f"SM?(0x{self.flags & 0xFF:02X})")

    @property
    def machine_name(self) -> str:
        return ELF_MACHINE_NAMES.get(self.machine, f"0x{self.machine:04X}")

    @property
    def is_cuda(self) -> bool:
        return self.machine == 190

    def section_named(self, name: str) -> Optional[Section]:
        for sec in self.sections:
            if sec.name == name:
                return sec
        return None


def parse_image(data: bytes, base: int) -> Optional[ElfImage]:
    """Parse one little-endian ELF64 image at ``base``; None if invalid."""
    if base + 0x40 > len(data) or data[base:base + 4] != ELF_MAGIC:
        return None
    if data[base + 4] != 2 or data[base + 5] != 1:  # ELF64, little-endian
        return None

    header_values = {}
    for key, _label, off, size in ELF64_HEADER_FIELDS:
        fmt = {2: "<H", 4: "<I", 8: "<Q"}[size]
        header_values[key] = struct.unpack_from(fmt, data, base + off)[0]

    image = ElfImage(base=base, machine=header_values["e_machine"],
                     flags=header_values["e_flags"],
                     header_values=header_values)

    shoff = header_values["e_shoff"]
    shentsize = header_values["e_shentsize"]
    shnum = header_values["e_shnum"]
    shstrndx = header_values["e_shstrndx"]
    if shnum == 0 or shentsize < 0x40 or shstrndx >= shnum:
        return image
    table = base + shoff
    if table + shnum * shentsize > len(data):
        return image

    strtab_hdr = table + shstrndx * shentsize
    strtab_off = base + struct.unpack_from("<Q", data, strtab_hdr + 0x18)[0]

    for i in range(shnum):
        hdr = table + i * shentsize
        name_idx, sh_type = struct.unpack_from("<II", data, hdr)
        flags, = struct.unpack_from("<Q", data, hdr + 0x08)
        offset, size = struct.unpack_from("<QQ", data, hdr + 0x18)
        link, info = struct.unpack_from("<II", data, hdr + 0x28)
        entsize, = struct.unpack_from("<Q", data, hdr + 0x38)
        image.sections.append(Section(
            index=i, name=read_cstring(data, strtab_off + name_idx),
            sh_type=sh_type, flags=flags, offset=offset, size=size,
            link=link, info=info, entsize=entsize, base=base))

    # Symbols (first SHT_SYMTAB section)
    for sec in image.sections:
        if sec.sh_type != 2:  # SHT_SYMTAB
            continue
        strsec = image.sections[sec.link] if sec.link < len(image.sections) else None
        entsize = sec.entsize or 24
        count = sec.size // entsize if entsize else 0
        for idx in range(count):
            off = sec.abs_offset + idx * entsize
            if off + 24 > len(data):
                break
            st_name, = struct.unpack_from("<I", data, off)
            st_info = data[off + 4]
            st_shndx, = struct.unpack_from("<H", data, off + 6)
            st_value, st_size = struct.unpack_from("<QQ", data, off + 8)
            name = ""
            if strsec is not None and st_name < strsec.size:
                name = read_cstring(data, strsec.abs_offset + st_name)
            image.symbols.append(Symbol(idx, name, st_info, st_shndx,
                                        st_value, st_size))
        break
    return image


def scan_images(data: bytes) -> List[ElfImage]:
    """All little-endian ELF64 images in the file (host binary + cubins)."""
    images = []
    start = 0
    while True:
        idx = data.find(ELF_MAGIC, start)
        if idx == -1:
            break
        image = parse_image(data, idx)
        if image is not None:
            images.append(image)
        start = idx + 1
    return images


def parse_nvinfo_records(data: bytes, image: ElfImage) -> List[NvInfoRecord]:
    records = []
    for sec in image.sections:
        if not sec.name.startswith(".nv.info"):
            continue
        pos = sec.abs_offset
        end = min(sec.abs_offset + sec.size, len(data))
        while pos + 4 <= end:
            fmt, attr = data[pos], data[pos + 1]
            value16, = struct.unpack_from("<H", data, pos + 2)
            if fmt == 0x04:  # EIFMT_SVAL: u16 payload length follows
                length = value16
                payload = pos + 4
            else:  # FLAG / HVAL: the u16 IS the value
                length = 0
                payload = pos + 2
            record = NvInfoRecord(section=sec.name, abs_offset=pos, format=fmt,
                                  attr=attr, length=length, data_offset=payload)
            _decode_record(data, image, record, value16)
            records.append(record)
            pos += 4 + length
            pos += (-pos) % 4  # records are 4-byte aligned
            if fmt not in NVINFO_FORMAT_NAMES and length == 0:
                break  # unknown format: avoid spinning on garbage
    return records


def _decode_record(data: bytes, image: ElfImage, rec: NvInfoRecord,
                   value16: int) -> None:
    parts = []
    payload = rec.data_offset
    remaining = rec.length

    if rec.format == 0x04 and rec.attr in NVINFO_SYMBOL_ATTRS and remaining >= 4:
        sym_idx, = struct.unpack_from("<I", data, payload)
        if sym_idx < len(image.symbols):
            rec.symbol = image.symbols[sym_idx].name or f"#{sym_idx}"
            parts.append(f"sym {rec.symbol}")
        payload += 4
        remaining -= 4
        rec.data_offset = payload

    if rec.format == 0x01:
        parts.append("flag present")
    elif rec.format == 0x03:
        parts.append(f"value {value16} (0x{value16:X})")
        rec.scalar = ("value", rec.abs_offset + 2, 2)
    elif rec.format == 0x04:
        if rec.attr == 0x2f and remaining >= 4:
            v, = struct.unpack_from("<I", data, payload)
            parts.append(f"register count {v}")
            rec.scalar = ("register count", payload, 4)
        elif rec.attr in (0x11, 0x12, 0x1e, 0x37) and remaining >= 4:
            v, = struct.unpack_from("<I", data, payload)
            parts.append(f"value {v} (0x{v:X})")
            rec.scalar = ("value", payload, 4)
        elif rec.attr == 0x0a and remaining >= 8:
            a, b = struct.unpack_from("<II", data, payload)
            parts.append(f"cbank offset 0x{b >> 16:X} size 0x{b & 0xFFFF:X}")
        elif rec.attr in (0x1c, 0x28, 0x31, 0x34) and remaining >= 4:
            n = remaining // 4
            offs = struct.unpack_from(f"<{n}I", data, payload)
            shown = ", ".join(f"0x{o:X}" for o in offs[:6])
            parts.append(f"{n} offsets: {shown}" + (" …" if n > 6 else ""))
        else:
            parts.append(f"{remaining} bytes: "
                         + data[payload:payload + min(remaining, 12)].hex(" ")
                         + (" …" if remaining > 12 else ""))
    rec.summary = " · ".join(parts)


def parse_notes(data: bytes, image: ElfImage) -> List[NoteEntry]:
    notes = []
    for sec in image.sections:
        if sec.sh_type != 7 and not sec.name.startswith(".note"):  # SHT_NOTE
            continue
        pos = sec.abs_offset
        end = min(sec.abs_offset + sec.size, len(data))
        while pos + 12 <= end:
            namesz, descsz, n_type = struct.unpack_from("<III", data, pos)
            name_off = pos + 12
            name = read_cstring(data, name_off) if namesz else ""
            desc_off = name_off + namesz + ((-namesz) % 4)
            if desc_off + descsz > end:
                break
            notes.append(NoteEntry(section=sec.name, abs_offset=pos,
                                   n_type=n_type, name=name,
                                   desc=bytes(data[desc_off:desc_off + descsz])))
            pos = desc_off + descsz + ((-descsz) % 4)
    return notes
