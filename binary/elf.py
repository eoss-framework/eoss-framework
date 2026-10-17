"""Pure-Python CUDA ELF parsing: locate .text sections inside CUDA binaries.

Handles both standalone cubins and host fatbinaries with embedded cubins
(nested ELF images are discovered by scanning for the ELF magic).

Canonical implementation (absorbed from VisualModifier/patch_sass.py).
"""

import struct
from typing import List, Tuple

ELF_MAGIC = b"\x7fELF"
EM_CUDA = 190
EI_CLASS_64 = 2
SHT_PROGBITS = 1
SHF_EXECINSTR = 0x4

#: (abs_file_offset, size, section_name)
TextSection = Tuple[int, int, str]


def read_cstring(data: bytes, offset: int) -> str:
    """Read a null-terminated string from data at offset."""
    end = offset
    while end < len(data) and data[end] != 0:
        end += 1
    return data[offset:end].decode("ascii", errors="replace")


def _parse_elf64_text_sections(data: bytes, elf_offset: int) -> List[TextSection]:
    """Parse a single ELF64 CUDA image at ``elf_offset`` within ``data``.

    Returns a list of (abs_offset, size, name) for executable .text sections.
    """
    base = elf_offset

    # Bounds check for ELF header (64 bytes minimum)
    if base + 0x40 > len(data):
        return []

    # Verify magic
    if data[base:base + 4] != ELF_MAGIC:
        return []

    # EI_CLASS must be 2 (64-bit)
    if data[base + 4] != EI_CLASS_64:
        return []

    # e_machine at offset 0x12 must be EM_CUDA
    e_machine = struct.unpack_from("<H", data, base + 0x12)[0]
    if e_machine != EM_CUDA:
        return []

    # Section header info
    e_shoff = struct.unpack_from("<Q", data, base + 0x28)[0]
    e_shentsize = struct.unpack_from("<H", data, base + 0x3A)[0]
    e_shnum = struct.unpack_from("<H", data, base + 0x3C)[0]
    e_shstrndx = struct.unpack_from("<H", data, base + 0x3E)[0]

    if e_shnum == 0 or e_shentsize == 0:
        return []

    sh_table = base + e_shoff

    # Load section-name string table
    if e_shstrndx >= e_shnum:
        return []
    shstr_hdr = sh_table + e_shstrndx * e_shentsize
    if shstr_hdr + 0x28 > len(data):
        return []
    shstr_off = struct.unpack_from("<Q", data, shstr_hdr + 0x18)[0]
    shstr_size = struct.unpack_from("<Q", data, shstr_hdr + 0x20)[0]
    strtab_abs = base + shstr_off

    if strtab_abs + shstr_size > len(data):
        return []

    results = []
    for i in range(e_shnum):
        hdr_off = sh_table + i * e_shentsize
        if hdr_off + 0x28 > len(data):
            break

        sh_name_idx = struct.unpack_from("<I", data, hdr_off + 0x00)[0]
        sh_type = struct.unpack_from("<I", data, hdr_off + 0x04)[0]
        sh_flags = struct.unpack_from("<Q", data, hdr_off + 0x08)[0]
        sh_offset = struct.unpack_from("<Q", data, hdr_off + 0x18)[0]
        sh_size = struct.unpack_from("<Q", data, hdr_off + 0x20)[0]

        name = read_cstring(data, strtab_abs + sh_name_idx)

        if name.startswith(".text") and sh_type == SHT_PROGBITS and (sh_flags & SHF_EXECINSTR):
            results.append((base + sh_offset, sh_size, name))

    return results


def find_cuda_text_sections(data: bytes) -> List[TextSection]:
    """Scan an entire file for nested CUDA ELFs and return all .text sections.

    Returns a list of (abs_file_offset, size, section_name).
    """
    sections = []
    search_start = 0

    while True:
        idx = data.find(ELF_MAGIC, search_start)
        if idx == -1:
            break
        sections.extend(_parse_elf64_text_sections(data, idx))
        search_start = idx + 1

    return sections


def iter_instructions(data: bytes, section: TextSection):
    """Yield (abs_file_offset, 16-byte instruction) pairs for a .text section."""
    sec_off, sec_size, _name = section
    for i in range(0, sec_size, 16):
        abs_pos = sec_off + i
        if abs_pos + 16 > len(data):
            break
        yield abs_pos, bytes(data[abs_pos:abs_pos + 16])
