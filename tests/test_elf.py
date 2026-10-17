"""Unit tests for eoss.binary.elf using a synthetic CUDA ELF64 fixture."""

import struct

from eoss.binary import elf


def build_synthetic_cubin(text_bytes: bytes = b"\x00" * 64) -> bytes:
    """Build a minimal ELF64 with e_machine=EM_CUDA and one .text section."""
    shstrtab = b"\x00.text\x00.shstrtab\x00"
    name_text = 1
    name_shstr = 7

    ehdr_size = 64
    text_off = ehdr_size
    shstr_off = text_off + len(text_bytes)
    shoff = shstr_off + len(shstrtab)
    if shoff % 8:
        shoff += 8 - (shoff % 8)

    # ELF header
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2          # EI_CLASS = 64-bit
    ehdr[5] = 1          # EI_DATA = LE
    ehdr[6] = 1          # EI_VERSION
    struct.pack_into("<H", ehdr, 0x12, 190)   # e_machine = EM_CUDA
    struct.pack_into("<Q", ehdr, 0x28, shoff)  # e_shoff
    struct.pack_into("<H", ehdr, 0x3A, 64)     # e_shentsize
    struct.pack_into("<H", ehdr, 0x3C, 3)      # e_shnum
    struct.pack_into("<H", ehdr, 0x3E, 2)      # e_shstrndx

    def section(name, sh_type, flags, offset, size):
        hdr = bytearray(64)
        struct.pack_into("<I", hdr, 0x00, name)
        struct.pack_into("<I", hdr, 0x04, sh_type)
        struct.pack_into("<Q", hdr, 0x08, flags)
        struct.pack_into("<Q", hdr, 0x18, offset)
        struct.pack_into("<Q", hdr, 0x20, size)
        return hdr

    null_sh = bytearray(64)
    text_sh = section(name_text, 1, 0x4, text_off, len(text_bytes))   # PROGBITS+EXEC
    shstr_sh = section(name_shstr, 3, 0, shstr_off, len(shstrtab))   # STRTAB

    data = bytearray()
    data += ehdr
    data += text_bytes
    data += shstrtab
    data += b"\x00" * (shoff - len(data))
    data += null_sh + text_sh + shstr_sh
    return bytes(data)


class TestElf:
    def test_finds_text_section(self):
        data = build_synthetic_cubin(b"\xAB" * 64)
        sections = elf.find_cuda_text_sections(data)
        assert len(sections) == 1
        off, size, name = sections[0]
        assert name == ".text"
        assert size == 64
        assert data[off:off + 4] == b"\xAB" * 4

    def test_ignores_non_cuda(self):
        data = bytearray(build_synthetic_cubin())
        struct.pack_into("<H", data, 0x12, 62)  # e_machine = x86-64
        assert elf.find_cuda_text_sections(bytes(data)) == []

    def test_ignores_32bit(self):
        data = bytearray(build_synthetic_cubin())
        data[4] = 1  # EI_CLASS = 32-bit
        assert elf.find_cuda_text_sections(bytes(data)) == []

    def test_iter_instructions(self):
        text = bytes(range(16)) * 4
        data = build_synthetic_cubin(text)
        sections = elf.find_cuda_text_sections(data)
        instrs = list(elf.iter_instructions(data, sections[0]))
        assert len(instrs) == 4
        assert instrs[0][1] == bytes(range(16))
        assert instrs[1][0] - instrs[0][0] == 16

    def test_garbage(self):
        assert elf.find_cuda_text_sections(b"\x7fELFgarbage") == []
        assert elf.find_cuda_text_sections(b"") == []
