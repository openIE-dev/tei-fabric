#!/usr/bin/env python3
"""Minimal ELF -> UF2 converter for RP2040/RP2350, stdlib only.

Fallback for hosts without picotool. Reads PT_LOAD segments with physical
addresses in flash (0x10000000..0x20000000), concatenates them, and emits
standard 256-byte-payload UF2 blocks tagged with the chip's UF2 family.

Usage: python3 scripts/elf2uf2.py <input.elf> <output.uf2> [family]

family: rp2040 (default here) | rp2350-arm-s
"""

import struct
import sys

UF2_MAGIC_START0 = 0x0A324655  # "UF2\n"
UF2_MAGIC_START1 = 0x9E5D5157
UF2_MAGIC_END = 0x0AB16F30
UF2_FLAG_FAMILY_ID = 0x00002000
FAMILIES = {
    "rp2040": 0xE48BFF56,
    "rp2350-arm-s": 0xE48BFF59,
}
DEFAULT_FAMILY = "rp2040"
PAYLOAD = 256

FLASH_BASE = 0x10000000
FLASH_END = 0x20000000


def flash_segments(elf: bytes):
    if elf[:4] != b"\x7fELF" or elf[4] != 1:  # ELFCLASS32
        sys.exit("not a 32-bit ELF")
    (phoff,) = struct.unpack_from("<I", elf, 0x1C)
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x2A)
    segs = []
    for i in range(phnum):
        ptype, off, _vaddr, paddr, filesz, _memsz, _flags, _align = struct.unpack_from(
            "<IIIIIIII", elf, phoff + i * phentsize
        )
        if ptype == 1 and filesz > 0 and FLASH_BASE <= paddr < FLASH_END:
            segs.append((paddr, elf[off : off + filesz]))
    segs.sort()
    return segs


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    family = sys.argv[3] if len(sys.argv) == 4 else DEFAULT_FAMILY
    if family not in FAMILIES:
        sys.exit(f"unknown family {family!r}; one of: {', '.join(FAMILIES)}")
    family_id = FAMILIES[family]

    elf = open(sys.argv[1], "rb").read()
    segs = flash_segments(elf)
    if not segs:
        sys.exit("no flash PT_LOAD segments found")

    # Build (addr, payload) chunks, merging contiguous segments.
    base, image = segs[0][0], bytearray()
    for paddr, data in segs:
        if paddr != base + len(image):
            # Non-contiguous: pad with 0xFF (erased flash) — fine for
            # the small gaps a linker emits, conservative otherwise.
            gap = paddr - (base + len(image))
            if gap < 0 or gap > 1 << 20:
                sys.exit(f"segment at 0x{paddr:08x} not mergeable")
            image += b"\xff" * gap
        image += data

    nblocks = (len(image) + PAYLOAD - 1) // PAYLOAD
    out = bytearray()
    for bn in range(nblocks):
        chunk = image[bn * PAYLOAD : (bn + 1) * PAYLOAD]
        block = struct.pack(
            "<IIIIIIII",
            UF2_MAGIC_START0,
            UF2_MAGIC_START1,
            UF2_FLAG_FAMILY_ID,
            base + bn * PAYLOAD,
            PAYLOAD,
            bn,
            nblocks,
            family_id,
        )
        block += chunk.ljust(476, b"\x00")
        block += struct.pack("<I", UF2_MAGIC_END)
        assert len(block) == 512
        out += block

    open(sys.argv[2], "wb").write(out)
    print(
        f"{sys.argv[2]}: {nblocks} blocks, {len(image)} bytes "
        f"@ 0x{base:08x}, family {family}"
    )


if __name__ == "__main__":
    main()
