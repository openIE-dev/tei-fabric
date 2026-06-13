MEMORY {
    /*
     * The RP2040 boot ROM loads exactly 256 bytes from the start of
     * flash — the boot2 second-stage bootloader that configures the
     * external QSPI flash chip for XIP. embassy-rp emits the blob
     * (selected by the boot2-* feature) into the `.boot2` section,
     * which embassy-rp's `link-rp.x` places at ORIGIN(BOOT2).
     * (On the RP2350 this whole mechanism is replaced by IMAGE_DEF.)
     */
    BOOT2 : ORIGIN = 0x10000000, LENGTH = 0x100
    /*
     * The Feather RP2040 carries 8 MiB external QSPI flash (GD25Q64C),
     * the Pico 1 2 MiB (W25Q16JV); 2 MiB is the safe shared default
     * and this image is tiny either way.
     */
    FLASH : ORIGIN = 0x10000100, LENGTH = 2048K - 0x100
    /*
     * SRAM0-SRAM3, striped mapping (good default for load balance).
     */
    RAM : ORIGIN = 0x20000000, LENGTH = 256K
    /*
     * Direct-mapped banks 4/5 — reserved for jobs that want predictable
     * access times (e.g. per-core stacks). Unused by this image.
     */
    SRAM4 : ORIGIN = 0x20040000, LENGTH = 4K
    SRAM5 : ORIGIN = 0x20041000, LENGTH = 4K
}

SECTIONS {
    /* ### Picotool 'Binary Info' header
     *
     * Goes after .vector_table, to keep it in the first 512 bytes of
     * flash where picotool can find it. (RP2040 layout — the RP2350
     * uses .start_block/.end_block instead.)
     */
    .boot_info : ALIGN(4)
    {
        KEEP(*(.boot_info));
    } > FLASH

} INSERT AFTER .vector_table;

/* move .text to start /after/ the boot info */
_stext = ADDR(.boot_info) + SIZEOF(.boot_info);

SECTIONS {
    /* ### Picotool 'Binary Info' Entries
     *
     * Picotool looks through this block (as we have pointers to it in
     * our header) to find interesting information.
     */
    .bi_entries : ALIGN(4)
    {
        __bi_entries_start = .;
        KEEP(*(.bi_entries));
        . = ALIGN(4);
        __bi_entries_end = .;
    } > FLASH
} INSERT AFTER .text;
