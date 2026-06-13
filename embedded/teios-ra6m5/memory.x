/* Renesas RA6M5 (R7FA6M5BH: 2 MB flash, 512 KB SRAM).
 *
 * BENCH-PENDING: bare-metal layout (app at 0x0). The Portenta C33 ships
 * with Arduino's bootloader in low flash; flashing through it places the
 * app above the bootloader (set FLASH ORIGIN to that offset). The teiOS
 * image itself is layout-agnostic.
 */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 2048K
  RAM   : ORIGIN = 0x20000000, LENGTH = 512K
}
