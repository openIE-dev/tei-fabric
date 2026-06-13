/* nRF52832 (512 KB flash, 64 KB RAM).
 *
 * BENCH-PENDING: this is the bare-metal layout (app at 0x0, no
 * SoftDevice). The Arduino Nicla bootloader occupies the low flash
 * sectors and a real Nicla flash places the app ABOVE it (offset set by
 * the installed bootloader); set FLASH ORIGIN accordingly when flashing
 * through that bootloader. The teiOS image itself is layout-agnostic.
 */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 512K
  RAM   : ORIGIN = 0x20000000, LENGTH = 64K
}
