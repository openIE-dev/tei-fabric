/* Arduino Portenta H7 (STM32H747XI, M7 core). teiOS lives ABOVE Arduino's
   MCUboot-class bootloader, which occupies the first 128 KiB of bank 1 and
   jumps to the application at 0x08040000 (the ArduinoCore-mbed
   PORTENTA_H7_M7 MBED_APP_START). We claim from there to the end of the
   2 MiB flash. RAM: AXI SRAM (D1, 0x24000000, 512 KiB). The reset handler
   sets VTOR to this base so the vector table is found. */
MEMORY
{
  FLASH : ORIGIN = 0x08040000, LENGTH = 1792K
  RAM   : ORIGIN = 0x24000000, LENGTH = 512K
}
