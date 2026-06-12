# TEI Embedded — flashable target matrix

**Status**: exploration draft (companion to EMBEDDED-ROADMAP.md) · 2026-06-11
**Caveat**: assembled from catalog knowledge current to early 2026; board
revisions and tool versions need a verification pass before E0 pins them.

## 0. What "our Yocto" means here

A recipe-driven image builder — call it the **TEI forge** for now — whose
output is *not Linux* but a reproducible, flashable TEI runtime artifact
per board: the `no_std` core + board support + calibrated (or default)
energy tables, baked into whatever the board's turnkey flash path expects.
Like BitBake, a recipe names a MACHINE and produces an image; unlike
BitBake, the image is a UF2, a .bin, a .hex, an FPGA bitstream, or a
bare-metal SD card image. **The artifact type IS the UX**, so targets are
tiered by flash path, best first.

| Artifact | Flash UX | Tooling the forge invokes |
|---|---|---|
| `.uf2` | drag-drop, zero install | none (mass-storage bootloader) |
| `.bin/.hex` | one CLI line | esptool/espflash, probe-rs, dfu-util, teensy_loader_cli, nrfutil |
| `.bit/.fs` | one CLI line | openFPGALoader (vendor-free) |
| `.img` | flash SD card | dd / Raspberry Pi Imager |

---

## Tier A — UF2 drag-drop (the Arduino-grade turnkey path)

| Board | Silicon | On-die substrates for dispatch | Notes |
|---|---|---|---|
| **Raspberry Pi Pico 2 / 2 W** | RP2350 (2×M33 **or** 2×Hazard3 RISC-V, switchable) | 3×PIO (12 SMs), DMA w/ sniffer-CRC, HSTX, dual-arch cores | **E1 flagship.** Core-vs-core vs PIO dispatch on one $5 board |
| Raspberry Pi Pico / W / WH | RP2040 (2×M0+) | 2×PIO (8 SMs), DMA | the install base |
| Adafruit Feather RP2040 (+DVI/CAN/RFM variants) | RP2040 | PIO, DMA | Feather form factor, huge tutorial reach |
| Adafruit Feather nRF52840 / Sense | nRF52840 (M4F) | PPI hardware event routing, radio, on-die USB | UF2 bootloader; PPI = "CPU-asleep" substrate |
| Adafruit Feather M4 / ItsyBitsy M4 | SAMD51 (M4F) | DMAC, EVSYS event system, CCL lookup logic | EVSYS = free event routing |
| Adafruit Feather ESP32-S2/S3 (TinyUF2) | ESP32-S2/S3 | ULP-RISC-V/FSM coprocessor, AES/SHA accel, vector ext (S3) | UF2 via TinyUF2; native path is esptool |
| Seeed XIAO RP2040 / RP2350 / nRF52840 (Sense) / SAMD21 | as named | as families above | thumbnail-size, classroom favorite |
| SparkFun Thing Plus / Pro Micro RP2040·RP2350 | RP2040/RP2350 | PIO, DMA | Qwiic ecosystem |
| Pimoroni Pico-family boards | RP2040/RP2350 | PIO, DMA | |
| Arduino Nano RP2040 Connect | RP2040 + NINA Wi-Fi | PIO, DMA + radio | Arduino-branded RP2 |
| Fomu | **iCE40UP5K FPGA** | the entire fabric is the substrate | FPGA that enumerates as USB + DFU/UF2-class flow — FPGA with Feather-grade UX |

## Tier B — one-line CLI flash

### ESP family (esptool.py / espflash — Rust-native flasher)
| Board | Silicon | Substrates | Notes |
|---|---|---|---|
| **ESP32-C6-DevKitC** | C6 (HP RISC-V 160 MHz + **LP RISC-V core**) | LP core, DMA, AES/SHA, 802.15.4+WiFi6 | **E-phase pick**: HP-vs-LP dispatch + radio upload of calibration reports |
| ESP32-P4-Function-EV | P4 (2×HP 400 MHz + LP core, no radio) | LP core, 2D-DMA, PPA, crypto | the compute-heavy ESP |
| ESP32-S3-DevKitC | S3 (2×LX7) | ULP-RISC-V, vector instructions, crypto | TinyML community |
| ESP32-C3 / H2 DevKits | single RISC-V | crypto, DMA | $2-4 class |
| classic ESP32-DevKitC | 2×LX6 | ULP-FSM, crypto | the install base |

### Nordic / STM32 / others (probe-rs — Rust-native, one tool for all)
| Board | Silicon | Substrates | Notes |
|---|---|---|---|
| nRF52840-DK / dongle | M4F | **PPI**, radio, on-die USB | nrfutil DFU on dongle; PPK2 attaches for T1 calibration |
| nRF5340-DK | M33 app + M33 net | dual-core dispatch, DPPI | |
| **nRF54L15-DK** | M33 + **RISC-V VPR coprocessors** | VPR "FLPR/PPR" cores, DPPI | the most TEI-shaped MCU shipping: dispatchable RISC-V helpers |
| STM32 Nucleo (any) | F4/L4/H7/**U5**/H5… | DMA, **FMAC** (filter-math accel), **CORDIC**, U5's LPBAM autonomous chains | on-board ST-LINK; probe-rs flashes; U5 LPBAM = work-while-asleep |
| **STM32N6 Nucleo/DK** | N6 (M55 + **Ethos-U55 NPU**) | NPU, Helium SIMD, DMA | the NPU column on an ST board |
| NXP FRDM-MCXN947 | 2×M33 + **Ethos-U55** + DSP | NPU, PowerQuad DSP, eIQ | NPU column, NXP channel |
| Teensy 4.0 / 4.1 | i.MX RT1062 (M7 @600 MHz) | 2×FlexIO, DMA, CM7 dual-issue | teensy_loader_cli; the performance MCU |
| Arduino UNO R4 / classic UNO R3 | RA4M1 / ATmega328P | DAC/CTSU / —— | reach play only; R3 via avrdude |
| TI MSP430FR LaunchPads | MSP430FR (+ **LEA** accel) | LEA vector math, FRAM | **EnergyTrace = the T0 measurement reference** |
| TI MSPM0 LaunchPads | M0+ | analog blocks | $0.5-class parts |
| **SiLabs xG24 Dev Kit** | EFR32MG24 (M33) | AI/ML accel (MVP), radio | **on-board AEM = T0 energy telemetry without bench gear** |
| Ambiq Apollo4/5 EVB | M4F/M55 subthreshold | low-power GPU (4), Helium (5) | the µW-class floor; J-Link |
| Alif Ensemble DevKit | M55×2 + **Ethos-U55×2** | dual NPU, dual core | most aggressive NPU-class part |
| WCH CH32V003 dev board | RV32EC @48 MHz | —— | $0.10 CPU; minichlink/wchisp; the "TEI on ten cents" stunt |
| Sipeed Longan Nano | GD32VF103 (RV32IMAC) | DMA | dfu-util; cheap RISC-V |

## Tier C — FPGA via open toolchain (openFPGALoader + Yosys/nextpnr)

The FPGA targets are special: the forge doesn't just flash firmware, it
ships **soft substrates** — LiteX/VexRiscv or NEORV32 SoCs with TEI event
counters synthesized into the fabric (a ledger the hardware itself
maintains). This is the rehearsal for purpose-built TEI silicon.

| Board | FPGA | Open flow | Notes |
|---|---|---|---|
| **iCEBreaker** | iCE40UP5K | icestorm (100% open) | the open-FPGA teaching board |
| iCEstick / UPduino / Alchitry Cu | iCE40 | icestorm | $20-50 |
| **ULX3S** | ECP5-85F | prjtrellis | the open-flow flagship; RISC-V SoC capable |
| **OrangeCrab** | ECP5-25F | prjtrellis, **DFU** | **ECP5 in Feather form factor** — FPGA-to-Feather literally |
| Colorlight i5/i9 | ECP5 | prjtrellis | $15-30 repurposed LED controllers |
| Butterstick | ECP5-85F | prjtrellis | |
| **Tang Nano 9K / 20K** | Gowin GW1NR/GW2A | **Apicula** | $15-30; huge hobby momentum |
| Fomu | iCE40UP5K | icestorm | also in Tier A for UX |
| QuickLogic Qomu / SparkFun QuickLogic Thing+ | EOS S3 (M4 + eFPGA) | **QORC — vendor-open incl. FPGA** | MCU+FPGA hybrid, fully open |
| Arty A7 / S7, Basys3, Cmod A7 | Xilinx 7-series | openXC7 (maturing) or Vivado (scriptable); openFPGALoader flashes | gateway to the university market |
| PYNQ-Z2 / Zybo (bare-metal PS) | Zynq-7000 | Vivado; bare-metal A9 + fabric | no-Linux Zynq is viable |
| DE10-Nano | Cyclone V | Quartus (scriptable) | MiSTer community = flashing culture exists |

Soft-core menu for these: VexRiscv/VexiiRiscv (LiteX), NEORV32 (VHDL,
exceptionally documented), picorv32, SERV (bit-serial — the *lowest-joule
soft core*, a perfect TEI demonstration in itself).

## Tier D — Pi-class, bare metal (no Linux, SD-image artifact)

The GPU-boot trick makes every Raspberry Pi a "drag an image, boot our
runtime" machine — the firmware loads `kernel*.img` from FAT; that file is
simply our flat binary. Turnkey UX identical to Raspberry Pi OS, contents
100% TEI.

| Board | Silicon | Bare-metal state | Notes |
|---|---|---|---|
| **Raspberry Pi Zero 2 W** | BCM2710 (4×A53) | mature (circle, rust crates) | $15; the image-flash flagship |
| Raspberry Pi 4 / CM4 | BCM2711 (4×A72) | mature | |
| Raspberry Pi 5 | BCM2712 + **RP1** | workable; RP1 southbridge adds I/O complexity | document as "later" |
| Raspberry Pi 3 | BCM2837 | mature | the docs/tutorials corpus |
| BeagleBone Black / BeaglePlay | AM335x / AM62 | bare-metal possible; **PRU-ICSS** (2×200 MHz deterministic RT cores) | PRUs are a first-class substrate — the original "CPU-asleep worker" |
| i.MX8M Plus EVK (M7 core) | A53×4 + M7 + NPU | M7 bare metal via RPMsg-less boot | asymmetric-core dispatch story |

## Priority picks (matches EMBEDDED-ROADMAP phasing)

1. **Pico 2 (RP2350)** — Tier A UX + richest dispatch story (M33 vs
   Hazard3 vs PIO) + Rust/Embassy first-class. E1.
2. **ESP32-C6** — HP-vs-LP dispatch + Wi-Fi calibration upload to the
   fabric. E2 alongside Arduino/MicroPython bindings.
3. **Raspberry Pi Zero 2 W** — the SD-image turnkey artifact; `teid`-
   without-Linux proof that "our Yocto" is real. E3.
4. **iCEBreaker + OrangeCrab** — soft-substrate bitstreams with
   fabric-maintained ledgers. E3/E4.
5. **xG24 + MSP430FR** — the two T0 energy-measurement references that
   anchor everyone else's calibration tables. Alongside E1 bench kit.
6. **STM32N6 or MCX N947** — the NPU column. E5.

## Verification pass needed before E0

- Confirm current board revisions/availability (Pico 2 W variants, P4
  devkit status, nRF54L15-DK GA, STM32N6 retail boards, Alif access).
- probe-rs target coverage for each Tier B part (it moves fast).
- TinyUF2 coverage map (which ESP32-S3 boards ship it by default).
- Apicula/openXC7 maturity for the specific FPGA parts listed.
- RPi 5 bare-metal ecosystem state (RP1 drivers outside Linux).
