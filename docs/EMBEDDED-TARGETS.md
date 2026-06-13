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

## The bench — David's actual inventory (plans target THESE first)

The E-phases build against boards in hand, not boards on order.

| Board (owned) | Silicon / substrates | Flash path | E-role |
|---|---|---|---|
| **Portenta H7 + H7 Lite** | STM32H747: **M7@480 + M4@240 dual-core** + STM32 hw-CRC peripheral, DMA2D/Chrom-ART, hw JPEG | **DFU bootloader → WebDFU = W1 browser-flashable** (double-tap reset). **LIVE forge target** — Studio compiles custom apps → `.bin` (dfu-util to 0x08040000) | **E1b SHIPPED (compile-verified)**: M7-software CRC32 vs the STM32 hw-CRC peripheral raced, priced, dispatched, true DWT CYCCNT. M4-core offload is the documented inter-core stretch. USB-HS/ULPI + 480 MHz RCC are bench-pending |
| **Adafruit Feathers (various)** | RP2040 (PIO + **DMA sniffer**, M0+ → timer-proxy cycles), nRF52840 (PPI), SAMD51 (EVSYS+CCL), ESP32-class | UF2 drag-drop (W2, every browser) | **E1a — first real ledger**: teios-rp2350 ports to RP2040 Feather with minimal delta (same embassy-rp, same sniffer demo) |
| **Particle Tachyon** | QCM6490: big.LITTLE + Hexagon NPU + Adreno + 5G; SPMI PMIC ADCs (candidate T0) | in-house (tachyon-os boot chain; EDL/fastboot) | teiOS-on-Snapdragon (Tier 0) |
| **OpenMV AE3** | Alif E3: M55-HP + M55-HE + **dual Ethos-U55** (open Vela) | SETOOLS / OpenMV tooling | the NPU column + vision bundle (Tier 0) |
| **Nicla Voice** | **Syntiant NDP120** always-on audio NPU + nRF52832 host (M4F @ 64 MHz) | Arduino/BLE DFU (nRF52832 has **no USB** — transport is UART/BLE, not USB CDC) | **the µW dispatch story**: NDP listens at sub-mW while the host sleeps — "sleep is a substrate" made audible. teiOS host path = embassy-nrf (nrf52832); the NDP120 is a fixed-function NN substrate priced by inference |
| **Nicla Sense ME** | **Bosch BHI260AP** programmable sensor-fusion hub + BME688 + nRF52832 host (M4F @ 64 MHz) | Arduino/BLE DFU (no USB on nRF52832) | the **sensor-fusion-as-substrate** row: BHI260's self-learning AI core runs fusion while the host sleeps. Same embassy-nrf host path as Nicla Voice; BHI260 priced by fusion-op |
| **Nano Matter (community preview)** | SiLabs MGM240S (EFR32MG24): **MVP AI accel**, 802.15.4/BLE | Arduino core (onboard DAP) | MVP-accel substrate + Matter radio; xG24 silicon w/o WSTK AEM (no T0 on-board) |
| **Portenta C33** | Renesas RA6M5 (M33) + ESP32-C3 radio | DFU / Arduino | Renesas-family beachhead; radio calibration upload |
| **Coral Dev Board Mini** | MT8167S (4×A35) + **Edge TPU** (~2 TOPS/W), Mendel Linux | SD/fastboot (Linux-class) | teid path: Edge TPU J/inference rows via the delegate's accel_invocations |
| **Raspberry Pi Zero 2 W** | BCM2710 4×A53 | **SD card, flat files on FAT (W2)** | E3 — the bare-metal image flagship (unchanged, in hand) |

Bench gaps to note honestly: no RP2350 (Hazard3/mixed-arch demo waits or
ships untested-on-hardware), no PPK2/bench energy meter confirmed —
first measured-tier joules likely come from an INA228 breakout (~$10)
or the Tachyon's PMIC ADCs.

**Forge-target status (what Studio's CODE→BUILD compiles today):**

| Board | Forge target | Why / blocker |
|---|---|---|
| Feather RP2040, Pico | ✅ live (`.uf2`) | embassy-rp, native USB CDC, mass-storage flash |
| Portenta H7 / H7 Lite | ✅ live (`.bin`, DFU) | embassy-stm32; USB-HS/RCC bench-pending but the build + image are real |
| Nicla Voice, Nicla Sense ME | ✅ live (`.bin`, E1c) | embassy-nrf nrf52832; **UART transport** (no USB) — first non-USB-CDC image. Workload hashed from flash (64 KB RAM). Single M4 substrate + accelerator menu entry ("sleep is a substrate"). UART pins + flash base bench-pending |
| Portenta C33 | ✅ live (`.bin`, E1d) | Renesas RA6M5 (M33) — bare-metal `cortex-m-rt` + `ra6m5-pac` (no embassy HAL). Real M33-software-vs-CRCA-hardware CRC race. Transport = semihosting (SCI UART is the bench step) |
| Tachyon, OpenMV AE3, Coral Mini, Pi Zero 2W | Tier 0 / in-house | own boot chains (tachyon-os, SETOOLS, Linux images), not the MCU forge pattern |

## Tier 0 — boards already under OpenIE control (in-house flows)

These jump every queue: control code exists today in `openie-fpga`
(pure-Rust, no vendor tools), so teiOS integration is wiring, not
porting. They are also the first REAL exotic substrates — the fabric's
simulated columns get hardware calibration targets.

| Board | Silicon | In-house flow (openie-fpga) | TEI significance |
|---|---|---|---|
| **SynSense Xylo Audio 2/3** (SYNS61201/65302) | neuromorphic ASIC — 1000/992 CuBa-LIF neurons | **`ofpga-xylo`: complete open driver** — 370+ register map, RAM bank layouts, bitshift-decay neuron model, cycle-accurate simulator, SPI programming path, USB, `ofpga-snn` compiler | **The fabric's neuromorphic column in physical silicon.** Real spike/SOP ledgers + measured joules from an actual SNN chip — the spiking dialect's DEFAULT_ACTIVITY and SOP_J calibrated against hardware, not Loihi papers |
| **AMD Kria KV260** (K26 SOM, XCZU5EV) | Zynq UltraScale+ | **Pure-Rust bitstream toolchain, no Vivado** — CGRA/DLRA/RRA designs compile→place→route→program on the real die; ICAP hardware CRC accepts every word; PCAP readback verifies config bits (SILICON_VALIDATION.md, 2026-04). Known gap: PS↔PL data bridge for functional I/O is the named next milestone | The forge's UltraScale+ recipe is ALREADY OPEN-FLOW (was listed as vendor-tool Tier C2 — wrong for us). Soft substrates with fabric-synthesized ledger counters on a $250 board, license-free |
| **Particle Tachyon** (Qualcomm QCM6490/SC7280, 5G SBC ~$149) | 8×Kryo big.LITTLE + Hexagon NPU (~12 TOPS) + Adreno + 5G modem | **`tachyon-os`: a bare-metal AArch64 Rust OS, in-house** — hands off from Particle's signed PBL→XBL→ABL chain and owns the AP; GENI UART, GICv3, ARMv8 timers, PSCI, ramoops, SPMI/PMIC (PMK8350/PM7325/PMR735A) crates + clk/DMA/dwc3/IOMMU/TLMM kernel modules; modem/DSP/GPU driven as signed blobs over QRTR/QMI/FastRPC | **teiOS on Snapdragon-class silicon already exists in embryo** ("not Linux at all", literally). Richest substrate set in the matrix: big.LITTLE dispatch (the Linux-EAS lineage, finally below Linux) + Hexagon NPU + GPU + sensors-DSP. SPMI PMIC ADCs = candidate T0 self-telemetry. The invisible-edge-mesh node compute (QCM6490) and the Qualcomm-AI-Hub-analogue row, on one board |
| **OpenMV AE3** (Alif Ensemble E3: M55 + Ethos-U55) | MCU + the matrix's only open-toolchain NPU | In David's possession (via Kwabena Agyeman, OpenMV founder — direct channel); AE3 firmware development + carrier pinmap staged in `ofpga-foundry/hw` (order/bringup plans, GenX320 + Lepton + Iridium module row) | The OpenMV-analogue row of the ecosystem map made concrete: vision domain bundle on Ethos-U (open Vela flow) with the Joulo camera lineage; `ofpga-cam` (pure-Rust U3V/GigE/CSI-2 machine-vision drivers) seeds the same domain |

## Tier A — UF2 drag-drop (the Arduino-grade turnkey path)

| Board | Silicon | On-die substrates for dispatch | Notes |
|---|---|---|---|
| **Raspberry Pi Pico 2 / 2 W** | RP2350 — per-core ARCHSEL: 2×M33, 2×Hazard3, **or 1×M33 + 1×Hazard3 simultaneously** (reset-switchable) | 3×PIO (12 SMs), 16-ch DMA w/ sniffer-CRC, HSTX | **E1 flagship — a 4-substrate live-dispatch demo** (M33 + Hazard3 + PIO + DMA) on a $5–7 board. Caveat: pico-sdk has no out-of-box mixed-binary build |
| Raspberry Pi Pico / W / WH | RP2040 (2×M0+) | 2×PIO (8 SMs), DMA | the install base |
| Adafruit Feather RP2040 (+DVI/CAN/RFM variants) | RP2040 | PIO, DMA | Feather form factor, huge tutorial reach |
| Adafruit Feather nRF52840 / Sense | nRF52840 (M4F) | PPI hardware event routing, radio, on-die USB | UF2 bootloader; PPI = "CPU-asleep" substrate |
| Adafruit Feather M4 / ItsyBitsy M4 | SAMD51 (M4F) | DMAC, EVSYS event system, CCL lookup logic | EVSYS = free event routing |
| Adafruit Feather ESP32-S2/S3 (TinyUF2) | ESP32-S2/S3 | ULP-RISC-V/FSM coprocessor, AES/SHA accel, vector ext (S3) | UF2 via TinyUF2; native path is esptool |
| Seeed XIAO RP2040 / RP2350 / nRF52840 (Sense) / SAMD21 | as named | as families above (note: SAMD21 = 12-ch EVSYS, NO CCL) | thumbnail-size, classroom favorite |
| SparkFun Thing Plus / Pro Micro RP2040·RP2350 | RP2040/RP2350 | PIO, DMA | Qwiic ecosystem |
| Pimoroni Pico-family boards | RP2040/RP2350 | PIO, DMA | |
| Arduino Nano RP2040 Connect | RP2040 + NINA Wi-Fi | PIO, DMA + radio | Arduino-branded RP2 |
| Fomu | **iCE40UP5K FPGA** | the entire fabric is the substrate | FPGA with Feather-grade DFU UX — but project dormant, stock uncertain; OrangeCrab is the living equivalent |

## Tier B — one-line CLI flash

### ESP family (esptool.py / espflash — Rust-native flasher)
| Board | Silicon | Substrates | Notes |
|---|---|---|---|
| **ESP32-C6-DevKitC** | C6 (HP RISC-V 160 MHz + **LP RISC-V core**) | LP core, DMA, AES/SHA, 802.15.4+WiFi6 | **E-phase pick**: HP-vs-LP dispatch + radio upload of calibration reports |
| ESP32-P4X-Function-EV (~$60; original board obsolete) | P4 (2×HP RV32IMAFC @400 MHz + PIE SIMD + LP core @40 MHz, no radio) | LP core, 2D-DMA, PPA, crypto | the compute-heavy ESP |
| ESP32-S3-DevKitC | S3 (2×LX7) | ULP-RISC-V, vector instructions, crypto | TinyML community |
| ESP32-C3 / H2 DevKits | single RISC-V | crypto, DMA | $2-4 class |
| classic ESP32-DevKitC | 2×LX6 | ULP-FSM, crypto | the install base |

### Nordic / STM32 / others (probe-rs — Rust-native, one tool for all)
| Board | Silicon | Substrates | Notes |
|---|---|---|---|
| nRF52840-DK / dongle | M4F | **PPI**, radio, on-die USB | nrfutil DFU on dongle; PPK2 attaches for T1 calibration |
| nRF5340-DK | M33 app + M33 net | dual-core dispatch, DPPI | |
| **nRF54L15-DK** ($39) | M33 + **FLPR VPR core** (RV32EMC @128 MHz; PPR is nRF54H20-only) | FLPR (Zephyr `cpuflpr` target, NCS soft-peripherals), DPPI | among the most TEI-shaped MCUs shipping: a dispatchable RISC-V helper core |
| STM32 Nucleo (G4 / H72x-H73x / H5 / **U5**) | the FMAC+CORDIC families (NOT F4/L4/H743/H503) | DMA, **FMAC** (≈1 MAC/cycle — value is full CPU offload via DMA; 7–11× vs CMSIS-DSP measured on G474), **CORDIC** (5–10× vs sw sin/cos), U5 LPBAM autonomous chains (~10 µA ADC sampling in Stop 2) | on-board ST-LINK; probe-rs flashes. **No on-die current/energy meter on any STM32** (ST-confirmed: shunt/IDD jumper or STLINK-V3PWR only) → T1 |
| **STM32N6 Nucleo (~$56) / DK (~$185)** | N6 (M55 @800 MHz + **ST Neural-ART NPU**, 600 GOPS — proprietary, NOT Ethos-U) | NPU (closed ST toolchain), Helium SIMD, DMA | NPU column; MLPerf-Tiny measured 156–444 µJ/inference |
| NXP FRDM-MCXN947 (~$26) | 2×M33 @150 MHz + **eIQ Neutron NPU** (NXP proprietary, NOT Ethos-U) + PowerQuad | NPU (closed NXP tools), PowerQuad (FFT/FIR/CORDIC) | NPU column, NXP channel |
| Teensy 4.0 / 4.1 | i.MX RT1062 (M7 @600 MHz) | 2×FlexIO, DMA, CM7 dual-issue | teensy_loader_cli; the performance MCU |
| Arduino UNO R4 / classic UNO R3 | RA4M1 / ATmega328P | DAC/CTSU / —— | reach play only; R3 via avrdude |
| TI MSP430FR LaunchPads (FR5994/FR6989-class) | MSP430FR (+ **LEA** accel, 36× energy on FFT measured) | LEA vector math, FRAM | **EnergyTrace = the bench-calibration (T1) anchor, NOT T0**: the charge-pulse counter lives in the eZ-FET/MSP-FET/XDS110 debug probe — firmware can never read its own joules (TI-confirmed). ET++ state-correlation only on FR59xx/69xx + CC13xx/26xx (the latter via XDS110) |
| TI MSPM0 LaunchPads | M0+ | analog blocks | $0.5-class parts |
| **SiLabs xG24 Pro Kit** (PK6010A — NOT the Dev Kit, which lacks AEM) | EFR32MG24 (M33) | MVP AI/ML accel (≈6× energy vs M33), radio | **The confirmed T0 path: target firmware reads its own AEM current via BSP_CurrentGet()/BSP_VoltageGet() over the board-controller channel** (BCP; verified in simplicity_sdk source, not deprecated). One bench-verify pending on the BRD4002A WPK board-controller servicing legacy BCP packets |
| Ambiq Apollo4/5 EVB | M4F/M55 subthreshold | low-power GPU (4), Helium (5) | the µW-class floor; J-Link |
| Alif Ensemble DK-E7 | M55-HP+**Ethos-U55-256** & M55-HE+**Ethos-U55-128** | dual NPU, dual core | **the only true Ethos-U part in this matrix** (open Vela toolchain; ST/NXP NPUs need closed tools). Published: 76× energy vs M55-alone MobileNetV2. E4/E6/E8 successors ship Ethos-U85 |
| WCH CH32V003 dev board | RV32EC @48 MHz | —— | $0.10 CPU; minichlink/wchisp; the "TEI on ten cents" stunt |
| Sipeed Longan Nano | GD32VF103 (RV32IMAC) | DMA | dfu-util; cheap RISC-V |

## Tier C — FPGA boards, the full sub-$500 field

The FPGA targets are special: the forge doesn't just flash firmware, it
ships **soft substrates** — LiteX/VexRiscv or NEORV32 SoCs with TEI event
counters synthesized into the fabric (a ledger the hardware itself
maintains). This is the rehearsal for purpose-built TEI silicon.
Prices are early-2026 street; verify before E-phase pinning.

**Web-flash path legend**: DFU = browser-flashable today via webdfu (W1);
bridge = teiProbe SPI/JTAG (W3); vendor = vendor cable/tool only (forge
emits the bitstream; flashing needs the bridge or a CLI fallback).

### C1 — fully open toolchain (Yosys + nextpnr; the forge owns the whole flow)

| Board | FPGA | ~$ | Flow | Web flash | Notes |
|---|---|---|---|---|---|
| **iCEBreaker** (+ bitsy) | iCE40UP5K | 75 | icestorm | bridge | the open-FPGA teaching board |
| iCEstick | iCE40HX1K | 40 | icestorm | bridge | tiny classic |
| **UPduino v3.1** | iCE40UP5K | 30 | icestorm | bridge | cheapest UP5K |
| TinyFPGA BX | iCE40LP8K | 38 | icestorm | own USB bootloader | stock intermittent |
| Go Board (Nandland) | iCE40HX1K | 65 | icestorm | bridge | tutorial-rich |
| Alchitry Cu | iCE40HX8K | 50 | icestorm | bridge | stackable Io/Br shields |
| Olimex iCE40HX8K-EVB | iCE40HX8K | 50 | icestorm | bridge | EU supply |
| Fomu | iCE40UP5K | 35 | icestorm | **DFU** | dormant/stock-uncertain; historical |
| **ULX3S** (12F–85F) | ECP5 | 115–250 | prjtrellis | **DFU-class (fujprog/esp32 OTA)** | the open-flow flagship |
| **OrangeCrab** (25F/85F) | ECP5 | 99–129 | prjtrellis | **DFU (foboot)** | **ECP5 in Feather form factor; the zero-programmer flow** |
| Butterstick | ECP5-85F | 120 | prjtrellis | DFU (foboot-class) | high-IO syzygy |
| **Colorlight i5 / i9 / i9+** | ECP5-25/45K | 15–50 (+~20 ext board) | prjtrellis | bridge | recycled LED controllers — the cheapest real ECP5s |
| icesugar-pro | ECP5-25F | 60 | prjtrellis | drag-drop (iCELink MSC!) | SD-card-sized |
| LogicBone | ECP5-45F | 130 | prjtrellis | DFU | BeagleBone form factor |
| Lattice ECP5-EVN | ECP5-85F | 115 | prjtrellis | bridge | official eval, cheap for 85F |
| **Tang Nano 9K / 20K** | Gowin GW1NR-9 / GW2A-18 | 15 / 30 | **Apicula 0.32** (PLL/BRAM/DSP OK) | bridge (BL702 USB-JTAG; openFPGALoader from git) | huge hobby momentum |
| Tang Nano 1K / 4K | GW1NZ/GW1NSR | 10–15 | Apicula (partial) | bridge | impulse-buy tier |
| Tang Primer 20K / 25K | GW2A / GW5A-25 | 30–45 | Apicula (20K yes; 25K/GW5A maturing) | bridge | SODIMM SOM + dock |
| Tang Mega 60K / 138K | GW5AT | 60–250 | vendor (GW5A Apicula WIP) | bridge | PCIe-class hobby boards |
| OLIMEX/CCGM GateMate eval | Cologne Chip CCGM1A1 | 50–110 | **Yosys synth + free vendor P&R** | bridge | the European open-ish fabric; EU sovereignty angle |
| CrossLink-NX EVN | Lattice Nexus | 130 | prjoxide (maturing) | bridge | Nexus open flow |

### C2 — free vendor toolchain (forge scripts it; bitstream still reproducible)

| Board | FPGA | ~$ | Tool | Web flash | Notes |
|---|---|---|---|---|---|
| **Arty A7-35/100** | Artix-7 | 130–300 | Vivado free (or openXC7, no STA) | bridge | the university standard |
| Arty S7 / Cmod A7 / Cmod S7 | Spartan/Artix-7 | 80–150 | Vivado free | bridge | breadboard-able 7-series |
| Basys3 | Artix-7 35T | 160 | Vivado free | bridge | intro-course classic |
| Nexys A7-100T | Artix-7 100T | 350 | Vivado free | bridge | bigger classrooms |
| **EBAZ4205** | Zynq-7010 | 20–35 | Vivado free | bridge | recycled miner control board — the $25 Zynq |
| QMTech boards (Artix/Kintex/Zynq/Cyclone) | XC7A35T…**XC7K325T** | 30–120 | Vivado free (K325T needs openXC7 or license workarounds) | bridge | AliExpress value kings; Kintex-325T under $120 |
| Alchitry Au / Au+ | Artix-7 35T/100T | 100–300 | Vivado free | bridge | maker-friendly Xilinx |
| Numato Mimas A7 | Artix-7 50T | 80 | Vivado free | bridge | |
| PYNQ-Z2 / Zybo Z7-10/20 / Cora Z7 | Zynq-7000 | 110–300 | Vivado free | bridge | bare-metal A9 + fabric (no-Linux Zynq is viable) |
| Arty Z7-20 | Zynq-7020 | 240 | Vivado free | bridge | |
| **Kria KV260** | Zynq UltraScale+ K26 | 250 | **openie-fpga pure-Rust flow (no Vivado!)** — see Tier 0 | bridge | astonishing $/fabric; bare-metal A53/R5F possible |
| ZUBoard 1CG | Zynq US+ 1CG | 160 | Vivado free | bridge | cheapest UltraScale+ |
| DE10-Nano | Cyclone V SoC | 225 | Quartus Lite | bridge | MiSTer community = flashing culture |
| DE10-Lite | MAX10 | 85–140 | Quartus Lite | bridge | academic staple |
| DE0-Nano | Cyclone IV | 100 | Quartus Lite | bridge | legacy but everywhere |
| Cyclone 10 LP eval | C10LP | 90 | Quartus Lite | bridge | |
| **Efinix Xyloni** | Trion T8 | 35 | Efinity (free license) | bridge | the $35 alt-architecture |
| Efinix Ti60 F225 dev | Titanium Ti60 | 150 | Efinity | bridge | efficiency-class fabric |
| **PolarFire SoC Discovery** | MPFS095T | 132 | Libero (free Silver) | bridge | **hard RISC-V cores + fabric** — a TEI-shaped SoC |
| PolarFire SoC Icicle | MPFS250T | 499 | Libero free | bridge | just inside the cap; the serious RISC-V+FPGA board |
| Renesas ForgeFPGA eval | SLG47910 | 50 | Go Configure (free) | bridge | sub-$1 FPGA class |

Soft-core menu: **NEORV32** (the standout — weekly releases, June 2026 HPM
rework gives 13 hardware performance counters via standard CSRs; CI setups
for iCEBreaker/UPduino/OrangeCrab/ULX3S/Tang Nano 9K+20K), VexRiscv /
**VexiiRiscv** (the successor, merged in mainline LiteX; LiteX CSRStatus
peripherals = ~15-line custom ledger counters with auto-generated firmware
accessors), SERV (bit-serial, ~198 LUT — the *lowest-joule soft core*),
picorv32 (caretaker mode). Apicula 0.32 (2026-04) fully supports Tang
Nano 9K/20K incl. PLL/BRAM/DSP. The zero-hardware-programmer flow:
**LiteX+VexRiscv on OrangeCrab via its foboot DFU bootloader.**

FPGA priority within the web-only doctrine: **OrangeCrab and ULX3S first**
(DFU = direct W1 browser flash of bitstreams), **Tang Nano 9K/20K and
Colorlight** as the price-floor mass tier (W3 bridge), **iCEBreaker** for
teaching, **EBAZ4205/KV260** as the value outliers, **PolarFire Discovery**
for the hard-RISC-V story. Every C1 board's bitstream is fully
reproducible by the forge with zero vendor licenses.

## Tier D — Pi-class, bare metal (no Linux, SD-image artifact)

The GPU-boot trick makes every Raspberry Pi a "drag an image, boot our
runtime" machine — the firmware loads `kernel*.img` from FAT; that file is
simply our flat binary. Turnkey UX identical to Raspberry Pi OS, contents
100% TEI.

| Board | Silicon | Bare-metal state | Notes |
|---|---|---|---|
| **Raspberry Pi Zero 2 W** | BCM2710 (4×A53) | mature (circle, rust crates) | $15; the image-flash flagship. **Zero power telemetry** (schematic-verified: dumb bucks, no sense resistors) — cycles-proxy T1 only |
| Raspberry Pi 4 / CM4 | BCM2711 (4×A72) | mature | |
| Raspberry Pi 5 | BCM2712 + **RP1** | workable; RP1 southbridge adds I/O complexity | document as "later" — but note: **PMIC ADCs are runtime-readable** (`vcgencmd pmic_read_adc`; per-rail V/I) → teiOS on Pi 5 can self-calibrate (caveat: 5V/USB path bypasses the PMIC) |
| Raspberry Pi 3 | BCM2837 | mature | the docs/tutorials corpus |
| BeagleBone Black / BeaglePlay | AM335x (PRU 2×200 MHz) / AM625 (PRUSS 2× up to 333 MHz, no ICSSG) | bare-metal possible; **PRU-ICSS** deterministic RT cores | PRUs = first-class substrate; AM62x PRU Academy launched 2026-04 |
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
5. **xG24 Pro Kit + MSP430FR5994** — the two T0 energy-measurement
   references that anchor everyone else's calibration tables (the xG24
   *Dev Kit* lacks AEM — Pro Kit / WPK mainboard required). Alongside E1
   bench kit.
6. **An NPU board** — Alif DK-E7 if the open Vela toolchain matters (the
   only Ethos-U in the matrix); STM32N6 (Neural-ART) or MCX N947 (eIQ
   Neutron) for reach, both behind closed vendor compilers. E5.

## Verification pass needed before E0

- Confirm current board revisions/availability (Pico 2 W variants, P4
  devkit status, nRF54L15-DK GA, STM32N6 retail boards, Alif access).
- probe-rs target coverage for each Tier B part (it moves fast).
- TinyUF2 coverage map (which ESP32-S3 boards ship it by default).
- Apicula/openXC7 maturity for the specific FPGA parts listed.
- RPi 5 bare-metal ecosystem state (RP1 drivers outside Linux).
