//! YOUR teiOS app — the file Studio's CODE tab edits (Nicla nRF52832).
//!
//! One scheduler pass: run the Hash primitive (CRC32 over the workload)
//! on the M4 software substrate, price it into a ledger, and dispatch.
//! The nRF52832 has a single general-compute core, so the M4 is the only
//! runnable substrate today; the always-on accelerator (NDP120/BHI260)
//! is a priced cost-table menu entry. When an offload kernel lands, add
//! `tei.run_on(SUBSTRATE_ACCEL, …)` and a `tei.check(...)` between them.

use crate::fw::tei::{Tei, TeiError};
use teios_nrf52832::{PRIMITIVE_HASH, SUBSTRATE_M4};

pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let _m4 = tei.run_on(SUBSTRATE_M4, PRIMITIVE_HASH).await?;
    tei.dispatch(PRIMITIVE_HASH).await?;
    tei.sleep_ms(1000).await;
    Ok(())
}
