//! YOUR teiOS app — the file Studio's CODE tab edits (Portenta C33).
//!
//! One scheduler pass: race the Hash primitive (CRC32 over the workload)
//! on the Cortex-M33 software path vs the RA6M5 CRCA hardware-CRC
//! peripheral, check they agree, dispatch the cheaper one. Synchronous
//! (no async runtime on the RA family).

use crate::fw::tei::{Tei, TeiError};
use teios_ra6m5::{PRIMITIVE_HASH, SUBSTRATE_CRC_HW, SUBSTRATE_M33};

pub fn app(tei: &mut Tei) -> Result<(), TeiError> {
    let m33 = tei.run_on(SUBSTRATE_M33, PRIMITIVE_HASH)?;
    let hw = tei.run_on(SUBSTRATE_CRC_HW, PRIMITIVE_HASH)?;
    tei.check(m33.result, hw.result)?;
    tei.dispatch(PRIMITIVE_HASH)?;
    // teiOS dispatches to the substrate it measured cheapest (M33 vs CRCA),
    // then publishes the calibrated prices home.
    tei.run(PRIMITIVE_HASH)?;
    tei.publish(PRIMITIVE_HASH)?;
    tei.sleep_ms(1000);
    Ok(())
}
