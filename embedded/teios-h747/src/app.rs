//! YOUR teiOS app — the file Studio's CODE tab edits (Portenta H7).
//!
//! One scheduler pass: race the Hash primitive (CRC32 over the workload)
//! on the Cortex-M7 software path vs the STM32 hardware CRC peripheral,
//! check they agree, dispatch the cheaper one. The M4 core is a priced
//! menu entry in the cost table; offloading a run to it is the
//! inter-core bring-up stretch (see fw.rs), so the default app does not
//! call it yet — when it lands, add `tei.run_on(SUBSTRATE_M4, …)`.

use crate::fw::tei::{Tei, TeiError};
use teios_h747::{PRIMITIVE_HASH, SUBSTRATE_CRC_HW, SUBSTRATE_M7};

pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let m7 = tei.run_on(SUBSTRATE_M7, PRIMITIVE_HASH).await?;
    let hw = tei.run_on(SUBSTRATE_CRC_HW, PRIMITIVE_HASH).await?;
    tei.check(m7.result, hw.result).await?;
    tei.dispatch(PRIMITIVE_HASH).await?;
    tei.sleep_ms(1000).await;
    Ok(())
}
