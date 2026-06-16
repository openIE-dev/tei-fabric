//! YOUR teiOS app — the only file Studio's CODE tab edits.
//!
//! `app(tei)` runs once per scheduler pass. The `tei` harness is the
//! whole safe surface: run a primitive on a substrate (returns a
//! ledger and emits its line), check results across substrates, emit
//! the dispatch verdict, sleep. Everything else — USB, the substrate
//! implementations, the per-second driver loop — is fixed by teiOS.
//!
//! This default body calibrates then dispatches: run CRC32 (the Hash
//! primitive) on each on-die substrate so the teiOS runtime measures both
//! and re-prices its cost table, check they agree, show the verdict — then
//! call `tei.run()` and let the runtime dispatch to whichever substrate it
//! now knows is cheapest. Edit freely — the forge validates and builds
//! whatever you write here (the fixed Cargo.toml means you cannot add
//! crates; the API below is what you have).

use crate::fw::tei::{Tei, TeiError};
use teios_app_rp2040::{PRIMITIVE_HASH, SUBSTRATE_CPU, SUBSTRATE_DMA};

/// One scheduler pass. Returns `Err` only on USB disconnect (teiOS
/// re-waits for the host and calls you again).
pub async fn app(tei: &mut Tei<'_, '_>) -> Result<(), TeiError> {
    // Calibration sweep: run the Hash primitive on each substrate. Each
    // call prices the run into a ledger, streams its JSON line, and folds
    // the (measured, if a meter is present) joules back into the runtime's
    // cost table.
    let cpu = tei.run_on(SUBSTRATE_CPU, PRIMITIVE_HASH).await?;
    let dma = tei.run_on(SUBSTRATE_DMA, PRIMITIVE_HASH).await?;

    // The two substrates must agree on the answer.
    tei.check(cpu.result, dma.result).await?;

    // The verdict, straight from the now-re-priced cost table.
    tei.dispatch(PRIMITIVE_HASH).await?;

    // The teiOS call: the runtime dispatches to the cheapest substrate it
    // measured above — we no longer name one.
    tei.run(PRIMITIVE_HASH).await?;

    // Publish the calibrated prices home — Studio relays them to the fabric,
    // where they land in the HUB cost surface + FLEET roster.
    tei.publish(PRIMITIVE_HASH).await?;

    tei.sleep_ms(1000).await;
    Ok(())
}
