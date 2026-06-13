//! YOUR teiOS app — the only file Studio's CODE tab edits.
//!
//! `app(tei)` runs once per scheduler pass. The `tei` harness is the
//! whole safe surface: run a primitive on a substrate (returns a
//! ledger and emits its line), check results across substrates, emit
//! the dispatch verdict, sleep. Everything else — USB, the substrate
//! implementations, the per-second driver loop — is fixed by teiOS.
//!
//! This default body is behaviour-identical to the shipped teios-rp2040
//! image: race CRC32 (the Hash primitive) on the CPU vs the DMA sniffer,
//! check they agree, dispatch the cheaper one. Edit freely — the forge
//! validates and builds whatever you write here (the fixed Cargo.toml
//! means you cannot add crates; the API below is what you have).

use crate::fw::tei::{Tei, TeiError};
use teios_app_rp2040::{PRIMITIVE_HASH, SUBSTRATE_CPU, SUBSTRATE_DMA};

/// One scheduler pass. Returns `Err` only on USB disconnect (teiOS
/// re-waits for the host and calls you again).
pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    // run the Hash primitive on each on-die substrate; each call prices
    // the run into a ledger and streams its JSON line to Studio.
    let cpu = tei.run_on(SUBSTRATE_CPU, PRIMITIVE_HASH).await?;
    let dma = tei.run_on(SUBSTRATE_DMA, PRIMITIVE_HASH).await?;

    // the two substrates must agree on the answer
    tei.check(cpu.result, dma.result).await?;

    // the verdict: lowest measured joules wins
    tei.dispatch(PRIMITIVE_HASH).await?;

    tei.sleep_ms(1000).await;
    Ok(())
}
