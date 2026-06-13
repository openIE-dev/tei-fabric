//! The token denylist — layer 2 of the security model (see crate docs).
//!
//! Each entry names a construct that reaches host I/O or escapes the
//! no-`std` skeleton *at build time* (const-eval and macro expansion run
//! on the build host). This is a conservative token scan, not a parser:
//! its job is to make the obvious abuse paths impossible, while the
//! fixed-deps invariant and the seatbelt sandbox carry the real weight.

/// Max app source size — a teiOS app body is small; anything larger is
/// almost certainly an attempt to smuggle something or to blow up the
/// type checker. 16 KiB is generous for hand-written app logic.
pub const MAX_SOURCE_BYTES: usize = 16 * 1024;

/// (needle, why-it's-denied). Matched case-sensitively against the
/// source with string and `//` line-comment contents stripped first, so
/// a comment mentioning `unsafe` doesn't trip it.
pub const DENYLIST: &[(&str, &str)] = &[
    ("unsafe", "raw memory / FFI — out of the safe app surface"),
    ("asm!", "inline assembly runs arbitrary instructions"),
    ("core::arch", "arch intrinsics / asm escape"),
    (
        "include!",
        "reads & compiles an arbitrary host file at expand time",
    ),
    (
        "include_bytes!",
        "reads an arbitrary host file at expand time",
    ),
    (
        "include_str!",
        "reads an arbitrary host file at expand time",
    ),
    (
        "env!",
        "leaks a host env var into the binary at compile time",
    ),
    ("option_env!", "leaks a host env var at compile time"),
    (
        "extern",
        "extern crate/block pulls beyond the fixed skeleton deps",
    ),
    (
        "std::",
        "the skeleton is no_std; std implies a host-shaped escape",
    ),
    (
        "::std",
        "the skeleton is no_std; std implies a host-shaped escape",
    ),
    (
        "proc_macro",
        "proc-macro code executes on the host at expand time",
    ),
    (
        "#[no_mangle]",
        "symbol games against the fixed linker layout",
    ),
    (
        "#[unsafe(",
        "unsafe attribute (e.g. link_section) escapes the surface",
    ),
    (
        "\\u{",
        "unicode path-escape that could defeat this token scan",
    ),
    ("build_rs", "no user build scripts; deps are fixed"),
];

/// Strip string literals (\" … \" and raw r\"…\") and `//` comments so the
/// denylist scans only live code. Block comments are left in place
/// (rare in app bodies) and would only cause a conservative reject.
fn strip_strings_and_line_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let b = src.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // line comment
        if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        // char or string literal — skip its contents (keep delimiters as
        // spaces so token boundaries survive)
        if b[i] == b'"' {
            out.push(' ');
            i += 1;
            while i < b.len() {
                if b[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if b[i] == b'"' {
                    break;
                }
                i += 1;
            }
            out.push(' ');
            i += 1;
            continue;
        }
        out.push(b[i] as char);
        i += 1;
    }
    out
}

/// Validate a user `app.rs` body. `Ok(())` means it's safe to splice and
/// build; `Err(msg)` carries the first violation for the user.
pub fn validate(app_source: &str) -> Result<(), String> {
    if app_source.len() > MAX_SOURCE_BYTES {
        return Err(format!(
            "app source is {} bytes; the limit is {} bytes",
            app_source.len(),
            MAX_SOURCE_BYTES
        ));
    }
    // The raw source is scanned for the unicode-escape needle (it would
    // be invisible after stripping); everything else scans live code.
    if app_source.contains("\\u{") {
        return Err("`\\u{…}` escapes are not allowed (they can defeat the safety scan)".into());
    }
    let code = strip_strings_and_line_comments(app_source);
    for (needle, why) in DENYLIST {
        if *needle == "\\u{" {
            continue; // handled above on the raw source
        }
        if code.contains(needle) {
            return Err(format!("`{needle}` is not allowed: {why}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_a_clean_app_body() {
        let ok = r#"
            // race the hash on every substrate, dispatch the winner
            let cpu = tei.run_on("cpu", PRIMITIVE_HASH).await;
            let dma = tei.run_on("dma-sniffer", PRIMITIVE_HASH).await;
            tei.check(cpu.result, dma.result);
            tei.dispatch(PRIMITIVE_HASH);
            tei.sleep_ms(1000).await;
        "#;
        assert!(validate(ok).is_ok(), "{:?}", validate(ok));
    }

    #[test]
    fn rejects_each_denied_construct() {
        let cases = [
            "unsafe { *(0 as *mut u32) = 1; }",
            "core::arch::asm!(\"nop\");",
            "let _ = include_bytes!(\"/etc/passwd\");",
            "let _ = include_str!(\"/etc/passwd\");",
            "include!(\"/tmp/evil.rs\");",
            "let _ = env!(\"HOME\");",
            "let _ = option_env!(\"SECRET\");",
            "extern crate alloc;",
            "std::process::exit(0);",
            "let x: ::std::vec::Vec<u8> = ::std::vec::Vec::new();",
            "#[no_mangle] fn x() {}",
            "#[unsafe(link_section=\".x\")] static X: u8 = 0;",
        ];
        for c in cases {
            assert!(validate(c).is_err(), "should reject: {c}");
        }
    }

    #[test]
    fn the_word_unsafe_in_a_comment_or_string_is_fine() {
        assert!(validate("// this is safe, not unsafe\nlet x = 1;").is_ok());
        assert!(validate(r#"tei.log("running unsafe-free pass");"#).is_ok());
    }

    #[test]
    fn rejects_oversize() {
        let big = "a".repeat(MAX_SOURCE_BYTES + 1);
        assert!(validate(&big).is_err());
    }

    #[test]
    fn rejects_unicode_escape_even_split_intent() {
        assert!(validate(r#"let _ = "\u{75}";"#).is_err());
    }
}
