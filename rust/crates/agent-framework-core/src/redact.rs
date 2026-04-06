// Copyright (c) Microsoft. All rights reserved.

//! Helpers for redacting sensitive data out of strings that may get logged
//! or embedded into user-visible errors.
//!
//! Provider HTTP errors contain attacker-controllable response bodies.
//! Before we embed them into [`AgentError`](crate::error::AgentError) we
//! truncate and scrub them for known secret patterns.

/// Maximum number of bytes from an error body that we include in an error
/// message. Keeps logs/errors bounded and limits any exfiltration surface.
pub const MAX_ERROR_BODY_LEN: usize = 2048;

/// Well-known prefixes for credentials that might accidentally land in an
/// error body (e.g. if a misconfigured proxy echoes request headers).
const SECRET_PREFIXES: &[&str] = &[
    "Bearer ", "bearer ", "sk-", "sk_", "xoxb-", "xoxp-", "ghp_", "ghs_", "AKIA",
];

/// Trim `body` to at most [`MAX_ERROR_BODY_LEN`] bytes and replace any token
/// immediately following a known secret prefix with `[REDACTED]`.
///
/// Intentionally simple: operates on whitespace-bounded tokens and doesn't
/// use regex. The goal is defence-in-depth — typical provider error bodies
/// never contain these substrings, but a misconfigured upstream proxy could
/// echo headers back, and this prevents keys from reaching logs.
pub fn scrub_error_body(body: &str) -> String {
    let truncated = truncate_to_bytes(body, MAX_ERROR_BODY_LEN);
    scrub_secrets(&truncated)
}

fn truncate_to_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    // Find a char boundary <= max so we don't slice mid-UTF-8.
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}... [truncated {} bytes]", &s[..end], s.len() - end)
}

fn scrub_secrets(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let mut matched = false;
        for prefix in SECRET_PREFIXES {
            let pb = prefix.as_bytes();
            if i + pb.len() <= bytes.len() && &bytes[i..i + pb.len()] == pb {
                // Emit the prefix and redact until the next whitespace/delimiter.
                out.push_str(prefix);
                let mut j = i + pb.len();
                while j < bytes.len() {
                    let b = bytes[j];
                    if b.is_ascii_whitespace() || b == b'"' || b == b'\'' || b == b',' || b == b'}' || b == b')' {
                        break;
                    }
                    j += 1;
                }
                if j > i + pb.len() {
                    out.push_str("[REDACTED]");
                }
                i = j;
                matched = true;
                break;
            }
        }
        if !matched {
            // Push the next char (ASCII fast-path falls back for UTF-8).
            let ch = s[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_long_bodies() {
        let long = "x".repeat(MAX_ERROR_BODY_LEN + 500);
        let s = scrub_error_body(&long);
        assert!(s.len() <= MAX_ERROR_BODY_LEN + 64); // truncated + short suffix
        assert!(s.contains("[truncated"));
    }

    #[test]
    fn preserves_short_bodies() {
        assert_eq!(scrub_error_body("hello"), "hello");
    }

    #[test]
    fn scrubs_bearer_token() {
        let input = r#"{"error": "Bearer sk-abc123 rejected"}"#;
        let out = scrub_error_body(input);
        assert!(!out.contains("sk-abc123"));
        assert!(out.contains("Bearer [REDACTED]"));
    }

    #[test]
    fn scrubs_sk_prefixed_keys() {
        let input = "debug: key=sk-proj-abc123xyz and sk-ant-def456";
        let out = scrub_error_body(input);
        assert!(!out.contains("abc123xyz"));
        assert!(!out.contains("def456"));
        assert!(out.contains("sk-[REDACTED]"));
    }

    #[test]
    fn scrubs_in_json_quotes() {
        let input = r#"{"authorization": "Bearer sk-abc"}"#;
        let out = scrub_error_body(input);
        assert!(!out.contains("sk-abc"));
    }

    #[test]
    fn respects_utf8_boundaries() {
        // 2049 bytes with a multi-byte char at the end.
        let mut s = "a".repeat(MAX_ERROR_BODY_LEN - 2);
        s.push('é'); // 2 bytes
        s.push('é');
        let out = scrub_error_body(&s);
        // Should not panic on truncation.
        assert!(out.contains("[truncated") || out.len() == s.len());
    }
}
