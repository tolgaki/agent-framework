// Copyright (c) Microsoft. All rights reserved.

//! A string newtype that redacts its contents in `Debug` output and zeroizes
//! its backing memory on drop.
//!
//! Provider configs hold API keys. If a config is logged (e.g. via
//! `tracing::debug!("config: {:?}", cfg)`), we don't want the key to leak.
//! Wrapping the key in [`SecretString`] makes accidental exposure opt-in:
//! you must call [`SecretString::expose`] explicitly.
//!
//! On drop, the underlying bytes are overwritten using the `zeroize` crate
//! so the key does not linger in freed heap memory for later process-memory
//! dumps / core files.

use std::fmt;

use zeroize::Zeroize;

/// A string whose `Debug` / `Display` never print its contents, and whose
/// memory is zeroized when dropped.
///
/// `Clone` is implemented because configs are routinely cloned at agent
/// construction. Each clone is independently zeroized on drop.
pub struct SecretString(String);

impl SecretString {
    /// Wrap a string as a secret.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Expose the underlying string. Use with care.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Consume the secret and return the underlying string.
    ///
    /// The returned `String` is NOT zeroized when it's dropped — callers
    /// that care about that should use [`expose`](Self::expose) and copy
    /// into their own zeroizing container instead.
    pub fn into_inner(self) -> String {
        // Avoid running our Drop (which would zeroize).
        std::mem::take(&mut std::mem::ManuallyDrop::new(self).0)
    }
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl Drop for SecretString {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("SecretString([REDACTED])")
    }
}

impl fmt::Display for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[REDACTED]")
    }
}

impl From<String> for SecretString {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for SecretString {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_is_redacted() {
        let s = SecretString::new("super-secret-key");
        assert_eq!(format!("{:?}", s), "SecretString([REDACTED])");
        assert_eq!(format!("{}", s), "[REDACTED]");
    }

    #[test]
    fn expose_returns_inner() {
        let s = SecretString::new("value");
        assert_eq!(s.expose(), "value");
    }

    #[test]
    fn clone_is_independent() {
        let a = SecretString::new("k");
        let b = a.clone();
        assert_eq!(a.expose(), "k");
        assert_eq!(b.expose(), "k");
        drop(a);
        // b should still be valid after a was dropped (and zeroized).
        assert_eq!(b.expose(), "k");
    }
}
