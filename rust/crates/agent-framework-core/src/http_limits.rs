// Copyright (c) Microsoft. All rights reserved.

//! Size limits for HTTP response bodies.
//!
//! Without a cap, an attacker-controlled `base_url` (or a compromised proxy /
//! DNS) can return unbounded bytes and OOM the process. These constants and
//! helpers give providers a consistent way to enforce a ceiling.

/// Default maximum bytes to read from a non-streaming JSON response body.
///
/// 16 MiB is generous for chat completions (largest real responses are ~100 KiB)
/// while bounding the exfiltration/OOM surface.
pub const DEFAULT_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Default cap on total streaming-state size per SSE request, in bytes.
///
/// Accumulated text + tool-call argument fragments are counted against this.
/// If the cap is exceeded, the stream terminates with an error.
pub const DEFAULT_MAX_STREAM_STATE_BYTES: usize = 32 * 1024 * 1024;
