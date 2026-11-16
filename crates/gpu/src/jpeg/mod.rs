//! JPEG header parsing for the VA-API decoder.
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`headers`] | Walk a JPEG marker stream; extract DQT, DHT, SOF, SOS, DRI. |

pub(crate) mod headers;

#[cfg(test)]
pub(crate) mod test_fixtures;
