//! Backend selection policy and session configuration.

mod policy;
mod session;
#[cfg(test)]
mod tests;

pub use policy::{BackendPolicy, ParseBackendPolicyError};
#[cfg(device_pages)]
pub(crate) use session::is_all;
pub use session::{DEFAULT_VAAPI_DEVICE, GPU_PAGE_OPEN_WAIT, SessionConfig};
