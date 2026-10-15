//! Core building blocks shared by every sonicdiar crate.
//!
//! * [`types`]: sources, stream formats and block metadata.
//! * [`clock`]: a host-monotonic clock in nanoseconds that device timestamps can be mapped onto.
//! * [`drift`]: estimates a device's true sample rate against the host clock.
//! * [`ring`]: a lock-free SPSC ring that is safe to use from real-time audio callbacks.
//! * [`timeline`]: turns device packet timestamps into a gap-free, overlap-free timeline.

pub mod clock;
pub mod drift;
pub mod error;
pub mod ring;
pub mod timeline;
pub mod types;

pub use clock::HostTime;
pub use error::{Error, Result};
pub use types::{BlockFlags, BlockMeta, Source, StreamFormat};
