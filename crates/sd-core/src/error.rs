use thiserror::Error;

/// Errors produced by `sd-core`.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum Error {
    #[error("invalid stream format: {0}")]
    InvalidFormat(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
