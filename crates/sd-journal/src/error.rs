use std::path::PathBuf;

use sd_core::Source;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum JournalError {
    #[error("disk full while writing {path}")]
    DiskFull {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("I/O error on {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid journal manifest {path}: {reason}")]
    BadManifest { path: PathBuf, reason: String },
    #[error("source {0} is not part of this journal")]
    UnknownSource(Source),
    #[error("block for {source_name} has {got} samples, not a multiple of {channels} channels")]
    BadBlock {
        source_name: Source,
        got: usize,
        channels: u16,
    },
    #[error("journal directory {0} already contains a journal")]
    AlreadyExists(PathBuf),
    #[error("invalid configuration: {0}")]
    Config(String),
    #[error("WAV export failed: {0}")]
    Wav(#[from] hound::Error),
}

pub type Result<T, E = JournalError> = std::result::Result<T, E>;

impl JournalError {
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        let path = path.into();
        if is_disk_full(&source) {
            JournalError::DiskFull { path, source }
        } else {
            JournalError::Io { path, source }
        }
    }

    /// True for out-of-space conditions, which the engine handles by pausing and retrying.
    pub fn is_disk_full(&self) -> bool {
        matches!(self, JournalError::DiskFull { .. })
    }
}

fn is_disk_full(e: &std::io::Error) -> bool {
    if e.kind() == std::io::ErrorKind::StorageFull || e.kind() == std::io::ErrorKind::QuotaExceeded
    {
        return true;
    }
    // ERROR_DISK_FULL (112) / ERROR_HANDLE_DISK_FULL (39) on Windows; ENOSPC / EDQUOT on Unix.
    match e.raw_os_error() {
        #[cfg(windows)]
        Some(112) | Some(39) => true,
        #[cfg(target_os = "linux")]
        Some(28) | Some(122) => true,
        #[cfg(target_vendor = "apple")]
        Some(28) | Some(69) => true,
        _ => false,
    }
}
