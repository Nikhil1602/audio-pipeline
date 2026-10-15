//! The C API of sonicdiar (`include/sonicdiar.h`): a thin layer over the
//! [`sonicdiar`] crate that passes configurations, events and lists as JSON.
//!
//! Every entry point catches panics, so none unwinds into C; errors are returned as codes with
//! the message kept per thread for `sd_last_error`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use sonicdiar_sdk::{Config, Error, Events, Session};

pub const SD_OK: c_int = 0;
pub const SD_ERR_ARGUMENT: c_int = -1;
pub const SD_ERR_CONFIG: c_int = -2;
pub const SD_ERR_ENGINE: c_int = -3;
pub const SD_ERR_MODELS: c_int = -4;
pub const SD_ERR_DEVICES: c_int = -5;
pub const SD_ERR_STATE: c_int = -6;
pub const SD_ERR_PANIC: c_int = -7;
/// Bumped on incompatible changes of the C contract.
pub const SD_API_VERSION: c_int = 1;

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn fail(code: c_int, message: impl Into<String>) -> c_int {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(message.into()));
    code
}

fn code_of(e: &Error) -> c_int {
    match e {
        Error::Config(_) => SD_ERR_CONFIG,
        Error::Engine(_) => SD_ERR_ENGINE,
        Error::Models(_) => SD_ERR_MODELS,
        Error::Devices(_) => SD_ERR_DEVICES,
    }
}

/// Runs `f`, turning errors and panics into codes.
fn guard(f: impl FnOnce() -> Result<(), (c_int, String)>) -> c_int {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SD_OK,
        Ok(Err((code, message))) => fail(code, message),
        Err(panic) => {
            let message = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            fail(SD_ERR_PANIC, format!("internal error: {message}"))
        }
    }
}

fn sdk(e: Error) -> (c_int, String) {
    (code_of(&e), e.to_string())
}

/// # Safety
/// `p` must be NULL or a NUL-terminated string valid for the call.
unsafe fn arg<'a>(p: *const c_char, name: &str) -> Result<&'a str, (c_int, String)> {
    if p.is_null() {
        return Err((SD_ERR_ARGUMENT, format!("{name} is NULL")));
    }
    // SAFETY: guaranteed by the caller.
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map_err(|_| (SD_ERR_ARGUMENT, format!("{name} is not valid UTF-8")))
}

/// # Safety
/// `p` must be NULL or a NUL-terminated string valid for the call.
unsafe fn opt_arg<'a>(p: *const c_char, name: &str) -> Result<Option<&'a str>, (c_int, String)> {
    if p.is_null() {
        Ok(None)
    } else {
        // SAFETY: guaranteed by the caller.
        unsafe { arg(p, name) }.map(Some)
    }
}

fn out_string(s: String) -> *mut c_char {
    // Interior NULs cannot occur in JSON or paths we produce; replace defensively.
    CString::new(s.replace('\0', " "))
        .expect("no interior NUL")
        .into_raw()
}

/// # Safety
/// `out` must be a valid pointer to writable storage for one pointer, or NULL.
unsafe fn put(out: *mut *mut c_char, s: Option<String>) -> Result<(), (c_int, String)> {
    if out.is_null() {
        return Err((SD_ERR_ARGUMENT, "output pointer is NULL".into()));
    }
    // SAFETY: guaranteed by the caller.
    unsafe { *out = s.map_or(std::ptr::null_mut(), out_string) };
    Ok(())
}

fn json<T: serde::Serialize>(v: &T) -> Result<String, (c_int, String)> {
    serde_json::to_string(v).map_err(|e| (SD_ERR_PANIC, e.to_string()))
}

/// A session behind the C handle.
pub struct SdSession {
    session: Mutex<Option<Session>>,
    events: Events,
}

// ---------------------------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "C" fn sd_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(sonicdiar_sdk::version()).expect("version"))
        .as_ptr()
}

#[unsafe(no_mangle)]
pub extern "C" fn sd_api_version() -> c_int {
    SD_API_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn sd_last_error() -> *mut c_char {
    LAST_ERROR.with(|e| e.borrow().clone().map_or(std::ptr::null_mut(), out_string))
}

/// # Safety
/// `s` must be NULL or a string returned by this library, not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: allocated by `CString::into_raw` in `out_string`, freed once.
        drop(unsafe { CString::from_raw(s) });
    }
}

/// # Safety
/// `kind` must be a NUL-terminated string; `json_out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_list_devices(kind: *const c_char, json_out: *mut *mut c_char) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let kind = unsafe { arg(kind, "kind") }?;
        let devices = sonicdiar_sdk::list_devices(kind).map_err(sdk)?;
        // SAFETY: as documented.
        unsafe { put(json_out, Some(json(&devices)?)) }
    })
}

/// # Safety
/// String arguments must be NUL-terminated (`only` may be NULL); `progress` may be NULL and is
/// called on the calling thread with `user`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_models_fetch(
    models_dir: *const c_char,
    group: *const c_char,
    only: *const c_char,
    progress: Option<extern "C" fn(*const c_char, *mut c_void)>,
    user: *mut c_void,
) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let (dir, group, only) = unsafe {
            (
                arg(models_dir, "models_dir")?,
                arg(group, "group")?,
                opt_arg(only, "only")?,
            )
        };
        let mut report = |p: sonicdiar_sdk::models::Progress| {
            let Some(cb) = progress else {
                return;
            };
            use sonicdiar_sdk::models::Progress as P;
            let v = match p {
                P::AlreadyPresent { id } => serde_json::json!({"step": "present", "id": id}),
                P::Downloading {
                    id,
                    url,
                    bytes,
                    license,
                } => serde_json::json!({"step": "downloading", "id": id, "url": url,
                    "bytes": bytes, "license": license}),
                P::Verified { id, sha256 } => {
                    serde_json::json!({"step": "verified", "id": id, "sha256": sha256})
                }
                P::BundleFiles { id, done, total } => {
                    serde_json::json!({"step": "files", "id": id, "done": done, "total": total})
                }
            };
            let s = CString::new(v.to_string()).expect("JSON has no NUL");
            cb(s.as_ptr(), user);
        };
        sonicdiar_sdk::models::fetch(Path::new(dir), group, only, &mut report).map_err(sdk)
    })
}

/// # Safety
/// `models_dir` must be NUL-terminated; `json_out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_models_list(
    models_dir: *const c_char,
    json_out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let dir = unsafe { arg(models_dir, "models_dir") }?;
        let list = sonicdiar_sdk::models::list(Path::new(dir));
        // SAFETY: as documented.
        unsafe { put(json_out, Some(json(&list)?)) }
    })
}

/// # Safety
/// `config_json` must be NUL-terminated; `out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_start(
    config_json: *const c_char,
    out: *mut *mut SdSession,
) -> c_int {
    guard(|| {
        if out.is_null() {
            return Err((SD_ERR_ARGUMENT, "out is NULL".into()));
        }
        // SAFETY: as documented.
        let cfg = Config::from_json(unsafe { arg(config_json, "config_json") }?).map_err(sdk)?;
        let session = Session::start(cfg).map_err(sdk)?;
        let handle = Box::new(SdSession {
            events: session.events(),
            session: Mutex::new(Some(session)),
        });
        // SAFETY: `out` is valid (checked above).
        unsafe { *out = Box::into_raw(handle) };
        Ok(())
    })
}

/// # Safety
/// `s` must be a live session handle; `json_out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_next_event(
    s: *mut SdSession,
    timeout_ms: c_int,
    json_out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: a live handle, shared (`&`) across threads by design.
        let s = unsafe { s.as_ref() }.ok_or((SD_ERR_ARGUMENT, "session is NULL".into()))?;
        let timeout = Duration::from_millis(timeout_ms.max(0) as u64);
        let event = s.events.next(timeout).map(|e| json(&e)).transpose()?;
        // SAFETY: as documented.
        unsafe { put(json_out, event) }
    })
}

/// # Safety
/// `s` must be a live session handle; `source` and `device` NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_set_device(
    s: *mut SdSession,
    source: *const c_char,
    device: *const c_char,
) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let s = unsafe { s.as_ref() }.ok_or((SD_ERR_ARGUMENT, "session is NULL".into()))?;
        // SAFETY: as documented.
        let (source, device) = unsafe { (arg(source, "source")?, arg(device, "device")?) };
        let guard = s.session.lock().unwrap_or_else(|p| p.into_inner());
        let session = guard
            .as_ref()
            .ok_or((SD_ERR_STATE, "the session has stopped".into()))?;
        session.set_device(source, device).map_err(sdk)
    })
}

/// # Safety
/// `s` must be a live session handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_pause(s: *mut SdSession) -> c_int {
    // SAFETY: as documented.
    unsafe { set_paused(s, true) }
}

/// # Safety
/// `s` must be a live session handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_resume(s: *mut SdSession) -> c_int {
    // SAFETY: as documented.
    unsafe { set_paused(s, false) }
}

/// # Safety
/// `s` must be NULL or a live session handle.
unsafe fn set_paused(s: *mut SdSession, paused: bool) -> c_int {
    guard(|| {
        // SAFETY: the caller's contract.
        let s = unsafe { s.as_ref() }.ok_or((SD_ERR_ARGUMENT, "session is NULL".into()))?;
        let guard = s.session.lock().unwrap_or_else(|p| p.into_inner());
        let session = guard
            .as_ref()
            .ok_or((SD_ERR_STATE, "the session has stopped".into()))?;
        if paused {
            session.pause().map_err(sdk)
        } else {
            session.resume().map_err(sdk)
        }
    })
}

/// # Safety
/// `s` must be a live session handle; `path_out` a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_journal_dir(
    s: *mut SdSession,
    path_out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let s = unsafe { s.as_ref() }.ok_or((SD_ERR_ARGUMENT, "session is NULL".into()))?;
        let guard = s.session.lock().unwrap_or_else(|p| p.into_inner());
        let session = guard
            .as_ref()
            .ok_or((SD_ERR_STATE, "the session has stopped".into()))?;
        let path = session.journal_dir().to_string_lossy().into_owned();
        // SAFETY: as documented.
        unsafe { put(path_out, Some(path)) }
    })
}

/// # Safety
/// `s` must be a live session handle; `summary_out` NULL or a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_stop(
    s: *mut SdSession,
    summary_out: *mut *mut c_char,
) -> c_int {
    guard(|| {
        // SAFETY: as documented.
        let s = unsafe { s.as_ref() }.ok_or((SD_ERR_ARGUMENT, "session is NULL".into()))?;
        let mut guard = s.session.lock().unwrap_or_else(|p| p.into_inner());
        let mut session = guard
            .take()
            .ok_or((SD_ERR_STATE, "the session has already stopped".into()))?;
        let summary = session.stop().map_err(sdk)?;
        if !summary_out.is_null() {
            // SAFETY: checked non-NULL; valid by contract.
            unsafe { put(summary_out, Some(json(&summary)?)) }?;
        }
        Ok(())
    })
}

/// # Safety
/// `s` must be NULL or a live session handle; it is invalid afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_session_free(s: *mut SdSession) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: allocated by `sd_session_start`, freed once; dropping stops the session.
        drop(unsafe { Box::from_raw(s) });
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn take(p: *mut c_char) -> String {
        // SAFETY: a string from this library, freed once.
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        // SAFETY: as above.
        unsafe { sd_string_free(p) };
        s
    }

    #[test]
    fn version_and_errors_cross_the_boundary() {
        // SAFETY: a static NUL-terminated string.
        let version = unsafe { CStr::from_ptr(sd_version()) };
        assert!(!version.to_bytes().is_empty());
        let mut out = std::ptr::null_mut();
        // SAFETY: valid arguments.
        let rc = unsafe { sd_list_devices(c"sideways".as_ptr(), &mut out) };
        assert_eq!(rc, SD_ERR_CONFIG);
        assert!(take(sd_last_error()).contains("input"));
        // SAFETY: NULL arguments are reported, not dereferenced.
        let rc = unsafe { sd_session_start(std::ptr::null(), &mut std::ptr::null_mut()) };
        assert_eq!(rc, SD_ERR_ARGUMENT);
        // SAFETY: valid arguments; invalid JSON field.
        let rc = unsafe { sd_session_start(c"{\"sytem\": 1}".as_ptr(), &mut std::ptr::null_mut()) };
        assert_eq!(rc, SD_ERR_CONFIG);
    }

    #[test]
    fn models_list_is_json() {
        let mut out = std::ptr::null_mut();
        // SAFETY: valid arguments.
        let rc = unsafe { sd_models_list(c"nowhere".as_ptr(), &mut out) };
        assert_eq!(rc, SD_OK);
        let v: serde_json::Value = serde_json::from_str(&take(out)).unwrap();
        assert!(v.as_array().is_some_and(|a| !a.is_empty()));
    }
}
