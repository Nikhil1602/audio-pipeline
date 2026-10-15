//! The native part of the `sonicdiar` Python package (`sonicdiar._native`). The
//! Python layer (`python/sonicdiar/__init__.py`) turns the JSON strings into dicts.
//!
//! Blocking calls (waiting for events, stopping, downloading) release the GIL.

use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use sonicdiar::{Config, Events, Session};

create_exception!(
    _native,
    SonicDiarError,
    PyException,
    "An error from SonicDiar."
);

fn err(e: sonicdiar::Error) -> PyErr {
    SonicDiarError::new_err(e.to_string())
}

fn json<T: serde::Serialize>(v: &T) -> PyResult<String> {
    serde_json::to_string(v).map_err(|e| SonicDiarError::new_err(e.to_string()))
}

/// A recording session (use `sonicdiar.Session`).
#[pyclass(module = "sonicdiar._native")]
struct NativeSession {
    session: Mutex<Option<Session>>,
    events: Events,
}

#[pymethods]
impl NativeSession {
    #[new]
    fn new(py: Python<'_>, config_json: &str) -> PyResult<Self> {
        let cfg = Config::from_json(config_json).map_err(err)?;
        let session = py.detach(|| Session::start(cfg)).map_err(err)?;
        Ok(Self {
            events: session.events(),
            session: Mutex::new(Some(session)),
        })
    }

    /// The next event as JSON, waiting up to `timeout` seconds; None if none arrived.
    fn next_event(&self, py: Python<'_>, timeout: f64) -> PyResult<Option<String>> {
        let timeout = Duration::from_secs_f64(timeout.max(0.0));
        let events = self.events.clone();
        py.detach(|| events.next(timeout))
            .map(|e| json(&e))
            .transpose()
    }

    fn set_device(&self, source: &str, device: &str) -> PyResult<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| SonicDiarError::new_err("the session has stopped"))?;
        s.set_device(source, device).map_err(err)
    }

    fn pause(&self) -> PyResult<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| SonicDiarError::new_err("the session has stopped"))?;
        s.pause().map_err(err)
    }

    fn resume(&self) -> PyResult<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| SonicDiarError::new_err("the session has stopped"))?;
        s.resume().map_err(err)
    }

    fn journal_dir(&self) -> PyResult<String> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| SonicDiarError::new_err("the session has stopped"))?;
        Ok(s.journal_dir().to_string_lossy().into_owned())
    }

    /// Stops the session; returns the summary as JSON. Remaining events stay readable.
    fn stop(&self, py: Python<'_>) -> PyResult<String> {
        let mut session = {
            let mut guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
            guard
                .take()
                .ok_or_else(|| SonicDiarError::new_err("the session has already stopped"))?
        };
        let summary = py.detach(|| session.stop()).map_err(err)?;
        json(&summary)
    }

    fn is_running(&self) -> bool {
        self.session
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }
}

#[pyfunction]
fn version() -> &'static str {
    sonicdiar::version()
}

/// Devices of kind "input" or "output", as JSON.
#[pyfunction]
fn list_devices(py: Python<'_>, kind: &str) -> PyResult<String> {
    let devices = py.detach(|| sonicdiar::list_devices(kind)).map_err(err)?;
    json(&devices)
}

/// Every known model with whether it is present in `models_dir`, as JSON.
#[pyfunction]
fn list_models(py: Python<'_>, models_dir: &str) -> PyResult<String> {
    let list = py.detach(|| sonicdiar::models::list(Path::new(models_dir)));
    json(&list)
}

/// Downloads and verifies a model group into `models_dir`; `progress(json)` is called for
/// each step.
#[pyfunction]
#[pyo3(signature = (models_dir, group, only=None, progress=None))]
fn fetch_models(
    py: Python<'_>,
    models_dir: &str,
    group: &str,
    only: Option<&str>,
    progress: Option<Py<PyAny>>,
) -> PyResult<()> {
    py.detach(|| {
        let mut report = |p: sonicdiar::models::Progress| {
            let Some(cb) = &progress else {
                return;
            };
            use sonicdiar::models::Progress as P;
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
            Python::attach(|py| {
                if let Err(e) = cb.call1(py, (v.to_string(),)) {
                    e.print(py);
                }
            });
        };
        sonicdiar::models::fetch(Path::new(models_dir), group, only, &mut report)
    })
    .map_err(err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeSession>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(list_devices, m)?)?;
    m.add_function(wrap_pyfunction!(list_models, m)?)?;
    m.add_function(wrap_pyfunction!(fetch_models, m)?)?;
    m.add("SonicDiarError", m.py().get_type::<SonicDiarError>())?;
    Ok(())
}
