//! The native addon of the `sonicdiar` npm package. The JavaScript layer (`index.js`)
//! parses the JSON strings and offers an async iterator over events.
//!
//! Waiting for events, stopping and downloading run on libuv's thread pool (Promises), so the
//! JavaScript thread never blocks.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi_derive::napi;
use sonicdiar::{Config, Events, Session};

fn err(e: sonicdiar::Error) -> Error {
    Error::new(Status::GenericFailure, e.to_string())
}

fn json<T: serde::Serialize>(v: &T) -> Result<String> {
    serde_json::to_string(v).map_err(|e| Error::new(Status::GenericFailure, e.to_string()))
}

#[napi]
pub fn version() -> String {
    sonicdiar::version().to_string()
}

/// Devices of kind "input" or "output", as JSON.
#[napi]
pub fn list_devices(kind: String) -> Result<String> {
    json(&sonicdiar::list_devices(&kind).map_err(err)?)
}

/// Every known model with whether it is present in `models_dir`, as JSON.
#[napi]
pub fn list_models(models_dir: String) -> Result<String> {
    json(&sonicdiar::models::list(&PathBuf::from(models_dir)))
}

pub struct FetchModels {
    dir: PathBuf,
    group: String,
    only: Option<String>,
}

impl Task for FetchModels {
    /// The steps, as a JSON array.
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        use sonicdiar::models::Progress as P;
        let mut steps = Vec::new();
        sonicdiar::models::fetch(&self.dir, &self.group, self.only.as_deref(), &mut |p| {
            steps.push(match p {
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
            });
        })
        .map_err(err)?;
        json(&steps)
    }

    fn resolve(&mut self, _env: Env, output: String) -> Result<String> {
        Ok(output)
    }
}

/// Downloads and verifies a model group into `models_dir`; resolves to the steps as JSON.
#[napi(ts_return_type = "Promise<string>")]
pub fn fetch_models(
    models_dir: String,
    group: String,
    only: Option<String>,
) -> AsyncTask<FetchModels> {
    AsyncTask::new(FetchModels {
        dir: PathBuf::from(models_dir),
        group,
        only,
    })
}

pub struct NextEvent {
    events: Events,
    timeout: Duration,
}

impl Task for NextEvent {
    type Output = Option<String>;
    type JsValue = Option<String>;

    fn compute(&mut self) -> Result<Option<String>> {
        self.events.next(self.timeout).map(|e| json(&e)).transpose()
    }

    fn resolve(&mut self, _env: Env, output: Option<String>) -> Result<Option<String>> {
        Ok(output)
    }
}

pub struct Stop {
    session: Arc<Mutex<Option<Session>>>,
}

impl Task for Stop {
    type Output = String;
    type JsValue = String;

    fn compute(&mut self) -> Result<String> {
        let mut session = self
            .session
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .take()
            .ok_or_else(|| Error::new(Status::GenericFailure, "the session has already stopped"))?;
        json(&session.stop().map_err(err)?)
    }

    fn resolve(&mut self, _env: Env, output: String) -> Result<String> {
        Ok(output)
    }
}

/// A recording session (use `Session` from the package).
#[napi]
pub struct NativeSession {
    session: Arc<Mutex<Option<Session>>>,
    events: Events,
}

#[napi]
impl NativeSession {
    /// Starts a session from a JSON configuration. Loading speech models can take a few
    /// seconds; it happens here.
    #[napi(constructor)]
    pub fn new(config_json: String) -> Result<Self> {
        let cfg = Config::from_json(&config_json).map_err(err)?;
        let session = Session::start(cfg).map_err(err)?;
        Ok(Self {
            events: session.events(),
            session: Arc::new(Mutex::new(Some(session))),
        })
    }

    /// The next event as JSON, waiting up to `timeout_ms`; null if none arrived.
    #[napi(ts_return_type = "Promise<string | null>")]
    pub fn next_event(&self, timeout_ms: u32) -> AsyncTask<NextEvent> {
        AsyncTask::new(NextEvent {
            events: self.events.clone(),
            timeout: Duration::from_millis(timeout_ms as u64),
        })
    }

    #[napi]
    pub fn set_device(&self, source: String, device: String) -> Result<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "the session has stopped"))?;
        s.set_device(&source, &device).map_err(err)
    }

    #[napi]
    pub fn pause(&self) -> Result<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "the session has stopped"))?;
        s.pause().map_err(err)
    }

    #[napi]
    pub fn resume(&self) -> Result<()> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "the session has stopped"))?;
        s.resume().map_err(err)
    }

    #[napi]
    pub fn journal_dir(&self) -> Result<String> {
        let guard = self.session.lock().unwrap_or_else(|p| p.into_inner());
        let s = guard
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "the session has stopped"))?;
        Ok(s.journal_dir().to_string_lossy().into_owned())
    }

    /// Stops the session; resolves to the summary as JSON. Remaining events stay readable.
    #[napi(ts_return_type = "Promise<string>")]
    pub fn stop(&self) -> AsyncTask<Stop> {
        AsyncTask::new(Stop {
            session: self.session.clone(),
        })
    }

    #[napi]
    pub fn is_running(&self) -> bool {
        self.session
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_some()
    }
}
