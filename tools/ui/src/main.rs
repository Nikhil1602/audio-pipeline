//! `sonicdiar-ui`: a web page on this computer for trying the SDK. Start, pause, resume and stop a
//! session, pick the language and devices, and watch the live transcript with speaker labels.
//!
//! The server listens on 127.0.0.1 only. Every API call must carry the token embedded in the
//! page and a local `Host` header, so other web sites cannot drive it (a cross-site request
//! cannot send the custom header without a CORS preflight, which is never granted) and DNS
//! rebinding is refused.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::{Cursor, Read};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sonicdiar::{Config, Event, Events, Session, Summary, models};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

const PAGE: &str = include_str!("page.html");
/// How long a poll waits for new events.
const POLL_WAIT: Duration = Duration::from_millis(250);
const MAX_BODY: u64 = 1 << 20;

#[derive(Parser)]
#[command(
    name = "sonicdiar-ui",
    version,
    about = "A web page for trying SonicDiar"
)]
struct Args {
    /// Models directory; the page can download what is missing into it.
    #[arg(long, default_value = "models")]
    models: PathBuf,
    /// Port on 127.0.0.1 (0: any free port).
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Do not open the page in the browser.
    #[arg(long)]
    no_browser: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Phase {
    Idle,
    Starting,
    Recording,
    Paused,
    Stopping,
}

struct State {
    phase: Phase,
    /// Increments with every session; the page starts a new transcript when it changes.
    session: u64,
    /// The current session's events in order (levels and health are kept as latest values).
    log: Vec<Value>,
    levels: serde_json::Map<String, Value>,
    health: Value,
    started: Option<Instant>,
    /// The clock stops at the stop request.
    frozen: Option<f64>,
    downloading: Option<String>,
    download_error: Option<String>,
    /// Increments when a download finishes, so the page reloads the model list.
    models_rev: u64,
    /// Hands the stop result to the running session's event pump.
    stopper: Option<mpsc::Sender<Result<Summary, String>>>,
}

struct App {
    models: PathBuf,
    token: String,
    hosts: [String; 2],
    state: Mutex<State>,
    changed: Condvar,
    session: Mutex<Option<Session>>,
}

impl App {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn session(&self) -> MutexGuard<'_, Option<Session>> {
        self.session.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn elapsed(st: &State) -> f64 {
        st.frozen
            .unwrap_or_else(|| st.started.map_or(0.0, |t| t.elapsed().as_secs_f64()))
    }

    /// Appends a page-level event (`ui_*`) to the session log.
    fn note(&self, st: &mut State, kind: &str, mut fields: Value) {
        if let Value::Object(m) = &mut fields {
            m.insert("type".into(), kind.into());
        }
        st.log.push(fields);
        self.changed.notify_all();
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    let server = match Server::http(("127.0.0.1", args.port)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "cannot listen on 127.0.0.1:{}: {e} (another sonicdiar-ui running? try --port 0)",
                args.port
            );
            return ExitCode::FAILURE;
        }
    };
    let port = server.server_addr().to_ip().map_or(args.port, |a| a.port());
    let app = Arc::new(App {
        models: args.models.clone(),
        token: token(),
        hosts: [format!("127.0.0.1:{port}"), format!("localhost:{port}")],
        state: Mutex::new(State {
            phase: Phase::Idle,
            session: 0,
            log: Vec::new(),
            levels: Default::default(),
            health: Value::Null,
            started: None,
            frozen: None,
            downloading: None,
            download_error: None,
            models_rev: 0,
            stopper: None,
        }),
        changed: Condvar::new(),
        session: Mutex::new(None),
    });
    let url = format!("http://127.0.0.1:{port}/");
    println!("SonicDiar {} test page: {url}", sonicdiar::version());
    println!("models directory: {}", args.models.display());
    println!("Press Ctrl+C to quit.");
    if !args.no_browser {
        open_browser(&url);
    }
    for rq in server.incoming_requests() {
        let app = app.clone();
        std::thread::spawn(move || handle(&app, rq));
    }
    ExitCode::SUCCESS
}

/// 128 random bits from the standard library's per-process random hasher keys.
fn token() -> String {
    let mut out = String::new();
    for i in 0..2u64 {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(i);
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos()),
        );
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

fn open_browser(url: &str) {
    let r = if cfg!(windows) {
        std::process::Command::new("cmd")
            .args(["/C", "start", "", url])
            .spawn()
    } else if cfg!(target_os = "macos") {
        std::process::Command::new("open").arg(url).spawn()
    } else {
        std::process::Command::new("xdg-open").arg(url).spawn()
    };
    if r.is_err() {
        println!("Open {url} in a browser.");
    }
}

type Reply = Response<Cursor<Vec<u8>>>;

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("valid header")
}

fn reply(status: u16, content_type: &str, body: Vec<u8>) -> Reply {
    Response::from_data(body)
        .with_status_code(StatusCode(status))
        .with_header(header("Content-Type", content_type))
        .with_header(header("Cache-Control", "no-store"))
        .with_header(header("X-Content-Type-Options", "nosniff"))
}

fn json_reply(status: u16, v: &Value) -> Reply {
    reply(status, "application/json", v.to_string().into_bytes())
}

fn error(status: u16, message: impl std::fmt::Display) -> Reply {
    json_reply(status, &json!({ "error": message.to_string() }))
}

fn header_value<'a>(rq: &'a Request, name: &'static str) -> Option<&'a str> {
    rq.headers()
        .iter()
        .find(|h| h.field.equiv(name))
        .map(|h| h.value.as_str())
}

fn handle(app: &Arc<App>, mut rq: Request) {
    let local = header_value(&rq, "Host").is_some_and(|h| app.hosts.iter().any(|a| a == h));
    let resp = if !local {
        error(403, "not a local request")
    } else {
        let url = rq.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let path = path.to_string();
        let query = query.to_string();
        if *rq.method() == Method::Get && (path == "/" || path == "/index.html") {
            let page = PAGE.replace("__SD_TOKEN__", &app.token);
            reply(200, "text/html; charset=utf-8", page.into_bytes()).with_header(header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
                 connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'",
            ))
        } else if !path.starts_with("/api/") {
            error(404, "not found")
        } else if header_value(&rq, "X-SD-Token") != Some(app.token.as_str()) {
            error(403, "missing or wrong token (reload the page)")
        } else {
            let mut body = String::new();
            let read = rq
                .as_reader()
                .take(MAX_BODY)
                .read_to_string(&mut body)
                .is_ok();
            if !read {
                error(400, "unreadable request body")
            } else {
                api(app, rq.method().clone(), &path, &query, &body)
            }
        }
    };
    let _ = rq.respond(resp);
}

fn api(app: &Arc<App>, method: Method, path: &str, query: &str, body: &str) -> Reply {
    match (method, path) {
        (Method::Get, "/api/info") => info(app),
        (Method::Get, "/api/poll") => poll(app, query),
        (Method::Post, "/api/start") => start(app, body),
        (Method::Post, "/api/pause") => pause(app, true),
        (Method::Post, "/api/resume") => pause(app, false),
        (Method::Post, "/api/stop") => stop(app),
        (Method::Post, "/api/device") => device(app, body),
        (Method::Post, "/api/models/fetch") => fetch(app, body),
        _ => error(404, "no such API call"),
    }
}

fn info(app: &App) -> Reply {
    let devices = |kind| match sonicdiar::list_devices(kind) {
        Ok(d) => json!(d),
        Err(e) => json!({ "error": e.to_string() }),
    };
    json_reply(
        200,
        &json!({
            "version": sonicdiar::version(),
            "models_dir": std::path::absolute(&app.models).unwrap_or(app.models.clone()),
            "models": models::list(&app.models),
            "languages": sonicdiar::languages(),
            "inputs": devices("input"),
            "outputs": devices("output"),
            "defaults": Config::default(),
            "transcription_defaults": sonicdiar::Transcription::default(),
        }),
    )
}

fn query_u64(query: &str, key: &str) -> Option<u64> {
    query
        .split('&')
        .filter_map(|kv| kv.split_once('='))
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.parse().ok())
}

/// The events after `after` of session `session` (all of them if the session changed),
/// waiting up to [`POLL_WAIT`] for some to arrive, plus the current status.
fn poll(app: &App, query: &str) -> Reply {
    let session = query_u64(query, "session");
    let after = query_u64(query, "after").unwrap_or(0) as usize;
    let mut st = app.state();
    let fresh = |st: &State| session != Some(st.session) || st.log.len() > after;
    if !fresh(&st) {
        st = app
            .changed
            .wait_timeout_while(st, POLL_WAIT, |st| !fresh(st))
            .unwrap_or_else(|p| p.into_inner())
            .0;
    }
    let from = if session == Some(st.session) {
        after.min(st.log.len())
    } else {
        0
    };
    json_reply(
        200,
        &json!({
            "session": st.session,
            "phase": st.phase,
            "elapsed": App::elapsed(&st),
            "from": from,
            "events": &st.log[from..],
            "levels": st.levels,
            "health": st.health,
            "downloading": st.downloading,
            "download_error": st.download_error,
            "models_rev": st.models_rev,
        }),
    )
}

fn start(app: &Arc<App>, body: &str) -> Reply {
    let mut cfg = match Config::from_json(if body.trim().is_empty() { "{}" } else { body }) {
        Ok(c) => c,
        Err(e) => return error(400, e),
    };
    cfg.models_dir = app.models.clone();
    cfg.levels = true;
    cfg.health = true;
    let id = {
        let mut st = app.state();
        if st.phase != Phase::Idle {
            return error(409, "a session is already running");
        }
        st.phase = Phase::Starting;
        st.session += 1;
        st.log.clear();
        st.levels.clear();
        st.health = Value::Null;
        st.started = None;
        st.frozen = None;
        app.changed.notify_all();
        st.session
    };
    let app = app.clone();
    std::thread::spawn(move || {
        let loaded = Session::start(cfg);
        let mut st = app.state();
        match loaded {
            Ok(s) => {
                let events = s.events();
                *app.session() = Some(s);
                let (tx, rx) = mpsc::channel();
                st.stopper = Some(tx);
                st.phase = Phase::Recording;
                st.started = Some(Instant::now());
                app.note(&mut st, "ui_started", json!({ "at": 0.0 }));
                drop(st);
                let pump_app = app.clone();
                std::thread::spawn(move || pump(&pump_app, id, events, rx));
            }
            Err(e) => {
                st.phase = Phase::Idle;
                app.note(&mut st, "ui_error", json!({ "message": e.to_string() }));
            }
        }
    });
    json_reply(200, &json!({ "session": id }))
}

/// Moves the session's events into the log until the session has stopped and every event
/// was read; then adds the summary and returns to idle.
fn pump(app: &App, id: u64, events: Events, stopped: mpsc::Receiver<Result<Summary, String>>) {
    let take = |e: Event| {
        let mut st = app.state();
        if st.session != id {
            return;
        }
        match &e {
            Event::Level {
                source,
                rms_dbfs,
                peak_dbfs,
            } => {
                if let Ok(Value::String(k)) = serde_json::to_value(source) {
                    st.levels.insert(k, json!([rms_dbfs, peak_dbfs]));
                }
            }
            Event::Health { .. } => {
                st.health = serde_json::to_value(&e).unwrap_or(Value::Null);
            }
            _ => {
                if let Ok(v) = serde_json::to_value(&e) {
                    st.log.push(v);
                    app.changed.notify_all();
                }
            }
        }
    };
    loop {
        if let Some(e) = events.next(Duration::from_millis(200)) {
            take(e);
            continue;
        }
        let Ok(result) = stopped.try_recv() else {
            continue;
        };
        // Stopped: everything the session produced is queued; read it all.
        while let Some(e) = events.next(Duration::ZERO) {
            take(e);
        }
        let mut st = app.state();
        match result {
            Ok(summary) => {
                let at = App::elapsed(&st);
                app.note(
                    &mut st,
                    "ui_summary",
                    json!({ "at": at, "summary": summary }),
                );
            }
            Err(e) => app.note(&mut st, "ui_error", json!({ "message": e })),
        }
        st.phase = Phase::Idle;
        app.changed.notify_all();
        return;
    }
}

fn pause(app: &App, paused: bool) -> Reply {
    let mut st = app.state();
    let ok_phase = if paused {
        Phase::Recording
    } else {
        Phase::Paused
    };
    if st.phase != ok_phase {
        return error(409, format!("cannot do that while {:?}", st.phase));
    }
    let result = match app.session().as_ref() {
        Some(s) if paused => s.pause(),
        Some(s) => s.resume(),
        None => return error(409, "no session"),
    };
    if let Err(e) = result {
        return error(500, e);
    }
    st.phase = if paused {
        Phase::Paused
    } else {
        Phase::Recording
    };
    let at = App::elapsed(&st);
    app.note(
        &mut st,
        if paused { "ui_paused" } else { "ui_resumed" },
        json!({ "at": at }),
    );
    json_reply(200, &json!({ "phase": st.phase }))
}

fn stop(app: &Arc<App>) -> Reply {
    let mut st = app.state();
    if !matches!(st.phase, Phase::Recording | Phase::Paused) {
        return error(409, format!("cannot stop while {:?}", st.phase));
    }
    let Some(mut session) = app.session().take() else {
        return error(409, "no session");
    };
    st.phase = Phase::Stopping;
    let at = App::elapsed(&st);
    st.frozen = Some(at);
    app.note(&mut st, "ui_stopping", json!({ "at": at }));
    let tx = st.stopper.take();
    drop(st);
    std::thread::spawn(move || {
        let result = session.stop().map_err(|e| e.to_string());
        if let Some(tx) = tx {
            let _ = tx.send(result);
        }
    });
    json_reply(200, &json!({ "phase": Phase::Stopping }))
}

#[derive(Deserialize)]
struct DeviceRequest {
    source: String,
    device: String,
}

fn device(app: &App, body: &str) -> Reply {
    let rq: DeviceRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return error(400, e),
    };
    match app.session().as_ref() {
        Some(s) => match s.set_device(&rq.source, &rq.device) {
            Ok(()) => json_reply(200, &json!({})),
            Err(e) => error(400, e),
        },
        None => error(409, "no session"),
    }
}

#[derive(Deserialize)]
struct FetchRequest {
    id: String,
}

fn fetch(app: &Arc<App>, body: &str) -> Reply {
    let rq: FetchRequest = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => return error(400, e),
    };
    {
        let mut st = app.state();
        if st.downloading.is_some() {
            return error(409, "a download is already running");
        }
        st.downloading = Some(rq.id.clone());
        st.download_error = None;
        app.changed.notify_all();
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let result = models::fetch(&app.models, "all", Some(&rq.id), &mut |p| {
            if let models::Progress::Downloading { id, .. } = p {
                app.state().downloading = Some(id);
            }
        });
        let mut st = app.state();
        st.downloading = None;
        st.download_error = result.err().map(|e| e.to_string());
        st.models_rev += 1;
        app.changed.notify_all();
    });
    json_reply(200, &json!({}))
}
