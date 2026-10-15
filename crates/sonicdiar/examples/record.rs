//! Records for a few seconds and prints every event as JSON (one per line), then the summary.
//!
//! ```text
//! cargo run --release -p sonicdiar --example record -- 15 '{"transcription": {}}'
//! ```
//! Arguments: seconds (default 10) and a configuration as JSON (default `{}`).

use std::time::{Duration, Instant};

use sonicdiar::{Config, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .with_writer(std::io::stderr)
            .init();
    }
    let mut args = std::env::args().skip(1);
    let seconds: u64 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(10);
    let config = Config::from_json(&args.next().unwrap_or_else(|| "{}".into()))?;
    eprintln!("sonicdiar {}", sonicdiar::version());
    let mut session = Session::start(config)?;
    eprintln!(
        "recording {seconds} s into {}",
        session.journal_dir().display()
    );
    let until = Instant::now() + Duration::from_secs(seconds);
    let t0 = Instant::now();
    while Instant::now() < until {
        if let Some(event) = session.next_event(Duration::from_millis(200)) {
            eprint!("[{:6.2}s] ", t0.elapsed().as_secs_f64());
            println!("{}", serde_json::to_string(&event)?);
        }
    }
    eprintln!("[{:6.2}s] stopping", t0.elapsed().as_secs_f64());
    let summary = session.stop()?;
    eprintln!("[{:6.2}s] stopped", t0.elapsed().as_secs_f64());
    // Transcripts of speech that was still queued when the session stopped.
    while let Some(event) = session.next_event(Duration::ZERO) {
        println!("{}", serde_json::to_string(&event)?);
    }
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
