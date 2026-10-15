//! `sdctl models fetch|verify|list`: download pinned model files and runtime libraries and check
//! their SHA-256 (see [`sonicdiar::models`]; the manifest is built into the library).

use std::path::PathBuf;
use std::process::ExitCode;

use sonicdiar::models::{self, Progress};

use crate::CliResult;

#[derive(clap::Subcommand)]
pub enum Cmd {
    /// Download missing or mismatching models (and this host's ONNX Runtime library).
    Fetch {
        #[arg(long, default_value = "models")]
        dir: PathBuf,
        /// Model group: "core" (what the pipeline needs; the default), "asr" (Whisper speech
        /// recognition models, large), "diar" (speaker diarization), "indic" (IndicConformer,
        /// 2.5 GB, gated: needs a Hugging Face token in HF_TOKEN or from `hf auth login`), or
        /// "all".
        #[arg(long, default_value = "core")]
        group: String,
        /// Only this model id (overrides --group).
        #[arg(long)]
        only: Option<String>,
    },
    /// Check every listed file's size and SHA-256.
    Verify {
        #[arg(long, default_value = "models")]
        dir: PathBuf,
        #[arg(long, default_value = "core")]
        group: String,
    },
    /// List every model with its size, licence and whether it is present.
    List {
        #[arg(long, default_value = "models")]
        dir: PathBuf,
    },
}

fn print(p: Progress) {
    match p {
        Progress::AlreadyPresent { id } => println!("  ok      {id:<20} already present"),
        Progress::Downloading {
            id,
            url,
            bytes,
            license,
        } => println!("  fetch   {id:<20} {url} ({bytes} bytes, {license})"),
        Progress::Verified { sha256, .. } => println!("          verified sha256 {sha256}"),
        Progress::BundleFiles { done, total, .. } => {
            println!("          {done}/{total} files verified")
        }
    }
}

pub fn run(cmd: Cmd) -> CliResult {
    match cmd {
        Cmd::Fetch { dir, group, only } => {
            models::fetch(&dir, &group, only.as_deref(), &mut print)?;
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Verify { dir, group } => {
            let mut all = true;
            for (id, ok) in models::verify(&dir, &group) {
                all &= ok;
                println!("  {} {id}", if ok { "ok     " } else { "MISSING" });
            }
            Ok(if all {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        Cmd::List { dir } => {
            for m in models::list(&dir) {
                println!(
                    "  {} {:<28} {:<8} {:>8.1} MB  {}",
                    if m.present { "ok     " } else { "-      " },
                    m.id,
                    m.group,
                    m.size as f64 / 1e6,
                    m.license
                );
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}
