//! Models and native runtimes: download, verify, list.
//!
//! Model weights and native runtimes are never committed or bundled. The manifest
//! (`models/manifest.json`, compiled into the library) pins each file to a release URL, size and
//! SHA-256 (runtimes: the upstream archive's hash *and* the extracted library's hash); anything
//! that does not match is rejected, not used and not kept. Multi-file models ("bundles") pin a
//! repository revision and list every file's size and hash in a lock file, also compiled in.
//!
//! Groups: `core` (what the pipeline needs: voice activity detection and this platform's ONNX
//! Runtime), `asr` (Whisper speech recognition models, large), `diar` (speaker diarization),
//! `indic` (IndicConformer for 11 Indian languages, 2.5 GB, gated on Hugging Face), `all`.

use std::io::Read;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

const MANIFEST: &str = include_str!("../../../models/manifest.json");
/// Bundle lock files referenced by the manifest, by name.
const LOCKS: &[(&str, &str)] = &[(
    "indicconformer.lock.json",
    include_str!("../../../models/indicconformer.lock.json"),
)];

#[derive(Deserialize)]
struct Manifest {
    models: Vec<Entry>,
    #[serde(default)]
    runtimes: Vec<Runtime>,
    #[serde(default)]
    bundles: Vec<Bundle>,
}

/// A model made of many files from one Hugging Face repository revision.
#[derive(Deserialize)]
struct Bundle {
    id: String,
    group: String,
    /// Lock file listing the files.
    lock: String,
    /// Where the files go, relative to the models directory.
    dir: String,
    license: String,
    /// The repository is gated: downloads need the user's Hugging Face token.
    #[serde(default)]
    gated: bool,
}

#[derive(Deserialize)]
struct BundleLock {
    repo: String,
    revision: String,
    subdir: String,
    files: Vec<LockedFile>,
}

#[derive(Deserialize)]
struct LockedFile {
    file: String,
    size: u64,
    sha256: String,
}

impl Bundle {
    fn lock(&self) -> Result<BundleLock> {
        let text = LOCKS
            .iter()
            .find(|(name, _)| *name == self.lock)
            .map(|(_, text)| *text)
            .ok_or_else(|| Error::Models(format!("lock file {} is not built in", self.lock)))?;
        serde_json::from_str(text).map_err(|e| Error::Models(format!("{}: {e}", self.lock)))
    }
}

#[derive(Deserialize)]
struct Entry {
    id: String,
    #[serde(default = "core_group")]
    group: String,
    file: String,
    url: String,
    sha256: String,
    size: u64,
    license: String,
}

/// A native runtime library shipped inside an upstream release archive.
#[derive(Deserialize)]
struct Runtime {
    id: String,
    os: String,
    arch: String,
    archive_url: String,
    archive_sha256: String,
    archive_size: u64,
    member: String,
    file: String,
    sha256: String,
    size: u64,
    license: String,
}

impl Runtime {
    fn for_this_host(rts: &[Runtime]) -> Option<&Runtime> {
        rts.iter()
            .find(|r| r.os == std::env::consts::OS && r.arch == std::env::consts::ARCH)
    }
}

fn core_group() -> String {
    "core".into()
}

fn manifest() -> Manifest {
    serde_json::from_str(MANIFEST).expect("the built-in manifest is valid JSON")
}

/// One downloadable item, as listed by [`list`].
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    /// `core`, `asr`, `diar`, `indic`, or `runtime`.
    pub group: String,
    /// File (or, for bundles, directory) inside the models directory.
    pub file: String,
    pub size: u64,
    pub license: String,
    /// Present and verified (size and SHA-256).
    pub present: bool,
}

/// What [`fetch`] is doing, for progress reporting.
#[derive(Debug, Clone, PartialEq)]
pub enum Progress {
    AlreadyPresent {
        id: String,
    },
    Downloading {
        id: String,
        url: String,
        bytes: u64,
        license: String,
    },
    Verified {
        id: String,
        sha256: String,
    },
    BundleFiles {
        id: String,
        done: usize,
        total: usize,
    },
}

fn selected(e: &Entry, group: &str, only: Option<&str>) -> bool {
    match only {
        Some(id) => e.id == id,
        None => group == "all" || e.group == group,
    }
}

/// Lists every item in the manifest (and this platform's runtime) with whether it is present.
pub fn list(dir: &Path) -> Vec<ModelInfo> {
    let m = manifest();
    let mut out: Vec<ModelInfo> = m
        .models
        .iter()
        .map(|e| ModelInfo {
            id: e.id.clone(),
            group: e.group.clone(),
            file: e.file.clone(),
            size: e.size,
            license: e.license.clone(),
            present: file_ok(dir, &e.file, e.size, &e.sha256),
        })
        .collect();
    for b in &m.bundles {
        if let Ok(lock) = b.lock() {
            let bdir = dir.join(&b.dir);
            out.push(ModelInfo {
                id: b.id.clone(),
                group: b.group.clone(),
                file: b.dir.clone(),
                size: lock.files.iter().map(|f| f.size).sum(),
                license: b.license.clone(),
                present: lock
                    .files
                    .iter()
                    .all(|f| file_ok(&bdir, &f.file, f.size, &f.sha256)),
            });
        }
    }
    if let Some(r) = Runtime::for_this_host(&m.runtimes) {
        out.push(ModelInfo {
            id: r.id.clone(),
            group: "runtime".into(),
            file: r.file.clone(),
            size: r.size,
            license: r.license.clone(),
            present: file_ok(dir, &r.file, r.size, &r.sha256),
        });
    }
    out
}

/// Downloads the missing or mismatching files of `group` (or only `only`, an id), and this
/// platform's ONNX Runtime library, into `dir`. Every file is checked against the manifest.
pub fn fetch(
    dir: &Path,
    group: &str,
    only: Option<&str>,
    progress: &mut dyn FnMut(Progress),
) -> Result<()> {
    std::fs::create_dir_all(dir).map_err(|e| Error::Models(format!("{}: {e}", dir.display())))?;
    let m = manifest();
    for e in m.models.iter().filter(|e| selected(e, group, only)) {
        if file_ok(dir, &e.file, e.size, &e.sha256) {
            progress(Progress::AlreadyPresent { id: e.id.clone() });
            continue;
        }
        progress(Progress::Downloading {
            id: e.id.clone(),
            url: e.url.clone(),
            bytes: e.size,
            license: e.license.clone(),
        });
        download_file(dir, &e.id, &e.file, &e.url, e.size, &e.sha256, None)?;
        progress(Progress::Verified {
            id: e.id.clone(),
            sha256: e.sha256.clone(),
        });
    }
    for b in m
        .bundles
        .iter()
        .filter(|b| only.map_or(group == "all" || b.group == group, |id| b.id == id))
    {
        fetch_bundle(dir, b, progress)?;
    }
    if let Some(r) = Runtime::for_this_host(&m.runtimes) {
        if file_ok(dir, &r.file, r.size, &r.sha256) {
            progress(Progress::AlreadyPresent { id: r.id.clone() });
        } else {
            progress(Progress::Downloading {
                id: r.id.clone(),
                url: r.archive_url.clone(),
                bytes: r.archive_size,
                license: r.license.clone(),
            });
            let archive = download(&r.archive_url, r.archive_size)?;
            let got = sha256_bytes(&archive);
            if archive.len() as u64 != r.archive_size || got != r.archive_sha256 {
                return Err(Error::Models(format!(
                    "{}: archive sha256 {got} ({} bytes) does not match the manifest",
                    r.id,
                    archive.len()
                )));
            }
            let lib = extract(&archive, &r.archive_url, &r.member)?;
            let got = sha256_bytes(&lib);
            if lib.len() as u64 != r.size || got != r.sha256 {
                return Err(Error::Models(format!(
                    "{}: extracted {} has sha256 {got}, not the manifest's",
                    r.id, r.member
                )));
            }
            install(dir, &r.file, &lib).map_err(|e| Error::Models(format!("{}: {e}", r.file)))?;
            progress(Progress::Verified {
                id: r.id.clone(),
                sha256: got,
            });
        }
    }
    Ok(())
}

/// Checks every file of `group` and this platform's runtime: (id, present and verified).
pub fn verify(dir: &Path, group: &str) -> Vec<(String, bool)> {
    let wanted = |g: &str| group == "all" || g == group || g == "runtime";
    list(dir)
        .into_iter()
        .filter(|i| wanted(&i.group))
        .map(|i| (i.id, i.present))
        .collect()
}

/// The user's Hugging Face token: `HF_TOKEN`, else the file `hf auth login` writes
/// (`$HF_HOME/token`, default `~/.cache/huggingface/token`). It is only ever sent to
/// huggingface.co, and not forwarded on redirects (ureq's default).
fn hf_token() -> Option<String> {
    if let Ok(t) = std::env::var("HF_TOKEN") {
        return Some(t.trim().to_string()).filter(|t| !t.is_empty());
    }
    let home = std::env::var_os("HF_HOME").map(PathBuf::from).or_else(|| {
        std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
            .map(|h| PathBuf::from(h).join(".cache").join("huggingface"))
    })?;
    let t = std::fs::read_to_string(home.join("token")).ok()?;
    Some(t.trim().to_string()).filter(|t| !t.is_empty())
}

fn fetch_bundle(models: &Path, b: &Bundle, progress: &mut dyn FnMut(Progress)) -> Result<()> {
    let lock = b.lock()?;
    let dir = models.join(&b.dir);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Models(format!("{}: {e}", dir.display())))?;
    let missing: Vec<&LockedFile> = lock
        .files
        .iter()
        .filter(|f| !file_ok(&dir, &f.file, f.size, &f.sha256))
        .collect();
    if missing.is_empty() {
        progress(Progress::AlreadyPresent { id: b.id.clone() });
        return Ok(());
    }
    progress(Progress::Downloading {
        id: b.id.clone(),
        url: format!("https://huggingface.co/{}@{}", lock.repo, lock.revision),
        bytes: missing.iter().map(|f| f.size).sum(),
        license: b.license.clone(),
    });
    let token = if b.gated {
        Some(hf_token().ok_or_else(|| {
            Error::Models(format!(
                "{} is gated on Hugging Face: accept its terms at https://huggingface.co/{}, then \
                 set HF_TOKEN or run `hf auth login`",
                b.id, lock.repo
            ))
        })?)
    } else {
        None
    };
    for (i, f) in missing.iter().enumerate() {
        let url = format!(
            "https://huggingface.co/{}/resolve/{}/{}/{}",
            lock.repo, lock.revision, lock.subdir, f.file
        );
        download_file(
            &dir,
            &b.id,
            &f.file,
            &url,
            f.size,
            &f.sha256,
            token.as_deref(),
        )?;
        if (i + 1) % 50 == 0 || i + 1 == missing.len() {
            progress(Progress::BundleFiles {
                id: b.id.clone(),
                done: i + 1,
                total: missing.len(),
            });
        }
    }
    Ok(())
}

fn net(e: impl std::fmt::Display) -> Error {
    Error::Models(e.to_string())
}

/// Streams `url` into `<dir>/<file>.part` while hashing, checks size and SHA-256, then renames
/// it into place. Large models never sit in memory, and a failed download leaves no file.
fn download_file(
    dir: &Path,
    id: &str,
    file: &str,
    url: &str,
    size: u64,
    sha256: &str,
    bearer: Option<&str>,
) -> Result<()> {
    use std::io::Write;
    let part = dir.join(format!("{file}.part"));
    let mut req = ureq::get(url);
    if let Some(t) = bearer {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let mut body = req.call().map_err(net)?.into_body();
    let mut reader = body.as_reader();
    let mut f = std::fs::File::create(&part).map_err(net)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n_total = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(net)?;
        if n == 0 {
            break;
        }
        n_total += n as u64;
        if n_total > size {
            break;
        }
        h.update(&buf[..n]);
        f.write_all(&buf[..n]).map_err(net)?;
    }
    f.sync_all().map_err(net)?;
    drop(f);
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if n_total != size || got != sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(Error::Models(format!(
            "{id}: {file}: downloaded {n_total} bytes with sha256 {got}; expected {size} bytes, {sha256}"
        )));
    }
    std::fs::rename(&part, dir.join(file)).map_err(net)?;
    Ok(())
}

fn sha256_bytes(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_file(p: &Path) -> std::io::Result<String> {
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

fn file_ok(dir: &Path, file: &str, size: u64, sha256: &str) -> bool {
    let p = dir.join(file);
    p.metadata().is_ok_and(|m| m.len() == size) && sha256_file(&p).is_ok_and(|h| h == sha256)
}

fn download(url: &str, max: u64) -> Result<Vec<u8>> {
    let mut body = ureq::get(url).call().map_err(net)?.into_body();
    body.with_config().limit(max + 1).read_to_vec().map_err(net)
}

/// Writes atomically (temp file + rename) so a crash never leaves a partial file in place.
fn install(dir: &Path, file: &str, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = dir.join(format!("{file}.part"));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, dir.join(file))
}

/// Extracts one member from a .zip or .tgz archive held in memory.
fn extract(archive: &[u8], url: &str, member: &str) -> Result<Vec<u8>> {
    if url.ends_with(".zip") {
        let mut z = zip::ZipArchive::new(std::io::Cursor::new(archive)).map_err(net)?;
        let mut f = z.by_name(member).map_err(net)?;
        let mut out = Vec::with_capacity(f.size() as usize);
        std::io::copy(&mut f, &mut out).map_err(net)?;
        return Ok(out);
    }
    let mut t = tar::Archive::new(flate2::read::GzDecoder::new(archive));
    for e in t.entries().map_err(net)? {
        let mut e = e.map_err(net)?;
        if e.path().map_err(net)?.to_string_lossy() == member {
            let mut out = Vec::new();
            std::io::copy(&mut e, &mut out).map_err(net)?;
            return Ok(out);
        }
    }
    Err(Error::Models(format!("{member} not found in {url}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_built_in_manifest_parses_and_has_this_platforms_runtime() {
        let m = manifest();
        assert!(m.models.iter().any(|e| e.id == "silero-vad-16k"));
        for b in &m.bundles {
            b.lock().expect("bundle lock is built in");
        }
        let host = (std::env::consts::OS, std::env::consts::ARCH);
        if matches!(
            host,
            ("windows", "x86_64")
                | ("linux", "x86_64")
                | ("macos", "aarch64")
                | ("macos", "x86_64")
        ) {
            assert!(Runtime::for_this_host(&m.runtimes).is_some());
        }
    }

    #[test]
    fn lists_missing_files_as_absent() {
        let items = list(Path::new("this/does/not/exist"));
        assert!(!items.is_empty());
        assert!(items.iter().all(|i| !i.present));
    }
}
