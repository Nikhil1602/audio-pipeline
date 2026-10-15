'use strict';
/**
 * SonicDiar: meeting audio capture for Node.js and Electron.
 *
 * Microphone and system audio (what the computer plays) on Windows, macOS and Linux, with echo
 * cancellation, voice activity detection, live transcription in 15 languages and speaker
 * labels.
 *
 *   const sd = require('sonicdiar');
 *   await sd.fetchModels('models', 'core');          // once
 *   const session = new sd.Session({ transcription: null });
 *   for await (const event of session.events({ seconds: 10 })) console.log(event);
 *   console.log(session.summary);
 *
 * Events are objects with a `type` ("device_started", "speech", "partial", "final",
 * "speaker_revision", ...); see docs/sdk.md for every field.
 */
const path = require('path');

function loadNative() {
  const name = `sonicdiar.${process.platform}-${process.arch}.node`;
  const candidates = [path.join(__dirname, 'prebuilds', name), path.join(__dirname, name)];
  for (const file of candidates) {
    try {
      return require(file);
    } catch (e) {
      if (e.code !== 'MODULE_NOT_FOUND') throw e;
    }
  }
  throw new Error(
    `sonicdiar has no native build for ${process.platform}-${process.arch} ` +
      `(looked for ${candidates.join(', ')})`,
  );
}

const native = loadNative();

/** The library version. */
function version() {
  return native.version();
}

/** Audio devices of kind "input" (microphones) or "output" (system audio). */
function listDevices(kind = 'input') {
  return JSON.parse(native.listDevices(kind));
}

/** Every known model with its size, licence and whether it is present. */
function listModels(modelsDir = 'models') {
  return JSON.parse(native.listModels(String(modelsDir)));
}

/**
 * Downloads and verifies a model group ("core", "asr", "diar", "indic", "all") or one model
 * (`only`) into `modelsDir`. Resolves to the steps taken.
 */
async function fetchModels(modelsDir = 'models', group = 'core', only = undefined) {
  return JSON.parse(await native.fetchModels(String(modelsDir), group, only ?? null));
}

/** A recording session. `config` is an object (or JSON string); `{}` uses the defaults. */
class Session {
  constructor(config = {}) {
    const text = typeof config === 'string' ? config : JSON.stringify(config);
    this._native = new native.NativeSession(text);
    /** The summary, once stopped. */
    this.summary = null;
  }

  /** The crash-safe recording on disk. */
  get journalDir() {
    return this._native.journalDir();
  }

  get running() {
    return this._native.isRunning();
  }

  /** The next event, waiting up to `timeoutMs`; null if none arrived. */
  async nextEvent(timeoutMs = 200) {
    const s = await this._native.nextEvent(timeoutMs);
    return s === null ? null : JSON.parse(s);
  }

  /**
   * Yields events, for `seconds` (then stops the session) or until it is stopped elsewhere.
   * After a stop, the remaining events (transcripts of queued speech) are yielded too.
   */
  async *events({ seconds = undefined, timeoutMs = 200 } = {}) {
    const until = seconds === undefined ? Infinity : Date.now() + seconds * 1000;
    while (this.running && Date.now() < until) {
      const event = await this.nextEvent(timeoutMs);
      if (event !== null) yield event;
    }
    if (this.running) await this.stop();
    for (;;) {
      const event = await this.nextEvent(0);
      if (event === null) return;
      yield event;
    }
  }

  /** Switches "mic" or "system" to a device id (from listDevices) or "default". */
  setDevice(source, device) {
    this._native.setDevice(source, device);
  }

  /**
   * Pauses: audio is replaced by silence (nothing recorded or transcribed) until resume();
   * event times keep counting from the start.
   */
  pause() {
    this._native.pause();
  }

  /** Resumes after pause(). */
  resume() {
    this._native.resume();
  }

  /**
   * Stops capture and finishes transcribing queued speech (whose events stay readable with
   * nextEvent). Resolves to the summary.
   */
  async stop() {
    this.summary = JSON.parse(await this._native.stop());
    return this.summary;
  }
}

module.exports = { version, listDevices, listModels, fetchModels, Session };
