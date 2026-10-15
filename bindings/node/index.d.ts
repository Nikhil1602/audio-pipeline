/** sonicdiar: meeting audio capture. See docs/sdk.md for every configuration field and event. */

export type SourceName = 'mic' | 'system';

export interface Device {
  id: string;
  name: string;
  kind: 'input' | 'output';
  is_default: boolean;
  sample_rate: number | null;
  channels: number | null;
}

export interface ModelInfo {
  id: string;
  group: string;
  file: string;
  size: number;
  license: string;
  present: boolean;
}

export interface Transcription {
  language?: string;
  whisper_model?: string | null;
  indic?: boolean;
  indic_lm?: string | null;
  partials?: boolean;
  numbers_as_digits?: boolean;
  gpu?: boolean;
  gpu_device?: number;
  threads?: number | null;
}

export interface Config {
  models_dir?: string;
  journal_dir?: string | null;
  mic?: string;
  system?: string;
  processing?: boolean;
  echo_cancellation?: 'auto' | 'on' | 'off';
  gain_control?: boolean;
  transcription?: Transcription | null;
  diarization?: { overlap?: boolean } | null;
  levels?: boolean;
  health?: boolean;
}

export interface Word { start: number; end: number; text: string }
export interface Part { start: number; end: number; speaker: string; text: string }
export interface Overlap { start: number; end: number; speaker: string }

export type Event =
  | { type: 'device_started'; source: SourceName; device_id: string; device_name: string }
  | { type: 'device_switching'; source: SourceName; reason: string }
  | { type: 'no_device'; source: SourceName }
  | { type: 'permission_needed'; source: SourceName; what: string; confirmed: boolean }
  | { type: 'capture_problem'; source: SourceName; message: string }
  | { type: 'level'; source: SourceName; rms_dbfs: number; peak_dbfs: number }
  | { type: 'speech'; source: SourceName; start: number; end: number }
  | { type: 'partial'; source: SourceName; utterance: number; start: number; end: number; text: string; language: string }
  | {
      type: 'final';
      source: SourceName;
      utterance: number;
      start: number;
      end: number;
      text: string;
      language: string;
      language_probabilities: [string, number][];
      words: Word[];
      speaker: string | null;
      parts: Part[];
      overlaps: Overlap[];
      echo_unverified: boolean;
    }
  | { type: 'speaker_revision'; source: SourceName; utterance: number; speaker: string; parts: Part[]; overlaps: Overlap[] }
  | { type: 'bleed_removed'; source: SourceName; utterance: number; start: number; end: number; text: string }
  | { type: 'transcription_skipped'; source: SourceName; utterance: number; start: number; end: number; reason: string }
  | { type: 'audio_dropped'; source: SourceName; frames: number }
  | { type: 'disk_space_low'; free_bytes: number }
  | { type: 'journal_error'; message: string; disk_full: boolean }
  | { type: 'journal_recovered' }
  | { type: 'journal_paused'; free_bytes: number }
  | { type: 'journal_resumed'; free_bytes: number }
  | { type: 'warning'; message: string }
  | { type: 'health'; elapsed: number; sources: SourceHealth[] };

export interface SourceHealth {
  source: SourceName;
  timeline: number;
  dropped_frames: number;
  device_glitches: number;
  restarts: number;
  drift_ppm: number | null;
}

export interface Summary {
  journal_dir: string;
  duration: number;
  sources: SourceHealth[];
  journaled_frames: number;
  journal_errors: number;
  complete: boolean;
}

export function version(): string;
export function listDevices(kind?: 'input' | 'output'): Device[];
export function listModels(modelsDir?: string): ModelInfo[];
export function fetchModels(modelsDir?: string, group?: string, only?: string): Promise<object[]>;

export class Session {
  constructor(config?: Config | string);
  readonly journalDir: string;
  readonly running: boolean;
  summary: Summary | null;
  nextEvent(timeoutMs?: number): Promise<Event | null>;
  events(options?: { seconds?: number; timeoutMs?: number }): AsyncGenerator<Event>;
  setDevice(source: SourceName, device: string): void;
  pause(): void;
  resume(): void;
  stop(): Promise<Summary>;
}
