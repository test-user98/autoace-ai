/** Mirrors the envelope built by `run_batch_remote` in modal_app.py. */

export const SCHEMA_FIELDS = [
  "emotional_tone",
  "emotional_intensity",
  "background_noise_present",
  "background_noise_type",
  "background_noise_severity",
  "audio_quality",
  "speaker_overlap_present",
  "long_silence_present",
  "confidence",
] as const;

export type SchemaField = (typeof SCHEMA_FIELDS)[number];

export type Prediction = Record<SchemaField, string | number | boolean>;

export type BatchReport = {
  ok: boolean;
  labeled: boolean;
  summary: string;
  item_count: number;
  missing_audio: string[];
  unmatched_audio: string[];
  unsupported: string[];
  duplicate_rows: string[];
  name_collisions: string[];
  bad_labels: { name: string; reason: string }[];
  errors: string[];
};

export type RowResult = {
  name: string;
  prediction: Prediction | null;
  evidence: Record<string, unknown>;
  expected: Record<string, unknown> | null;
  error: string | null;
  elapsed_s: number;
  duration_s: number;
};

export type RunEnvelope = {
  status: "done";
  system_version: string;
  report: BatchReport | null;
  rows: RowResult[];
  summary: {
    total: number;
    succeeded: number;
    failed: number;
    total_audio_s: number;
    total_elapsed_s: number;
    realtime_factor: number;
  };
  timings: {
    download_s: number;
    extract_s: number;
    run_batch_s: number;
    total_s: number;
    zip_bytes: number;
    first_input_on_container: boolean;
    container_idle_before_input_s: number;
    gpu: string;
  };
  csv: string;
  json: string;
};

export type StatusResponse =
  | RunEnvelope
  | { status: "running"; call_id: string }
  | { status: "error"; error: string }
  | { error: string };
