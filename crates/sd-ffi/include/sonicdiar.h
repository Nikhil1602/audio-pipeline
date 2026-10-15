/*
 * SonicDiar C API.
 *
 * Meeting audio capture: microphone and system audio, echo cancellation, voice activity
 * detection, live transcription and speaker labels, on Windows, macOS and Linux.
 *
 * Conventions
 *  - Strings are UTF-8 and NUL-terminated. Strings the library returns through `char **`
 *    out-parameters are owned by the caller and must be released with sd_string_free().
 *  - Functions return SD_OK (0) on success or a negative SD_ERR_* code; sd_last_error()
 *    then describes the most recent error on the calling thread.
 *  - Configurations, events, summaries, device and model lists are JSON; their fields are
 *    documented in docs/sdk.md. Events look like {"type": "final", "source": "mic", ...}.
 *  - Every function may be called from any thread. A session may be read with
 *    sd_session_next_event() on one thread while another thread calls sd_session_stop().
 *
 * Typical use
 *    sd_models_fetch("models", "core", NULL, NULL, NULL);   // once
 *    sd_session *s;
 *    sd_session_start("{\"transcription\": null}", &s);
 *    while (recording) {
 *        char *ev;
 *        if (sd_session_next_event(s, 200, &ev) == SD_OK && ev) { use(ev); sd_string_free(ev); }
 *    }
 *    char *summary;
 *    sd_session_stop(s, &summary);
 *    // Transcripts of speech still queued at the stop:
 *    char *ev;
 *    while (sd_session_next_event(s, 0, &ev) == SD_OK && ev) { use(ev); sd_string_free(ev); }
 *    sd_session_free(s);
 */
#ifndef SONICDIAR_H
#define SONICDIAR_H

#ifdef __cplusplus
extern "C" {
#endif

/* The version of this API's contract; bumped on incompatible changes. */
#define SD_API_VERSION 1

#define SD_OK 0
#define SD_ERR_ARGUMENT (-1) /* NULL pointer, invalid UTF-8, unknown value */
#define SD_ERR_CONFIG (-2)   /* invalid configuration (JSON or values) */
#define SD_ERR_ENGINE (-3)   /* the engine failed to start or stop */
#define SD_ERR_MODELS (-4)   /* downloading or verifying models failed */
#define SD_ERR_DEVICES (-5)  /* listing devices failed */
#define SD_ERR_STATE (-6)    /* e.g. stopping a session twice */
#define SD_ERR_PANIC (-7)    /* an internal error; the message says where */

typedef struct sd_session sd_session;

/* Progress of sd_models_fetch(): one JSON object per step, e.g.
 * {"step": "downloading", "id": "silero-vad-16k", "bytes": 1289603, ...}. */
typedef void (*sd_progress_fn)(const char *json, void *user);

/* The library version, e.g. "0.1.0". Static; do not free. */
const char *sd_version(void);
/* SD_API_VERSION of the library (compare with the header's). */
int sd_api_version(void);

/* The last error on this thread, or NULL. Free with sd_string_free(). */
char *sd_last_error(void);
void sd_string_free(char *s);

/* Devices of kind "input" (microphones) or "output" (system audio), as a JSON array. */
int sd_list_devices(const char *kind, char **json_out);

/* Downloads (and verifies) model group "core", "asr", "diar", "indic" or "all" - or only the
 * model `only` (may be NULL) - into models_dir. `progress` may be NULL. */
int sd_models_fetch(const char *models_dir, const char *group, const char *only,
                    sd_progress_fn progress, void *user);
/* Every known model with its size, licence and whether it is present, as a JSON array. */
int sd_models_list(const char *models_dir, char **json_out);

/* Starts a session from a JSON configuration ("{}" for the defaults). */
int sd_session_start(const char *config_json, sd_session **out);
/* The next event as JSON, waiting up to timeout_ms. *json_out is NULL if none arrived in time
 * (and, after sd_session_stop(), once every event was read). */
int sd_session_next_event(sd_session *s, int timeout_ms, char **json_out);
/* Switches source "mic" or "system" to a device id or "default". */
int sd_session_set_device(sd_session *s, const char *source, const char *device);
/* Pauses: audio is replaced by silence (nothing recorded or transcribed) until resumed; event
 * times keep counting from the start. */
int sd_session_pause(sd_session *s);
int sd_session_resume(sd_session *s);
/* The journal directory (the crash-safe recording), as a string. */
int sd_session_journal_dir(sd_session *s, char **path_out);
/* Stops capture and finishes transcribing queued speech; *summary_out (may be NULL) receives
 * the summary as JSON. The events produced meanwhile remain readable. */
int sd_session_stop(sd_session *s, char **summary_out);
/* Frees the session (stopping it first if needed). */
void sd_session_free(sd_session *s);

#ifdef __cplusplus
}
#endif

#endif /* SONICDIAR_H */
