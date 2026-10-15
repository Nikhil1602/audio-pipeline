/*
 * Smoke test of the C API: version, errors, device and model lists, and (with the argument
 * "record") a 3-second session without processing, printing every event.
 *
 *   cc -I crates/sd-ffi/include crates/sd-ffi/examples/smoke.c -L target/release -lsonicdiar
 */
#include <stdio.h>
#include <string.h>
#include "sonicdiar.h"

#ifdef _WIN32
#include <windows.h>
static void sleep_ms(int ms) { Sleep(ms); }
#else
#include <time.h>
static void sleep_ms(int ms) {
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&t, NULL);
}
#endif

static int failures = 0;

static void check(int ok, const char *what) {
    printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) failures++;
}

static void print_error(void) {
    char *e = sd_last_error();
    printf("     error: %s\n", e ? e : "(none)");
    sd_string_free(e);
}

int main(int argc, char **argv) {
    printf("SonicDiar %s, API %d (header %d)\n", sd_version(), sd_api_version(),
           SD_API_VERSION);
    check(sd_api_version() == SD_API_VERSION, "API version matches the header");

    /* Errors come back as codes with a message. */
    char *out = NULL;
    check(sd_list_devices("sideways", &out) == SD_ERR_CONFIG, "bad device kind is rejected");
    print_error();
    sd_session *s = NULL;
    check(sd_session_start("{\"sytem\": \"none\"}", &s) == SD_ERR_CONFIG,
          "unknown config field is rejected");
    print_error();

    /* Lists are JSON. */
    int rc = sd_models_list("models", &out);
    check(rc == SD_OK && out && out[0] == '[', "models list is a JSON array");
    sd_string_free(out);
    rc = sd_list_devices("output", &out);
    if (rc == SD_OK) {
        printf("     outputs: %.200s\n", out);
        sd_string_free(out);
    } else {
        printf("     (no device list here: %d)\n", rc);
        print_error();
    }

    if (argc > 1 && strcmp(argv[1], "record") == 0) {
        rc = sd_session_start("{\"processing\": false}", &s);
        check(rc == SD_OK, "session starts");
        if (rc != SD_OK) {
            print_error();
            return 1;
        }
        int started = 0;
        for (int i = 0; i < 15; i++) {
            char *ev = NULL;
            if (sd_session_next_event(s, 200, &ev) == SD_OK && ev) {
                printf("     event: %.160s\n", ev);
                if (strstr(ev, "\"device_started\"")) started++;
                sd_string_free(ev);
            }
        }
        sleep_ms(500);
        char *summary = NULL;
        rc = sd_session_stop(s, &summary);
        check(rc == SD_OK && summary, "session stops with a summary");
        if (summary) printf("     summary: %.300s\n", summary);
        sd_string_free(summary);
        check(sd_session_stop(s, NULL) == SD_ERR_STATE, "stopping twice is an error");
        char *ev = NULL;
        while (sd_session_next_event(s, 0, &ev) == SD_OK && ev) sd_string_free(ev);
        sd_session_free(s);
        check(started > 0, "a device started");
    }
    printf("%s\n", failures ? "SMOKE FAILED" : "SMOKE PASSED");
    return failures ? 1 : 0;
}
