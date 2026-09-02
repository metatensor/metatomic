// Strings, errors, and units
// ==========================
//
// The C API uses a few small helpers that show up in almost every integration:
// heap-allocated strings (:c:type:`mta_string_t`), a thread-local last-error
// slot, and unit conversion factors for engine/model unit mismatches.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

// %%
//
// Strings
// -------
//
// :c:func:`mta_string_create` copies a UTF-8 C string.
// :c:func:`mta_string_view` borrows it, and :c:func:`mta_string_free` releases
// it (``NULL`` is a no-op).

static int demo_strings(void) {
    mta_string_t hello = mta_string_create("Angstrom");
    if (hello == NULL) {
        return EXIT_FAILURE;
    }
    printf("string view: %s\n", mta_string_view(hello));
    mta_string_free(hello);
    mta_string_free(NULL);
    return EXIT_SUCCESS;
}

// %%
//
// Unit conversion
// ---------------
//
// :c:func:`mta_unit_conversion_factor` parses unit expressions and returns the
// multiplicative factor from ``from_unit`` to ``to_unit``. Dimensions must
// match; see :ref:`units` for the expression grammar.

static int demo_units(void) {
    double factor = 0.0;

    if (mta_unit_conversion_factor("m", "m", &factor) != MTA_SUCCESS) {
        return EXIT_FAILURE;
    }
    printf("m -> m: %.1f\n", factor);

    if (mta_unit_conversion_factor("kJ/mol", "eV", &factor) != MTA_SUCCESS) {
        return EXIT_FAILURE;
    }
    printf("kJ/mol -> eV: %.12f\n", factor);

    if (mta_unit_conversion_factor("Angstrom", "nm", &factor) != MTA_SUCCESS) {
        return EXIT_FAILURE;
    }
    printf("Angstrom -> nm: %.3f\n", factor);

    return EXIT_SUCCESS;
}

// %%
//
// Errors
// ------
//
// On failure, functions return a non-zero :c:type:`mta_status_t` and store a
// message that :c:func:`mta_last_error` can read. Plugins and models should
// call :c:func:`mta_set_last_error` before returning an error status.

static int demo_errors(void) {
    double factor = 0.0;
    mta_status_t status = mta_unit_conversion_factor("m", "kg", &factor);
    if (status == MTA_SUCCESS) {
        fprintf(stderr, "expected a dimension mismatch\n");
        return EXIT_FAILURE;
    }

    const char* message = NULL;
    const char* origin = NULL;
    mta_last_error(&message, &origin, NULL);
    printf("status: %d\n", (int)status);
    printf("error: %s\n", message != NULL ? message : "(none)");

    mta_set_last_error(
        "tutorial-triggered error",
        "demo_errors",
        NULL,
        NULL
    );
    message = NULL;
    origin = NULL;
    mta_last_error(&message, &origin, NULL);
    printf("custom error: %s (origin=%s)\n", message, origin);
    return EXIT_SUCCESS;
}

// %%
//
// Formatting metadata
// -------------------
//
// :c:func:`mta_format_metadata` turns a JSON :c:type:`ModelMetadata` document
// into a human-readable summary suitable for logging.

static int demo_format_metadata(void) {
    /* Use the published metatensor / metatomic citation — the same DOI as in
       CITATION.cff — so the formatted output is something you could actually
       paste into a paper or log. */
    const char* json =
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"tutorial-demo\","
        "\"authors\": ["
        "  \"Filippo Bigi\","
        "  \"Joseph W. Abbott\","
        "  \"Philip Loche\","
        "  \"et al.\""
        "],"
        "\"description\": "
        "\"Illustrative metadata for the C API strings tutorial, "
        "citing the metatensor and metatomic paper.\","
        "\"references\": {"
        "  \"model\": ["
        "    \"F. Bigi et al., J. Chem. Phys. 164, 064113 (2026), "
        "https://doi.org/10.1063/5.0304911\""
        "  ],"
        "  \"architecture\": ["
        "    \"metatensor and metatomic: foundational libraries for "
        "interoperable atomistic machine learning\""
        "  ],"
        "  \"implementation\": ["
        "    \"https://github.com/metatensor/metatomic\""
        "  ]"
        "},"
        "\"extra\": {"
        "  \"doi\": \"10.1063/5.0304911\""
        "}"
        "}";

    mta_string_t printed = NULL;
    if (mta_format_metadata(json, &printed) != MTA_SUCCESS) {
        const char* message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "format_metadata failed: %s\n", message);
        return EXIT_FAILURE;
    }
    printf("%s", mta_string_view(printed));
    mta_string_free(printed);
    return EXIT_SUCCESS;
}

// %%

int main(void) {
    if (demo_strings() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (demo_units() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (demo_errors() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (demo_format_metadata() != EXIT_SUCCESS) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

// %%
//
// Expected output
// ---------------
//
// ::
//
//     string view: Angstrom
//     m -> m: 1.0
//     kJ/mol -> eV: 0.010364269656
//     Angstrom -> nm: 0.100
//     status: 1
//     error: invalid parameter: dimension mismatch in unit conversion: 'm' has dimension [L] but 'kg' has dimension [M]
//     custom error: tutorial-triggered error (origin=demo_errors)
//     This is the tutorial-demo model
//     ===============================
//
//     Illustrative metadata for the C API strings tutorial, citing the metatensor and
//     metatomic paper.
//
//     Model authors
//     -------------
//
//     - Filippo Bigi
//     - Joseph W. Abbott
//     - Philip Loche
//     - et al.
//
//     Model references
//     ----------------
//
//     Please cite the following references when using this model:
//     - about this specific model:
//       * F. Bigi et al., J. Chem. Phys. 164, 064113 (2026),
//         https://doi.org/10.1063/5.0304911
//     - about the architecture of this model:
//       * metatensor and metatomic: foundational libraries for interoperable
//         atomistic machine learning
//     - about the implementation of this model:
//       * https://github.com/metatensor/metatomic
