/*
 * A force_t backend driven entirely through metatomic's C API
 * (metatomic-core, see include/metatomic_force.h).
 *
 * This is a learning exercise: everything here is written from scratch
 * against `external/metatomic/metatomic-core/include/metatomic.h`, closely
 * following the patterns shown in metatomic's own C API tutorials
 * (examples/c/1-create-system.c, 2-add-model.c, 2-using-system.c,
 * 3-add-plugin.c on the metatomic-core branch), but is NOT a copy of any of
 * them -- in particular the tutorial's shifted-LJ model only computes
 * energy; this one also fills in the "positions" gradient block (i.e.
 * forces), since symd actually needs them to integrate the equations of
 * motion.
 *
 * Two things happen here:
 *
 *  1. A toy shifted Lennard-Jones `mta_model_t` is implemented and
 *     registered in-process as a metatomic plugin (`mtm_lj_*` functions,
 *     `build_metatomic`). Its math is intentionally the same shifted LJ
 *     symd's own `src/lj_force.c` implements, so the two can be diffed
 *     directly on identical input -- see NOTES-metatomic.md.
 *
 *  2. `metatomic_gather_forces` plays the role of the *engine*: it wraps
 *     symd's positions into an `mta_system_t`, attaches a pair list built
 *     from symd's own neighbor list (`nlist_parameters_t`, the same one
 *     `build_nlj` uses), calls `mta_execute_model` (unit conversion and
 *     consistency checking on top of the model's `execute_inner`), and
 *     copies the resulting energy/forces back into symd's arrays.
 */

#include "metatomic_force.h"
#include "mtm_simple_array.h"

#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <metatomic.h>

/* All atoms are tagged with this atomic type; the toy model does not care
 * about chemistry, only about distances. */
#define MTM_ATOMIC_TYPE 12

typedef struct {
    double sigma;
    double epsilon;
    double cutoff; /* = sqrt(nlist->rcut), the *true* LJ cutoff (not the
                       larger cell-list skin radius) */
    double shift;  /* energy shift so the pair term is exactly 0 at cutoff */
} MtmLennardJones;

static double mtm_lj_shift(double cutoff, double sigma, double epsilon) {
    double sigma_rc = sigma / cutoff;
    double sigma_rc_6 = sigma_rc * sigma_rc * sigma_rc;
    sigma_rc_6 *= sigma_rc_6;
    return 4.0 * epsilon * (sigma_rc_6 * sigma_rc_6 - sigma_rc_6);
}

/* Pair energy and the force it exerts *on the first atom* of the pair
 * (`force_on_i`), given the displacement vector `d = pos_j - pos_i`.
 * The force on the second atom is simply `-force_on_i` (Newton's third
 * law) -- this matches how symd's own `lj()` helper in src/lj_force.c is
 * used, but is derived independently here rather than copied from it. */
static void mtm_lj_pair(
    double dx, double dy, double dz,
    const MtmLennardJones *lj,
    double *energy,
    double force_on_i[3]
) {
    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 <= 0.0 || r2 >= lj->cutoff * lj->cutoff) {
        *energy = 0.0;
        force_on_i[0] = force_on_i[1] = force_on_i[2] = 0.0;
        return;
    }

    double inv2 = (lj->sigma * lj->sigma) / r2;
    double inv6 = inv2 * inv2 * inv2;
    double inv12 = inv6 * inv6;
    *energy = 4.0 * lj->epsilon * (inv12 - inv6) - lj->shift;

    /* dE/d(r^2) = (12 * epsilon / r^2) * (inv6 - 2 * inv12); the shift is a
     * constant and drops out. Force on atom i is -dE/d(pos_i), and since
     * r^2 = |pos_j - pos_i|^2, d(r^2)/d(pos_i) = -2 * d, giving
     * F_i = 2 * dE/d(r^2) * d = (24 * epsilon / r^2) * (inv6 - 2 * inv12) * d. */
    double dedr2 = (12.0 * lj->epsilon / r2) * (inv6 - 2.0 * inv12);
    force_on_i[0] = 2.0 * dedr2 * dx;
    force_on_i[1] = 2.0 * dedr2 * dy;
    force_on_i[2] = 2.0 * dedr2 * dz;
}

/* Pair-list cutoffs in JSON use the IEEE-754 bit pattern as a hex string
 * (see metatomic's :ref:`core-json-pair-options`), so the engine sees the
 * exact `double`. */
static void mtm_format_f64_hex(double value, char *buf, size_t n) {
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    snprintf(buf, n, "0x%" PRIx64, bits);
}

static void mtm_format_pair_options(const MtmLennardJones *lj, char *buf, size_t n) {
    char cutoff_hex[32];
    mtm_format_f64_hex(lj->cutoff, cutoff_hex, sizeof(cutoff_hex));
    snprintf(
        buf,
        n,
        "{"
        "\"type\": \"metatomic_pair_options\","
        "\"cutoff\": \"%s\","
        "\"full_list\": false,"
        "\"strict\": true,"
        "\"requestors\": [\"symd-lennard-jones\"]"
        "}",
        cutoff_hex
    );
}

/* ------------------------------------------------------------------ */
/* mta_model_t callbacks                                              */
/* ------------------------------------------------------------------ */

static mta_status_t mtm_lj_unload(void *model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

static mta_status_t mtm_lj_metadata(const void *model_data, mta_string_t *out) {
    (void)model_data;
    *out = mta_string_create(
        "{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"symd-lennard-jones\","
        "\"authors\": [\"symd metatomic C API exercise\"],"
        "\"description\": \"Shifted Lennard-Jones potential, wired up as a "
        "metatomic model to drive symd's MD loop through the C API\","
        "\"references\": {\"model\": [], \"architecture\": [], \"implementation\": []},"
        "\"extra\": {\"potential\": \"shifted-lennard-jones\"}"
        "}"
    );
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t mtm_lj_capabilities(const void *model_data, mta_string_t *out) {
    const MtmLennardJones *lj = (const MtmLennardJones *)model_data;
    char json[512];
    snprintf(
        json,
        sizeof(json),
        "{"
        "\"type\": \"metatomic_model_capabilities\","
        "\"outputs\": [{"
        "  \"type\": \"metatomic_quantity\","
        "  \"name\": \"energy\","
        "  \"unit\": \"eV\","
        "  \"gradients\": [\"positions\"],"
        "  \"sample_kind\": \"system\""
        "}],"
        "\"atomic_types\": [%d],"
        "\"interaction_range\": %.17g,"
        "\"length_unit\": \"Angstrom\","
        "\"supported_devices\": [\"cpu\"],"
        "\"dtype\": \"float64\""
        "}",
        MTM_ATOMIC_TYPE,
        lj->cutoff
    );
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t mtm_lj_requested_pair_lists(const void *model_data, mta_string_t *out) {
    const MtmLennardJones *lj = (const MtmLennardJones *)model_data;
    char object[512];
    char json[520];
    mtm_format_pair_options(lj, object, sizeof(object));
    snprintf(json, sizeof(json), "[%s]", object);
    *out = mta_string_create(json);
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t mtm_lj_requested_inputs(const void *model_data, mta_string_t *out) {
    (void)model_data;
    *out = mta_string_create("[]");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

static mta_status_t mtm_fail_mts(const char *what, const char *origin) {
    const char *message = NULL;
    mts_last_error(&message, NULL, NULL);
    mta_set_last_error(message != NULL ? message : what, origin, NULL, NULL);
    return MTA_METATENSOR_ERROR;
}

/* Read a metatensor block's data array as a flat, contiguous float64
 * buffer. `*view` must be released with `(*view)->deleter(*view)` once
 * `*data` is no longer needed. */
static mta_status_t mtm_block_f64_view(
    const mts_block_t *block,
    const double **data,
    const uintptr_t **shape,
    uintptr_t *ndim,
    DLManagedTensorVersioned **view
) {
    mts_array_t array;
    memset(&array, 0, sizeof(array));
    if (mts_block_data((mts_block_t *)block, &array) != MTS_SUCCESS) {
        return mtm_fail_mts("failed to read block values", "mtm_block_f64_view");
    }
    if (array.shape == NULL || array.as_dlpack == NULL) {
        return mtm_fail_mts("block array is missing shape/as_dlpack", "mtm_block_f64_view");
    }
    if (array.shape(array.ptr, shape, ndim) != MTS_SUCCESS) {
        return mtm_fail_mts("failed to get block shape", "mtm_block_f64_view");
    }

    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    *view = NULL;
    if (array.as_dlpack(array.ptr, view, cpu, NULL, version) != MTS_SUCCESS || *view == NULL) {
        return mtm_fail_mts("failed to export block values via DLPack", "mtm_block_f64_view");
    }
    if ((*view)->dl_tensor.dtype.code != kDLFloat || (*view)->dl_tensor.dtype.bits != 64) {
        (*view)->deleter(*view);
        *view = NULL;
        mta_set_last_error("block values must be float64", "mtm_block_f64_view", NULL, NULL);
        return MTA_DLPACK_ERROR;
    }

    *data = (const double *)((char *)(*view)->dl_tensor.data + (*view)->dl_tensor.byte_offset);
    return MTA_SUCCESS;
}

/* Build the "positions" gradient block: samples = [sample, system, atom]
 * (one row per atom, all referring to value-row 0 / system 0), components =
 * [xyz], properties = "energy" (matching the parent block). Values are
 * -force (see docs/src/quantities/energy.rst: "'positions' gradients will
 * contain the negative of the forces"). */
static mts_block_t *mtm_positions_gradient(const double *forces, uintptr_t n_atoms) {
    int32_t *sample_values = malloc(sizeof(int32_t) * n_atoms * 3);
    if (sample_values == NULL) {
        return NULL;
    }
    for (uintptr_t j = 0; j < n_atoms; j++) {
        sample_values[3 * j + 0] = 0;         /* sample: row in the energy block */
        sample_values[3 * j + 1] = 0;         /* system */
        sample_values[3 * j + 2] = (int32_t)j; /* atom */
    }
    const char *sample_names[] = {"sample", "system", "atom"};
    const mts_labels_t *samples = mtm_labels(sample_names, 3, sample_values, n_atoms);
    free(sample_values);
    if (samples == NULL) {
        return NULL;
    }

    int32_t xyz_values[] = {0, 1, 2};
    const char *xyz_name[] = {"xyz"};
    const mts_labels_t *xyz = mtm_labels(xyz_name, 1, xyz_values, 3);
    if (xyz == NULL) {
        mts_labels_free(samples);
        return NULL;
    }

    int32_t energy_value = 0;
    const char *energy_name[] = {"energy"};
    const mts_labels_t *properties = mtm_labels(energy_name, 1, &energy_value, 1);
    if (properties == NULL) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        return NULL;
    }

    double *data = malloc(sizeof(double) * n_atoms * 3);
    if (data == NULL) {
        mts_labels_free(samples);
        mts_labels_free(xyz);
        mts_labels_free(properties);
        return NULL;
    }
    for (uintptr_t j = 0; j < n_atoms; j++) {
        for (int k = 0; k < 3; k++) {
            data[j * 3 + k] = -forces[j * 3 + k];
        }
    }

    uintptr_t shape[3] = {n_atoms, 3, 1};
    mts_array_t values = mtm_array_nd(data, shape, 3, kDLFloat, 64);
    const mts_labels_t *components[] = {xyz};
    mts_block_t *gradient = mts_block(values, samples, components, 1, properties);
    if (gradient == NULL && values.destroy != NULL) {
        values.destroy(values.ptr);
    }
    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(properties);
    return gradient;
}

/* Build the "energy" output TensorMap: a single block with one "system"
 * sample and one "energy" property, plus a "positions" gradient holding
 * `-forces`. Mirrors the tutorial's `scalar_tensormap`, extended with the
 * gradient the tutorial itself doesn't compute. */
static mts_tensormap_t *mtm_energy_tensormap(double energy, const double *forces, uintptr_t n_atoms) {
    double *value = malloc(sizeof(double));
    if (value == NULL) {
        return NULL;
    }
    *value = energy;

    const mts_labels_t *samples = mtm_labels_single_zero("system");
    const mts_labels_t *properties = mtm_labels_single_zero("energy");
    const mts_labels_t *keys = mtm_labels_single_zero("_");
    if (samples == NULL || properties == NULL || keys == NULL) {
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        free(value);
        return NULL;
    }

    mts_array_t values = mtm_array_2d(value, 1, 1, kDLFloat, 64);
    mts_block_t *block = mts_block(values, samples, NULL, 0, properties);
    if (block == NULL) {
        if (values.destroy != NULL) {
            values.destroy(values.ptr);
        }
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        return NULL;
    }

    mts_block_t *gradient = mtm_positions_gradient(forces, n_atoms);
    if (gradient == NULL || mts_block_add_gradient(block, "positions", gradient) != MTS_SUCCESS) {
        mts_block_free(gradient);
        mts_block_free(block);
        mts_labels_free(samples);
        mts_labels_free(properties);
        mts_labels_free(keys);
        return NULL;
    }

    mts_block_t *blocks[] = {block};
    mts_tensormap_t *tensor = mts_tensormap(keys, blocks, 1);
    mts_labels_free(samples);
    mts_labels_free(properties);
    mts_labels_free(keys);
    if (tensor == NULL) {
        mts_block_free(block);
    }
    return tensor;
}

/* This toy model only ever supports exactly one system at a time -- that is
 * all `metatomic_gather_forces` below ever asks for, and it keeps the
 * per-atom sample bookkeeping in `mtm_positions_gradient` simple. */
static mta_status_t mtm_lj_execute_inner(
    void *model_data,
    const mta_system_t *const *systems,
    uintptr_t systems_count,
    const mts_labels_t *selected_atoms,
    const char *requested_outputs_json,
    mts_tensormap_t **outputs,
    uintptr_t outputs_count
) {
    (void)selected_atoms;
    (void)requested_outputs_json;

    if (model_data == NULL) {
        mta_set_last_error("model_data is NULL", "mtm_lj_execute_inner", NULL, NULL);
        return MTA_INVALID_PARAMETER_ERROR;
    }
    if (systems_count != 1 || systems == NULL || systems[0] == NULL) {
        mta_set_last_error(
            "this toy model only supports exactly one system per call",
            "mtm_lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }
    if (outputs_count > 0 && outputs == NULL) {
        mta_set_last_error("outputs is NULL but outputs_count is not 0", "mtm_lj_execute_inner", NULL, NULL);
        return MTA_INVALID_PARAMETER_ERROR;
    }

    const MtmLennardJones *lj = (const MtmLennardJones *)model_data;
    const mta_system_t *system = systems[0];

    uintptr_t n_atoms = 0;
    if (mta_system_size(system, &n_atoms) != MTA_SUCCESS) {
        return mtm_fail_mts("mta_system_size failed", "mtm_lj_execute_inner");
    }

    char options[512];
    mtm_format_pair_options(lj, options, sizeof(options));
    const mts_block_t *pairs = NULL;
    mta_status_t status = mta_system_get_pairs(system, options, &pairs);
    if (status != MTA_SUCCESS) {
        /* A missing neighbor list is an error, not a silent zero energy. */
        return status;
    }

    /* pair displacement vectors, shape [n_pairs, 3, 1] */
    const double *disp = NULL;
    const uintptr_t *disp_shape = NULL;
    uintptr_t disp_ndim = 0;
    DLManagedTensorVersioned *disp_view = NULL;
    status = mtm_block_f64_view(pairs, &disp, &disp_shape, &disp_ndim, &disp_view);
    if (status != MTA_SUCCESS) {
        return status;
    }
    if (disp_ndim != 3 || disp_shape[1] != 3 || disp_shape[2] != 1) {
        disp_view->deleter(disp_view);
        mta_set_last_error(
            "pair list values must have shape [n_pairs, 3, 1]",
            "mtm_lj_execute_inner",
            NULL,
            NULL
        );
        return MTA_INVALID_PARAMETER_ERROR;
    }
    uintptr_t n_pairs = disp_shape[0];

    /* pair atom indices: samples are ["first_atom", "second_atom", ...] */
    const mts_labels_t *pair_samples = mts_block_labels(pairs, 0);
    if (pair_samples == NULL) {
        disp_view->deleter(disp_view);
        return mtm_fail_mts("failed to get pair list samples", "mtm_lj_execute_inner");
    }
    const int32_t *sample_values = NULL;
    uintptr_t sample_count = 0;
    uintptr_t sample_size = 0;
    if (mts_labels_values_cpu(pair_samples, &sample_values, &sample_count, &sample_size) != MTS_SUCCESS
        || sample_count != n_pairs || sample_size < 2) {
        mts_labels_free(pair_samples);
        disp_view->deleter(disp_view);
        return mtm_fail_mts("failed to read pair list sample values", "mtm_lj_execute_inner");
    }

    double *forces = calloc(n_atoms * 3, sizeof(double));
    if (forces == NULL) {
        mts_labels_free(pair_samples);
        disp_view->deleter(disp_view);
        mta_set_last_error("out of memory", "mtm_lj_execute_inner", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }

    double energy = 0.0;
    for (uintptr_t p = 0; p < n_pairs; p++) {
        double dx = disp[3 * p + 0];
        double dy = disp[3 * p + 1];
        double dz = disp[3 * p + 2];

        double pair_energy;
        double force_on_i[3];
        mtm_lj_pair(dx, dy, dz, lj, &pair_energy, force_on_i);
        energy += pair_energy;

        int32_t i = sample_values[p * sample_size + 0];
        int32_t j = sample_values[p * sample_size + 1];
        for (int k = 0; k < 3; k++) {
            forces[(uintptr_t)i * 3 + k] += force_on_i[k];
            forces[(uintptr_t)j * 3 + k] -= force_on_i[k];
        }
    }

    mts_labels_free(pair_samples);
    disp_view->deleter(disp_view);

    for (uintptr_t idx = 0; idx < outputs_count; idx++) {
        outputs[idx] = mtm_energy_tensormap(energy, forces, n_atoms);
        if (outputs[idx] == NULL) {
            for (uintptr_t prev = 0; prev < idx; prev++) {
                mts_tensormap_free(outputs[prev]);
                outputs[prev] = NULL;
            }
            free(forces);
            return mtm_fail_mts("failed to build energy TensorMap", "mtm_lj_execute_inner");
        }
    }

    free(forces);
    return MTA_SUCCESS;
}

static mta_status_t mtm_lj_load_model(
    const char *load_from,
    const char *options_json,
    mta_model_t *model
) {
    (void)options_json;
    if (strcmp(load_from, "symd-lennard-jones") != 0) {
        return MTA_MODEL_NOT_SUPPORTED_ERROR;
    }

    MtmLennardJones *data = malloc(sizeof(MtmLennardJones));
    if (data == NULL) {
        mta_set_last_error("out of memory", "mtm_lj_load_model", NULL, NULL);
        return MTA_INTERNAL_ERROR;
    }
    /* filled in by build_metatomic once `data` is set on the model */
    data->sigma = 0.0;
    data->epsilon = 0.0;
    data->cutoff = 0.0;
    data->shift = 0.0;

    model->data = data;
    model->unload = mtm_lj_unload;
    model->capabilities = mtm_lj_capabilities;
    model->metadata = mtm_lj_metadata;
    model->requested_pair_lists = mtm_lj_requested_pair_lists;
    model->requested_inputs = mtm_lj_requested_inputs;
    model->execute_inner = mtm_lj_execute_inner;
    return MTA_SUCCESS;
}

static const char *MTM_PLUGIN_NAME = "symd-metatomic-lj";

static mta_status_t mtm_plugin_load_model(
    const char *load_from,
    const char *options_json,
    struct mta_model_t *model
) {
    return mtm_lj_load_model(load_from, options_json, model);
}

/* ------------------------------------------------------------------ */
/* force_t plumbing                                                    */
/* ------------------------------------------------------------------ */

typedef struct {
    mta_model_t model;
    nlist_parameters_t *nlist;
} metatomic_parameters_t;

static double metatomic_gather_forces(run_params_t *params, double *positions, double *forces) {
    unsigned int n_dims = N_DIMS;
    unsigned int n_particles = params->n_particles;
    unsigned int n_total = n_particles + params->n_ghost_particles;
    force_t *force_p = params->force_parameters;
    metatomic_parameters_t *mp = (metatomic_parameters_t *)force_p->parameters;
    nlist_parameters_t *nlist = mp->nlist;

    update_nlist(positions, params->box->box_size, n_dims, n_particles, params->n_ghost_particles, nlist);
    double rcut2 = nlist->rcut; /* already squared, see build_nlist_params */

    /* Collect the pairs within the *true* cutoff (the cell/Verlet list
     * itself uses a slightly larger skin radius) -- same filtering
     * nlj_gather_forces does in src/lj_force.c. */
    unsigned int max_pairs = 0;
    for (unsigned int i = 0; i < n_particles; i++) {
        max_pairs += nlist->nlist_count[i];
    }
    int32_t *pair_atoms = malloc(sizeof(int32_t) * 2 * (max_pairs > 0 ? max_pairs : 1));
    double *pair_disp = malloc(sizeof(double) * 3 * (max_pairs > 0 ? max_pairs : 1));

    unsigned int n_pairs = 0;
    unsigned int offset = 0;
    for (unsigned int i = 0; i < n_particles; i++) {
        for (unsigned int n = offset; n - offset < nlist->nlist_count[i]; n++) {
            unsigned int j = nlist->nlist[n];
            double disp[3] = {0.0, 0.0, 0.0};
            double r2 = 0.0;
            for (unsigned int k = 0; k < n_dims; k++) {
                double diff = positions[j * n_dims + k] - positions[i * n_dims + k];
                disp[k] = diff;
                r2 += diff * diff;
            }
            if (r2 <= rcut2) {
                pair_atoms[2 * n_pairs + 0] = (int32_t)i;
                pair_atoms[2 * n_pairs + 1] = (int32_t)j;
                pair_disp[3 * n_pairs + 0] = disp[0];
                pair_disp[3 * n_pairs + 1] = disp[1];
                pair_disp[3 * n_pairs + 2] = disp[2];
                n_pairs++;
            }
        }
        offset += nlist->nlist_count[i];
    }

    /* Wrap symd's own positions (real + ghost images, exactly as `lj`/`nlj`
     * see them) into an `mta_system_t`. We hand metatomic a flat,
     * non-periodic system: symd already materializes periodic images as
     * separate "ghost" atoms, so there is no periodicity left for
     * metatomic to handle itself. */
    double *positions_3d = malloc(sizeof(double) * n_total * 3);
    int32_t *types = malloc(sizeof(int32_t) * n_total);
    for (unsigned int a = 0; a < n_total; a++) {
        for (unsigned int k = 0; k < 3; k++) {
            positions_3d[a * 3 + k] = (k < n_dims) ? positions[a * n_dims + k] : 0.0;
        }
        types[a] = MTM_ATOMIC_TYPE;
    }
    double cell[9] = {0};
    bool pbc[3] = {false, false, false};

    int64_t pos_shape[2] = {(int64_t)n_total, 3};
    int64_t types_shape[1] = {(int64_t)n_total};
    int64_t cell_shape[2] = {3, 3};
    int64_t pbc_shape[1] = {3};

    DLManagedTensorVersioned *positions_tensor =
        mtm_dlpack_view(positions_3d, 2, pos_shape, (DLDataType){kDLFloat, 64, 1});
    DLManagedTensorVersioned *types_tensor =
        mtm_dlpack_view(types, 1, types_shape, (DLDataType){kDLInt, 32, 1});
    DLManagedTensorVersioned *cell_tensor =
        mtm_dlpack_view(cell, 2, cell_shape, (DLDataType){kDLFloat, 64, 1});
    DLManagedTensorVersioned *pbc_tensor =
        mtm_dlpack_view(pbc, 1, pbc_shape, (DLDataType){kDLBool, 8, 1});

    mta_system_t *system = NULL;
    mta_status_t status =
        mta_system_create("Angstrom", types_tensor, positions_tensor, cell_tensor, pbc_tensor, &system);
    /* `mta_system_create` takes ownership of the DLPack *views* (it will
     * call their deleter), but our `mtm_dlpack_view` deleter only releases
     * the view's own bookkeeping (shape/strides), not the underlying
     * buffer -- see the comment in mtm_simple_array.h. `positions_3d` and
     * `types` are still ours to free, whether or not system creation
     * succeeded. `cell`/`pbc` are stack arrays, nothing to free. */
    if (status != MTA_SUCCESS) {
        const char *message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "mta_system_create failed: %s\n", message != NULL ? message : "(no message)");
        free(positions_3d);
        free(types);
        free(pair_atoms);
        free(pair_disp);
        return 0.0;
    }

    /* Attach the pair list. */
    const char *sample_names[] = {
        "first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"
    };
    int32_t *pair_samples = malloc(sizeof(int32_t) * 5 * (n_pairs > 0 ? n_pairs : 1));
    for (unsigned int p = 0; p < n_pairs; p++) {
        pair_samples[5 * p + 0] = pair_atoms[2 * p + 0];
        pair_samples[5 * p + 1] = pair_atoms[2 * p + 1];
        pair_samples[5 * p + 2] = 0;
        pair_samples[5 * p + 3] = 0;
        pair_samples[5 * p + 4] = 0;
    }
    const mts_labels_t *psamples = mtm_labels(sample_names, 5, pair_samples, n_pairs > 0 ? n_pairs : 0);
    free(pair_samples);

    int32_t xyz_values[] = {0, 1, 2};
    const char *xyz_name[] = {"xyz"};
    const mts_labels_t *pxyz = mtm_labels(xyz_name, 1, xyz_values, 3);
    int32_t distance_value = 0;
    const char *distance_name[] = {"distance"};
    const mts_labels_t *pdistance = mtm_labels(distance_name, 1, &distance_value, 1);

    double *disp_data = malloc(sizeof(double) * 3 * (n_pairs > 0 ? n_pairs : 1));
    memcpy(disp_data, pair_disp, sizeof(double) * 3 * n_pairs);
    uintptr_t disp_shape[3] = {n_pairs, 3, 1};
    mts_array_t disp_array = mtm_array_nd(disp_data, disp_shape, 3, kDLFloat, 64);
    const mts_labels_t *pcomponents[] = {pxyz};
    mts_block_t *pair_block = mts_block(disp_array, psamples, pcomponents, 1, pdistance);
    if (pair_block == NULL && disp_array.destroy != NULL) {
        disp_array.destroy(disp_array.ptr);
    }
    mts_labels_free(psamples);
    mts_labels_free(pxyz);
    mts_labels_free(pdistance);
    free(pair_atoms);
    free(pair_disp);

    char options[512];
    MtmLennardJones *lj = (MtmLennardJones *)mp->model.data;
    mtm_format_pair_options(lj, options, sizeof(options));
    status = mta_system_add_pairs(system, options, pair_block);
    if (status != MTA_SUCCESS) {
        mta_system_free(system);
        free(positions_3d);
        free(types);
        return 0.0;
    }

    mts_tensormap_t *output = NULL;
    const mta_system_t *systems[1] = {system};
    /* mta_execute_model handles unit conversion + consistency checking on
     * top of the model's execute_inner -- route through it like a real
     * engine would, rather than calling execute_inner directly. */
    static const char *requested_outputs = "[{"
        "\"type\": \"metatomic_quantity\","
        "\"name\": \"energy\","
        "\"unit\": \"eV\","
        "\"gradients\": [\"positions\"],"
        "\"sample_kind\": \"system\""
        "}]";
    status = mta_execute_model(mp->model, systems, 1, NULL, requested_outputs, true, &output, 1);
    if (status != MTA_SUCCESS || output == NULL) {
        const char *message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "metatomic model execution failed: %s\n", message != NULL ? message : "(no message)");
        mta_system_free(system);
        free(positions_3d);
        free(types);
        return 0.0;
    }

    /* Read the energy value and the "positions" gradient back out. */
    double energy = 0.0;
    mts_block_t *block = NULL;
    if (mts_tensormap_block_by_id(output, &block, 0) == MTS_SUCCESS && block != NULL) {
        const double *values = NULL;
        const uintptr_t *shape = NULL;
        uintptr_t ndim = 0;
        DLManagedTensorVersioned *view = NULL;
        if (mtm_block_f64_view(block, &values, &shape, &ndim, &view) == MTA_SUCCESS) {
            energy = values[0];
            view->deleter(view);
        }

        mts_block_t *gradient = NULL;
        if (mts_block_gradient(block, "positions", &gradient) == MTS_SUCCESS && gradient != NULL) {
            const double *grad_values = NULL;
            const uintptr_t *grad_shape = NULL;
            uintptr_t grad_ndim = 0;
            DLManagedTensorVersioned *grad_view = NULL;
            if (mtm_block_f64_view(gradient, &grad_values, &grad_shape, &grad_ndim, &grad_view) == MTA_SUCCESS) {
                const mts_labels_t *grad_samples = mts_block_labels(gradient, 0);
                const int32_t *grad_sample_values = NULL;
                uintptr_t grad_sample_count = 0;
                uintptr_t grad_sample_size = 0;
                if (grad_samples != NULL
                    && mts_labels_values_cpu(
                           grad_samples, &grad_sample_values, &grad_sample_count, &grad_sample_size
                       ) == MTS_SUCCESS) {
                    /* zero real particles' forces before accumulating */
                    for (unsigned int a = 0; a < n_particles; a++) {
                        for (unsigned int k = 0; k < n_dims; k++) {
                            forces[a * n_dims + k] = 0.0;
                        }
                    }
                    for (uintptr_t r = 0; r < grad_sample_count; r++) {
                        int32_t atom = grad_sample_values[r * grad_sample_size + 2];
                        if ((unsigned int)atom >= n_particles) {
                            continue; /* ghost image, matches lj/nlj dropping ghost forces */
                        }
                        for (unsigned int k = 0; k < n_dims; k++) {
                            /* gradient = -force */
                            forces[(unsigned int)atom * n_dims + k] -= grad_values[r * 3 + k];
                        }
                    }
                }
                if (grad_samples != NULL) {
                    mts_labels_free(grad_samples);
                }
                grad_view->deleter(grad_view);
            }
        }
    }

    mts_tensormap_free(output);
    mta_system_free(system);
    free(positions_3d);
    free(types);

    return energy;
}

static void metatomic_free_forces(force_t *force) {
    if (force == NULL) {
        return;
    }
    metatomic_parameters_t *mp = (metatomic_parameters_t *)force->parameters;
    if (mp != NULL) {
        if (mp->model.unload != NULL && mp->model.data != NULL) {
            mp->model.unload(mp->model.data);
        }
        if (mp->nlist != NULL) {
            free_nlist(mp->nlist);
        }
        free(mp);
    }
    free(force);
}

force_t *build_metatomic(double epsilon, double sigma, nlist_parameters_t *nlist) {
    static bool plugin_registered = false;
    if (!plugin_registered) {
        struct mta_plugin_t plugin = {
            .abi_version = MTA_ABI_VERSION,
            .name = MTM_PLUGIN_NAME,
            .load_model = mtm_plugin_load_model,
        };
        if (mta_register_plugin(plugin) != MTA_SUCCESS) {
            fprintf(stderr, "failed to register the symd metatomic plugin\n");
            return NULL;
        }
        plugin_registered = true;
    }

    metatomic_parameters_t *mp = malloc(sizeof(metatomic_parameters_t));
    mp->nlist = nlist;

    mta_status_t status = mta_load_model("symd-lennard-jones", "{}", MTM_PLUGIN_NAME, &mp->model);
    if (status != MTA_SUCCESS) {
        const char *message = NULL;
        mta_last_error(&message, NULL, NULL);
        fprintf(stderr, "failed to load the symd metatomic model: %s\n", message != NULL ? message : "(no message)");
        free(mp);
        return NULL;
    }

    MtmLennardJones *lj = (MtmLennardJones *)mp->model.data;
    lj->sigma = sigma;
    lj->epsilon = epsilon;
    lj->cutoff = sqrt(nlist->rcut);
    lj->shift = mtm_lj_shift(lj->cutoff, sigma, epsilon);

    force_t *force = malloc(sizeof(force_t));
    force->gather = metatomic_gather_forces;
    force->free = metatomic_free_forces;
    force->parameters = mp;
    return force;
}
