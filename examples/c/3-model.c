// .. _c-tutorial-model:
//
// Defining a custom model
// =======================
//
// We will now explore how to implement a metatomic model purely in C. While
// most users will never need to do this, it is useful to understand the
// underlying mechanics, and when writing new bindings to the metatomic C API.
//
// In the C API, a metatomic model is represented by the :c:type:`mta_model_t`
// struct. This struct contains a ``void*`` data pointer, and multiple function
// pointers, making what's sometimes called a "vtable" or `virtual table
// <https://en.wikipedia.org/wiki/Virtual_method_table>`_. Each of the function
// pointer take the data pointer as a first argument, allowing the model to
// store its own private data and state. The vtable contains functions to query
// the model's metadata, capabilities, requested inputs, pair-list options, and
// requested outputs, as well as a function to execute the model.
//
// The model in this tutorial will provide a single ``"energy"`` output,
// computing a shifted Lennard-Jones pair potential:
//
// .. math::
//
//     E = 4 \epsilon \left[
//         \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}
//     \right] - E_{\mathrm{shift}},
//
// with :math:`E_{\mathrm{shift}}` chosen so the energy is zero at the cutoff
// :math:`r_c`.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <inttypes.h>

#include <metatomic.h>
#include <metatensor/dlpack/dlpack.h>

// %%
//
// Some helpers for the tutorial

#include "utils/dlpack.h"       // tensor_from_data
#include "utils/mts_array.h"    // make_mts_array

// Read a mts_block_t values as a DLPack tensor. The DLpack tensor must be
// released with `view->deleter(view)` when done.
static mts_status_t block_dlpack_data(const mts_block_t* block, DLManagedTensorVersioned** view) {
    struct mts_array_t array = {0};
    mts_status_t status = mts_block_data((mts_block_t*)block, &array);
    if (status != MTS_SUCCESS) {
        return status;
    }

    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    status = array.as_dlpack(array.ptr, view, cpu, NULL, version);

    if (array.destroy != NULL) {
        array.destroy(array.ptr);
    }

    return status;
}

// Read a mts_labels_t values as a DLPack tensor. The DLpack tensor must be
// released with `view->deleter(view)` when done.
static mts_status_t labels_dlpack_data(const mts_labels_t* labels, DLManagedTensorVersioned** view) {
    struct mts_array_t array = {0};
    mts_status_t status = mts_labels_values(labels, &array);
    if (status != MTS_SUCCESS) {
        return status;
    }

    DLDevice cpu = {.device_type = kDLCPU, .device_id = 0};
    DLPackVersion version = {.major = DLPACK_MAJOR_VERSION, .minor = DLPACK_MINOR_VERSION};
    status = array.as_dlpack(array.ptr, view, cpu, NULL, version);

    if (array.destroy != NULL) {
        array.destroy(array.ptr);
    }

    return status;
}

// Get a CPU pointer to the float64 data in a DLPack tensor. The tensor must be
// contiguous and have the right type.
static double* dlpack_double_data(DLManagedTensorVersioned* view) {
    assert(view != NULL);
    assert(view->dl_tensor.device.device_type == kDLCPU);
    assert(view->dl_tensor.dtype.code == kDLFloat);
    assert(view->dl_tensor.dtype.bits == 64);
    assert(view->dl_tensor.dtype.lanes == 1);
    assert((view->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0);

    assert(view->dl_tensor.ndim >= 1);
    assert(view->dl_tensor.shape != NULL);
    assert(view->dl_tensor.strides == NULL || view->dl_tensor.strides[view->dl_tensor.ndim - 1] == 1);

    return (double*)((uint8_t*)view->dl_tensor.data + view->dl_tensor.byte_offset);
}

// Get a CPU pointer to the int32 data in a DLPack tensor. The tensor must be
// contiguous and have the right type.
static int32_t* dlpack_int32_data(DLManagedTensorVersioned* view) {
    assert(view != NULL);
    assert(view->dl_tensor.device.device_type == kDLCPU);
    assert(view->dl_tensor.dtype.code == kDLInt);
    assert(view->dl_tensor.dtype.bits == 32);
    assert(view->dl_tensor.dtype.lanes == 1);
    assert((view->flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0);

    assert(view->dl_tensor.ndim >= 1);
    assert(view->dl_tensor.shape != NULL);
    assert(view->dl_tensor.strides == NULL || view->dl_tensor.strides[view->dl_tensor.ndim - 1] == 1);

    return (int32_t*)((uint8_t*)view->dl_tensor.data + view->dl_tensor.byte_offset);
}

// %%
//
// The model's physics
// -------------------
//
// Let's start by implementing the Lennard-Jones potential, nothing too fancy
// here. The model's private data will be a struct storing the parameters
// required by the computation.

typedef struct LennardJones {
    double sigma;
    double epsilon;
    double cutoff;
    double shift;
} LennardJones;

// Compute the shift so the potential is zero at the cutoff.
static double lj_shift(double cutoff, double sigma, double epsilon) {
    double x = sigma / cutoff;
    double x6 = x * x * x * x * x * x;
    return 4.0 * epsilon * (x6 * x6 - x6);
}

// Compute the energy and force contribution from one pair of atoms. The force
// is returned on the first atom; the second atom's force is the same vector,
// negated.
static void lj_pair(
    const LennardJones* parameters,
    const double distance[3],
    double* energy,
    double force_on_first[3]
) {
    double r2 = distance[0] * distance[0]
              + distance[1] * distance[1]
              + distance[2] * distance[2];

    double cutoff2 = parameters->cutoff * parameters->cutoff;
    if (r2 <= 0.0 || r2 >= cutoff2) {
        *energy = 0.0;
        force_on_first[0] = force_on_first[1] = force_on_first[2] = 0.0;
        return;
    }

    double sigma_r_2 = (parameters->sigma * parameters->sigma) / r2;
    double sigma_r_6 = sigma_r_2 * sigma_r_2 * sigma_r_2;
    double sigma_r_12 = sigma_r_6 * sigma_r_6;
    *energy = 4.0 * parameters->epsilon * (sigma_r_12 - sigma_r_6) - parameters->shift;

    // dE/d(r^2) = (12 epsilon / r^2) (inv6 - 2 inv12)
    // r^2 = |pos_2 - pos_1|^2, so d(r^2)/d(pos_1) = -2 d
    // force on atom 1 is -dE/d(pos_1) = 2 dE/d(r^2) d
    double dedr2 = (12.0 * parameters->epsilon / r2) * (sigma_r_6 - 2.0 * sigma_r_12);
    force_on_first[0] = 2.0 * dedr2 * distance[0];
    force_on_first[1] = 2.0 * dedr2 * distance[1];
    force_on_first[2] = 2.0 * dedr2 * distance[2];
}

// %%
//
// Creating the ``mta_model_t`` callbacks
// --------------------------------------
//
// We need to implement the following callbacks for the model:
//
// - :c:func:`mta_model_t.unload`: free the model's private data
// - :c:func:`mta_model_t.metadata`: return a JSON string describing the model
// - :c:func:`mta_model_t.capabilities`: return a JSON string describing the
//   model's capabilities
// - :c:func:`mta_model_t.requested_pair_lists`: return a JSON string describing
//   the pair-list options requested by the model
// - :c:func:`mta_model_t.requested_inputs`: return a JSON string describing any
//   extra inputs
// - :c:func:`mta_model_t.execute_inner`: compute the model's outputs for the
//   given systems
//
// All the callbacks take the model's private data pointer as their first
// argument, and return a status code. :c:func:`mta_model_t.unload` is
// straightforward: it just frees the private data.

static mta_status_t lj_unload(void* model_data) {
    free(model_data);
    return MTA_SUCCESS;
}

// %%
//
// :c:func:`mta_model_t.metadata` returns the human-facing description of the
// model: name, authors, references, etc. All the information is passed around
// as JSON strings, which the caller takes ownership of. The expected JSON
// structure is documented in :ref:`this page <core-json-model-metadata>`.
//
// :c:type:`mta_string_t` is used to allocate and pass strings around instead of
// plain NULL-terminated ``const char*`` so that metatomic can manage the memory
// in a way that is compatible with all supported languages.


static mta_status_t lj_metadata(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("{"
        "\"type\": \"metatomic_model_metadata\","
        "\"name\": \"Lennard-Jones model for tutorials\","
        "\"authors\": [\"metatomic authors\"],"
        "\"references\": {\"model\": [], \"architecture\": [], \"implementation\": []}"
    "}");
    return (*out != NULL) ? MTA_SUCCESS : MTA_INTERNAL_ERROR;
}

// %%
//
// :c:func:`mta_model_t.capabilities` tells an engine what the model can
// compute. Here the energy output advertises a ``"positions"`` gradient,
// meaning callers can request forces from this model. For a short range model
// like this one, the ``"interaction_range"`` is set to the cutoff.
//
// Other notable information in the capabilities is the list of supported atomic
// types, the supported devices, and the data type used for all inputs and
// outputs.
//
// If any of the callback in :c:type:`mta_model_t` fail, they should return a
// non-success status code and set an error message with
// :c:func:`mta_set_last_error`. The error message is a UTF-8 string, which can
// be retrieved by the caller with :c:func:`mta_last_error`.

static mta_status_t error(mta_status_t status, const char* message) {
    assert(status != MTA_SUCCESS);
    mta_set_last_error(
        /*message=*/message,
        /*origin=*/"model tutorial",
        /*data=*/NULL,
        /*data_deleter=*/NULL
    );
    return status;
}

static mta_status_t lj_capabilities(const void* model_data, mta_string_t* out) {
    const LennardJones* parameters = (const LennardJones*)model_data;
    char json[512];
    int printed = snprintf(json, sizeof(json),
        "{"
            "\"type\": \"metatomic_model_capabilities\","
            "\"outputs\": [{"
                "\"type\": \"metatomic_quantity\","
                "\"name\": \"energy\","
                "\"unit\": \"eV\","
                "\"gradients\": [\"positions\"],"
                "\"sample_kind\": \"system\""
            "}],"
            "\"atomic_types\": [1],"
            "\"interaction_range\": %.17g,"
            "\"length_unit\": \"Angstrom\","
            "\"supported_devices\": [\"cpu\"],"
            "\"dtype\": \"float64\""
        "}",
        parameters->cutoff
    );

    if (printed < 0 || (size_t)printed >= sizeof(json)) {
        return error(MTA_MODEL_ERROR, "failed to format capabilities JSON");
    }

    *out = mta_string_create(json);

    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate capabilities JSON");
    }

    return MTA_SUCCESS;
}

// %%
//
// The model can request pair lists (neighor lists) from the engine, which will
// be computed by the engine and attached to the :c:type:`mta_system_t` before
// calling the model. Multiple pair lists with different options can all be
// requested simulataneously, and the engine will compute them all.
//
// These requests are also exchanged as JSON. Here we request a half list (each
// pair once) with the model's cutoff. The cutoff is represented by its IEEE-754
// bit pattern to make sure the value can not be changed by serialization.

// format the pair options for the given cutoff
static mta_status_t format_pair_options(double cutoff, char* buffer, size_t size) {
    // extract the bit pattern of the cutoff
    uint64_t bits;
    memcpy(&bits, &cutoff, sizeof(bits));
    // format the bits as hex string
    char cutoff_hex[32];
    snprintf(cutoff_hex, sizeof(cutoff_hex), "0x%" PRIx64, bits);

    int printed = snprintf(buffer, size, "{"
            "\"type\": \"metatomic_pair_list_options\","
            " \"cutoff\": \"%s\","
            " \"full_list\": false,"
            " \"strict\": true,"
            "\"requestors\": [\"lj-tutorial\"]"
        "}",
        cutoff_hex
    );

    if (printed < 0 || (size_t)printed >= size) {
        return error(MTA_MODEL_ERROR, "failed to format pair list options JSON");
    }

    return MTA_SUCCESS;
}

static mta_status_t lj_requested_pair_lists(const void* model_data, mta_string_t* out) {
    const LennardJones* parameters = (const LennardJones*)model_data;

    char json[512] = {0};
    mta_status_t status = format_pair_options(parameters->cutoff, json, sizeof(json));
    if (status != MTA_SUCCESS) {
        return status;
    }

    char json_array[512] = {0};
    int printed = snprintf(json_array, sizeof(json_array), "[%s]", json);

    if (printed < 0 || (size_t)printed >= sizeof(json_array)) {
        return error(MTA_MODEL_ERROR, "failed to format pair list options JSON");
    }

    *out = mta_string_create(json_array);

    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate pair list options JSON");
    }

    return MTA_SUCCESS;
}

// %%
//
// Finallty, the model can also request additional inputs from the engine, which
// are passed as a JSON-formatted list of :ref:`core-json-quantity` objects.
// Here we don't need any extra inputs, so we return an empty list.

static mta_status_t lj_requested_inputs(const void* model_data, mta_string_t* out) {
    (void)model_data;
    *out = mta_string_create("[]");  // no extra inputs beyond the system
    if (*out == NULL) {
        return error(MTA_MEMORY_ERROR, "failed to allocate requested inputs JSON");
    }
    return MTA_SUCCESS;
}

// %%
//
// ``mta_model_t.execute_inner``
// -----------------------------
//
// The most important callback in :c:type:`mta_model_t` is
// :c:func:`mta_model_t.execute_inner`, which actually executes the model.
//
// It is named ``execute_inner`` to indicate that it should not be directly
// called by the engine, but rather through the free function
// :c:func:`mta_execute_model`, which handles unit conversion and consistency
// checks.
//
// This function takes the following parameters:
//
// - ``model_data`` is a pointer to the model data, here ``LennardJones*``;
// - ``systems`` is an array of systems on which the model shoudl be executed;
// - ``systems_count`` is the size of the ``systems`` array;
// - ``selected_atoms`` is an optional ``mts_labels_t`` object with "system" and
//   "atom" dimensions, indicating which atoms should be included in the
//   outputs. If ``NULL``, all atoms participate in the output;
// - ``requested_outputs_json`` is a JSON-encoded list of
//   :ref:`core-json-quantity` that the model should compute as outputs;
// - ``outputs`` is an array (of the same size as ``requested_outputs_json``)
//   with space for all requested outputs;
// - ``outputs_count`` is the size of the ``output`` array.

// helper to create the energy tensormap output from the energy and forces
static mta_status_t create_energy_tensormap(
    double energy,
    const double* energy_gradient,
    uintptr_t n_atoms,
    mts_tensormap_t** tensor
);

static mta_status_t lj_execute_inner(
    void* model_data,
    const mta_system_t* const* systems,
    uintptr_t systems_count,
    const mts_labels_t* selected_atoms,
    const char* requested_outputs_json,
    mts_tensormap_t** outputs,
    uintptr_t outputs_count
) {
    if (selected_atoms != NULL) {
        return error(MTA_INVALID_PARAMETER_ERROR, "this model does not support selected_atoms");
    }

    if (outputs_count == 0) {
        // no output requested
        return MTA_SUCCESS;
    }

    // if some output was requested, we assume it is the total energy, this
    // should be checked properly in an actual model
    (void)requested_outputs_json;

    if (systems_count != 1) {
        return error(MTA_INVALID_PARAMETER_ERROR, "this model only supports a single system");
    }

    const LennardJones* parameters = (const LennardJones*)model_data;
    const mta_system_t* system = systems[0];

    uintptr_t n_atoms = 0;
    assert(mta_system_size(system, &n_atoms) == MTA_SUCCESS);


    // get the pair list the engine computed, using the pair options JSON as a key
    char options_json[512] = {0};
    mta_status_t status = format_pair_options(parameters->cutoff, options_json, sizeof(options_json));
    if (status != MTA_SUCCESS) {
        return status;
    }

    const mts_block_t* pairs = NULL;
    status = mta_system_get_pairs(system, options_json, &pairs);
    if (status != MTA_SUCCESS) {
        return status;
    }

    // extract the array from the block, and get a CPU pointer to the values.
    // The array is a 2D array of shape (n_pairs, 3, 1), with the 3D distance
    // vector for each pair.
    DLManagedTensorVersioned* distances_dlpack = NULL;
    mts_status_t mts_status = block_dlpack_data(pairs, &distances_dlpack);
    if (mts_status != MTS_SUCCESS) {
        return error(MTA_METATENSOR_ERROR, "failed to get pair list distances as DLPack tensor");
    }
    assert(distances_dlpack != NULL);
    assert(distances_dlpack->dl_tensor.ndim == 3);
    assert(distances_dlpack->dl_tensor.shape[1] == 3);
    assert(distances_dlpack->dl_tensor.shape[2] == 1);

    double* distances = dlpack_double_data(distances_dlpack);
    int64_t n_pairs = distances_dlpack->dl_tensor.shape[0];

    // get the samples associated with the pairs `mts_block_t`. These contain
    // the indices of the two atoms in each pair, which we will need to compute
    // the forces.
    const mts_labels_t* pairs_samples = mts_block_labels(pairs, 0);
    DLManagedTensorVersioned* samples_dlpack = NULL;
    mts_status = labels_dlpack_data(pairs_samples, &samples_dlpack);
    if (mts_status != MTS_SUCCESS) {
        return error(MTA_METATENSOR_ERROR, "failed to get pair list samples as DLPack tensor");
    }
    assert(samples_dlpack != NULL);
    assert(samples_dlpack->dl_tensor.ndim == 2);
    // the dimensions of the samples are [i, j, shift_a, shift_b, shift_c]
    assert(samples_dlpack->dl_tensor.shape[1] == 5);

    int32_t* samples = dlpack_int32_data(samples_dlpack);

    double* energy_gradient = calloc(n_atoms * 3, sizeof(double));
    double energy = 0.0;
    for (int64_t pair_i = 0; pair_i < n_pairs; pair_i++) {
        double vector[3] = {
            distances[3 * pair_i + 0],
            distances[3 * pair_i + 1],
            distances[3 * pair_i + 2]
        };

        double pair_energy = 0.0;
        double force_on_first[3] = {0.0, 0.0, 0.0};
        lj_pair(parameters, vector, &pair_energy, force_on_first);
        energy += pair_energy;

        int32_t i = samples[pair_i * 5 + 0];
        int32_t j = samples[pair_i * 5 + 1];
        for (int d = 0; d < 3; d++) {
            energy_gradient[i * 3 + d] -= force_on_first[d];
            energy_gradient[j * 3 + d] += force_on_first[d];  // Newton's third law
        }
    }

    // fill the output
    status = create_energy_tensormap(energy, energy_gradient, n_atoms, outputs);

    // cleanup
    free(energy_gradient);

    if (distances_dlpack->deleter != NULL) {
        distances_dlpack->deleter(distances_dlpack);
    }
    if (samples_dlpack->deleter != NULL) {
        samples_dlpack->deleter(samples_dlpack);
    }

    mts_labels_free(pairs_samples);

    return status;
}

// %%
//
// .. raw:: html
//
//   <details><summary>Implementation of <code>create_energy_tensormap()</code></summary>

static mta_status_t create_energy_tensormap(
    double energy,
    const double* energy_gradient,
    uintptr_t n_atoms,
    mts_tensormap_t** tensor
) {
    DLDataType f64_dtype = {.code = kDLFloat, .bits = 64, .lanes = 1};
    DLDataType i32_dtype = {.code = kDLInt, .bits = 32, .lanes = 1};

    // create all the labels and arrays for the energy block
    mts_array_t values = make_mts_array(&energy, (uintptr_t[]){1, 1}, 2, f64_dtype);

    // the samples are a single-element array with a single "system" sample
    int32_t zero = 0;
    mts_array_t samples_values = make_mts_array(&zero, (uintptr_t[]){1, 1}, 2, i32_dtype);

    const char* system_dims[] = {"system"};
    const mts_labels_t* samples = mts_labels(system_dims, 1, samples_values);
    if (samples == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create samples labels");
    }

    // properties are a single-element array with a single "energy" property
    struct mts_array_t properties_values = make_mts_array(&zero, (uintptr_t[]){1, 1}, 2, i32_dtype);
    const char* energy_dims[] = {"energy"};
    const mts_labels_t* properties = mts_labels(energy_dims, 1, properties_values);
    if (properties == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create properties labels");
    }

    // create the energy block with no components
    mts_block_t* block = mts_block(values, samples, NULL, 0, properties);
    if (block == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create energy block");
    }

    // samples for the gradients of the energy
    int32_t* gradient_samples_scratch = malloc(sizeof(int32_t) * n_atoms * 3);
    for (uintptr_t i = 0; i < n_atoms; i++) {
        gradient_samples_scratch[3 * i + 0] = 0;          // parent energy sample (row 0)
        gradient_samples_scratch[3 * i + 1] = 0;          // system index
        gradient_samples_scratch[3 * i + 2] = (int32_t)i; // atom i
    }
    const char* grad_sample_dims[] = {"sample", "system", "atom"};
    mts_array_t grad_sample_values = make_mts_array(
        gradient_samples_scratch, (uintptr_t[]){n_atoms, 3}, 2, i32_dtype
    );
    free(gradient_samples_scratch);

    const mts_labels_t* gradient_samples = mts_labels(grad_sample_dims, 3, grad_sample_values);
    if (gradient_samples == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create gradient samples labels");
    }

    // components for the gradients of the energy
    int32_t xyz_values[] = {0, 1, 2};
    struct mts_array_t xyz_array = make_mts_array(
        xyz_values, (uintptr_t[]){3, 1}, 2, i32_dtype
    );
    const char* xyz_dims[] = {"xyz"};
    const mts_labels_t* xyz = mts_labels(xyz_dims, 1, xyz_array);
    if (xyz == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create xyz components labels");
    }

    // create the gradient block and attach it to the energy block
    mts_array_t gradient_values = make_mts_array(
        energy_gradient, (uintptr_t[]){n_atoms, 3, 1}, 3, f64_dtype
    );

    mts_block_t* gradient_block = mts_block(gradient_values, gradient_samples, &xyz, 1, properties);
    if (gradient_block == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create energy gradient block");
    }

    mts_status_t status = mts_block_add_gradient(block, "positions", gradient_block);
    if (status != MTS_SUCCESS) {
        return error(MTA_METATENSOR_ERROR, "failed to attach energy gradient block");
    }

    // create the keys for the energy tensormap
    struct mts_array_t key_array = make_mts_array(&zero, (uintptr_t[]){1, 1}, 2, i32_dtype);
    const char* key_dims[] = {"_"};  // energy is always a single-block map
    const mts_labels_t* keys = mts_labels(key_dims, 1, key_array);
    if (keys == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create keys labels");
    }

    mts_block_t* blocks[] = {block};
    *tensor = mts_tensormap(keys, blocks, 1);
    if (*tensor == NULL) {
        return error(MTA_METATENSOR_ERROR, "failed to create energy tensormap");
    }

    mts_labels_free(samples);
    mts_labels_free(properties);
    mts_labels_free(xyz);
    mts_labels_free(keys);
    return MTA_SUCCESS;
}

// %%
//
// .. raw:: html
//
//   </details>
//
//
// Running the model
// -----------------
//
// We now have all the building blocks to create a model. This would typically
// be done through a :c:type:`mta_plugin_t`, which we will explore in more
// details in the :ref:`next tutorial <c-tutorial-plugin>`.

static mta_model_t make_lennard_jones_model(double cutoff) {
    LennardJones* data = malloc(sizeof(LennardJones));
    assert(data != NULL);
    data->sigma = 1.0;
    data->epsilon = 1.0;
    data->cutoff = cutoff;
    data->shift = lj_shift(data->cutoff, data->sigma, data->epsilon);

    mta_model_t model = {
        .data = data,
        .unload = lj_unload,
        .metadata = lj_metadata,
        .capabilities = lj_capabilities,
        .requested_pair_lists = lj_requested_pair_lists,
        .requested_inputs = lj_requested_inputs,
        .execute_inner = lj_execute_inner,
    };
    return model;
}

// %%
//
//
// We'll use a system containing two atoms ``distance`` apart along *z*, with a
// single pair between them. See the previous tutorials
// (:ref:`c-tutorial-create-system` and :ref:`c-tutorial-use-system`) for more
// details.


static mta_system_t* make_two_atom_system(double distance, double cutoff);

// %%
//
// .. raw:: html
//
//   <details><summary>Implementation of <code>make_two_atom_system()</code></summary>

static mta_system_t* make_two_atom_system(double distance, double cutoff) {
    DLDataType i32_dtype = {.code = kDLInt, .bits = 32, .lanes = 1};
    DLDataType bool_dtype = {.code = kDLBool, .bits = 8, .lanes = 1};
    DLDataType f64_dtype = {.code = kDLFloat, .bits = 64, .lanes = 1};

    static double positions_data[6];
    positions_data[0] = 0.0; positions_data[1] = 0.0; positions_data[2] = 0.0;
    positions_data[3] = 0.0; positions_data[4] = 0.0; positions_data[5] = distance;

    // non-periodic: cell must be all zeros
    static double cell_data[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    static int32_t types_data[] = {1, 1};
    static bool pbc_data[] = {false, false, false};

    DLManagedTensorVersioned* positions = tensor_from_data(
        positions_data, (int64_t[]){2, 3}, 2, f64_dtype
    );
    DLManagedTensorVersioned* cell = tensor_from_data(
        cell_data, (int64_t[]){3, 3}, 2, f64_dtype
    );
    DLManagedTensorVersioned* types = tensor_from_data(
        types_data, (int64_t[]){2}, 1, i32_dtype
    );
    DLManagedTensorVersioned* pbc = tensor_from_data(
        pbc_data, (int64_t[]){3}, 1, bool_dtype
    );

    mta_system_t* system = NULL;
    mta_status_t create_status = mta_system_create("Angstrom", types, positions, cell, pbc, &system);
    if (create_status != MTA_SUCCESS) {
        const char* error_message = NULL;
        mta_last_error(&error_message, NULL, NULL);
        fprintf(stderr, "failed to create system: %s\n", error_message);
        return NULL;
    }

    int32_t pair_samples[] = {0, 1, 0, 0, 0};
    struct mts_array_t samples_array = make_mts_array(
        pair_samples, (uintptr_t[]){1, 5}, 2, i32_dtype
    );
    const char* sample_dims[] = {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"};
    const mts_labels_t* samples = mts_labels(sample_dims, 5, samples_array);

    int32_t xyz_values[] = {0, 1, 2};
    struct mts_array_t xyz_array = make_mts_array(
        xyz_values, (uintptr_t[]){3, 1}, 2, i32_dtype
    );
    const char* xyz_dims[] = {"xyz"};
    const mts_labels_t* xyz = mts_labels(xyz_dims, 1, xyz_array);
    const mts_labels_t* components[] = {xyz};

    int32_t zero = 0;
    struct mts_array_t prop_array = make_mts_array(
        &zero, (uintptr_t[]){1, 1}, 2, i32_dtype
    );
    const char* distance_dims[] = {"distance"};
    const mts_labels_t* properties = mts_labels(distance_dims, 1, prop_array);

    double disp_data[] = {0.0, 0.0, distance};
    struct mts_array_t values = make_mts_array(
        disp_data, (uintptr_t[]){1, 3, 1}, 3, f64_dtype
    );
    mts_block_t* pairs = mts_block(values, samples, components, 1, properties);

    char options[512] = {0};
    format_pair_options(cutoff, options, sizeof(options));
    mta_system_add_pairs(system, options, pairs);

    mts_labels_free(samples);
    mts_labels_free(xyz);
    mts_labels_free(properties);
    return system;
}

// %%
//
// .. raw:: html
//
//   </details>

int main(void) {

// %%
//
// Now, we create the model and a system with two atoms 1.3 σ apart, just past
// the potential minimum, so they attract. We then use
// :c:func:`mta_execute_model` to run the model on the system, and request the
// energy and its gradient with respect to the atomic positions.

double cutoff = 3.0;

mta_model_t model = make_lennard_jones_model(cutoff);
mta_system_t* system = make_two_atom_system(/*distance=*/1.3, cutoff);
const mta_system_t* systems[] = {system};

const char* requested_outputs = "[{"
    "\"type\": \"metatomic_quantity\","
    "\"name\": \"energy\","
    "\"unit\": \"eV\","
    "\"gradients\": [\"positions\"],"
    "\"sample_kind\": \"system\""
"}]";

mts_tensormap_t* output = NULL;
mta_status_t status;
status = mta_execute_model(
    /*model=*/model,
    /*systems=*/systems,
    /*systems_count=*/1,
    /*selected_atoms=*/NULL,
    /*requested_outputs_json=*/requested_outputs,
    /*check_consistency=*/true,
    /*outputs=*/&output,
    /*outputs_count=*/1
);

bool failed = false;
DLManagedTensorVersioned* energy_dlpack = NULL;
DLManagedTensorVersioned* gradient_dlpack = NULL;

if (status != MTA_SUCCESS) {
    const char* error_message = NULL;
    mta_last_error(&error_message, NULL, NULL);
    fprintf(stderr, "failed to run model: %s\n", error_message);

    failed = true;
    goto cleanup;
}

// %%
//
// Finally, we can look into the output and extract the energy and forces. The
// energy is a single scalar, while the forces are a 2×3 array (2 atoms, 3
// dimensions). The forces are the negative of the gradient of the energy with
// respect to the atomic positions.

mts_block_t* block = NULL;
mts_status_t mts_status = mts_tensormap_block_by_id(output, &block, 0);
if (mts_status != MTS_SUCCESS) {
    fprintf(stderr, "failed to get energy block from output\n");
    failed = true;
    goto cleanup;
}

mts_status = block_dlpack_data(block, &energy_dlpack);
if (mts_status != MTS_SUCCESS) {
    fprintf(stderr, "failed to get energy block as DLPack tensor\n");
    failed = true;
    goto cleanup;
}
double energy = dlpack_double_data(energy_dlpack)[0];
assert(fabs(energy - (-0.651537)) < 1e-6);

mts_block_t* gradient_block = NULL;
mts_status = mts_block_gradient(block, "positions", &gradient_block);
if (mts_status != MTS_SUCCESS) {
    fprintf(stderr, "failed to get gradient block\n");
    failed = true;
    goto cleanup;
}

mts_status = block_dlpack_data(gradient_block, &gradient_dlpack);
if (mts_status != MTS_SUCCESS) {
    fprintf(stderr, "failed to get gradient block as DLPack tensor\n");
    failed = true;
    goto cleanup;
}
double* gradient = dlpack_double_data(gradient_dlpack);

// -forces on the first atom
assert(gradient[0] == 0.0);
assert(gradient[1] == 0.0);
assert(fabs(gradient[2] - (-2.239980)) < 1e-6);

// -forces on the second atom
assert(gradient[3] == 0.0);
assert(gradient[4] == 0.0);
assert(fabs(gradient[5] - (2.239980)) < 1e-6);

// %%
//

cleanup:

if (energy_dlpack != NULL && energy_dlpack->deleter != NULL) {
    energy_dlpack->deleter(energy_dlpack);
}

if (gradient_dlpack != NULL && gradient_dlpack->deleter != NULL) {
    gradient_dlpack->deleter(gradient_dlpack);
}

mts_tensormap_free(output);
mta_system_free(system);
model.unload(model.data);

if (failed) {
    return EXIT_FAILURE;
} else {
    return EXIT_SUCCESS;
}

}
