#ifndef METATOMIC_TORCH_SYSTEM_HPP
#define METATOMIC_TORCH_SYSTEM_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor/torch.hpp>
#include <metatomic.hpp>

#include "metatomic/torch/exports.h"

namespace metatomic_torch {

class PairListOptionsHolder;
/// TorchScript will always manipulate `PairListOptions` through a `torch::intrusive_ptr`
using PairListOptions = torch::intrusive_ptr<PairListOptionsHolder>;

class SystemHolder;
/// TorchScript will always manipulate `SystemHolder` through a `torch::intrusive_ptr`
using System = torch::intrusive_ptr<SystemHolder>;

/// Options for the calculation of a neighbor list
class METATOMIC_TORCH_EXPORT PairListOptionsHolder final: public torch::CustomClassHolder {
public:
    /// Create `PairListOptions` with the given `cutoff` value, and the flags `full_list` and `strict`.
    ///
    /// `requestor` can be used to store information about who requested the
    /// neighbor list.
    PairListOptionsHolder(double cutoff, bool full_list, bool strict, std::string requestor = "");

    PairListOptionsHolder(metatomic::PairListOptions options): options_(std::move(options)) {}
    ~PairListOptionsHolder() override = default;

    /// Spherical cutoff radius for this neighbor list, in the units of the model
    double cutoff() const {
        return options_.cutoff();
    }

    /// Should the list be a full neighbor list (contains both the pair i->j
    /// and j->i) or a half neighbor list (contains only the pair i->j)
    bool full_list() const {
        return options_.full_list();
    }

    /// Should the list guarantee that *only* atoms within the cutoff are
    // included (strict) or possibly also include pairs beyond the cutoff (non-strict)
    bool strict() const {
        return options_.strict();
    }

    /// Get the list of strings describing who requested this neighbor list
    std::vector<std::string> requestors() const {
        return options_.requestors();
    }

    /// Add a new requestor to the list of who requested this neighbor list
    void add_requestor(std::string requestor);

    /// Get the underlying `metatomic::PairListOptions` object
    const metatomic::PairListOptions& as_metatomic() const {
        return options_;
    }

    // ======= Legacy functions ========= //

    /// Get the length unit of the model, this will be used in `engine_cutoff`.
    const std::string& length_unit() const {
        return length_unit_;
    }

    /// Set the length unit to a new value.
    ///
    /// This is typically called by `AtomisticModel`, and should not
    /// need to be set by users directly.
    void set_length_unit(std::string length_unit);

    /// Spherical cutoff radius for this neighbor list, in the units of the engine
    double engine_cutoff(const std::string& engine_length_unit) const;

    /// Implementation of Python's `__repr__`
    std::string repr() const;
    /// Implementation of Python's `__str__`
    std::string str() const;

    /// Serialize a `PairListOptions` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `PairListOptions` from a JSON string.
    static PairListOptions from_json(const std::string& json);
    static PairListOptions from_legacy_json(const std::string& json);

private:
    metatomic::PairListOptions options_;
    std::string length_unit_;

    friend bool operator==(const PairListOptions& lhs, const PairListOptions& rhs);
};

/// Check `PairListOptions` for equality. The `requestors` list and the
/// `length_unit` are ignored when checking for equality
inline bool operator==(const PairListOptions& lhs, const PairListOptions& rhs) {
    return lhs->options_ == rhs->options_;
}

/// Check `PairListOptions` for inequality.
inline bool operator!=(const PairListOptions& lhs, const PairListOptions& rhs) {
    return !(lhs == rhs);
}


/// Custom autograd node going (positions, cell) => neighbor list distances
///
/// This does not actually computes the neighbor list, but registers a new node
/// in the overall computational graph, allowing to re-use neighbor list
/// computed outside of torch. The public interface for this is the
/// `register_autograd_neighbors()` function.
class METATOMIC_TORCH_EXPORT NeighborsAutograd: public torch::autograd::Function<NeighborsAutograd> {
public:
    /// This does nothing unless `check_consistency` is true, in which case it
    /// checks that the values in `neighbors` can be re-computed with the
    /// samples and positions/cell
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor positions,
        torch::Tensor cell,
        metatensor_torch::TensorBlock neighbors,
        bool check_consistency
    );

    /// Compute the gradient of the output w.r.t. positions/cell
    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<torch::Tensor> outputs_grad
    );
};

/// Register a new autograd node going from (`system.positions`, `system.cell`)
/// to the `neighbors` distance vectors.
///
/// This does not recompute the distance vectors, but work as-if all the data in
/// `neighbors.values` was computed directly from `system.positions` and
/// `system.cell`, allowing downstream models to use it directly with full
/// autograd integration.
///
/// `check_consistency` can be set to `true` to run a handful of additional
/// checks in case the data in neighbors does not follow what's expected.
METATOMIC_TORCH_EXPORT void register_autograd_neighbors(
    System system,
    metatensor_torch::TensorBlock neighbors,
    bool check_consistency
);

/// A System contains all the information about an atomistic system; and should
/// be used as the input of atomistic models.
class METATOMIC_TORCH_EXPORT SystemHolder final: public torch::CustomClassHolder {
public:
    /// Create a `SystemHolder` with the given `types`, `positions` and
    /// `cell`.
    ///
    /// @param length_unit unit of length used by `positions` and `cell`
    /// @param types 1D tensor of 32-bit integer representing the particles
    ///        identity. For atoms, this is typically their atomic numbers.
    /// @param positions 2D tensor of shape (len(types), 3) containing the
    ///        Cartesian positions of all particles in the system.
    /// @param cell 2D tensor of shape (3, 3), describing the bounding box/unit
    ///        cell of the system. Each row should be one of the bounding box
    ///        vector; and columns should contain the x, y, and z components of
    ///        these vectors (i.e. the cell should be given in row-major order).
    /// @param pbc 1D tensor of 3 boolean values, indicating if the system is
    ///        periodic along the directions defined by cell axes `a`, `b` and
    ///        `c`, respectively.
    SystemHolder(
        std::string length_unit,
        torch::Tensor types,
        torch::Tensor positions,
        torch::Tensor cell,
        torch::Tensor pbc
    );
    ~SystemHolder() override = default;

    /// Swap the contents of this `SystemHolder` with another one.
    void swap(SystemHolder& other) noexcept;

    /// Get the underlying `metatomic::System` object
    const metatomic::System& as_metatomic() const {
        return system_;
    }

    /// Move the `metatomic::System` out of this class.
    ///
    /// This leaves this `SystemHolder` in an empty state that should no longer
    /// be used.
    metatomic::System release();

    /// Create a `System` from a pre-existing `metatomic::System`.
    ///
    /// The system is moved into the new holder, taking ownership of the
    /// underlying data.
    static System from_metatomic(metatomic::System system);

    /// Create a `System` which is a view of a pre-existing `metatomic::System`.
    ///
    /// The view does not own the underlying data. The `parent` is kept alive as
    /// long as this system is alive.
    static System view_from_metatomic(metatomic::System system, torch::IValue parent);

    /// Get the length unit used by the positions, cell, and pair lists of this
    /// system.
    std::string length_unit() const {
        return system_.length_unit();
    }

    /// Get the number of particles in this system
    int64_t size() const {
        return static_cast<int64_t>(this->system_.size());
    }

    /// Get the particle types for all particles in the system.
    torch::Tensor types() const {
        return types_;
    }

    /// Get the positions for the atoms in the system.
    torch::Tensor positions() const {
        return positions_;
    }

    /// Unit cell/bounding box of the system.
    torch::Tensor cell() const {
        return cell_;
    }

    /// Periodic boundary conditions for the system.
    torch::Tensor pbc() const {
        return pbc_;
    }

    /// Get the device used by all the data in this `System`
    torch::Device device() const {
        return positions_.device();
    }

    /// Get the dtype used by all the floating point data in this `System`
    torch::Dtype scalar_type() const {
        return positions_.scalar_type();
    }

    /// Move all the data in this `System` to the given `dtype` and `device`.
    ///
    /// `non_blocking` has the same semantics as the corresponding parameter for
    /// `torch::Tensor::to`.
    System to(
        torch::optional<torch::Dtype> dtype = torch::nullopt,
        torch::optional<torch::Device> device = torch::nullopt,
        bool non_blocking = false
    ) const;

    /// Wrapper of the `to` function to enable using it with positional
    /// parameters from Python; for example `to(dtype)`, `to(device)`,
    /// `to(dtype, device=device)`, `to(dtype, device)`, `to(device, dtype)`,
    /// etc.
    System to_positional(
        torch::IValue positional_1,
        torch::IValue positional_2,
        torch::optional<torch::Dtype> dtype,
        torch::optional<torch::Device> device,
        bool non_blocking
    ) const;

    /// Add a new pair list in this system corresponding to the given `options`.
    ///
    /// The pair list should have the following samples: "first_atom",
    /// "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c",
    /// containing the index of the first and second atom and the number of cell
    /// vector a/b/c to add to the positions difference to get the pair vector.
    ///
    /// The `pairs` should also have a single component "xyz" with values `[0,
    /// 1, 2]`; and a single property "distance" with value 0.
    ///
    /// The pair values must contain the distance vector from the first to the
    /// second atom, i.e. `positions[second_atom] - positions[first_atom] +
    /// cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c`.
    void add_pairs(PairListOptions options, metatensor_torch::TensorBlock pairs);

    /// Retrieve a previously stored pair list with the given options, or
    /// throw an error if no such pair list exists.
    static metatensor_torch::TensorBlock pairs(System self, PairListOptions options);

    /// Get the options for all pair lists registered with this `System`
    std::vector<PairListOptions> known_pairs() const;

    /// Add custom data to this system, stored as `TensorMap`.
    ///
    /// @param name name of the data
    /// @param data the data to store. Ownership of `data` is transferred to
    ///        this `System`.
    void add_custom_data(std::string name, metatensor_torch::TensorMap data);

    /// Retrieve custom data stored in this System, or throw an error.
    static metatensor_torch::TensorMap custom_data(System self, std::string name);

    /// Get the list of data registered with this `System`
    std::vector<std::string> known_custom_data() const;

    /// Implementation of `__str__` and `__repr__` for Python
    std::string str() const;

private:
    /// Create a `SystemHolder` from a pre-existing `metatomic::System`, and keep
    /// a reference to the `parent` object alive as long as this `SystemHolder`
    /// is alive.
    explicit SystemHolder(
        metatomic::System system,
        torch::IValue parent
    );

    metatomic::System system_;
    torch::IValue parent_;

    // Store the data as torch tensors so we keep the gradient tracking
    // information. If this system was created from torch tensors, these will be
    // views of the original tensors; if it was created from a
    // metatomic::System, these will be new tensors created from the underlying
    // data through DLPack.
    torch::Tensor types_;
    torch::Tensor positions_;
    torch::Tensor cell_;
    torch::Tensor pbc_;

    friend class torch::intrusive_ptr<SystemHolder>;
};

}

#endif
