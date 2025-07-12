.. _engine-plumed:

PLUMED
======


.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://www.plumed.org/
     - In the official (development) version

The `metatomic` interface for `PLUMED <https://www.plumed.org/>`_ allows using an
exported :py:class:`AtomisticModel` to compute arbitrary functions of the 
atomic coordinates, using them as 
collective variables to perform advanced sampling such as metadynamics.
Below we also provide a minimal example of the implementation of a minimalist,
model that is compatible with this interface. See also `this recipe
<https://atomistic-cookbook.org/examples/metatomic-plumed/metatomic-plumed.html>`_
for more realistic, complex demonstrations.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

See the official `installation instruction`_ in the documentation of PLUMED.

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The model must provide a :ref:`features <features-output>` output, and it is
important that this output has a fixed size, and that the size can be determined
by executing the model with an empty system (as this is how PLUMED 
determines internally the size of a CV). 
A minimal example of a model that computes the distance between two atoms is
given below. Note how the capabilities and outputs of the model are defined
to create an :py:class:`AtomisticModel`  from a bare `torch.nn.Module`, 
before exporting it as a torchscript file.

.. code-block:: python

    import torch
    from typing import Dict, List, Optional

    import metatensor.torch as mts
    import metatomic.torch as mta

    class Distance(torch.nn.Module):
        def requested_neighbor_lists(self) -> List[mta.NeighborListOptions]:
            # request a neighbor list to be computed and stored in the 
            # system passed to `forward`. In this case we return an empty
            # list because we don't use neighbor lists
            return []

        def forward(self, 
            systems: List[mta.System],
            outputs: Dict[str, mta.ModelOutput],
            selected_atoms: Optional[mts.Labels],
        ) -> Dict[str, mts.TensorMap]:

            if "features" not in outputs:
                return {}

            if outputs["features"].per_atom:
                raise ValueError("per-atoms features are not supported in this model")

            # PLUMED will first call the model with 0 atoms to get the size of the
            # output, so we need to handle this case first
            if len(systems[0]) == 0:
                # prepares an empty TensorMap with the correct metadata
                keys = mts.Labels("_", torch.tensor([[0]]))
                block = mts.TensorBlock(
                        # this shape will be used by PLUMED to know that the CV only has 1 entry
                        torch.zeros((0, 1), dtype=torch.float64),
                        samples=mts.Labels("structure", torch.zeros((0, 1), dtype=torch.int32)),
                        components=[],
                        properties=mts.Labels("distance", torch.zeros((1, 1), dtype=torch.int32)),
                    )

                return {"features": mts.TensorMap(keys, [block])}

            if selected_atoms is None:
                raise ValueError("the model should provide an atom index selection")

            if len(selected_atoms) != 2:
                raise ValueError("the model should be given two atoms to compute a distance")
        
            atom_ids = selected_atoms.column("atom").squeeze()    
            values = []
            for system in systems:
                if len(system.positions) < selected_atoms.values.max():
                    raise ValueError("System size is too small for the selected atoms indices")
                distance = torch.sqrt(((system.positions[atom_ids[0]]-system.positions[atom_ids[1]])**2).sum())
                values.append(distance)

            # creates a tensor map to contain the values and return 
            keys = mts.Labels("_", torch.tensor([[0]]))
            block = mts.TensorBlock(
                        torch.stack(values, dim=0).reshape(-1,1),
                        samples=mts.Labels("structure", torch.arange(len(systems),
                                    dtype=torch.int32).reshape((-1, 1))),
                        components=[],
                        properties=mts.Labels("distance", torch.zeros((1, 1), 
                                    dtype=torch.int32)),
                    )

            return {"features": mts.TensorMap(keys, [block])}

    # instantiates the model, describes its metadata, and export
    module = Distance()

    # metatdata about the model itself
    metadata = mta.ModelMetadata(
        name="Distance",
        description="Computes the distance between two selected atoms",
    )

    # metatdata about what the model can do
    capabilities = mta.ModelCapabilities(
        length_unit="Angstrom",
        outputs={"features": mta.ModelOutput(per_atom=False)},
        atomic_types=[0],
        interaction_range=torch.inf,
        supported_devices=["cpu", "cuda"],
        dtype="float64",
    )

    model = mta.AtomisticModel(
        module=module.eval(),
        metadata=metadata,
        capabilities=capabilities,
    )

    model.save("mta-distance.pt")


How to use the model in PLUMED
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the official `syntax reference`_ in the PLUMED documentation.
An example of a PLUMED input to load the model above could read

.. code-block::

    dist: METATOMIC ...
        MODEL=mta-distance.pt
        SPECIES1=1-416  # no need for species
        SPECIES_TO_TYPES=0  # map everything to zero
        SELECTED_ATOMS=401,402  # indices of atoms (1-based)
    ...


.. _installation instruction: https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_t_o_m_i_c_m_o_d.html
.. _syntax reference: https://www.plumed.org/doc-v2.10/user-doc/html/_m_e_t_a_t_o_m_i_c.html
