"""
.. _c-tutorial-symd-engine:

Case study: driving symd through the C API
===========================================

The other C tutorials show how to build a system, attach a pair list, and
(most recently) define and run a model in isolation. This one closes the loop
with a real, independent
simulation engine: `symd <https://github.com/whitead/symd>`_, a small C
molecular-dynamics code.

symd has its own force plugin ABI (a ``force_t`` vtable: a ``gather``
function that fills a ``forces`` array and returns the energy, plus a
``free``). Wiring a new ``force_type: "metatomic"`` into it means writing the
engine side of the C API for real. The walkthrough below follows the same
12-step data flow as the :ref:`torch engine/model diagram <model-dataflow>`
on the overview page, point by point, so it should read as "the same
contract, a different language" rather than a separate story.

The full integration lives in symd's own tree, on the ``feat/metatomic-c-api``
branch (not part of this repository). This page does not run that binary.
It walks through the C API contract using the vendored source, then reports
a previously computed NVE check on a small free 3D cluster -- chosen so
there is nothing for symd's neighbor list or symmetry machinery to hide.
"""

# %%
#
# The bridge, 12 points at a time
# ---------------------------------
#
# Every snippet below is quoted directly out of symd's real
# ``src/metatomic_force.c`` (via ``literalinclude``, not retyped), so it
# can't drift out of sync with what actually runs. Larger ones are
# collapsed by default -- click to expand.
#
# .. literalinclude:: /_include/metatomic_force.c
#    :language: c
#    :lines: 30-39
#    :caption: metatomic_force.c, top of file
#
# ``metatomic_force.h`` itself pulls in symd's own ``force.h`` (the
# ``force_t`` vtable) and ``nlist.h`` (symd's neighbor list). Before the
# 12 points: the physics the model actually computes, independently
# derived rather than copied from symd's own ``lj()`` (so the two can be
# cross-checked, see the results below):
#
# .. literalinclude:: /_include/metatomic_force.c
#    :language: c
#    :lines: 66-93
#    :caption: mtm_lj_pair() -- shifted Lennard-Jones, energy and force
#
# #. **The engine loads an exported model.** symd doesn't load a file --
#    it registers an in-process plugin and loads it by name, once, in
#    ``build_metatomic()``:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 783-800
#
# #. **The engine requests and gets the model's capabilities.**
#    :c:func:`mta_execute_model` does this itself (calling ``mtm_lj_capabilities``)
#    as part of consistency checking -- symd's own code never calls it directly.
# #. **The engine creates evaluation options.** The C API has no separate
#    options object; the equivalent is just the ``requested_outputs_json``
#    string handed straight to :c:func:`mta_execute_model` (point 9).
# #. **The engine creates a list of System.** ``mta_system_create`` wraps
#    symd's own position/cell/pbc arrays as DLPack views -- no copies:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 613-615
#
# #. **The engine asks the model for the neighbor lists it needs.**
#    Handled internally by :c:func:`mta_execute_model` (``mtm_lj_requested_pair_lists``
#    reports the cutoff as a ``PairListOptions`` JSON string).
# #. **The engine computes those neighbor lists and registers them.**
#    symd's own cutoff filtering, then attaching the result:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 670-673
#
#    .. details:: Show how symd's own neighbor list becomes a metatomic pair list
#
#       Collecting the pairs symd's cell/Verlet list already found, within the
#       model's *true* cutoff (the list itself uses a slightly larger skin
#       radius):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 549-582
#
#       Then packaging them as a metatomic pair-list block (displacement
#       vectors, plus the ``first_atom``/``second_atom`` sample labels):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 633-668
#
# #. **The engine asks for any extra required input data.** ``mtm_lj_requested_inputs``
#    returns ``[]`` -- this model needs nothing beyond positions.
# #. **The engine registers that extra data.** Nothing to do, since point 7
#    asked for nothing.
# #. **The engine calls the model.** Not ``forward()`` -- the C API's
#    equivalent is :c:func:`mta_execute_model` itself, which also does unit
#    conversion and consistency checking on top of whatever the model computes:
#
#    .. literalinclude:: /_include/metatomic_force.c
#       :language: c
#       :lines: 681-695
# #. **The model runs.** ``mtm_lj_execute_inner`` -- called internally by
#    :c:func:`mta_execute_model` through the ``execute_inner`` function pointer.
#
#    .. details:: Show the full mtm_lj_execute_inner()
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 358-486
#
# #. **The model returns its outputs.** A :py:class:`TensorMap`-shaped
#    ``"energy"`` output, with a ``"positions"`` gradient block attached --
#    that gradient *is* the force, and driving a real engine needs it filled
#    in. See :ref:`the previous tutorial <c-tutorial-add-model>` for how,
#    including a finite-difference check that it's actually correct.
#
#    .. details:: Show how the gradient block is built, and how the engine reads it back
#
#       Building it (one row per atom, values = ``-force``):
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 240-298
#
#       Reading it back out, on the engine side:
#
#       .. literalinclude:: /_include/metatomic_force.c
#          :language: c
#          :lines: 704-756
#
# #. **The engine runs backward() for gradients, if needed.** Not used here:
#    the C API has no autodiff pass. Forces come back already computed, as
#    the ``"positions"`` gradient block from point 11 -- exactly the
#    *explicit* gradients path the torch docs' own tip on this step mentions
#    as the alternative to backward-mode differentiation.

# %%
#
# Setting up a run
# ----------------
#
# The check uses eight particles in a small 3D cluster, in a box much larger
# than the interaction cutoff -- so there is nothing for symd's own neighbor
# list to do beyond simple distance checks, and nothing for the C API bridge
# to get wrong. ``p1`` is symd's trivial symmetry group (one member, the
# identity): it still goes through symd's normal group machinery, just
# without imposing any actual constraint.
#
# Both force types share the same starting configuration, timestep, and
# seed, in NVE (no thermostat, no pressure coupling):
#
# * 4000 steps, ``dt = 0.005``
# * 8 particles, 20³ box, cutoff 3σ
# * ``lj``: symd's own hand-written Lennard-Jones
# * ``metatomic``: the C API bridge above

# %%
#
# Results (precomputed)
# ---------------------
#
# These numbers come from a local ``symd3`` run of that setup. They are not
# regenerated when the documentation is built.
#
# Over 4000 NVE steps, the metatomic total energy stayed flat to about
# ``(max - min) / |mean| ≈ 0.07%``. The same trajectory with symd's own
# ``lj()`` conserves energy to the same quality. The two traces sit at a
# small constant offset from each other; that offset is not drift.
#
# The offset is the cutoff convention: symd's ``lj()`` shifts the *force* to
# zero smoothly at the cutoff, while the model in this tutorial only shifts
# the *energy* and truncates the force there. Both are standard Lennard-Jones
# cutoffs; they are not bit-identical.

print(
    "precomputed NVE check (4000 steps, 8-particle cluster):\n"
    "  metatomic (max-min)/|mean| of total energy ~ 0.07%\n"
    "  offset vs symd lj(): shifted-force vs shifted-potential cutoff"
)
