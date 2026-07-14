ORCA external optimizer example
================================

This directory contains wrappers that let `ORCA`_ drive geometry optimization
(and other workflows using the external-tool interface) with a metatomic
machine-learning potential.

The wrappers implement the file protocol documented in `orca-external-tools`_.
On each ORCA step they read ``*.extinp.tmp`` and the accompanying XYZ geometry,
evaluate energy and gradient with :py:class:`metatomic_ase.MetatomicCalculator`,
and write ``*.engrad`` back for ORCA.

.. _ORCA: https://www.faccts.de/orca/
.. _orca-external-tools: https://github.com/faccts/orca-external-tools#interface

Prerequisites
-------------

- ORCA 6 or newer (``ProgExt`` / ``Ext_Params`` in the input file)
- Python packages ``metatomic``, ``metatomic-ase``, and their dependencies
- An exported metatomic model (``.pt``), plus an ``extensions/`` directory if the
  model requires compiled extensions

Files
-----

``orca_common.py``
    Shared protocol parsing, unit conversion, and job evaluation logic.

``metatomic-orca-external``
    Standalone script invoked by ORCA via ``%method ProgExt``. Reloads the model
    on every ORCA call.

``metatomic-orca-server``
    Persistent HTTP server that keeps the model resident in memory.

``metatomic-orca-client``
    Thin ORCA-facing client that forwards jobs to ``metatomic-orca-server``.

``water_opt/water.xyz``
    Starting water geometry for a test optimization.

``water_opt/water_opt.inp``
    ORCA input template using the server/client setup. Edit paths before running.

Recommended setup (server/client)
---------------------------------

For production workflows (geometry optimization, NEB, GOAT), start a persistent
server so ORCA does not reload the PyTorch model on every energy/gradient call.

1. Install metatomic and metatomic-ase in the Python environment ORCA will use.

2. Start the server in one terminal (use ``--warmup`` to load the model
   immediately)::

       metatomic-orca-server \
           --model /path/to/model-md.pt \
           --extensions-directory /path/to/extensions \
           --device cuda \
           --warmup

   You can also set ``METATOMIC_MODEL``, ``METATOMIC_EXTENSIONS``, and
   ``METATOMIC_DEVICE`` instead of passing flags.

3. Edit ``water_opt/water_opt.inp``:

   - ``ProgExt`` must point to ``metatomic-orca-client`` (absolute path)
   - ``Ext_Params`` should pass ``-b hostname:port`` if not using the default
     ``127.0.0.1:8888``

   Example::

       %method
         ProgExt "/home/user/metatomic/python/examples/orca/metatomic-orca-client"
         Ext_Params "-b 127.0.0.1:8888"
       end

   Model paths are configured on the server. To override per job, add
   ``--model`` / ``--extensions-directory`` to ``Ext_Params``.

4. Run ORCA from the example directory::

       cd water_opt
       orca water_opt.inp > job.out

Standalone mode
---------------

For quick tests, ORCA can call ``metatomic-orca-external`` directly::

    %method
      ProgExt "/home/user/metatomic/python/examples/orca/metatomic-orca-external"
      Ext_Params "--model /home/user/models/model-md.pt --extensions-directory /home/user/models/extensions"
    end

Each ORCA step starts a new Python process and reloads the model, which is
simple but slow for long optimizations.

Parallelism: ORCA PAL, ``NCores``, and PyTorch threading
--------------------------------------------------------

ORCA does **not** parallelize the external program for you. It only reports how
many cores were allocated in each ``*.extinp.tmp`` file as ``NCores``. The
wrapper reads that value and configures CPU threading before each evaluation:

- ``torch.set_num_threads(NCores)``
- ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, and related
  variables set to ``NCores``

Set ``METATOMIC_DISABLE_THREADING_CONFIG=1`` if you prefer to manage these
variables yourself (for example in your job scheduler script).

Matching ORCA and metatomic resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a single geometry optimization (one external call at a time), choose PAL so
that ``NCores`` matches the CPU cores you want PyTorch to use:

.. code-block:: text

    ! ExtOpt Opt PAL4

    %pal
      nprocs 4
    end

    %method
      ProgExt "/path/to/metatomic-orca-client"
      Ext_Params "-b 127.0.0.1:8888"
    end

For multi-image workflows (NEB, GOAT, numerical frequencies), ORCA can run
several external calls in parallel. Use ``nprocs_group`` to set how many cores
each external call receives (see the `ORCA parallel manual`_):

.. code-block:: text

    ! ExtOpt NEB-CI PAL8

    %pal
      nprocs 8
      nprocs_group 4
    end

Here ORCA may launch two external evaluations at once, each with ``NCores = 4``.
Start **one** ``metatomic-orca-server`` per distinct ``NCores``/GPU combination,
or run standalone wrappers and let the scheduler place one process per core
group. Avoid oversubscribing: if ``NCores = 4``, do not let multiple concurrent
wrapper processes each spawn 4 OpenMP threads on the same 4 physical cores.

GPU evaluation
~~~~~~~~~~~~~~

For GPU inference, start the server with ``--device cuda`` (or set
``METATOMIC_DEVICE=cuda``). ORCA ``NCores`` then mainly controls CPU-side work
such as neighbor-list construction; the model forward pass runs on the GPU
selected by ``CUDA_VISIBLE_DEVICES`` on the server host.

Example server startup on a GPU node:

.. code-block:: bash

    export CUDA_VISIBLE_DEVICES=0
    metatomic-orca-server \
        --model /path/to/model-md.pt \
        --extensions-directory /path/to/extensions \
        --device cuda \
        --warmup

Keep ORCA ``PAL``/``NCores`` modest on GPU nodes unless CPU neighbor builds are
the bottleneck. A practical starting point is ``PAL1`` or ``PAL2`` for the
external call when using GPU inference.

.. _ORCA parallel manual: https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/parallel.html

Expected outputs
----------------

- ``water_opt.engrad`` — energy and gradient written each step
- ``water_opt.xyz`` — final optimized geometry
- ``water_opt_trj.xyz`` — optimization trajectory (if ORCA writes it)

Standalone test (without ORCA)
------------------------------

Smoke-test the standalone wrapper if ORCA has already created an
``*.extinp.tmp`` file, or craft one following the `interface specification`_::

    ./metatomic-orca-external water_opt_EXT.extinp.tmp \
        --model /path/to/model-md.pt \
        --extensions-directory /path/to/extensions

Test the client against a running server::

    metatomic-orca-server --model /path/to/model-md.pt --warmup
    metatomic-orca-client -b 127.0.0.1:8888 water_opt_EXT.extinp.tmp

.. _interface specification: https://github.com/faccts/orca-external-tools#interface

Troubleshooting
---------------

**ORCA cannot find the script**
    Use an absolute path in ``ProgExt``. ORCA's working directory may differ
    from where you launch the job.

**Connection error from the client**
    Ensure ``metatomic-orca-server`` is running and that ``-b`` matches the
    server bind address.

**Model or extensions not found**
    Pass absolute paths to ``metatomic-orca-server``, or set ``METATOMIC_MODEL``
    and ``METATOMIC_EXTENSIONS``.

**Point charges**
    ORCA point-charge files (``pointcharges.pc``) are not supported in this
    version.

**CPU oversubscription / slow runs**
    Check that ORCA ``PAL``/``NCores`` matches the threading configured by the
    wrapper. Use ``METATOMIC_DISABLE_THREADING_CONFIG=1`` only when setting
    ``OMP_NUM_THREADS``/``MKL_NUM_THREADS`` manually.

Related
-------

- `metatomic issue #228`_
- `ORCA external optimizer tutorial`_
- `orca-external-tools`_

.. _metatomic issue #228: https://github.com/metatensor/metatomic/issues/228
.. _ORCA external optimizer tutorial: https://www.faccts.de/docs/orca/6.1/tutorials/workflows/extopt.html
