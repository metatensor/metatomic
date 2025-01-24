.. _engine-chemiscope:

Chemiscope
==========

.. warning::

    Chemiscope still uses the old metatensor-based atomistic models, and has not
    yet been updated to execute metatomic models.

.. list-table::
   :header-rows: 1

   * - Official website
     - How is metatomic supported?
   * - https://chemiscope.org
     - In the original version

Supported model outputs
^^^^^^^^^^^^^^^^^^^^^^^

The :ref:`features <features-output>` output is supported, and can be used to
compute features for multiple structures in ``chemiscope.explore()``.

How to install the code
^^^^^^^^^^^^^^^^^^^^^^^

The code can be installed with the following command:

.. code-block:: bash

   pip install chemiscope[metatensor]


How to use the code
^^^^^^^^^^^^^^^^^^^

See the example from https://chemiscope.org/docs/examples/8-explore-with-metatensor.html.
