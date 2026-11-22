Miscellaneous
=============

Here are a couple of functions that are useful when writing an interface in a
simulation engine to use metatomic models.

.. autofunction:: metatomic.torch.pick_device

.. autofunction:: metatomic.torch.pick_output

Testing
-------

The flexible q-TIP4P/F water model is available as a pure PyTorch model, to test
the integration of metatomic models with simulation engines on a system with
realistic physics. Since this model uses fixed point charges, it can be used to
check that engines are able to handle models with long-range interactions.

.. autofunction:: metatomic.torch.testing.qtip4pf_model

.. autoclass:: metatomic.torch.testing.QTIP4PF
    :members:
    :exclude-members: forward
