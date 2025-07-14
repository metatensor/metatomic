import importlib.util
import os
import sys

import torch
from metatensor.torch import Labels

from metatomic.torch import (
    ModelEvaluationOptions,
    ModelOutput,
    System,
    load_atomistic_model,
)


EXAMPLES = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples")
)

DOCS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "docs")
)


def test_export_atomistic_model(tmp_path):
    """
    Check if the model defined in ``python/examples/1-export-atomistic-model.py`` works
    """
    os.chdir(tmp_path)

    # import example from full path
    spec = importlib.util.spec_from_file_location(
        "export_atomistic_model",
        os.path.join(EXAMPLES, "1-export-atomistic-model.py"),
    )

    export_atomistic_model = importlib.util.module_from_spec(spec)
    sys.modules["export_atomistic_model"] = export_atomistic_model
    spec.loader.exec_module(export_atomistic_model)

    # define properties for prediction
    system = System(
        types=torch.tensor([1]),
        positions=torch.tensor([[1.0, 1, 1]], dtype=torch.float64, requires_grad=True),
        cell=torch.zeros([3, 3], dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )

    outputs = {
        "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
    }

    # run bare model
    export_atomistic_model.model([system], outputs)

    # run exported model
    options = ModelEvaluationOptions(length_unit="Angstrom", outputs=outputs)
    export_atomistic_model.wrapper([system], options, check_consistency=True)

    # run exported and saved model
    export_atomistic_model.wrapper.save("exported-model.pt")
    atomistic_model = load_atomistic_model("exported-model.pt")
    atomistic_model([system], options, check_consistency=True)


def test_plumed_example(tmp_path):
    """
    Check if the model defined in ``docs/src/engines/plumed-model.py`` works
    """
    os.chdir(tmp_path)

    # import example from full path
    spec = importlib.util.spec_from_file_location(
        "plumed_model",
        os.path.join(DOCS, "src", "engines", "plumed-model.py"),
    )

    plumed_model = importlib.util.module_from_spec(spec)
    sys.modules["plumed_model"] = plumed_model
    spec.loader.exec_module(plumed_model)

    # define properties for prediction
    system = System(
        types=torch.tensor([0, 0, 0]),
        positions=torch.tensor(
            [[1.0, 1, 1], [2.0, 2, 2], [3.0, 3, 3]],
            dtype=torch.float64,
            requires_grad=True,
        ),
        cell=torch.zeros([3, 3], dtype=torch.float64),
        pbc=torch.tensor([False, False, False]),
    )

    outputs = {
        "features": ModelOutput(per_atom=False),
    }

    # run bare model
    selected_atoms = Labels(["system", "atom"], torch.tensor([[0, 1], [0, 0]]))
    plumed_model.distance([system], outputs, selected_atoms)

    # run exported model
    options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs=outputs,
        selected_atoms=selected_atoms,
    )
    plumed_model.model([system], options, check_consistency=True)

    # run exported and saved model
    atomistic_model = load_atomistic_model("mta-distance.pt")
    atomistic_model([system], options, check_consistency=True)
