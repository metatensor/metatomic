# Metatomic ecosystem regression tests

These tests run models from the metatomic ecosystem (uPET/PET-MAD, FlashMD,
ShiftML, …) that were **exported by previous versions of metatomic**, and check
that they still load and still produce the same outputs.

You can run the tests with:

```bash
tox -e regtests                           # all of them
tox -e regtests -- -k pet-mad-s-v1.5.0    # a single case
```

### Adding a model

`models.json` says *what* to build: the packages a model needs, and the code
that writes it out. Replace `<model-name>` with the name of the model.

```json
"<model-name>": {
    "dependencies": ["upet"],
    "source": "import upet; upet.save_upet(model='pet-mad', size='s', version='1.5.0')"
}
```

The `source` runs in an empty directory and must write exactly one `.pt` file;
its name does not matter.

```bash
python export-models.py --list
python export-models.py <model-name>
python export-models.py <model-name> --relock     # re-resolve the dependencies
```

This creates a virtual environment under `cache/venvs/`, installs the
dependencies in it, runs the ``source``, stores the model in
`build/<sha256>.pt`, and records in `models.lock` the exact version of *every*
package that was installed.

Once a model is in `models.lock`, building it again reuses those pinned versions
rather than resolving them afresh, so the same model can be rebuilt later even
as the upstream projects move on. Pass `--relock` to deliberately move to a
newer set of dependencies.

The next step is to upload the model so it can be used for tests. This requires
write access to the https://huggingface.co/metatensor/metatomic-regtests
repository. The address of the uploaded model is recorded in `models.lock`.

```bash
python export-models.py <model-name> --upload
```

### Adding a test case

Create `references/<case>/input.json`:

```json
{
  "model": "pet-mad-s-v1.5.0",
  "length_unit": "angstrom",
  "systems": [
    {
      "types": [14, ...],
      "positions": [
        [1.0, 2.0, 3.0],
        [...]
      ],
      "cell": [
        [9.0, 0.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 9.0]
      ],
      "pbc": [true, false, true]
    }
  ],
  "selected_atoms": null,
  "outputs": {
    "energy": {
      "unit": "eV",
      "sample_kind": "system",
      "gradients": ["positions", "strain"],
      "reference": "energy.mts",
      "rtol": 1e-6,
      "atol": 1e-6
    }
  }
}

```

Then generate the reference outputs:

```bash
python update-references.py --list
python update-references.py <name-of-case>
```
