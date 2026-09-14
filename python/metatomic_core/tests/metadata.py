import pytest

from metatomic import ModelMetadata, References


def test_references():
    references = References()
    assert references.model == []
    assert references.architecture == []
    assert references.implementation == []

    references.add("model", "doi:10.1234/test")
    references.add("architecture", "doi:10.1234/arch")
    references.add("implementation", "https://github.com/test")

    assert references.model == ["doi:10.1234/test"]
    assert references.architecture == ["doi:10.1234/arch"]
    assert references.implementation == ["https://github.com/test"]

    # the returned lists are copies
    references.model.append("doi:10.1234/other")
    assert references.model == ["doi:10.1234/test"]

    message = (
        "reference section must be 'model', 'architecture', or 'implementation', "
        "got 'wrong'"
    )
    with pytest.raises(ValueError, match=message):
        references.add("wrong", "doi:10.1234/test")

    message = "reference can not be empty string \\(in 'model' section\\)"
    with pytest.raises(ValueError, match=message):
        references.add("model", "")

    with pytest.raises(ValueError, match=message):
        References(model=["doi:10.1234/test", ""])


@pytest.fixture
def metadata():
    return ModelMetadata(
        name="test-model",
        description="A test model",
        authors=["Alice", "Bob <bob@test.com>"],
        references=References(
            model=["doi:10.1234/test"],
            architecture=["doi:10.1234/arch"],
            implementation=["https://github.com/test"],
        ),
        extra={"key1": "value1", "key2": "value2"},
    )


def test_model_metadata(metadata):
    assert metadata.name == "test-model"
    assert metadata.description == "A test model"
    assert metadata.authors == ["Alice", "Bob <bob@test.com>"]
    assert metadata.references.model == ["doi:10.1234/test"]
    assert metadata.extra == {"key1": "value1", "key2": "value2"}

    metadata.add_author("Charlie")
    assert metadata.authors == ["Alice", "Bob <bob@test.com>", "Charlie"]

    # the references can also be given as a plain dict
    metadata = ModelMetadata(references={"model": ["doi:10.1234/test"]})
    assert metadata.references.model == ["doi:10.1234/test"]
    assert metadata.references.architecture == []

    defaults = ModelMetadata()
    assert defaults.name == ""
    assert defaults.description == ""
    assert defaults.authors == []
    assert defaults.references == References()
    assert defaults.extra == {}


def test_model_metadata_errors():
    with pytest.raises(ValueError, match="author can not be empty string"):
        ModelMetadata(authors=["Alice", ""])

    with pytest.raises(ValueError, match="name must be a string"):
        ModelMetadata(name=42)

    with pytest.raises(ValueError, match="description must be a string"):
        ModelMetadata(description=42)

    with pytest.raises(ValueError, match="extra values must be strings"):
        ModelMetadata(extra={"key": 42})

    with pytest.raises(ValueError, match="extra keys must be strings"):
        ModelMetadata(extra={42: "value"})

    message = (
        "reference section must be 'model', 'architecture', or 'implementation', "
        "got 'wrong'"
    )
    with pytest.raises(ValueError, match=message):
        ModelMetadata(references={"wrong": []})

    message = "references must be a References or a dict"
    with pytest.raises(ValueError, match=message):
        ModelMetadata(references=42)


def test_model_metadata_roundtrip(metadata):
    data = metadata.to_dict()

    assert data == {
        "type": "metatomic_model_metadata",
        "name": "test-model",
        "authors": ["Alice", "Bob <bob@test.com>"],
        "description": "A test model",
        "references": {
            "model": ["doi:10.1234/test"],
            "architecture": ["doi:10.1234/arch"],
            "implementation": ["https://github.com/test"],
        },
        "extra": {"key1": "value1", "key2": "value2"},
    }

    assert ModelMetadata.from_dict(data) == metadata


def test_model_metadata_from_dict_errors(metadata):
    def corrupted(**kwargs):
        data = metadata.to_dict()
        data.update(kwargs)
        return data

    def without(key):
        data = metadata.to_dict()
        del data[key]
        return data

    cases = [
        ("not an object", "invalid JSON data for ModelMetadata, expected an object"),
        (
            corrupted(type="something-else"),
            "'type' in JSON for ModelMetadata must be 'metatomic_model_metadata'",
        ),
        (without("name"), "'name' in JSON for ModelMetadata must be a string"),
        (corrupted(name=42), "'name' in JSON for ModelMetadata must be a string"),
        (
            corrupted(authors="Alice"),
            "'authors' in JSON for ModelMetadata must be an array",
        ),
        (
            corrupted(authors=["Alice", 42]),
            "'authors' in JSON for ModelMetadata must be an array of strings",
        ),
        (
            without("description"),
            "'description' in JSON for ModelMetadata must be a string",
        ),
        (
            corrupted(extra="not-an-object"),
            "'extra' in JSON for ModelMetadata must be an object",
        ),
        (
            corrupted(extra={"key": 42}),
            "'extra' in JSON for ModelMetadata must be an object with string values",
        ),
        (
            corrupted(references="not-an-object"),
            "invalid JSON data for references in ModelMetadata, expected an object",
        ),
        (
            without("references"),
            "invalid JSON data for references in ModelMetadata, expected an object",
        ),
        (
            corrupted(references={"model": "doi:10.1234/test"}),
            "'model' in references of ModelMetadata must be an array",
        ),
        (
            corrupted(
                references={"model": [42], "architecture": [], "implementation": []}
            ),
            "'model' in references of ModelMetadata must be an array of strings",
        ),
        (
            corrupted(references={"model": [], "implementation": []}),
            "'architecture' in references of ModelMetadata must be an array",
        ),
    ]

    for data, message in cases:
        with pytest.raises(ValueError, match=message):
            ModelMetadata.from_dict(data)


def test_model_metadata_printing(metadata):
    expected = """This is the test-model model
============================

A test model

Model authors
-------------

- Alice
- Bob <bob@test.com>

Model references
----------------

Please cite the following references when using this model:
- about this specific model:
  * doi:10.1234/test
- about the architecture of this model:
  * doi:10.1234/arch
- about the implementation of this model:
  * https://github.com/test
"""
    assert str(metadata) == expected

    expected = """This is an unnamed model
========================
"""
    assert str(ModelMetadata()) == expected
