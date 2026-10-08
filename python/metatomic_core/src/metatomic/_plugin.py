import ctypes
import json
import os
from collections.abc import Mapping
from typing import Optional, Union

from ._c_api import mta_model_t
from ._c_lib import _get_library
from ._model import ExternalModel


def load_plugin(path: Union[str, os.PathLike]):
    """
    Load the shared library at ``path`` and register the plugin contained within.

    Loading the same file multiple times only registers the plugin once, and the next
    calls do nothing.

    :param path: path to the plugin shared library
    """
    lib = _get_library()
    lib.mta_load_plugin(str(path).encode("utf8"))


def load_model(
    load_from: str,
    options: Optional[Mapping[str, str]] = None,
    plugin_name: Optional[str] = None,
) -> ExternalModel:
    """
    Load a model from ``load_from`` with the given ``options``.

    If ``plugin_name`` is ``None``, metatomic will try to determine the correct plugin
    to use by checking the ``load_from`` parameter. If we can not determine the correct
    plugin, we then try to load the model with each registered plugin until one
    succeeds.

    If ``plugin_name`` is given, then we only try to load the model with the specified
    plugin, and raise an error if the plugin can not load the model.

    :param load_from: where to load the model from (e.g. a file path, a model name,
        etc.). The interpretation of this string is up to the plugin.
    :param options: optional dictionary of string keys and string values used to
        configure the model. The interpretation of these options is up to the plugin.
    :param plugin_name: optional name of the plugin to use for loading the model
    :return: the loaded model
    """
    if options is None:
        options_json = None
    else:
        options_json = json.dumps(dict(options)).encode("utf8")

    if plugin_name is not None:
        plugin_name = str(plugin_name).encode("utf8")

    lib = _get_library()
    model = mta_model_t()
    lib.mta_load_model(
        str(load_from).encode("utf8"), options_json, plugin_name, ctypes.byref(model)
    )

    return ExternalModel(model)
