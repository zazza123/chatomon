import logging

import chatomon
from chatomon import ChatomonConfigError, ChatomonError, MissingExtraError
from chatomon.errors import MissingAnnotationsError


def test_version_is_exposed():
    assert chatomon.__version__ == "0.1.0.dev0"


def test_library_logger_has_only_a_null_handler():
    handlers = logging.getLogger("chatomon").handlers
    assert handlers
    assert all(isinstance(handler, logging.NullHandler) for handler in handlers)


def test_missing_extra_error_names_the_extra():
    error = MissingExtraError("principled", "orbitals")
    assert isinstance(error, ImportError)
    assert isinstance(error, ChatomonError)
    assert "pip install 'chatomon[principled]'" in str(error)
    assert error.extra == "principled"


def test_config_error_is_a_value_error():
    assert issubclass(ChatomonConfigError, ValueError)


def test_missing_annotations_error_lists_what_is_missing():
    error = MissingAnnotationsError("judge", ["claims", "intents"])
    assert error.orbital == "judge"
    assert error.missing == ("claims", "intents")
    assert "'claims'" in str(error)
