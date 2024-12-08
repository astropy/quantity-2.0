"""Utility functions for the quantity package."""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Hashable, Mapping
from typing import Any, TypeVar, dataclass_transform, overload

import array_api_compat


def has_array_namespace(arg: object) -> bool:
    try:
        array_api_compat.array_namespace(arg)
    except TypeError:
        return False
    else:
        return True


# ===================================================================
# Dataclass utilities

_CT = TypeVar("_CT")


def field(
    *,
    converter: Callable[[Any], Any] | None = None,
    metadata: Mapping[Hashable, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Dataclass field with a converter argument.

    Parameters
    ----------
    converter : callable, optional
        A callable that converts the value of the field. This is added to the
        metadata of the field.
    metadata : Mapping[Hashable, Any], optional
        Additional metadata to add to the field.
        See `dataclasses.field` for more information.
    **kwargs : Any
        Additional keyword arguments to pass to `dataclasses.field`.

    """
    if converter is not None:
        # Check the converter
        if not callable(converter):
            msg = f"converter must be callable, got {converter!r}"
            raise TypeError(msg)

        # Convert the metadata to a mutable dict if it is not None.
        metadata = dict(metadata) if metadata is not None else {}

        if "converter" in metadata:
            msg = "Cannot specify 'converter' in metadata and as a keyword argument."
            raise ValueError(msg)

        # Add the converter to the metadata
        metadata["converter"] = converter

    return dataclasses.field(metadata=metadata, **kwargs)


def _process_dataclass(cls: type[_CT], **kwargs: Any) -> type[_CT]:
    # Make the dataclass from the class.
    # This does all the usual dataclass stuff.
    dcls: type[_CT] = dataclasses.dataclass(cls, **kwargs)

    # Compute the signature of the __init__ method
    sig = inspect.signature(dcls.__init__)
    # Eliminate the 'self' parameter
    sig = sig.replace(parameters=list(sig.parameters.values())[1:])
    # Store the signature on the __init__ method (Not assigning to __signature__
    # because that should have `self`).
    dcls.__init__._obj_signature_ = sig  # type: ignore[attr-defined]

    # Ensure that the __init__ method does conversion
    @functools.wraps(dcls.__init__)  # give it the same signature
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ba = self.__init__._obj_signature_.bind_partial(*args, **kwargs)
        ba.apply_defaults()  # so eligible for conversion

        # Convert the fields, if there's a converter
        for f in dataclasses.fields(self):
            k = f.name
            if k not in ba.arguments:  # mandatory field not provided?!
                continue  # defer the error to the dataclass __init__

            converter = f.metadata.get("converter")
            if converter is not None:
                ba.arguments[k] = converter(ba.arguments[k])

        #  Call the original dataclass __init__ method
        self.__init__.__wrapped__(self, *ba.args, **ba.kwargs)

    dcls.__init__ = __init__  # type: ignore[method-assign]

    return dcls


@overload
def dataclass(cls: type[_CT], /, **kwargs: Any) -> type[_CT]: ...


@overload
def dataclass(**kwargs: Any) -> Callable[[type[_CT]], type[_CT]]: ...


@dataclass_transform(field_specifiers=(dataclasses.Field, dataclasses.field, field))
def dataclass(
    cls: type[_CT] | None = None, /, **kwargs: Any
) -> type[_CT] | Callable[[type[_CT]], type[_CT]]:
    """Make a dataclass, supporting field converters.

    For more information about dataclasses see the `dataclasses` module.

    Parameters
    ----------
    cls : type | None, optional
        The class to transform into a dataclass.
        If None, returns a partial function that can be used as a decorator.
    **kwargs : Any
        Additional keyword arguments to pass to `dataclasses.dataclass`.

    """
    if cls is None:
        return functools.partial(_process_dataclass, **kwargs)
    return _process_dataclass(cls, **kwargs)
