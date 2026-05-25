"""Base model and HDF5 persistence for OFDTD phase 0.

This module deliberately keeps the public object layer Pydantic-based while
storing numerical arrays as xarray objects in a plain HDF5 hierarchy.

Design notes
------------
* Every OFDTDBaseModel is immutable and can contain nested OFDTDBaseModel
  objects.
* xarray.DataArray and xarray.Dataset fields are stored under /__ofdtd_arrays__.
* A JSON metadata tree under /__ofdtd__/metadata_json stores model structure,
  scalar fields, collection structure, class import paths, and references to
  the xarray HDF5 groups.
* Raw numpy arrays are rejected at the model-serialization layer; wrap arrays in
  xr.DataArray or xr.Dataset so dims/coords/attrs survive the round trip.
"""

from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import h5py
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

_FORMAT_NAME = "ofdtd.base_model.hdf5"
_FORMAT_VERSION = 1
_META_GROUP = "__ofdtd__"
_ARRAY_GROUP = "__ofdtd_arrays__"
_META_DATASET = "metadata_json"
_KIND_KEY = "__ofdtd_kind__"
_MODEL_KIND = "model"
_DATAARRAY_REF_KIND = "xarray_dataarray_ref"
_DATASET_REF_KIND = "xarray_dataset_ref"
_LIST_KIND = "list"
_TUPLE_KIND = "tuple"
_DICT_KIND = "dict"
_COMPLEX_KIND = "complex"
_PATH_KIND = "path"
_ENUM_KIND = "enum"
_NONFINITE_FLOAT_KIND = "nonfinite_float"


class OFDTDSerializationError(RuntimeError):
    """Raised when an OFDTD object cannot be serialized or deserialized."""


class OFDTDBaseModel(BaseModel):
    """Immutable, recursively serializable base class for OFDTD objects.

    Subclasses may contain other ``OFDTDBaseModel`` instances and may contain
    ``xarray.DataArray`` or ``xarray.Dataset`` instances anywhere in the model
    tree, including inside lists, tuples, or dictionaries.

    Example
    -------
    >>> class FieldData(OFDTDBaseModel):
    ...     ex: xr.DataArray
    ...
    >>> fd = FieldData(ex=xr.DataArray([1.0, 2.0], dims=("x",)))
    >>> fd.write_hdf5("field.h5", overwrite=True)
    >>> round_trip = FieldData.read_hdf5("field.h5")
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    _registry: ClassVar[dict[str, type["OFDTDBaseModel"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        OFDTDBaseModel._registry[_qualified_name(cls)] = cls

    def updated_copy(self: Self, **updates: Any) -> Self:
        """Return a validated copy with selected fields replaced.

        Pydantic's ``model_copy(update=...)`` intentionally skips validation of
        the updated values, so this helper reconstructs the model through
        ``model_validate`` instead.
        """

        data = {field_name: getattr(self, field_name) for field_name in type(self).model_fields}
        data.update(updates)
        return type(self).model_validate(data)

    def write_hdf5(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        compression: str | None = "gzip",
        compression_opts: int | None = 4,
    ) -> None:
        """Write this model tree to an HDF5 file.

        Parameters
        ----------
        path:
            Destination ``.h5``/``.hdf5`` path.
        overwrite:
            If ``False``, raise ``FileExistsError`` when ``path`` already
            exists.
        compression:
            HDF5 compression filter applied to non-scalar numeric array data.
            Set to ``None`` to disable compression.
        compression_opts:
            Compression level/options forwarded to h5py.
        """

        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(path, "w") as h5:
            writer = _HDF5ModelWriter(
                h5=h5,
                compression=compression,
                compression_opts=compression_opts,
            )
            metadata = _encode_value(self, writer)
            writer.write_metadata(metadata)

    @classmethod
    def read_hdf5(cls: type[Self], path: str | Path) -> Self:
        """Read an OFDTD model tree from an HDF5 file.

        The root object is reconstructed using the class path stored in the
        file. Calling this on a subclass asserts that the stored root object is
        an instance of that subclass.
        """

        path = Path(path)
        with h5py.File(path, "r") as h5:
            reader = _HDF5ModelReader(h5=h5)
            metadata = reader.read_metadata()
            obj = _decode_value(metadata, reader)

        if not isinstance(obj, cls):
            raise TypeError(
                f"HDF5 file contains {_qualified_name(type(obj))}, "
                f"which is not an instance of {_qualified_name(cls)}."
            )
        return obj

    # Convenience aliases for later Tidy3D-like API spelling.
    def to_hdf5(self, path: str | Path, **kwargs: Any) -> None:
        self.write_hdf5(path, **kwargs)

    @classmethod
    def from_hdf5(cls: type[Self], path: str | Path) -> Self:
        return cls.read_hdf5(path)


class _HDF5ModelWriter:
    def __init__(
        self,
        *,
        h5: h5py.File,
        compression: str | None,
        compression_opts: int | None,
    ) -> None:
        self.h5 = h5
        self.compression = compression
        self.compression_opts = compression_opts
        self.array_count = 0
        self.h5.require_group(_ARRAY_GROUP)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        meta_group = self.h5.require_group(_META_GROUP)
        meta_group.attrs["format"] = _FORMAT_NAME
        meta_group.attrs["format_version"] = _FORMAT_VERSION
        meta_group.attrs["root_kind"] = _MODEL_KIND

        payload = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if _META_DATASET in meta_group:
            del meta_group[_META_DATASET]
        meta_group.create_dataset(_META_DATASET, data=np.void(payload))

    def write_dataarray(self, value: xr.DataArray) -> str:
        group_path = self._next_array_group_path()
        group = self.h5.create_group(group_path)
        _write_xarray_dataarray(
            group,
            value,
            include_coords=True,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        return group_path

    def write_dataset(self, value: xr.Dataset) -> str:
        group_path = self._next_array_group_path()
        group = self.h5.create_group(group_path)
        _write_xarray_dataset(
            group,
            value,
            compression=self.compression,
            compression_opts=self.compression_opts,
        )
        return group_path

    def _next_array_group_path(self) -> str:
        group_path = f"/{_ARRAY_GROUP}/array_{self.array_count:08d}"
        self.array_count += 1
        return group_path


class _HDF5ModelReader:
    def __init__(self, *, h5: h5py.File) -> None:
        self.h5 = h5

    def read_metadata(self) -> dict[str, Any]:
        if _META_GROUP not in self.h5:
            raise OFDTDSerializationError(f"Missing HDF5 group /{_META_GROUP}")
        meta_group = self.h5[_META_GROUP]
        if meta_group.attrs.get("format") != _FORMAT_NAME:
            raise OFDTDSerializationError(
                f"Not an OFDTD base-model file: {meta_group.attrs.get('format')!r}"
            )
        version = int(meta_group.attrs.get("format_version", -1))
        if version != _FORMAT_VERSION:
            raise OFDTDSerializationError(
                f"Unsupported OFDTD HDF5 format version {version}; "
                f"expected {_FORMAT_VERSION}."
            )
        if _META_DATASET not in meta_group:
            raise OFDTDSerializationError(
                f"Missing HDF5 dataset /{_META_GROUP}/{_META_DATASET}"
            )

        raw = meta_group[_META_DATASET][()]
        if isinstance(raw, np.void):
            payload = bytes(raw).decode("utf-8")
        elif isinstance(raw, bytes):
            payload = raw.decode("utf-8")
        else:
            payload = str(raw)
        return json.loads(payload)

    def read_dataarray(self, group_path: str) -> xr.DataArray:
        if group_path not in self.h5:
            raise OFDTDSerializationError(f"Missing xarray DataArray group {group_path}")
        return _read_xarray_dataarray(self.h5[group_path])

    def read_dataset(self, group_path: str) -> xr.Dataset:
        if group_path not in self.h5:
            raise OFDTDSerializationError(f"Missing xarray Dataset group {group_path}")
        return _read_xarray_dataset(self.h5[group_path])


def _encode_value(value: Any, writer: _HDF5ModelWriter) -> Any:
    """Encode arbitrary supported model-field value into JSON metadata."""

    if isinstance(value, OFDTDBaseModel):
        fields: dict[str, Any] = {}
        for field_name in type(value).model_fields:
            fields[field_name] = _encode_value(getattr(value, field_name), writer)
        return {
            _KIND_KEY: _MODEL_KIND,
            "class": _qualified_name(type(value)),
            "fields": fields,
        }

    if isinstance(value, xr.DataArray):
        return {
            _KIND_KEY: _DATAARRAY_REF_KIND,
            "group": writer.write_dataarray(value),
        }

    if isinstance(value, xr.Dataset):
        return {
            _KIND_KEY: _DATASET_REF_KIND,
            "group": writer.write_dataset(value),
        }

    if isinstance(value, np.ndarray):
        raise TypeError(
            "Raw numpy.ndarray fields are not supported by OFDTDBaseModel HDF5 "
            "serialization. Wrap the array in xr.DataArray or xr.Dataset so "
            "dims/coords/attrs are explicit."
        )

    if isinstance(value, np.generic):
        return _encode_value(value.item(), writer)

    if isinstance(value, Path):
        return {_KIND_KEY: _PATH_KIND, "value": str(value)}

    if isinstance(value, Enum):
        return {
            _KIND_KEY: _ENUM_KIND,
            "class": _qualified_name(type(value)),
            "value": _encode_value(value.value, writer),
        }

    if isinstance(value, complex):
        return {_KIND_KEY: _COMPLEX_KIND, "real": value.real, "imag": value.imag}

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            marker = "nan"
        elif value > 0:
            marker = "inf"
        else:
            marker = "-inf"
        return {_KIND_KEY: _NONFINITE_FLOAT_KIND, "value": marker}

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, tuple):
        return {
            _KIND_KEY: _TUPLE_KIND,
            "items": [_encode_value(item, writer) for item in value],
        }

    if isinstance(value, list):
        return {
            _KIND_KEY: _LIST_KIND,
            "items": [_encode_value(item, writer) for item in value],
        }

    if isinstance(value, Mapping):
        return {
            _KIND_KEY: _DICT_KIND,
            "items": [
                [_encode_value(key, writer), _encode_value(item, writer)]
                for key, item in value.items()
            ],
        }

    raise TypeError(
        f"Unsupported field value {value!r} of type {_qualified_name(type(value))}. "
        "Add an encoder for this type or convert it before writing HDF5."
    )


def _decode_value(node: Any, reader: _HDF5ModelReader) -> Any:
    """Decode a JSON metadata node back into Python objects."""

    if not isinstance(node, dict) or _KIND_KEY not in node:
        return node

    kind = node[_KIND_KEY]

    if kind == _MODEL_KIND:
        model_cls = _resolve_model_class(node["class"])
        fields = {
            field_name: _decode_value(encoded_value, reader)
            for field_name, encoded_value in node["fields"].items()
        }
        return model_cls.model_validate(fields)

    if kind == _DATAARRAY_REF_KIND:
        return reader.read_dataarray(node["group"])

    if kind == _DATASET_REF_KIND:
        return reader.read_dataset(node["group"])

    if kind == _LIST_KIND:
        return [_decode_value(item, reader) for item in node["items"]]

    if kind == _TUPLE_KIND:
        return tuple(_decode_value(item, reader) for item in node["items"])

    if kind == _DICT_KIND:
        return {
            _decode_value(key, reader): _decode_value(value, reader)
            for key, value in node["items"]
        }

    if kind == _COMPLEX_KIND:
        return complex(float(node["real"]), float(node["imag"]))

    if kind == _PATH_KIND:
        return Path(node["value"])

    if kind == _ENUM_KIND:
        enum_cls = _resolve_class(node["class"])
        return enum_cls(_decode_value(node["value"], reader))

    if kind == _NONFINITE_FLOAT_KIND:
        value = node["value"]
        if value == "nan":
            return float("nan")
        if value == "inf":
            return float("inf")
        if value == "-inf":
            return float("-inf")
        raise OFDTDSerializationError(f"Unknown non-finite float marker: {value!r}")

    raise OFDTDSerializationError(f"Unknown encoded metadata kind: {kind!r}")


def _write_xarray_dataset(
    group: h5py.Group,
    value: xr.Dataset,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    group.attrs["xarray_kind"] = "Dataset"
    group.attrs["attrs_json"] = json.dumps(_json_safe(value.attrs), allow_nan=False)

    coords_group = group.create_group("coords")
    _write_named_dataarrays(
        coords_group,
        value.coords,
        include_coords=False,
        compression=compression,
        compression_opts=compression_opts,
    )

    data_vars_group = group.create_group("data_vars")
    _write_named_dataarrays(
        data_vars_group,
        value.data_vars,
        include_coords=False,
        compression=compression,
        compression_opts=compression_opts,
    )


def _read_xarray_dataset(group: h5py.Group) -> xr.Dataset:
    if group.attrs.get("xarray_kind") != "Dataset":
        raise OFDTDSerializationError(f"HDF5 group {group.name} is not an xarray Dataset")

    coords = _read_named_dataarrays(group["coords"])
    data_vars = _read_named_dataarrays(group["data_vars"])
    attrs = _json_restore(json.loads(group.attrs.get("attrs_json", "{}")))
    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def _write_named_dataarrays(
    group: h5py.Group,
    values: Mapping[Any, xr.DataArray],
    *,
    include_coords: bool,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    order: list[str] = []
    for index, (name, dataarray) in enumerate(values.items()):
        if not isinstance(name, str):
            raise TypeError(
                f"Only string xarray names are supported in HDF5 persistence; got {name!r}."
            )
        child_name = f"item_{index:08d}"
        order.append(child_name)
        child_group = group.create_group(child_name)
        child_group.attrs["mapping_key_json"] = json.dumps(name, allow_nan=False)
        _write_xarray_dataarray(
            child_group,
            dataarray,
            include_coords=include_coords,
            compression=compression,
            compression_opts=compression_opts,
        )
    group.attrs["order_json"] = json.dumps(order, allow_nan=False)


def _read_named_dataarrays(group: h5py.Group) -> dict[str, xr.DataArray]:
    result: dict[str, xr.DataArray] = {}
    order = json.loads(group.attrs.get("order_json", "[]"))
    for child_name in order:
        child_group = group[child_name]
        key = json.loads(child_group.attrs["mapping_key_json"])
        result[key] = _read_xarray_dataarray(child_group)
    return result


def _write_xarray_dataarray(
    group: h5py.Group,
    value: xr.DataArray,
    *,
    include_coords: bool,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    group.attrs["xarray_kind"] = "DataArray"
    group.attrs["name_json"] = json.dumps(value.name, allow_nan=False)
    group.attrs["dims_json"] = json.dumps(list(value.dims), allow_nan=False)
    group.attrs["attrs_json"] = json.dumps(_json_safe(value.attrs), allow_nan=False)

    _write_hdf5_array(
        group,
        "data",
        value.data,
        compression=compression,
        compression_opts=compression_opts,
    )

    coords_group = group.create_group("coords")
    if include_coords:
        _write_named_dataarrays(
            coords_group,
            value.coords,
            include_coords=False,
            compression=compression,
            compression_opts=compression_opts,
        )
    else:
        coords_group.attrs["order_json"] = json.dumps([], allow_nan=False)


def _read_xarray_dataarray(group: h5py.Group) -> xr.DataArray:
    if group.attrs.get("xarray_kind") != "DataArray":
        raise OFDTDSerializationError(f"HDF5 group {group.name} is not an xarray DataArray")

    data = _read_hdf5_array(group["data"])
    dims = tuple(json.loads(group.attrs["dims_json"]))
    name = json.loads(group.attrs.get("name_json", "null"))
    attrs = _json_restore(json.loads(group.attrs.get("attrs_json", "{}")))
    coords = _read_named_dataarrays(group["coords"])
    return xr.DataArray(data=data, dims=dims, coords=coords, name=name, attrs=attrs)


def _write_hdf5_array(
    group: h5py.Group,
    name: str,
    value: Any,
    *,
    compression: str | None,
    compression_opts: int | None,
) -> None:
    array = np.asarray(value)
    data: Any = array
    dtype: Any | None = None
    encoding: str | None = None
    original_dtype: str | None = None

    if array.dtype.kind in {"U", "S"}:
        encoding = "utf8_string"
        dtype = h5py.string_dtype(encoding="utf-8")
        data = array.astype(object)
    elif array.dtype.kind == "O":
        if _object_array_is_string_like(array):
            encoding = "utf8_string"
            dtype = h5py.string_dtype(encoding="utf-8")
            data = _object_array_to_string_array(array)
        else:
            raise TypeError(
                f"Object-dtype xarray data at {group.name}/{name} cannot be stored safely. "
                "Use numeric, bool, complex, datetime, timedelta, or string data."
            )
    elif array.dtype.kind == "M":
        encoding = "datetime64_as_int64"
        original_dtype = str(array.dtype)
        data = array.view("int64")
    elif array.dtype.kind == "m":
        encoding = "timedelta64_as_int64"
        original_dtype = str(array.dtype)
        data = array.view("int64")

    create_kwargs: dict[str, Any] = {}
    can_compress = (
        compression is not None
        and np.shape(data) != ()
        and encoding != "utf8_string"
        and np.asarray(data).size > 0
    )
    if can_compress:
        create_kwargs["compression"] = compression
        if compression_opts is not None:
            create_kwargs["compression_opts"] = compression_opts

    dataset = group.create_dataset(name, data=data, dtype=dtype, **create_kwargs)
    if encoding is not None:
        dataset.attrs["encoding"] = encoding
    if original_dtype is not None:
        dataset.attrs["original_dtype"] = original_dtype


def _read_hdf5_array(dataset: h5py.Dataset) -> np.ndarray | np.generic:
    encoding = dataset.attrs.get("encoding")

    if encoding == "utf8_string":
        value = dataset.asstr()[()]
        return np.asarray(value)

    value = dataset[()]
    if encoding in {"datetime64_as_int64", "timedelta64_as_int64"}:
        original_dtype = np.dtype(dataset.attrs["original_dtype"])
        return np.asarray(value).view(original_dtype)
    return value


def _object_array_is_string_like(array: np.ndarray) -> bool:
    for item in array.ravel():
        if item is not None and not isinstance(item, (str, bytes, np.str_, np.bytes_)):
            return False
    return True


def _object_array_to_string_array(array: np.ndarray) -> np.ndarray:
    def normalize(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, (bytes, np.bytes_)):
            return bytes(item).decode("utf-8")
        return str(item)

    return np.vectorize(normalize, otypes=[object])(array)


def _json_safe(value: Any) -> Any:
    """Convert xarray attrs to a JSON-safe representation."""

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if math.isnan(value):
            marker = "nan"
        elif value > 0:
            marker = "inf"
        else:
            marker = "-inf"
        return {_KIND_KEY: _NONFINITE_FLOAT_KIND, "value": marker}

    if isinstance(value, complex):
        return {_KIND_KEY: _COMPLEX_KIND, "real": value.real, "imag": value.imag}

    if isinstance(value, np.generic):
        return _json_safe(value.item())

    if isinstance(value, np.ndarray):
        return {
            _KIND_KEY: "json_numpy_array",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "value": value.tolist(),
        }

    if isinstance(value, Path):
        return {_KIND_KEY: _PATH_KIND, "value": str(value)}

    if isinstance(value, Enum):
        return {
            _KIND_KEY: _ENUM_KIND,
            "class": _qualified_name(type(value)),
            "value": _json_safe(value.value),
        }

    if isinstance(value, tuple):
        return {_KIND_KEY: _TUPLE_KIND, "items": [_json_safe(v) for v in value]}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}

    return {
        _KIND_KEY: "repr",
        "class": _qualified_name(type(value)),
        "value": repr(value),
    }


def _json_restore(value: Any) -> Any:
    if not isinstance(value, dict) or _KIND_KEY not in value:
        if isinstance(value, list):
            return [_json_restore(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_restore(v) for k, v in value.items()}
        return value

    kind = value[_KIND_KEY]
    if kind == _NONFINITE_FLOAT_KIND:
        return _decode_value(value, _NoArrayReader())
    if kind == _COMPLEX_KIND:
        return complex(value["real"], value["imag"])
    if kind == _PATH_KIND:
        return Path(value["value"])
    if kind == _ENUM_KIND:
        enum_cls = _resolve_class(value["class"])
        return enum_cls(_json_restore(value["value"]))
    if kind == _TUPLE_KIND:
        return tuple(_json_restore(v) for v in value["items"])
    if kind == "json_numpy_array":
        return np.asarray(value["value"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
    if kind == "repr":
        return value["value"]
    return value


class _NoArrayReader:
    """Tiny stand-in used only for restoring non-array attr markers."""


def _qualified_name(cls: type[Any]) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_model_class(path: str) -> type[OFDTDBaseModel]:
    if path in OFDTDBaseModel._registry:
        return OFDTDBaseModel._registry[path]
    cls = _resolve_class(path)
    if not isinstance(cls, type) or not issubclass(cls, OFDTDBaseModel):
        raise OFDTDSerializationError(f"Resolved class {path!r} is not an OFDTDBaseModel")
    OFDTDBaseModel._registry[path] = cls
    return cls


def _resolve_class(path: str) -> type[Any]:
    module_name, sep, qualname = path.partition(":")
    if not sep:
        raise OFDTDSerializationError(f"Invalid qualified class path {path!r}")
    module = importlib.import_module(module_name)
    obj: Any = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    if not isinstance(obj, type):
        raise OFDTDSerializationError(f"Qualified path {path!r} did not resolve to a class")
    return obj
