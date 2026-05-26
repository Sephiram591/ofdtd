from pathlib import Path

import numpy as np
import xarray_jax
import jax
import jax.numpy as jnp
import pytest
import xarray as xr
from pydantic import ValidationError

from ofdtd.base import OFDTDBaseModel


class FieldData(OFDTDBaseModel):
    ex: xr.DataArray


class SimulationData(OFDTDBaseModel):
    name: str
    field: FieldData
    fields: tuple[FieldData, ...]
    dataset: xr.Dataset
    output_path: Path
    phase: complex
    metadata: dict[str, object]


def test_nested_model_hdf5_round_trip(tmp_path: Path) -> None:
    x = jnp.linspace(0.0, 1.0, 5)
    ex = xr.DataArray(
        jnp.sin(2.0 * np.pi * x),
        dims=("x",),
        coords={"x": x},
        name="Ex",
        attrs={"units": "V/m", "scale": jnp.array([1.0, 2.0])},
    )
    ds = xr.Dataset(
        data_vars={"ex": ex},
        coords={"x": x},
        attrs={"description": "test dataset"},
    )
    data = SimulationData(
        name="round-trip",
        field=FieldData(ex=ex),
        fields=(FieldData(ex=ex),),
        dataset=ds,
        output_path=tmp_path / "field.h5",
        phase=1.0 + 2.0j,
        metadata={"alpha": 1.0, "missing": None},
    )

    path = tmp_path / "simulation_data.h5"
    data.write_hdf5(path)
    print(type(data.field.ex.data))
    loaded = SimulationData.read_hdf5(path)

    assert loaded.name == data.name
    assert loaded.field.ex.equals(ex)
    print(type(loaded.field.ex.data))
    assert isinstance(loaded.field.ex.data, jax.Array)
    assert loaded.fields[0].ex.equals(ex)
    assert loaded.dataset.equals(ds)
    assert loaded.output_path == data.output_path
    assert loaded.phase == data.phase
    assert loaded.metadata == data.metadata


def test_write_hdf5_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    field = FieldData(ex=xr.DataArray([1.0], dims=("x",)))
    path = tmp_path / "field.h5"

    field.write_hdf5(path)

    with pytest.raises(FileExistsError):
        field.write_hdf5(path)


def test_raw_numpy_array_fields_are_rejected(tmp_path: Path) -> None:
    class BadData(OFDTDBaseModel):
        values: object

    bad = BadData(values=np.array([1.0, 2.0]))

    with pytest.raises(TypeError, match="Raw numpy.ndarray"):
        bad.write_hdf5(tmp_path / "bad.h5")


def test_updated_copy_validates_updates() -> None:
    class PositiveValue(OFDTDBaseModel):
        value: int

    model = PositiveValue(value=1)
    updated = model.updated_copy(value=2)

    assert updated.value == 2

    with pytest.raises(ValidationError):
        model.updated_copy(value="not an int")
