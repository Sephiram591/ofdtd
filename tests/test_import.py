from ofdtd import OFDTDBaseModel as ExportedBaseModel
from ofdtd.base import OFDTDBaseModel, OFDTDSerializationError


def test_public_imports_are_available() -> None:
    assert ExportedBaseModel is OFDTDBaseModel
    assert issubclass(OFDTDSerializationError, RuntimeError)
