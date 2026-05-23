"""Configuration models for ofdtd."""

from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Configuration for a basic FDTD simulation.

    Parameters
    ----------
    nx:
        Number of cells in the x direction.
    ny:
        Number of cells in the y direction.
    nz:
        Number of cells in the z direction.
    dx:
        Grid spacing.
    dt:
        Time step.
    """

    nx: int = Field(gt=0, description="Number of cells in x.")
    ny: int = Field(gt=0, description="Number of cells in y.")
    nz: int = Field(gt=0, description="Number of cells in z.")
    dx: float = Field(gt=0, description="Grid spacing.")
    dt: float = Field(gt=0, description="Time step.")
