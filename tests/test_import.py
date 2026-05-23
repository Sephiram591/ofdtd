from ofdtd import SimulationConfig, make_grid


def test_make_grid_shape() -> None:
    config = SimulationConfig(nx=4, ny=5, nz=6, dx=0.1, dt=0.01)
    grid = make_grid(config)

    assert grid.shape == (4, 5, 6)
