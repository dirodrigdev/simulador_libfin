```python
import numpy as np
from simulador import SimulationConfig, AssetBucket, WithdrawalTramo, AdvancedSimulator

def test_simulator_shapes_and_positive_cpi():
    cfg = SimulationConfig(horizon_years=2, steps_per_year=12, initial_capital=1_000_000, n_sims=100, inflation_mean=0.01, inflation_vol=0.01, random_seed=42)
    assets = [AssetBucket("RV", 0.6, 0.10, 0.15), AssetBucket("RF", 0.4, 0.03, 0.05)]
    wds = [WithdrawalTramo(0, 2, 1000)]
    sim = AdvancedSimulator(cfg, assets, wds)
    paths, cpi = sim.run()
    assert paths.shape == (100, 2*12 + 1)
    assert cpi.shape == (100, 2*12 + 1)
    assert np.all(cpi > 0)
    assert np.all(np.isfinite(paths))

def test_reproducibility_with_seed():
    cfg1 = SimulationConfig(horizon_years=1, steps_per_year=12, initial_capital=100000, n_sims=50, inflation_mean=0.01, inflation_vol=0.01, random_seed=123)
    cfg2 = SimulationConfig(horizon_years=1, steps_per_year=12, initial_capital=100000, n_sims=50, inflation_mean=0.01, inflation_vol=0.01, random_seed=123)
    assets = [AssetBucket("RV", 0.6, 0.10, 0.15), AssetBucket("RF", 0.4, 0.03, 0.05)]
    wds = [WithdrawalTramo(0, 1, 1000)]
    sim1 = AdvancedSimulator(cfg1, assets, wds)
    sim2 = AdvancedSimulator(cfg2, assets, wds)
    p1, c1 = sim1.run()
    p2, c2 = sim2.run()
    assert np.allclose(p1, p2)
    assert np.allclose(c1, c2)
