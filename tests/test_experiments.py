"""End-to-end data generation checks, independent of plotting implementation."""
import json
import numpy as np
from lowrank.experiments import run_experiment


def test_go_nogo_saved_decisions_use_a_driven_reference(tmp_path):
    output = tmp_path / 'go.npz'
    run_experiment('go_nogo', output, size=128, trials=1, seed=0,
                   duration=10., orthogonal=True)
    with np.load(output) as data:
        assert data['decisions'][0, 0]
        assert not data['decisions'][0, 1]
        assert data['upper_states'][0, 0] > .5
        assert np.isfinite(data['traces']).all()
        assert 'driven-Go' in json.loads(str(data['metadata']))['decision_threshold']
