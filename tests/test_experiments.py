"""End-to-end data generation checks, independent of plotting implementation."""
import json
import numpy as np
from lowrank.experiments import run_experiment


def test_go_nogo_saved_decisions_use_a_theoretical_driven_reference(tmp_path):
    output = tmp_path / 'go.npz'
    run_experiment('go_nogo', output, size=128, trials=1, seed=0,
                   duration=10., orthogonal=True)
    with np.load(output) as data:
        assert data['decisions'][0, 0]
        assert not data['decisions'][0, 1]
        assert data['upper_states'][0, 0] > .5
        assert np.isfinite(data['traces']).all()
        assert 'theoretical driven-Go' in json.loads(str(data['metadata']))['decision_threshold']


def test_independent_realization_shards_match_combined_run(tmp_path):
    combined, shard = tmp_path / 'combined.npz', tmp_path / 'shard.npz'
    run_experiment('context', combined, size=32, seed=2, trials=2, orthogonal=True,
                   duration=.4, baseline_duration=.4, transient_duration=.4)
    run_experiment('context', shard, size=32, seed=2, trials=1, trial_start=1, orthogonal=True,
                   duration=.4, baseline_duration=.4, transient_duration=.4)
    with np.load(combined) as a, np.load(shard) as b:
        np.testing.assert_array_equal(a['traces'][1], b['traces'][0])
        np.testing.assert_array_equal(a['kappa'][1], b['kappa'][0])
        assert b['trial_ids'][0] == 1


def test_aggregation_sorts_realizations_and_rejects_duplicate_ids(tmp_path):
    import pytest
    from lowrank.experiments import aggregate_experiments
    first, second, joined = (tmp_path / f'{name}.npz' for name in ('first', 'second', 'joined'))
    for trial, output in enumerate((first, second)):
        run_experiment('context', output, size=32, seed=2, trials=1, trial_start=trial,
                       orthogonal=True, duration=.4, baseline_duration=.4, transient_duration=.4)
    aggregate_experiments([second, first], joined)
    with np.load(joined) as data:
        np.testing.assert_array_equal(data['trial_ids'], [0, 1])
        assert data['traces'].shape[0] == 2
        assert json.loads(str(data['metadata']))['trials'] == 2
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_experiments([first, first], joined)
