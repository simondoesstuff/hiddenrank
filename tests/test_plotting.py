"""Plot saved measurements with full grids and selected paper conditions."""
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lowrank.plotting import _integration_figure, _integration_groups, _integration_selection, _response_grid, plot_experiment


def _data(stimulus, contexts) -> dict[str, Any]:
    times = np.linspace(0, 5, 11)
    conditions = len(contexts)
    ramp = np.linspace(0, 1, len(times))
    return dict(times=times, stimulus=np.asarray(stimulus), contexts=np.asarray(contexts),
                traces=np.broadcast_to(ramp, (2, conditions, len(times))),
                projections=np.broadcast_to(ramp[:, None], (2, conditions, len(times), 3)),
                kappa=np.broadcast_to(ramp[:, None], (2, conditions, len(times), 2)),
                decisions=np.zeros((2, conditions)),
                metadata=json.dumps(dict(name="integration", size=128, trials=2, seed=0,
                    baseline_duration=1., stimulus_end=4., g=.8)))


def test_selection_finds_target_independently_of_grid_order():
    data = _data([[.9, .9], [.6, .1], [.6, .1], [.2, .2]], [1, 0, 1, 0])
    assert _integration_selection(data) == [1, 2]


def test_response_grid_respects_coordinates_and_rejects_sparse_subset():
    result = _response_grid(np.array([[1, 1], [0, 0], [1, 0], [0, 1]]), [4, 1, 2, 3])
    assert result is not None
    a, b, values = result
    np.testing.assert_array_equal(a, [0, 1])
    np.testing.assert_array_equal(b, [0, 1])
    np.testing.assert_array_equal(values, [[1, 2], [3, 4]])
    assert _response_grid(np.array([[0, 0], [1, 1]]), [0, 1]) is None


@pytest.mark.parametrize("full_grid", [False, True])
def test_integration_saved_figure_supports_representative_or_full_grid(tmp_path, full_grid):
    pairs = [[a, b] for a in [0., 1.] for b in [0., 1.]] if full_grid else [[.6, .1]]
    data = _data(pairs * 2, [0] * len(pairs) + [1] * len(pairs))
    source = tmp_path / "integration.npz"
    np.savez(source, **data)
    output = plot_experiment(source, tmp_path / "integration.svg")
    text = output.read_text()
    assert "population state" in text
    assert "Eq. 156 scaffold" in text
    assert "response probability" in text
    assert "Recurrent population overlaps" in text


def test_legacy_integration_archive_marks_missing_overlap_data():
    data = _data([[.6, .1], [.6, .1]], [0, 1])
    del data["kappa"]
    fig = _integration_figure(data, json.loads(data["metadata"]))
    assert any("unavailable" in text.get_text() for ax in fig.axes for text in ax.texts)
    plt.close(fig)


def test_scaffold_coordinates_preferred_to_recurrent_overlaps():
    data = _data([[.6, .1], [.6, .1]], [0, 1])
    data["scaffold_coordinates"] = np.ones((2, 2, 11, 2)) * 2
    meta = json.loads(data["metadata"])
    meta["architecture"] = "supplement"
    fig = _integration_figure(data, meta)
    axis = next(ax for ax in fig.axes if ax.get_title() == "Population scaffold trajectories")
    assert "scaffold coordinate" in axis.get_xlabel()
    # Grouped per-realization paths come first; the kappa fixture stays below one.
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), 2.)
    plt.close(fig)


def test_scaffold_ring_uses_architecture_specific_right_pattern_variance():
    from lowrank.plotting import _integration_ring
    paper = _integration_ring(architecture="paper")
    supplement = _integration_ring(architecture="supplement")
    assert np.linalg.norm(supplement[0]) > np.linalg.norm(paper[0])
    with pytest.raises(ValueError, match="architecture"):
        _integration_ring(architecture="invalid")


def test_grouped_paths_cancel_irrelevant_input_and_preserve_failed_trials():
    pairs = [[a, b] for a in [.1, .6] for b in [-1., 1.]]
    data = _data(pairs * 2, [0] * 4 + [1] * 4)
    # Context A tracks A; context B tracks B. Independent nuisance terms
    # cancel under equal averaging, including a failed/negative realization.
    values = np.empty((2, 8, 11, 3))
    for condition, (a, b) in enumerate(data["stimulus"]):
        context = data["contexts"][condition]
        relevant, nuisance = (a, b) if context == 0 else (b, a - .35)
        values[:, condition] = (np.array([-2., 4.]) + relevant + nuisance)[:, None, None]
    data["projections"] = values
    groups = _integration_groups(data, "projections")
    assert [(context, strength) for context, strength, _ in groups] == [
        (0, .1), (0, .6), (1, -1.), (1, 1.)]
    for _, strength, paths in groups:
        np.testing.assert_allclose(paths[:, 0, 0], [-2. + strength, 4. + strength])
        np.testing.assert_allclose(paths.mean(axis=0), 1. + strength)
    # Reordering and repeating a condition do not overweight its nuisance value.
    order = [7, 0, 4, 2, 1, 6, 3, 5, 0]
    shuffled = dict(data, stimulus=data["stimulus"][order], contexts=data["contexts"][order],
                    projections=values[:, order])
    for actual, expected in zip(_integration_groups(shuffled, "projections"), groups, strict=True):
        assert actual[:2] == expected[:2]
        np.testing.assert_allclose(actual[2], expected[2])


def test_grouped_paths_support_sparse_representative_conditions():
    data = _data([[.6, .1], [.6, .1]], [0, 1])
    groups = _integration_groups(data, "projections")
    assert [(context, strength) for context, strength, _ in groups] == [(0, .6), (1, .1)]
    np.testing.assert_array_equal(groups[0][2], data["projections"][:, 0])
    np.testing.assert_array_equal(groups[1][2], data["projections"][:, 1])
