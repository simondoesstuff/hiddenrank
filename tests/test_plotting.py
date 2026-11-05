"""Plot saved measurements with full grids and selected paper conditions."""
import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lowrank.plotting import _integration_figure, _integration_selection, _response_grid, plot_experiment


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
