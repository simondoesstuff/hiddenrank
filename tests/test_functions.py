import pytest

from example.functions import square


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (3.0, 9.0), (-3.0, 9.0), (2.5, 6.25)],
)
def test_square(value: float, expected: float) -> None:
    assert square(value) == pytest.approx(expected)
