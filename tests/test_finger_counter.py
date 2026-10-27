
import pytest

from src.utils.finger_counter import count_raised_fingers


def test_closed_fist():
    assert count_raised_fingers([False, False, False, False, False]) == 0


def test_open_palm():
    assert count_raised_fingers([True, True, True, True, True]) == 5


def test_three_fingers():
    assert count_raised_fingers([False, True, True, True, False]) == 3


def test_invalid_number_of_fingers():
    with pytest.raises(ValueError):
        count_raised_fingers([True, False])


def test_boolean_and_numeric_states():
    assert count_raised_fingers([1, 0, 1, 0, 1]) == 3
