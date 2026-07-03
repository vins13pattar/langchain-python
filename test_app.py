import pytest

from app import calculator


def test_calculator_add():
    assert calculator(2, 3, "add") == 5


def test_calculator_subtract():
    assert calculator(5, 3, "subtract") == 2


def test_calculator_multiply():
    assert calculator(4, 3, "multiply") == 12


def test_calculator_divide():
    assert calculator(10, 2, "divide") == 5


def test_calculator_divide_by_zero():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        calculator(10, 0, "divide")


def test_calculator_unsupported_operation():
    with pytest.raises(ValueError, match="Unsupported operation"):
        calculator(10, 2, "modulo")
