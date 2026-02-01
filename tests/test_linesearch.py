from math import sqrt
import pytest

from hypt.linesearch import GoldenSearch, LineSearch, NestedLineSearch


def test_nested_line_search_evaluates_expected_params_in_order():
    search = NestedLineSearch({
        'a': GoldenSearch(0, 1, 5),
        'b': LineSearch([0.0, 0.25, 0.5, 0.75, 1.0], patience=1),
        'c': 42,
    })

    def fobj(params):
        return params['a'] ** 2 + (params['b'] - 0.5) ** 2

    g = (sqrt(5) - 1) / 2
    expected_a_values = [1-g, g] + [g**i for i in range(3, 6)]
    expected_b_values = [0.0, 0.25, 0.5, 0.75]
    expected_params = [
        {'a': a, 'b': b, 'c': 42}
        for a in expected_a_values
        for b in expected_b_values
    ]

    for i, (params, expected) in enumerate(zip(search, expected_params)):
        assert params['a'] == pytest.approx(expected['a'])
        assert params['b'] == pytest.approx(expected['b'])
        assert params['c'] == expected['c']

        search.feedback(fobj(params))

    assert i + 1 == len(expected_params)
