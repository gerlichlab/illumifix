"""Tests of helpers functions for working with types and functions from the expression library"""

from collections.abc import Callable
from typing import TypeVar

import pytest
from expression import Result, result
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from illumifix import expression_utilities as expr_util

_TSource = TypeVar("_TSource")
_TError = TypeVar("_TError")


def gen_anything(exceptions: type | tuple[type, ...] = tuple()) -> SearchStrategy[object]:
    return (
        st.from_type(type).flatmap(st.from_type).filter(lambda obj: not isinstance(obj, exceptions))
    )


def gen_result(
    *,
    gen_bad: Callable[[], SearchStrategy[_TError]],
    gen_good: Callable[[], SearchStrategy[_TSource]],
    gen_coinflip: Callable[[], SearchStrategy[bool]] = st.booleans,
) -> SearchStrategy[Result[_TSource, _TError]]:
    return gen_coinflip().flatmap(
        lambda p: gen_good().map(Result.Ok) if p else gen_bad().map(Result.Error)
    )


@given(results=st.lists(gen_result(gen_bad=gen_anything, gen_good=gen_anything)))
def test_separate__yields_correct_number_of_failures_and_successes(
    results: list[Result[object, object]],
):
    """The elements should be correctly binned based on the subtype of result wrapper."""
    exp_num_bad: int = 0
    exp_num_good: int = 0
    for res in results:
        if res.is_error():
            exp_num_bad += 1
        else:
            exp_num_good += 1
    bads, goods = expr_util.separate(results)
    assert len(bads) == exp_num_bad
    assert len(goods) == exp_num_good


@given(results=st.lists(gen_result(gen_bad=st.integers, gen_good=st.integers)))
def test_separate__preserves_element_order(results: list[Result[object, object]]):
    """The order of the elements in each result should be the same as in the original collection."""
    bads, goods = expr_util.separate(results)
    iter_bad = iter(bads)
    iter_good = iter(goods)
    for i, res in enumerate(results):
        match res:
            case result.Result(tag="ok", ok=exp):
                obs = next(iter_good)
                if not (obs == exp or obs is exp):
                    pytest.fail(f"At {i}-th result, {obs} != {exp}")
            case result.Result(tag="error", error=exp):
                obs = next(iter_bad)
                if not (obs == exp or obs is exp):
                    pytest.fail(f"At {i}-th result, {obs} != {exp}")
    try:
        next(iter_bad)
    except StopIteration:
        pass  # all good, supply expectedly exhausted
    else:
        pytest.fail("Iterator over failures not exhausted")
    try:
        next(iter_bad)
    except StopIteration:
        pass  # all good, supply expectedly exhausted
    else:
        pytest.fail("Iterator over successes not exhausted")


@given(
    results=st.sampled_from(
        (
            st.booleans,
            lambda: st.just(True),  # noqa: FBT003
        )
    ).flatmap(
        lambda gen_flip: st.lists(
            gen_result(
                gen_good=gen_anything,
                gen_bad=gen_anything,
                gen_coinflip=gen_flip,
            )
        )
    )
)
def test_sequence_accumulate_errors__result_wrapper_is_correct(
    results: list[Result[object, object]],
):
    """If any element is a failure, the total result is a failure."""
    match expr_util.sequence_accumulate_errors(results):
        case result.Result(tag="ok", ok=_):
            assert all(res.is_ok() for res in results)
        case result.Result(tag="error", error=_):
            assert any(res.is_error() for res in results)


@given(
    results=st.lists(
        gen_result(
            gen_bad=st.integers,  # Constrain the element type for simpler equality checking / assertion.
            gen_good=gen_anything,
        )
    ).filter(lambda results: any(res.is_error() for res in results))
)
def test_sequence_accumulate_errors__preserves_element_order_of_failures(
    results: list[Result[object, object]],
):
    """The order of the failures should be the same as in the original collection."""
    match expr_util.sequence_accumulate_errors(results):
        case result.Result(tag="error", error=failures):
            assert [Result.Error(e) for e in failures] == [res for res in results if res.is_error()]
        case other:
            pytest.fail(f"Expected a Result.Error, but got {type(other).__name__}")


@given(
    results=st.lists(
        gen_result(
            gen_bad=gen_anything,
            # Constrain the element type for simpler equality checking / assertion.
            gen_good=st.integers,
            # Generate all Result.Ok values.
            gen_coinflip=lambda: st.just(True),  # noqa: FBT003
        )
    )
)
def test_sequence_accumulate_errors__preserves_element_order_of_successes(
    results: list[Result[object, object]],
):
    """The order of the successes should be the same as in the original collection."""
    match expr_util.sequence_accumulate_errors(results):
        case result.Result(tag="ok", ok=successes):
            assert list(map(Result.Ok, successes)) == results
        case other:
            pytest.fail(f"Expected a Result.Ok, but got {type(other).__name__}")
