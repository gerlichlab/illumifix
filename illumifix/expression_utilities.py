"""Extra utilities for working with functions and data types from the expression library"""

from collections.abc import Callable, Iterable
from typing import TypeVar

from expression import Result, curry_flip, result

_TInput = TypeVar("_TInput")
_TSource = TypeVar("_TSource")
_TError = TypeVar("_TError")


def separate(results: Iterable[Result[_TSource, _TError]]) -> tuple[list[_TError], list[_TSource]]:
    bads: list[_TError] = []
    goods: list[_TSource] = []
    for i, res in enumerate(results):
        match res:
            case result.Result(tag="ok", ok=g):
                goods.append(g)
            case result.Result(tag="error", error=e):
                bads.append(e)
            case unknown:
                raise TypeError(
                    f"Expected a Result, but got a {type(unknown).__name__} for the {i}-th element"
                )
    return bads, goods


def sequence_accumulate_errors(
    results: Iterable[Result[_TSource, _TError]],
) -> Result[list[_TSource], list[_TError]]:
    bads, goods = separate(results)
    return Result.Error(bads) if bads else Result.Ok(goods)


@curry_flip(1)
def traverse_accumulate_errors(
    inputs: Iterable[_TInput],
    func: Callable[[_TInput], Result[_TSource, _TError]],
) -> Result[list[_TSource], list[_TError]]:
    return sequence_accumulate_errors(map(func, inputs))
