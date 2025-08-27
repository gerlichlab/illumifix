"""Extra utilities for working with functions and data types from the expression library"""

from collections.abc import Callable, Iterable
from typing import TypeVar

from expression import Result, curry_flip, result

_TInput = TypeVar("_TInput")
_TSource = TypeVar("_TSource")
_TError = TypeVar("_TError")


def separate(results: Iterable[Result[_TSource, _TError]]) -> tuple[list[_TError], list[_TSource]]:
    """Separate the given collection into errors and Oks.

    Args:
        results (Iterable[Result[_TSource, _TError]]): the collection of wrapped values to split into errors and Oks.

    Raises:
        TypeError: if any element in the given collection isn't a result.Result

    Returns:
        tuple[list[_TError], list[_TSource]]: a pair of lists,
        the first of which is the errors and the second of which is the Oks

    """
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
    """Return a result.Error wrapping all such values if any exists, otherwise a result.Ok wrapping all Ok-wrapped values

    Args:
        results (Iterable[Result[_TSource, _TError]]): the collection of values to process

    Returns:
        Result[list[_TSource], list[_TError]]: a result.Ok wrapping all values if every one is a result.Ok,
        otherwise a result.Error wrapping a list of all result.Error-wrapped values

    """
    bads, goods = separate(results)
    return Result.Error(bads) if bads else Result.Ok(goods)


@curry_flip(1)
def traverse_accumulate_errors(
    inputs: Iterable[_TInput],
    func: Callable[[_TInput], Result[_TSource, _TError]],
) -> Result[list[_TSource], list[_TError]]:
    """Apply the given function to each element, returning a result.Ok-wrapped collection of resulting values if every application succeeds, otherwise a result.Error wrapping all error values.

    Args:
        inputs (Iterable[_TInput]): the elements to input to the given function
        func (Callable[[_TInput], Result[_TSource, _TError]]): the function to apply to each element

    Returns:
        Result[list[_TSource], list[_TError]]: either a result.Ok-wrapped collection of resulting values
        of every application succeeds, otherwise a result.Error wrapping all error values

    """
    return sequence_accumulate_errors(map(func, inputs))
