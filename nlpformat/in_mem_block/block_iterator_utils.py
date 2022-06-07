"""Iterator utils."""

from __future__ import division

import typing
import warnings
import random
import datetime

import numpy as np


def cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element


def sample_iterators(iterators: typing.List[typing.Iterator],
                     ratios: typing.List[int]) -> typing.Iterable[typing.Any]:
    """Retrieve info generated from the iterator(s) according to their
    sampling ratios.

    Params:
    -------
    iterators: list of iterators
        All iterators (one for each file).

    ratios: list of int
        The ratios with which to sample each iterator.

    Yields:
    -------
    item: Any
        Decoded bytes of features into its respective data types from
        an iterator (based off their sampling ratio).
    """
    iterators = [cycle(iterator) for iterator in iterators]
    ratios = np.array(ratios)
    ratios = ratios / ratios.sum()
    while True:
        choice = np.random.choice(len(ratios), p=ratios)
        yield next(iterators[choice])


def shuffle_iterator(iterator: typing.Iterator,
                     buffer_size: int) -> typing.Iterable[typing.Any]:
    """Shuffle elements contained in an iterator.

    Params:
    -------
    iterator: iterator
        The iterator.

    queue_size: int
        Length of buffer. Determines how many records are queued to
        sample from.

    Yields:
    -------
    item: Any
        Decoded bytes of the features into its respective data type (for
        an individual record) from an iterator.
    """
    random.seed()
    end_file = False

    while end_file is False:
        buffer = []
        try:
            for _ in range(buffer_size):
                record = next(iterator)
                buffer.append(record)

        except StopIteration:
            end_file = True

        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item
        
