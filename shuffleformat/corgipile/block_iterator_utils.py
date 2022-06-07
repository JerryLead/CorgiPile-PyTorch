"""Iterator utils."""

from __future__ import division

import typing
import warnings
import random
import datetime

import numpy as np

import torch.distributed as dist


def shuffle_iterator(iterator: typing.Iterator,
                     buffer_size: int) -> typing.Iterable[typing.Any]:
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


