"""Iterator utils."""

from __future__ import division

import typing
import warnings
import random
import io
import pickle

import numpy as np


def cycle(iterator_fn: typing.Callable) -> typing.Iterable[typing.Any]:
    """Create a repeating iterator from an iterator generator."""
    while True:
        for element in iterator_fn():
            yield element


def shuffle_iterator(iterator: typing.Iterator,
                     io_buffer: typing.List,
                     io_buffer_size: int,  # sliding_buffer
                     total_records_num: int,
                     old_buffer: typing.List,
                     select_ratio_from_old_buffer: float,
                      file_writer: io.FileIO
                     ) -> typing.Iterable[typing.Any]:
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
    #print(current_buffer_dataset_indexes)
    #select_ratio_from_old_buffer = float(num_records_from_old_buffer / total_records_num)
   
    # Read the first m items and fill the reservoir.
    current_dataset_index = 0
    current_buffer_index = 0

    try:
        if len(io_buffer) == 0:
            for _ in range(io_buffer_size):
                io_buffer.append(next(iterator))
                current_dataset_index += 1

    except StopIteration:
        warnings.warn("Number of elements in the iterator is less than the "
                      f"buffer size (N={io_buffer_size}).")

    old_buffer_len = len(old_buffer)

    # if old_buffer:
    #     print(old_buffer[0])
    # selected_record_count = 0
    # the old_buffer is empty in the first epoch
    

    while current_dataset_index < total_records_num:
        # first time: old_buffer_len == 0
        # The I/O Worker reads example tuple e from the database, and uses buffer A to do reservoir sampling. 
        # The dropped example d is used for the gradient step, with updates to a shared model.
        if old_buffer_len == 0 or random.random() >= select_ratio_from_old_buffer:
            # Read the first m items and fill the reservoir. Then, when we read the kth additional item (m + k overall),
            # we randomly select an integer s in [0, m + k).
            # If s < m, then we put the item at slot s; otherwise we send the item to SGD.
            
            index = random.randint(0, current_dataset_index)
            
            if index < io_buffer_size:
                item = io_buffer[index]
                io_buffer[index] = next(iterator)
            else:
                item = next(iterator)
            current_dataset_index += 1
            yield item
            # selected_record_count += 1

            # After the I/O Worker finishes one pass over the data, the buffers are swapped.
            # That is, the I/O Worker begins filling the buffer that the Memory Worker is using,
            # while the Memory Worker works on the buffer that has just been filled by the I/O Worker.
            if current_dataset_index == total_records_num:
                # if (len(old_buffer) == 0):
                #     for item in io_buffer:
                #         yield item
                old_buffer.clear()
                old_buffer.extend(io_buffer)
                io_buffer.clear()

                
        else:
            item = old_buffer[current_buffer_index]
            yield item
            current_buffer_index = (current_buffer_index + 1) % old_buffer_len
            # selected_record_count += 1

    if file_writer:
        pickle.dump(old_buffer, file_writer)
        #pickle.dump(io_buffer, file_writer)
        file_writer.close()


