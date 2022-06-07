"""Reader utils."""

import functools
import gzip
import io
import os
import struct
import typing

import numpy as np

from shuffleformat.tfrecord import example_pb2


def tfrecord_iterator(
    data_path: str,
    block_index_list: typing.List[typing.Tuple[int, int]],
    start_block_index: int,
    end_block_index: int,
    shard: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Iterable[memoryview]:
    """Create an iterator over the tfrecord dataset.

    Since the tfrecords file stores each example as bytes, we can
    define an iterator over `datum_bytes_view`, which is a memoryview
    object referencing the bytes.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str, optional, default=None
        Index file path. Can be set to None if no file is available.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    Yields:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
    file = io.open(data_path, "rb")
    
    length_bytes = bytearray(8)
    crc_bytes = bytearray(4)
    datum_bytes = bytearray(1024 * 1024)

    def read_records(start_offset=None, end_offset=None):
        nonlocal length_bytes, crc_bytes, datum_bytes

        if start_offset is not None:
            file.seek(start_offset)
        if end_offset is None:
            end_offset = os.path.getsize(data_path)
        while file.tell() < end_offset:
            if file.readinto(length_bytes) != 8:
                raise RuntimeError("Failed to read the record size.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the start token.")
            length, = struct.unpack("<Q", length_bytes)
            if length > len(datum_bytes):
                datum_bytes = datum_bytes.zfill(int(length * 1.5))
            datum_bytes_view = memoryview(datum_bytes)[:length]
            if file.readinto(datum_bytes_view) != length:
                raise RuntimeError("Failed to read the record.")
            if file.readinto(crc_bytes) != 4:
                raise RuntimeError("Failed to read the end token.")

            yield datum_bytes_view
           
    if shard is None:
        for (start_byte, end_byte) in block_index_list[start_block_index: end_block_index]:
            #print('[', start_block_index, end_block_index, '](', start_byte, end_byte, ')')
            yield from read_records(start_byte, end_byte)
    else:
        # e.g., [0, 71, 142, 213 | 284, 355, 426, 497 | 568, 639, 710, 781 | 852, 923, 994, 1065, 1136]
        # block[0, 4), block[4, 8), block[8, 12), block[12, 17)
        # start_block_index = 8, end_block_index = 12
        # num_blocks = 4
        num_blocks = end_block_index - start_block_index
        # shard_idx = 0, shard_count = 2
        shard_idx, shard_count = shard
        # start_index = 4 * 0 // 2 = 0, 4 * 1 // 2 = 2
        start_index = (num_blocks * shard_idx) // shard_count + start_block_index
        # end_index = 4 * 1 // 2 = 2, 4 * 2 // 2 = 4
        # partition[0 + 8, 2 + 8) = [8, 10), partition[2 + 8, 4 + 8) = [10, 12)
        end_index = (num_blocks * (shard_idx + 1)) // shard_count 
        
        if end_index > num_blocks:
            end_index = num_blocks

        end_index = end_index + start_block_index
        
        for (start_byte, end_byte) in block_index_list[start_index: end_index]:
            yield from read_records(start_byte, end_byte)
 

    file.close()


def process_feature(feature: example_pb2.Feature,
                    typename: str,
                    typename_mapping: dict,
                    key: str):
    # NOTE: We assume that each key in the example has only one field
    # (either "bytes_list", "float_list", or "int64_list")!
    field = feature.ListFields()[0]
    inferred_typename, value = field[0].name, field[1].value

    if typename is not None:
        tf_typename = typename_mapping[typename]
        if tf_typename != inferred_typename:
            reversed_mapping = {v: k for k, v in typename_mapping.items()}
            raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                        f"(should be '{reversed_mapping[inferred_typename]}').")

    if inferred_typename == "bytes_list":
        value = np.frombuffer(value[0], dtype=np.uint8)
    elif inferred_typename == "float_list":
        value = np.array(value, dtype=np.float32)
    elif inferred_typename == "int64_list":
        value = np.array(value, dtype=np.int64)
    return value


def extract_feature_dict(features, description, typename_mapping):
    if isinstance(features, example_pb2.FeatureLists):
        features = features.feature_list

        def get_value(typename, typename_mapping, key):
            feature = features[key].feature
            fn = functools.partial(process_feature, typename=typename,
                                   typename_mapping=typename_mapping, key=key)
            return list(map(fn, feature))
    elif isinstance(features, example_pb2.Features):
        features = features.feature

        def get_value(typename, typename_mapping, key):
            return process_feature(features[key], typename,
                                   typename_mapping, key)
    else:
        raise TypeError(f"Incompatible type: features should be either of type "
                        f"example_pb2.Features or example_pb2.FeatureLists and "
                        f"not {type(features)}")

    all_keys = list(features.keys())

    if description is None or len(description) == 0:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    processed_features = {}
    for key, typename in description.items():
        if key not in all_keys:
            raise KeyError(f"Key {key} doesn't exist (select from {all_keys})!")

        processed_features[key] = get_value(typename, typename_mapping, key)

    return processed_features


def tfrecord_block_loader(
    data_path: str,
    block_index_list: typing.List[typing.Tuple[int, int]],
    start_block_index: int,
    end_block_index: int,
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
    transform: typing.Callable[[dict], typing.Any] = None,
    shard: typing.Optional[typing.Tuple[int, int]] = None
) -> typing.Iterable[typing.Dict[str, np.ndarray]]:
    """Create an iterator over the (decoded) examples contained within
    the dataset.

    Decodes raw bytes of the features (contained within the dataset)
    into its respective format.

    Params:
    -------
    data_path: str
        TFRecord file path.

    index_path: str or None
        Index file path. Can be set to None if no file is available.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shard: tuple of ints, optional, default=None
        A tuple (index, count) representing worker_id and num_workers
        count. Necessary to evenly split/shard the dataset among many
        workers (i.e. >1).

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    Yields:
    -------
    features: dict of {str, np.ndarray}
        Decoded bytes of the features into its respective data type (for
        an individual record).
    """

    typename_mapping = {
        "byte": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    record_iterator = tfrecord_iterator(data_path, block_index_list, start_block_index, end_block_index, shard)


    if transform is None:
        for record in record_iterator:
            example = example_pb2.Example()
            example.ParseFromString(record)
            yield extract_feature_dict(example.features, description, typename_mapping)

    else:
        for record in record_iterator:
            example = example_pb2.Example()
            example.ParseFromString(record)
            yield transform(extract_feature_dict(example.features, description, typename_mapping))


