"""How activations are stored on disk.

Activations leave the model in bfloat16. numpy has no bfloat16, and float16 cannot hold
the dynamic range of Gemma's residual stream (|x| goes past the float16 ceiling of 65504
in the later layers), so bfloat16 tensors are stored as their raw 16-bit patterns inside
an int16 array. Lossless, still 2 bytes per element, readable with plain numpy.
"""

import numpy as np

# Store-dtype tags, as written into meta.json.
BF16_RAW = "bfloat16_raw_int16"
FLOAT32 = "float32"
INT32 = "int32"
BOOL = "bool"

_NUMPY_DTYPES = {BF16_RAW: np.int16, FLOAT32: np.float32, INT32: np.int32, BOOL: np.bool_}


def numpy_dtype(store_dtype: str) -> np.dtype:
    """On-disk numpy dtype for a store-dtype tag."""
    if store_dtype not in _NUMPY_DTYPES:
        raise ValueError(f"Unknown store dtype {store_dtype!r}, expected one of {sorted(_NUMPY_DTYPES)}")
    return np.dtype(_NUMPY_DTYPES[store_dtype])


def decode(array: np.ndarray, store_dtype: str) -> np.ndarray:
    """Widens raw bfloat16 patterns to float32; every other store dtype passes through.

    bfloat16 has the same 8-bit exponent as float32 and just 16 fewer mantissa bits, so
    this is an exact left shift into the high half of a float32, not a conversion.
    """
    if store_dtype != BF16_RAW:
        return array
    raw = np.ascontiguousarray(array).view(np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)
