from itertools import product

import pytest

from scanprops import param_matrix


@pytest.mark.parametrize(
    "lo, hi, no_infer, fp16, int8",
    [(1000, 2000, *flags) for flags in product((True, False), repeat=3)],
)
def test_param_matrix(lo, hi, no_infer, fp16, int8):
    res = list(param_matrix((lo, hi), no_infer, fp16, int8))
    nno_infer = 1 if no_infer else 0
    nfp16 = 2 if fp16 else 1
    nint8 = 2 if int8 else 1
    # 1024 & 2048 -> 2
    assert len(res) == 2 * (nfp16 * nint8 - int(fp16 and int8)) + nno_infer
