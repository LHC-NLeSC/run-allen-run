from itertools import product

import pytest

from scanprops import expand_range_2, param_matrix


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


@pytest.mark.parametrize(
    "lo, hi, expected", [(10, 250, (5, 16, 256)), (32, 512, (5, 32, 512))]
)
def test_expand_range_2(lo: int, hi: int, expected: tuple[int, int, int]):
    length, expected_lo, expected_hi = expected
    res = expand_range_2(lo, hi)
    assert len(res) == length
    assert res[0] == expected_lo
    assert res[-1] == expected_hi
