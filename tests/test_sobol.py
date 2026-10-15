import numpy as np
import pytest

from sagari import SobolGlobalDesign


def test_extend_continues_one_sequence(tmp_path):
    a = SobolGlobalDesign(4, 42, tmp_path / "s.json")
    first = a.extend(8)
    second = SobolGlobalDesign(4, 42, tmp_path / "s.json").extend(4)      # a new process resuming
    direct = SobolGlobalDesign(4, 42).extend(12)
    assert np.array_equal(np.vstack([first, second]), direct)


def test_peek_does_not_advance():
    a = SobolGlobalDesign(3, 1)
    assert np.array_equal(a.peek(4), a.extend(4))
    assert a.num_generated == 4


def test_state_mismatch(tmp_path):
    SobolGlobalDesign(4, 42, tmp_path / "s.json").extend(4)
    with pytest.raises(ValueError):
        SobolGlobalDesign(3, 42, tmp_path / "s.json")
    with pytest.raises(ValueError):
        SobolGlobalDesign(4, 43, tmp_path / "s.json")
