from collections import deque

import pytest

from leantree.repl_adapter.interaction import LeanProcess, LeanProcessException


class _DummyProc:
    def __init__(self, returncode):
        self.returncode = returncode


def test_assert_started_raises_process_exception_when_not_started():
    p = LeanProcess.__new__(LeanProcess)
    p._proc = None
    p._stderr_buffer = deque(maxlen=50)

    with pytest.raises(LeanProcessException, match="Subprocess not started"):
        p._assert_started()


def test_assert_started_raises_process_exception_when_process_terminated():
    p = LeanProcess.__new__(LeanProcess)
    p._proc = _DummyProc(returncode=134)
    p._stderr_buffer = deque(["panic line"], maxlen=50)

    with pytest.raises(LeanProcessException, match="terminated with exit code 134"):
        p._assert_started()
