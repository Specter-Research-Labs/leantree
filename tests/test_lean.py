"""
Tests for LeanHypothesis and related classes in leantree.core.lean
"""
import pytest

from leantree.core.lean import LeanHypothesis


class TestLeanHypothesisFromString:
    """Test that from_string + str() round-trip produces consistent hypotheses."""

    # Example hypotheses from the comment in LeanHypothesis.from_string
    EXAMPLE_SIMPLE = "hu : u ∈ { carrier := {u | E.IsSolution u}, add_mem' := ⋯, zero_mem' := ⋯ }.carrier"

    EXAMPLE_MULTILINE = """\
ih :
  ∀ (x : α),
    traverse F ({ head := x, tail := tl } * { head := y, tail := L2 }) =
      (fun x1 x2 => x1 * x2) <$> traverse F { head := x, tail := tl } <*> traverse F { head := y, tail := L2 }"""

    EXAMPLE_WITH_VALUE = """\
to𝕜 : (E →L[ℝ] ℝ) → E →L[𝕜] 𝕜 := fun fr =>
  let __LinearMap := (↑fr).extendTo𝕜';
  { toLinearMap := __LinearMap, cont := ⋯ }"""

    @pytest.mark.parametrize("hypothesis_str", [
        EXAMPLE_SIMPLE,
        EXAMPLE_MULTILINE,
        EXAMPLE_WITH_VALUE,
    ])
    def test_from_string_str_roundtrip(self, hypothesis_str: str):
        """Test that parsing and re-stringifying is idempotent after one pass."""
        # First round-trip
        hypotheses = LeanHypothesis.from_string(hypothesis_str)
        assert len(hypotheses) == 1
        hypothesis = hypotheses[0]
        stringified = str(hypothesis)

        # Second round-trip
        hypotheses_again = LeanHypothesis.from_string(stringified)
        assert len(hypotheses_again) == 1
        hypothesis_again = hypotheses_again[0]

        # The two parsed hypotheses should be identical
        assert hypothesis.user_name == hypothesis_again.user_name
        assert hypothesis.type == hypothesis_again.type
        assert hypothesis.value == hypothesis_again.value

        # And stringifying again should produce the same string
        assert str(hypothesis_again) == stringified
