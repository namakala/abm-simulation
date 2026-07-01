"""Guard against stale xfail decorators in test_pss10_comprehensive.

The env leak that originally required xfail marks on the PSS-10 bracket
notation tests is fixed by the complete_env_isolation autouse fixture in
conftest.py. These tests should now pass as regular tests.
"""

import src.python.tests.test_pss10_comprehensive as mod

_XFAIL_TESTS = [
    "test_bracket_notation_parsing",
    "test_backward_compatibility",
    "test_mixed_format_usage",
    "test_whitespace_handling",
]


def test_pss10_comprehensive_has_no_xfail_marks():
    """None of the PSS-10 comprehensive tests should carry an xfail mark."""
    still_marked = [
        name for name in _XFAIL_TESTS if any(m.name == "xfail" for m in getattr(getattr(mod, name), "pytestmark", []))
    ]
    assert still_marked == [], f"stale xfail marks remain on: {still_marked}"
