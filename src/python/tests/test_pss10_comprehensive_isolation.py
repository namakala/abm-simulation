"""Order-independence guard for test_pss10_comprehensive.

The bracket-notation tests build a Config from a temp .env file. When a
prior test leaves PSS10_ITEM_MEAN/SD in os.environ (e.g. a stray repo
.env loaded by reload_config), load_dotenv(override=False) cannot apply
the temp file values, so the tests silently read defaults. These tests
must therefore clear the PSS10 env keys before constructing Config.
"""

import os
import tempfile
from pathlib import Path

from src.python.config import Config

_POLLUTING = {
    "PSS10_ITEM_MEAN": "[1.43, 1.38, 1.51, 1.31, 1.50, 1.40, 1.43, 1.60, 1.14, 1.31]",
    "PSS10_ITEM_SD": "[0.89, 0.89, 0.93, 0.92, 0.80, 0.78, 0.78, 0.88, 0.91, 0.93]",
}

_ENV_CONTENT = (
    "PSS10_ITEM_MEAN=[2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]\n"
    "PSS10_ITEM_SD=[1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]\n"
)

_EXPECTED_MEANS = [2.1, 1.8, 2.3, 1.9, 2.2, 1.7, 2.0, 1.6, 2.4, 1.5]
_EXPECTED_SDS = [1.1, 0.9, 1.2, 1.0, 1.1, 0.8, 1.0, 0.9, 1.3, 0.8]


def test_bracket_notation_resists_env_pollution():
    """Temp .env must win even when PSS10 env vars are pre-set in os.environ."""
    os.environ.update(_POLLUTING)
    try:
        with tempfile.TemporaryDirectory() as d:
            env = Path(d) / ".env"
            env.write_text(_ENV_CONTENT)
            import src.python.tests.test_pss10_comprehensive as m

            # Reuse the test module's own helper-driven construction path by
            # exercising it the way its tests do: pop then construct.
            m._clear_pss10_env()
            cfg = Config(str(env))
            assert cfg.pss10_item_means == _EXPECTED_MEANS
            assert cfg.pss10_item_sds == _EXPECTED_SDS
    finally:
        for k in _POLLUTING:
            os.environ.pop(k, None)
