"""
Tests for demo file integrity.

Verifies that demo QMD files contain all required configuration keys
to avoid KeyError at runtime.
"""

import pathlib


# Required config keys that Person.__init__ accesses via direct dict lookup
REQUIRED_CONFIG_KEYS = {
    "initial_resilience_mean",
    "initial_resilience_sd",
    "initial_affect_mean",
    "initial_affect_sd",
    "initial_resources_mean",
    "initial_resources_sd",
    "stress_probability",
    "coping_success_rate",
}


class TestAgentInitializationDemoConfig:
    """Tests for config dict completeness in agent_initialization_demo.qmd."""

    @staticmethod
    def _find_config_section(lines):
        """Find the config dict block in the QMD file."""
        in_config = False
        config_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped == "config = {":
                in_config = True
                config_lines.append(stripped)
            elif in_config:
                config_lines.append(stripped)
                if stripped == "}":
                    break
        return config_lines

    def test_demo_config_includes_coping_success_rate(self):
        """agent_initialization_demo.qmd config must include coping_success_rate."""
        demo_path = (
            pathlib.Path(__file__).resolve().parents[3] / "src" / "python" / "demos" / "agent_initialization_demo.qmd"
        )
        content = demo_path.read_text()
        assert content.find('"coping_success_rate"') != -1, (
            "agent_initialization_demo.qmd config is missing 'coping_success_rate'.\n"
            "Add it to the config dict in the 'generate' cell."
        )

    def test_demo_config_has_all_required_keys(self):
        """agent_initialization_demo.qmd config must have all keys Person.__init__ expects."""
        demo_path = (
            pathlib.Path(__file__).resolve().parents[3] / "src" / "python" / "demos" / "agent_initialization_demo.qmd"
        )
        content = demo_path.read_text()

        # Find keys in the config dict by looking for quoted strings followed by colon
        keys_in_config = set()
        lines = content.split("\n")
        in_config = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("config = {"):
                in_config = True
                continue
            if in_config:
                if stripped == "}":
                    break
                # Extract all keys (quoted strings before colon) from this line
                idx = 0
                while True:
                    start = stripped.find('"', idx)
                    if start == -1:
                        break
                    end = stripped.find('"', start + 1)
                    if end == -1:
                        break
                    keys_in_config.add(stripped[start + 1 : end])
                    idx = end + 1

        missing = REQUIRED_CONFIG_KEYS - keys_in_config
        assert not missing, f"agent_initialization_demo.qmd config is missing keys: {missing}"
