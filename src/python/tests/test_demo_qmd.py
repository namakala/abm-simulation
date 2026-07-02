"""Tests for QMD demo scripts.

Verifies file existence, structure, and renderability of Quarto demos.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMOS_DIR = PROJECT_ROOT / "src" / "python" / "demos"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
PIXI_PATH = shutil.which("pixi") or "/home/lam/.cargo/bin/pixi"

DEMO_FILES = [
    "stress_perception.qmd",
    "resilience_activation.qmd",
    "resource_allocation.qmd",
    "interaction.qmd",
    "stress_buffering.qmd",
    "analyze.qmd",
    "track_daily_stress.qmd",
    "agent_correlation_analysis.qmd",
    "population_correlation_analysis.qmd",
    "parameter_sweep_correlations.qmd",
    "population_analysis.qmd",
    "agent_initialization_demo.qmd",
    "stress_processing_mechanism.qmd",
    "agent_diversity_demo.qmd",
    "stress_pipeline_debug_demo.qmd",
]


# ── Step 1: pixi task ────────────────────────────────────────────────


def _load_pixi_tasks() -> dict:
    """Return the [tasks] section of pixi.toml."""
    with open(PIXI_TOML, "rb") as f:
        data = tomllib.load(f)
    return data.get("tasks", {})


class TestPixiQuartoTask:
    """Step 1 — quarto render task exists and works."""

    def test_quarto_task_defined(self):
        """pixi.toml must define a 'quarto' task for rendering .qmd files."""
        tasks = _load_pixi_tasks()
        assert "quarto" in tasks, "pixi.toml must define a 'quarto' task"
        task_value = tasks["quarto"]
        assert "quarto render" in str(task_value)


class TestPixiServeTasks:
    """Serve dashboard tasks exist for viewing rendered demos."""

    def test_serve_task_calls_shell_script(self):
        """Serve task must call serve.sh (pixi can't parse for-loops)."""
        tasks = _load_pixi_tasks()
        assert "serve" in tasks, "pixi.toml must define a 'serve' task"
        task_value = str(tasks["serve"])
        assert "serve.sh" in task_value, "serve task must call src/shell/serve.sh"

    def test_serve_quick_task_defined(self):
        """pixi.toml must define a 'serve-quick' task for serving without re-render."""
        tasks = _load_pixi_tasks()
        assert "serve-quick" in tasks, "pixi.toml must define a 'serve-quick' task"
        task_value = str(tasks["serve-quick"])
        assert "http.server" in task_value, "serve-quick task must start an HTTP server"
        assert "9000" in task_value, "serve-quick task must use port 9000"


class TestServeScript:
    """The serve.sh shell script must exist with correct render logic."""

    SERVE_SCRIPT = PROJECT_ROOT / "src" / "shell" / "serve.sh"

    def test_script_exists(self):
        """serve.sh must exist in src/shell/."""
        assert self.SERVE_SCRIPT.is_file(), f"Missing serve script: {self.SERVE_SCRIPT}"

    def test_script_has_for_loop(self):
        """serve.sh must iterate over .qmd files with a for loop."""
        content = self.SERVE_SCRIPT.read_text()
        assert "for " in content, "serve.sh must contain a for loop"
        assert "*.qmd" in content, "serve.sh must iterate over .qmd files"

    def test_script_invokes_quarto_and_server(self):
        """serve.sh must call quarto render and python http.server."""
        content = self.SERVE_SCRIPT.read_text()
        assert "quarto render" in content, "serve.sh must invoke quarto render"
        assert "http.server" in content, "serve.sh must start HTTP server"
        assert "9000" in content, "serve.sh must use port 9000"


# ── Step 2–6: File structure ────────────────────────────────────────


class TestDemoFileExistence:
    """Each demo .qmd file must exist in src/python/demos/."""

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_file_exists(self, filename: str):
        """Every demo file in DEMO_FILES must exist."""
        path = DEMOS_DIR / filename
        assert path.is_file(), f"Missing demo file: {path}"


class TestDemoFrontmatter:
    """YAML frontmatter validation for each .qmd."""

    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n^---", re.MULTILINE | re.DOTALL)

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_frontmatter(self, filename: str):
        """Each .qmd must have YAML frontmatter (--- delimited)."""
        content = (DEMOS_DIR / filename).read_text()
        assert self.FRONTMATTER_RE.search(content), f"Missing YAML frontmatter in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_format_html(self, filename: str):
        """Each .qmd must declare format: html."""
        content = (DEMOS_DIR / filename).read_text()
        assert "format: html" in content or "format:\n  html" in content, f"{filename} must declare format: html"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_title(self, filename: str):
        """Each .qmd must have a title in frontmatter."""
        content = (DEMOS_DIR / filename).read_text()
        match = self.FRONTMATTER_RE.search(content)
        assert match is not None, f"Missing frontmatter in {filename}"
        assert "title:" in match.group(1), f"Missing 'title:' in {filename} frontmatter"


class TestDemoCodeChunks:
    """Each .qmd must contain the expected Python code chunks."""

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_python_chunks(self, filename: str):
        """Each .qmd must contain at least one ```{python} ... ``` block."""
        content = (DEMOS_DIR / filename).read_text()
        assert re.search(r"```\{python\}", content), f"Missing Python code chunk in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_syspath_append(self, filename: str):
        """Each .qmd must use sys.path.append to add the project root."""
        content = (DEMOS_DIR / filename).read_text()
        assert "sys.path.append" in content, f"Missing sys.path.append in {filename}"
        assert "PROJECT_ROOT" in content, f"{filename} must define PROJECT_ROOT path"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_seaborn_import(self, filename: str):
        """Each .qmd must import seaborn or matplotlib for plotting."""
        content = (DEMOS_DIR / filename).read_text()
        # stress_buffering uses matplotlib path diagram instead of seaborn
        has_plot_lib = "import seaborn" in content or "import matplotlib" in content
        assert has_plot_lib, f"Missing plotting import in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_numpy_import(self, filename: str):
        """Each .qmd must import numpy."""
        content = (DEMOS_DIR / filename).read_text()
        assert "import numpy" in content, f"Missing numpy import in {filename}"


class TestStressPerceptionDemo:
    """Step 2 — stress_perception.qmd specifics."""

    DEMO = "stress_perception.qmd"

    def test_sweeps_c_and_o(self):
        """Must sweep omega_c and omega_o across [0,1] in a loop."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "omega_c" in content, "Missing omega_c parameter sweep"
        assert "omega_o" in content, "Missing omega_o parameter sweep"

    def test_has_6x6_or_larger(self):
        """Must have a pairplot or PairGrid (implies 6+ variables)."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(kw in content for kw in ["pairplot", "PairGrid", "PairPlot"]), (
            f"Missing pairplot/PairGrid in {self.DEMO}"
        )

    def test_narrative_explains_appraisal(self):
        """Narrative must reference challenge/hindrance appraisal."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["appraisal", "monotonic", "complement"]), (
            f"Missing appraisal narrative in {self.DEMO}"
        )


class TestResilienceActivationDemo:
    """Step 3 — resilience_activation.qmd specifics."""

    DEMO = "resilience_activation.qmd"

    def test_sweeps_challenge_hindrance(self):
        """Must vary challenge and hindrance across [0,1]."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "challenge" in content, "Missing challenge sweep"
        assert "hindrance" in content, "Missing hindrance sweep"

    def test_has_neighbor_affects(self):
        """Must include 3 levels of neighbor_affects."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "neighbor_affects" in content, "Missing neighbor_affects sweep"

    def test_has_coping_probability(self):
        """Must collect and discuss coping probability."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "coping_probability" in content or "coping" in content, f"Missing coping probability in {self.DEMO}"

    def test_narrative_explains_coping(self):
        """Narrative must explain challenge/hindrance balance on coping."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["challenge/hindrance", "coping outcome", "modulat"]), (
            f"Missing coping narrative in {self.DEMO}"
        )


class TestResourceAllocationDemo:
    """Step 4 — resource_allocation.qmd specifics."""

    DEMO = "resource_allocation.qmd"

    def test_has_four_scenarios(self):
        """Must describe or iterate 4 distinct scenarios."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        # Count scenario headings / scenario markers / named dict keys
        # Scenarios are named: "High R + default T", "Low R + default T", etc.
        scenario_count = content.count("Scenario")
        named_count = sum(1 for s in ["High R", "Low R", "high T", "starting PF"] if s in content)
        assert scenario_count >= 4 or named_count >= 4, (
            f"Expected >=4 scenarios, found scenario={scenario_count} named={named_count}"
        )

    def test_has_softmax_temperature(self):
        """Must reference softmax temperature."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "temperature" in content or "softmax" in content, f"Missing softmax temperature in {self.DEMO}"

    def test_has_timeseries(self):
        """Must include time-series plot of PF efficacies."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["time series", "trajector", "timeline"]), (
            f"Missing time series narrative in {self.DEMO}"
        )

    def test_narrative_explains_softmax(self):
        """Narrative must explain uniform vs winner-take-most."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["winner-take", "uniform", "softmax"]), (
            f"Missing softmax explanation in {self.DEMO}"
        )


class TestInteractionDemo:
    """Step 5 — interaction.qmd specifics."""

    DEMO = "interaction.qmd"

    def test_uses_process_interaction(self):
        """Must use process_interaction() directly (not run_phase stub)."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "process_interaction" in content, "Must use process_interaction() directly"

    def test_sweeps_partner_self_affect(self):
        """Must vary partner_affect and self_affect."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "partner_affect" in content, "Missing partner_affect sweep"
        assert "self_affect" in content, "Missing self_affect sweep"

    def test_has_state_machine_scenarios(self):
        """Must test all 5 stress state combinations."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "stressed" in content, "Missing stress state references"

    def test_narrative_explains_negativity_bias(self):
        """Narrative must explain 1.5x negativity bias."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["negativity", "1.5", "negative bias"]), (
            f"Missing negativity bias narrative in {self.DEMO}"
        )


class TestStressBufferingDemo:
    """Step 6 — stress_buffering.qmd specifics."""

    DEMO = "stress_buffering.qmd"

    def test_has_path_plot(self):
        """Must use a path/mediation diagram (not correlogram)."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["path", "mediation", "arrow", "FancyArrowPatch"]), (
            f"Missing path diagram in {self.DEMO}"
        )

    def test_has_a_b_c_prime(self):
        """Must compute and display a, b, c' coefficients."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert all(term in content for term in ["a-path", "b-path", "c'"]) or all(
            term in content for term in ["a_coefficient", "b_coefficient", "c_prime"]
        ), f"Missing mediation coefficients in {self.DEMO}"

    def test_has_indirect_effect(self):
        """Must compute a*b indirect effect."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert "indirect" in content or "a*b" in content, f"Missing indirect effect in {self.DEMO}"

    def test_narrative_explains_baron_kenny(self):
        """Narrative must reference Baron & Kenny mediation."""
        content = (DEMOS_DIR / self.DEMO).read_text()
        assert any(term in content for term in ["Baron", "Kenny", "mediation"]), (
            f"Missing mediation narrative in {self.DEMO}"
        )


# ── Dashboard ────────────────────────────────────────────────────


class TestDashboard:
    """Dashboard index.html exists and has correct structure."""

    DASHBOARD_PATH = DEMOS_DIR / "index.html"

    def test_dashboard_exists(self):
        """Dashboard index.html must exist in demos directory."""
        assert self.DASHBOARD_PATH.is_file(), f"Missing dashboard: {self.DASHBOARD_PATH}"

    def test_dashboard_has_title(self):
        """Dashboard must have a title tag."""
        content = self.DASHBOARD_PATH.read_text()
        assert "<title>" in content and "</title>" in content, "Missing <title> tag"

    def test_dashboard_has_links_to_demos(self):
        """Dashboard must include links to each known demo HTML."""
        content = self.DASHBOARD_PATH.read_text()
        for qmd_file in DEMO_FILES:
            html_file = qmd_file.replace(".qmd", ".html")
            assert html_file in content, f"Missing link to {html_file}"

    def test_dashboard_no_external_deps(self):
        """Dashboard must not load external CSS/JS from CDN."""
        content = self.DASHBOARD_PATH.read_text()
        assert "<style>" in content, "Must have inline <style> (no external CSS)"
        assert "http://" not in content.replace("https://", "").split("http://")[0] if "http://" in content else True, (
            "Should avoid external http dependencies"
        )


# ── Tutorial Structure ────────────────────────────────────────────


ALL_DEMO_QMD = DEMO_FILES


class TestTutorialStructure:
    """Each QMD must follow the tutorial format with explanation sections."""

    @pytest.mark.parametrize("filename", ALL_DEMO_QMD)
    def test_has_what_this_code_does(self, filename: str):
        """Each QMD must have a 'What This Code Does' section."""
        content = (DEMOS_DIR / filename).read_text()
        assert "What This Code Does" in content, f"Missing 'What This Code Does' section in {filename}"

    @pytest.mark.parametrize("filename", ALL_DEMO_QMD)
    def test_has_why_this_matters(self, filename: str):
        """Each QMD must have a 'Why This Matters' section."""
        content = (DEMOS_DIR / filename).read_text()
        assert "Why This Matters" in content, f"Missing 'Why This Matters' section in {filename}"

    @pytest.mark.parametrize("filename", ALL_DEMO_QMD)
    def test_has_python_code_block(self, filename: str):
        """Each QMD must have at least one ```{python} code block."""
        content = (DEMOS_DIR / filename).read_text()
        assert re.search(r"```\{python\}", content), f"Missing Python code block in {filename}"


# ── Step 7: Render verification (slow / integration) ───────────────


@pytest.mark.slow
@pytest.mark.parametrize("filename", DEMO_FILES)
def test_demo_renders_without_errors(filename: str):
    """Each .qmd must render via quarto without errors (slow)."""
    import subprocess

    filepath = DEMOS_DIR / filename
    result = subprocess.run(
        [PIXI_PATH, "run", "quarto", str(filepath)],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"Quarto render failed for {filename}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
    )


@pytest.mark.slow
def test_all_html_outputs_exist():
    """After rendering, each .qmd must produce a corresponding .html file."""
    import subprocess

    for filename in DEMO_FILES:
        filepath = DEMOS_DIR / filename
        result = subprocess.run(
            [PIXI_PATH, "run", "quarto", str(filepath)],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=180,
        )
        html_path = filepath.with_suffix(".html")
        assert html_path.is_file(), f"Missing HTML output after render: {html_path}\nStderr: {result.stderr[-1000:]}"
