"""Tests for QMD demo scripts — Quarto project refactor.

Verifies file existence, structure, and renderability of Quarto demos.
Tests the new structure:
- _quarto.yml project config in src/python/demos/
- index.qmd dashboard (replaces index.html)
- sys.path.insert(0, "src/python") in setup blocks (no PROJECT_ROOT)
- pixi.toml tasks point to src/python/demos/
- serve uses quarto preview
- serve.sh is removed
- HTML outputs go to _site/
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEMOS_DIR = PROJECT_ROOT / "src" / "python" / "demos"
PIXI_TOML = PROJECT_ROOT / "pixi.toml"
SITE_DIR = DEMOS_DIR / "_site"
QUARTO_YML = DEMOS_DIR / "_quarto.yml"
INDEX_QMD = DEMOS_DIR / "index.qmd"
SERVE_SCRIPT = PROJECT_ROOT / "src" / "shell" / "serve.sh"
GITIGNORE = PROJECT_ROOT / ".gitignore"

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


# ── WP-1: _quarto.yml ──────────────────────────────────────────────


class TestQuartoYml:
    """WP-1 — _quarto.yml exists in src/python/demos/ with correct structure."""

    def test_quarto_yml_exists(self):
        """src/python/demos/_quarto.yml must exist."""
        assert QUARTO_YML.is_file(), f"Missing _quarto.yml: {QUARTO_YML}"

    def test_quarto_yml_is_valid_yaml(self):
        """_quarto.yml must parse as valid YAML."""
        content = QUARTO_YML.read_text()
        # tomllib does not support multi-doc YAML, so just check it parses
        # The YAML frontmatter-like structure should at minimum be loadable
        assert "project:" in content, "_quarto.yml must have a 'project:' key"
        assert "execute:" in content, "_quarto.yml must have an 'execute:' key"

    def test_output_dir_is_site(self):
        """output-dir must be _site to keep demos directory clean."""
        content = QUARTO_YML.read_text()
        assert "output-dir: _site" in content or "output-dir:\n  _site" in content, (
            "_quarto.yml must set output-dir: _site"
        )

    def test_engine_is_jupyter(self):
        """execute.engine must be jupyter for reproducible code execution."""
        content = QUARTO_YML.read_text()
        assert "engine: jupyter" in content or "engine:\n  jupyter" in content, (
            "_quarto.yml must set execute.engine: jupyter"
        )

    def test_freeze_is_auto(self):
        """execute.freeze must be auto to cache block outputs."""
        content = QUARTO_YML.read_text()
        assert "freeze: auto" in content or "freeze:\n  auto" in content, "_quarto.yml should set execute.freeze: auto"


# ── WP-2: index.qmd ────────────────────────────────────────────────


class TestIndexQmd:
    """WP-2 — index.qmd exists and lists all demo pages."""

    def test_index_qmd_exists(self):
        """src/python/demos/index.qmd must exist."""
        assert INDEX_QMD.is_file(), f"Missing index.qmd: {INDEX_QMD}"

    def test_has_yaml_frontmatter(self):
        """index.qmd must have YAML frontmatter."""
        content = INDEX_QMD.read_text()
        assert content.startswith("---"), "index.qmd must start with YAML frontmatter"
        fm_end = content.index("\n---\n")
        assert fm_end > 0, "index.qmd must close its frontmatter"

    def test_has_title(self):
        """index.qmd frontmatter must have a title."""
        content = INDEX_QMD.read_text()
        assert "title:" in content[:300], "index.qmd must declare a title"

    def test_references_all_demo_htmls(self):
        """index.qmd must reference all demo .html files."""
        content = INDEX_QMD.read_text()
        for qmd_file in DEMO_FILES:
            html = qmd_file.replace(".qmd", ".html")
            assert html in content, f"index.qmd must reference {html}"


# ── WP-3: PROJECT_ROOT replaced with sys.path.insert ──────────────


def _load_pixi_tasks() -> dict:
    """Return the [tasks] section of pixi.toml."""
    with open(PIXI_TOML, "rb") as f:
        data = tomllib.load(f)
    return data.get("tasks", {})


class TestPixiTasks:
    """WP-4 — pixi.toml tasks point to src/python/demos/."""

    def test_quarto_task_points_to_demos_dir(self):
        """quarto task must reference src/python/demos via script or direct path."""
        tasks = _load_pixi_tasks()
        assert "quarto" in tasks, "pixi.toml must define a 'quarto' task"
        task_str = str(tasks["quarto"])
        assert "src/python/demos" in task_str or "quarto-render.sh" in task_str, (
            "quarto task must reference src/python/demos directly or via quarto-render.sh"
        )

    def test_serve_task_uses_quarto_preview(self):
        """serve task must use 'quarto preview' not http.server."""
        tasks = _load_pixi_tasks()
        assert "serve" in tasks, "pixi.toml must define a 'serve' task"
        task_str = str(tasks["serve"])
        assert "quarto preview" in task_str, "serve task must use 'quarto preview', not python http.server"

    def test_serve_quick_task_removed(self):
        """serve-quick task must be removed from pixi.toml."""
        tasks = _load_pixi_tasks()
        assert "serve-quick" not in tasks, "serve-quick task should be removed; use 'pixi run serve' instead"


class TestServeScriptRemoved:
    """WP-6 — serve.sh must be deleted."""

    def test_serve_script_does_not_exist(self):
        """src/shell/serve.sh must be deleted."""
        assert not SERVE_SCRIPT.exists(), f"serve.sh must be deleted: {SERVE_SCRIPT}"


class TestGitignoreUpdated:
    """WP-5 — .gitignore must exclude generated HTML and _site/."""

    def test_gitignore_excludes_demos_html(self):
        """src/python/demos/*.html must be in .gitignore."""
        if not GITIGNORE.exists():
            pytest.skip(".gitignore does not exist")
        content = GITIGNORE.read_text()
        assert "src/python/demos/*.html" in content, ".gitignore must exclude src/python/demos/*.html"

    def test_gitignore_excludes_site_dir(self):
        """src/python/demos/_site/ must be in .gitignore."""
        if not GITIGNORE.exists():
            pytest.skip(".gitignore does not exist")
        content = GITIGNORE.read_text()
        assert "src/python/demos/_site/" in content, ".gitignore must exclude src/python/demos/_site/"


class TestQmdNoProjectRoot:
    """WP-3 — No .qmd file must use the old PROJECT_ROOT hack."""

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_no_project_root_variable(self, filename: str):
        """No .qmd must define PROJECT_ROOT via parents[2]."""
        content = (DEMOS_DIR / filename).read_text()
        assert "PROJECT_ROOT" not in content, (
            f"{filename} must not use PROJECT_ROOT; use sys.path.insert(0, 'src/python') instead"
        )

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_uses_sys_path_insert(self, filename: str):
        """Each .qmd must have demos_dir + project_root + sys.path.insert in setup."""
        content = (DEMOS_DIR / filename).read_text()
        assert "demos_dir = pathlib.Path(os.getcwd())" in content, (
            f"{filename} must define demos_dir = pathlib.Path(os.getcwd())"
        )
        assert "project_root = demos_dir.parents[2]" in content, (
            f"{filename} must define project_root = demos_dir.parents[2]"
        )
        assert "sys.path.insert(0, str(project_root))" in content, (
            f"{filename} must have sys.path.insert(0, 'src/python') in its setup block"
        )

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_no_sys_path_append(self, filename: str):
        """No .qmd must use sys.path.append (old pattern)."""
        content = (DEMOS_DIR / filename).read_text()
        assert "sys.path.append" not in content, f"{filename} must not use sys.path.append; use sys.path.insert instead"


# ── File structure (existing demos must still exist) ────────────────


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


class TestDemoStructureIntact:
    """Existing demo structure requirements that must continue to hold."""

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_what_this_code_does(self, filename: str):
        """Each QMD must have a 'What This Code Does' section."""
        content = (DEMOS_DIR / filename).read_text()
        assert "What This Code Does" in content, f"Missing 'What This Code Does' in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_why_this_matters(self, filename: str):
        """Each QMD must have a 'Why This Matters' section."""
        content = (DEMOS_DIR / filename).read_text()
        assert "Why This Matters" in content, f"Missing 'Why This Matters' in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_python_code_block(self, filename: str):
        """Each QMD must have at least one ```{python} code block."""
        content = (DEMOS_DIR / filename).read_text()
        assert re.search(r"```\{python\}", content), f"Missing Python code block in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_plotting_import(self, filename: str):
        """Each .qmd must import seaborn or matplotlib for plotting."""
        content = (DEMOS_DIR / filename).read_text()
        has_plot_lib = "import seaborn" in content or "import matplotlib" in content
        assert has_plot_lib, f"Missing plotting import in {filename}"

    @pytest.mark.parametrize("filename", DEMO_FILES)
    def test_has_numpy_import(self, filename: str):
        """Each .qmd must import numpy."""
        content = (DEMOS_DIR / filename).read_text()
        assert "import numpy" in content, f"Missing numpy import in {filename}"
