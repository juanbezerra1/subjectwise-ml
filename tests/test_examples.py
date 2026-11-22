"""The examples and the README snippets must only call things that exist."""

import ast
import re
from pathlib import Path

import pytest

import subjectwise as sw

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))
READMES = [p for p in (ROOT / "README.md", ROOT / "README.pt-BR.md") if p.exists()]


def _missing_attributes(source: str) -> list[str]:
    tree = ast.parse(source)
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sw"
    }
    return sorted(name for name in used if not hasattr(sw, name))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_scripts_use_only_the_public_api(path):
    assert _missing_attributes(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_python_snippets_parse_and_use_the_public_api(path):
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.DOTALL)
    assert blocks, "the README should show at least one Python snippet"
    for block in blocks:
        assert _missing_attributes(block) == []


@pytest.mark.parametrize("path", READMES, ids=lambda p: p.name)
def test_readme_cli_commands_exist(path):
    from typer.testing import CliRunner

    from subjectwise.cli import app

    commands = set(
        re.findall(r"^subjectwise (\w+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert {"demo", "check", "evaluate", "power"} <= commands
    for command in commands:
        assert CliRunner().invoke(app, [command, "--help"]).exit_code == 0
