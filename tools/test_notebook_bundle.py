"""Guard the notebook bundle: it must be current, and it must repair a stale session.

Two failure modes are covered:

1. the notebook was not regenerated after a module changed, so Colab would run
   code that no longer exists in the repository;
2. a session re-runs the source cell - which rewrites the files - but keeps the
   modules it already imported, so the traceback mixes two versions of one file
   and a fix appears not to work. The revision check cell must reload those
   modules, and must say so plainly when reloading cannot help.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.embed_sources import (  # noqa: E402
    BENCHMARK_CHECKED,
    BENCHMARK_MODULES,
    BENCHMARK_NOTEBOOK_PATH,
    NOTEBOOK_PATH,
    REVISION_NAME,
    benchmark_notebook_is_current,
    benchmark_sources,
    fingerprint,
    notebook_has_stored_outputs,
    notebook_is_current,
    read_sources,
)

STALE_REVISION = "stale0000000000"
PROJECT_PREFIXES = ("models", "matching", "pipeline")


def _text(cell: Dict[str, object]) -> str:
    return "".join(cell.get("source", []))


def _cells(path: str = NOTEBOOK_PATH) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        cells = json.load(handle).get("cells", [])
    source = [text for text in map(_text, cells) if "SOURCE_FILES = " in text]
    check = [text for text in map(_text, cells) if REVISION_NAME in text and "importlib" in text]
    assert source, "no source cell found in the notebook"
    assert check, "no revision check cell found in the notebook"
    return source[0], check[0]


def _code_texts(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as handle:
        cells = json.load(handle).get("cells", [])
    return [_text(cell) for cell in cells if cell.get("cell_type") == "code"]


def _payload(source_cell: str) -> str:
    start = source_cell.index("SOURCE_FILES = ") + len("SOURCE_FILES = ")
    return source_cell[start : source_cell.index("\n\nfor relative", start)].strip()


def test_notebook_is_up_to_date() -> None:
    sources = read_sources()
    assert notebook_is_current(sources), (
        f"notebook is stale for revision {fingerprint(sources)}; run: python tools/embed_sources.py"
    )
    print(f"notebook bundle is current ({fingerprint(sources)})")


def test_every_embedded_module_carries_a_revision() -> None:
    source_cell, _ = _cells()
    payload = json.loads(_payload(source_cell))
    assert set(payload) == set(read_sources()), "embedded file list differs from the project"
    for relative, text in payload.items():
        assert f"{REVISION_NAME} = " in text, f"{relative} carries no revision marker"
    assert f'EXPECTED_SOURCE_REVISION = "{fingerprint(read_sources())}"' in source_cell
    print(f"all {len(payload)} embedded modules carry a revision marker")


def test_check_cell_heals_a_stale_session() -> None:
    source_cell, check_cell = _cells()
    with _session(source_cell) as namespace:
        exec(compile(check_cell, "<check-cell-fresh>", "exec"), namespace)
        assert _revisions() == {name: fingerprint(read_sources()) for name in _checked(check_cell)}

        stale = _make_stale(namespace, source_cell)
        exec(compile(check_cell, "<check-cell-stale>", "exec"), stale)

        revisions = _revisions()
        assert set(revisions.values()) == {STALE_REVISION}, revisions
    print("a stale session is reloaded in place instead of failing")


def test_check_cell_reports_a_module_without_a_marker() -> None:
    source_cell, check_cell = _cells()
    with _session(source_cell) as namespace:
        exec(compile(check_cell, "<check-cell-fresh>", "exec"), namespace)
        del importlib.import_module("models.segmenter").NOTEBOOK_SOURCE_REVISION
        described = namespace["describe"]("models.segmenter", None)
    assert described == "  models.segmenter: imported before this notebook revision existed", described
    print("a module predating the marker is described in plain words")


def test_check_cell_asks_for_a_restart_when_reload_raises() -> None:
    message = _unrepairable_check(reload_behaviour="raise")
    assert message.startswith("Fix this first: Runtime > Restart session, then Run all."), message
    assert "Reloading models.feature_encoder failed" in message, message
    print("a failing reload is reported as a restart, naming the module")


def test_check_cell_reports_what_it_could_not_repair() -> None:
    message = _unrepairable_check(reload_behaviour="noop")
    assert "Fix this first: Runtime > Restart session, then Run all." in message, message
    assert "still runs different code" in message, message
    assert f"(expected {STALE_REVISION})" in message, "the message should show both revisions"
    print("a reload that changes nothing is reported instead of silently accepted")


def _unrepairable_check(reload_behaviour: str) -> str:
    source_cell, check_cell = _cells()
    with _session(source_cell) as namespace:
        exec(compile(check_cell, "<check-cell-fresh>", "exec"), namespace)
        stale = _make_stale(namespace, source_cell)

        real_reload = importlib.reload

        def fake_reload(target):
            # `importlib.reload` takes a module, not a name.
            name = getattr(target, "__name__", target)
            if reload_behaviour == "raise":
                raise ImportError(f"simulated failure for {name}")
            return target

        # The check cell runs its own `import importlib`, so the only way to make
        # its reload fail is to patch the module itself.
        importlib.reload = fake_reload
        try:
            exec(compile(check_cell, "<check-cell-unrepairable>", "exec"), stale)
        except RuntimeError as error:
            return str(error)
        finally:
            importlib.reload = real_reload
    raise AssertionError("the revision check accepted a session it could not repair")


def _make_stale(namespace: Dict[str, object], source_cell: str) -> Dict[str, object]:
    """Rewrite the bundle under a new revision while memory keeps the old modules."""
    stale = dict(namespace)
    rewritten = source_cell.replace(fingerprint(read_sources()), STALE_REVISION)
    exec(compile(rewritten, "<source-cell-stale>", "exec"), stale)
    return stale


def _checked(check_cell: str) -> List[str]:
    start = check_cell.index("CHECKED = (") + len("CHECKED = (")
    return [name.strip().strip('",') for name in check_cell[start : check_cell.index(")", start)].split("\n") if name.strip()]


def _revisions() -> Dict[str, Optional[str]]:
    return {
        name: getattr(importlib.import_module(name), REVISION_NAME, None)
        for name in sys.modules
        if name.startswith(PROJECT_PREFIXES) and "." in name
    }


@contextmanager
def _session(source_cell: str) -> Iterator[Dict[str, object]]:
    """Execute the source cell in a throwaway directory with a clean import table."""
    workdir = tempfile.mkdtemp(prefix="notebook_bundle_")
    previous = os.getcwd()
    saved = {name: module for name, module in sys.modules.items() if name.startswith(PROJECT_PREFIXES)}
    os.chdir(workdir)
    sys.path.insert(0, workdir)
    for name in saved:
        del sys.modules[name]
    try:
        namespace: Dict[str, object] = {}
        exec(compile(source_cell, "<source-cell>", "exec"), namespace)
        yield namespace
    finally:
        os.chdir(previous)
        if workdir in sys.path:
            sys.path.remove(workdir)
        for name in [name for name in sys.modules if name.startswith(PROJECT_PREFIXES)]:
            del sys.modules[name]
        sys.modules.update(saved)
        shutil.rmtree(workdir, ignore_errors=True)


def test_notebook_is_committed_without_outputs() -> None:
    assert not notebook_has_stored_outputs(), (
        "the notebook carries stored outputs; executed figures are base64 and bloat the diff. "
        "Run: python tools/embed_sources.py"
    )
    size_kb = os.path.getsize(NOTEBOOK_PATH) / 1024.0
    assert size_kb < 500, f"notebook is {size_kb:.0f} KB, which suggests embedded outputs"
    print(f"notebook is committed clean ({size_kb:.0f} KB, no stored outputs)")


def test_check_reports_stored_outputs() -> None:
    source_cell, _ = _cells()
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["noise"]}]
            cell["execution_count"] = 1
            break
    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False, encoding="utf-8") as handle:
        json.dump(notebook, handle)
        polluted = handle.name
    try:
        with _patched_notebook(polluted):
            assert notebook_is_current(read_sources()), "only the outputs changed"
            assert notebook_has_stored_outputs(), "stored outputs must be detected"
    finally:
        os.unlink(polluted)
    print("stored outputs are detected as notebook drift")


@contextmanager
def _patched_notebook(path: str) -> Iterator[None]:
    import tools.embed_sources as embed

    original = embed.NOTEBOOK_PATH
    embed.NOTEBOOK_PATH = path
    try:
        yield
    finally:
        embed.NOTEBOOK_PATH = original


def test_benchmark_notebook_is_up_to_date() -> None:
    assert benchmark_notebook_is_current(), (
        f"benchmark notebook is stale for revision "
        f"{fingerprint(benchmark_sources())}; run: python tools/embed_sources.py"
    )
    print(f"benchmark notebook is current ({fingerprint(benchmark_sources())})")


def test_benchmark_notebook_is_committed_without_outputs() -> None:
    assert not notebook_has_stored_outputs(BENCHMARK_NOTEBOOK_PATH), (
        "the benchmark notebook carries stored outputs; "
        "run: python tools/embed_sources.py"
    )
    size_kb = os.path.getsize(BENCHMARK_NOTEBOOK_PATH) / 1024.0
    assert size_kb < 500, f"benchmark notebook is {size_kb:.0f} KB, which suggests outputs"
    print(f"benchmark notebook is committed clean ({size_kb:.0f} KB, no stored outputs)")


def test_benchmark_notebook_embeds_the_phase2_modules() -> None:
    source_cell, check_cell = _cells(BENCHMARK_NOTEBOOK_PATH)
    payload = json.loads(_payload(source_cell))
    assert set(payload) == set(BENCHMARK_MODULES), "embedded file list differs from the project"
    for relative, text in payload.items():
        assert f"{REVISION_NAME} = " in text, f"{relative} carries no revision marker"
    expected = f'EXPECTED_SOURCE_REVISION = "{fingerprint(benchmark_sources())}"'
    assert expected in source_cell
    for module in BENCHMARK_CHECKED:
        assert f'"{module}"' in check_cell, f"revision check does not cover {module}"
    print(f"benchmark notebook embeds {len(payload)} modules and checks {len(BENCHMARK_CHECKED)}")


def test_benchmark_notebook_covers_the_workflow() -> None:
    """The notebook must do the four things the phase asks for, in code."""
    code = "\n".join(_code_texts(BENCHMARK_NOTEBOOK_PATH))
    required = {
        "mount Google Drive": "drive.mount(",
        "integrity gate": "check_dataset(",
        "resume-aware run": "run_split(",
        "smoke-test limit": "OSB_LIMIT",
        "worst-pair panels": "plot_benchmark_pair(",
        "provenance": "provenance.json",
        "write results back": "shutil.copytree(",
        "single-pipeline contract": "segment_object(",
    }
    for label, needle in required.items():
        assert needle in code, f"the benchmark notebook never runs {label} ({needle})"
    print(f"benchmark notebook covers {len(required)} workflow steps")


def main() -> int:
    test_notebook_is_up_to_date()
    test_notebook_is_committed_without_outputs()
    test_check_reports_stored_outputs()
    test_every_embedded_module_carries_a_revision()
    test_check_cell_heals_a_stale_session()
    test_check_cell_reports_a_module_without_a_marker()
    test_check_cell_asks_for_a_restart_when_reload_raises()
    test_check_cell_reports_what_it_could_not_repair()
    test_benchmark_notebook_is_up_to_date()
    test_benchmark_notebook_is_committed_without_outputs()
    test_benchmark_notebook_embeds_the_phase2_modules()
    test_benchmark_notebook_covers_the_workflow()
    print("NOTEBOOK BUNDLE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
