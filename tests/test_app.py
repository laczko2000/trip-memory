"""The screen code loads (a typo in it would stop the app from starting at all)."""

import compileall
from pathlib import Path


def test_all_code_compiles():
    assert compileall.compile_dir(Path(__file__).parent.parent / "trip_memory", quiet=1, force=True)


def test_the_screen_loads():
    from trip_memory import app

    assert callable(app.main)
