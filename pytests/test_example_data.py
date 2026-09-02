import os
import textwrap

import pytest


def _seed(path):
    """Pre-seed the download/extract markers so get_example_data() skips the network."""
    (path / "example_data.tar.gz").write_bytes(b"")
    extracted_dir = path / "example_data"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "data.py").write_text(
        textwrap.dedent("""
    def get_data():
      return {"MINI_TRAJSET": [("traj.nc", "top.pdb")]}
  """)
    )


def test_get_example_data_creates_a_missing_target_folder(tmp_path, monkeypatch):
    """The docs tell users to pass any writable path; it must not have to exist yet."""
    import subprocess

    import nearl

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)
    target = tmp_path / "fresh"
    assert not target.exists()

    # Reaching the data-index check means the old "Path ... does not exist" is gone.
    with pytest.raises(OSError, match="Data index file not found"):
        nearl.get_example_data(str(target))
    assert target.is_dir()


def test_get_example_data_does_not_change_the_working_directory(tmp_path):
    """It used to chdir() into the extracted folder and never come back."""
    import nearl

    _seed(tmp_path)
    before = os.getcwd()
    nearl.get_example_data(str(tmp_path))
    assert os.getcwd() == before


def test_get_example_data_when_invoked_from_a_script(tmp_path):
    """
    Regression test for nearl.get_example_data() failing with
    "ModuleNotFoundError: No module named 'data'" when the calling code is a
    script file (python foo.py) rather than `python -c "..."`.

    os.chdir()-ing into the extracted example-data folder does not put that
    folder on sys.path, so a plain `import data` only happened to work when
    sys.path already contained the (dynamically-resolved) current directory,
    e.g. under `python -c`. Under a real script, sys.path[0] is the script's
    own directory, so the import fails even though data.py exists on disk.

    This test avoids the network by pre-seeding the archive/extracted folder
    so get_example_data() skips straight to the `import data` step, and it
    is run the same way a user's script would be (as a subprocess script,
    not `python -c`) so it reproduces the actual reported failure mode.
    """
    # Pre-seed the "already downloaded" and "already extracted" markers so
    # get_example_data() skips the network entirely and exercises only the
    # data-index-loading step.
    (tmp_path / "example_data.tar.gz").write_bytes(b"")
    extracted_dir = tmp_path / "example_data"
    extracted_dir.mkdir()
    (extracted_dir / "data.py").write_text(
        textwrap.dedent("""
    def get_data():
      return {"MINI_TRAJSET": [("traj.nc", "top.pdb")]}
  """)
    )

    runner_script = tmp_path / "run_get_example_data.py"
    runner_script.write_text(
        textwrap.dedent(f"""
    import nearl
    paths = nearl.get_example_data({str(tmp_path)!r})
    assert paths == {{"MINI_TRAJSET": [("traj.nc", "top.pdb")]}}, paths
    print("OK")
  """)
    )

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, str(runner_script)],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"get_example_data() failed when run as a script:\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "OK" in result.stdout
