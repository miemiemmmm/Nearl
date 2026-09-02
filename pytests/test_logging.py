import io
import os
import logging

import pytest

import nearl


@pytest.fixture
def isolated_logger(tmp_path, monkeypatch):
  stdout_buf = io.StringIO()
  stderr_buf = io.StringIO()
  monkeypatch.setattr(nearl._stdout_handler, "stream", stdout_buf)
  monkeypatch.setattr(nearl._stderr_handler, "stream", stderr_buf)
  monkeypatch.setattr(nearl, "_file_handler", None)
  monkeypatch.setattr(nearl._package_logger, "handlers", [nearl._stdout_handler, nearl._stderr_handler])
  monkeypatch.setitem(nearl.CONFIG, "tempfolder", str(tmp_path))
  monkeypatch.setitem(nearl.CONFIG, "debug", False)
  monkeypatch.setitem(nearl.CONFIG, "verbose", False)
  return tmp_path, stdout_buf, stderr_buf


def test_log_info_goes_to_stdout(isolated_logger):
  _, out, err = isolated_logger
  nearl.log("plain info message")
  assert "plain info message" in out.getvalue()
  assert err.getvalue() == ""


def test_log_warning_goes_to_stderr(isolated_logger):
  _, out, err = isolated_logger
  nearl.log.warning("a real warning")
  assert out.getvalue() == ""
  assert "a real warning" in err.getvalue()


def test_log_error_goes_to_stderr(isolated_logger):
  _, out, err = isolated_logger
  nearl.log.error("a real error")
  assert out.getvalue() == ""
  assert "a real error" in err.getvalue()


def test_log_does_not_guess_level_from_text(isolated_logger):
  # log() must not classify by keyword content -- only log.warning/log.error/
  # log.debug raise the severity, otherwise "No errors found" misclassifies.
  _, out, err = isolated_logger
  nearl.log("No errors were found during validation")
  nearl.log("This is the warning-free control run")
  assert "No errors were found during validation" in out.getvalue()
  assert "This is the warning-free control run" in out.getvalue()
  assert err.getvalue() == ""


def test_log_debug_suppressed_by_default(isolated_logger):
  _, out, err = isolated_logger
  nearl.log.debug("hidden debug line")
  assert out.getvalue() == ""
  assert err.getvalue() == ""


def test_log_debug_shown_when_debug_enabled(isolated_logger, monkeypatch):
  _, out, _ = isolated_logger
  monkeypatch.setitem(nearl.CONFIG, "debug", True)
  nearl.log.debug("visible debug line")
  assert "visible debug line" in out.getvalue()


def test_log_zero_args_does_not_crash(isolated_logger):
  nearl.log()


def test_log_non_string_first_arg_does_not_crash(isolated_logger):
  _, out, _ = isolated_logger
  nearl.log(42, "message")
  assert "42 message" in out.getvalue()


def test_log_console_and_file_match(isolated_logger):
  tmp_path, out, _ = isolated_logger
  nearl.log("a traceable message")
  logfiles = list(tmp_path.glob("nearl.*.log"))
  assert len(logfiles) == 1
  assert logfiles[0].read_text().strip() == out.getvalue().strip()


def test_log_file_named_by_pid(isolated_logger):
  tmp_path, _, _ = isolated_logger
  nearl.log("trigger file creation")
  logfiles = list(tmp_path.glob("nearl.*.log"))
  assert len(logfiles) == 1
  assert logfiles[0].name == f"nearl.{os.getpid()}.log"


def test_log_file_override_routes_to_given_stream(isolated_logger):
  _, out, err = isolated_logger
  override = io.StringIO()
  nearl.log("forced elsewhere", file=override)
  assert out.getvalue() == ""
  assert err.getvalue() == ""
  assert "forced elsewhere" in override.getvalue()


def test_child_logger_propagates_to_same_handlers_and_file(isolated_logger):
  tmp_path, _, err = isolated_logger
  feat_logger = logging.getLogger("nearl.features")
  feat_logger.warning("child logger warning")
  assert "child logger warning" in err.getvalue()
  logfiles = list(tmp_path.glob("nearl.*.log"))
  assert "child logger warning" in logfiles[0].read_text()
