"""
Tests for the XWayland overlay helper and its parent-side proxy.

The protocol (JSON lines), the proxy's forwarding/fallback behavior, the
helper's entry point and the backend-selection rules are all pure Python:
no child process is spawned and no display is opened. The tray wiring is
checked at the source level the way test_host_process walks subprocess calls.
"""

import ast
import io
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from vocalinux.common_types import RecognitionState
from vocalinux.ui.dictation_overlay import (
    create_dictation_overlay,
    helper_backend_available,
)
from vocalinux.ui.overlay_helper import (
    CMD_ENABLED,
    CMD_LEVEL,
    CMD_QUIT,
    CMD_STATE,
    EVT_ERROR,
    EVT_READY,
    RemoteDictationOverlay,
    apply_command,
    decode_message,
    encode_message,
    main,
    read_commands,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_proc(events: str = '{"event":"ready"}\n') -> MagicMock:
    """Build a stand-in Popen whose handshake line is already queued."""
    proc = MagicMock()
    proc.stdin = MagicMock()
    proc.stdout = io.StringIO(events)
    proc.stderr = io.StringIO("")
    proc.poll.return_value = None
    return proc


def _proxy(proc: MagicMock, **kwargs) -> RemoteDictationOverlay:
    """Construct a proxy against the fake process, with synchronous posting."""
    return RemoteDictationOverlay(
        spawn=lambda: proc,
        post_to_main=lambda callback: callback(),
        **kwargs,
    )


class TestProtocol(unittest.TestCase):
    """JSON line encoding shared by the proxy and the helper."""

    def test_message_round_trip(self):
        message = {"cmd": CMD_STATE, "state": "PROCESSING"}
        self.assertEqual(decode_message(encode_message(message)), message)

    def test_decode_rejects_non_objects(self):
        for line in ("", "not json", "[1, 2]", '"text"', "42"):
            with self.subTest(line=line):
                self.assertIsNone(decode_message(line))


class TestApplyCommand(unittest.TestCase):
    """Command dispatch against an overlay (helper side and fallback side)."""

    def setUp(self):
        self.overlay = MagicMock()
        self.quit = MagicMock()

    def test_state_translates_to_the_enum(self):
        apply_command(self.overlay, {"cmd": CMD_STATE, "state": "LISTENING"}, self.quit)
        self.overlay.on_recognition_state.assert_called_once_with(RecognitionState.LISTENING)

    def test_unknown_state_is_skipped(self):
        apply_command(self.overlay, {"cmd": CMD_STATE, "state": "NOPE"}, self.quit)
        self.overlay.on_recognition_state.assert_not_called()

    def test_enabled_forwards_as_bool(self):
        apply_command(self.overlay, {"cmd": CMD_ENABLED, "enabled": True}, self.quit)
        self.overlay.set_enabled.assert_called_once_with(True)

    def test_level_is_coerced_to_float(self):
        apply_command(self.overlay, {"cmd": CMD_LEVEL, "level": "42.5"}, self.quit)
        self.overlay.set_audio_level.assert_called_once_with(42.5)

    def test_bad_level_is_skipped(self):
        apply_command(self.overlay, {"cmd": CMD_LEVEL, "level": "loud"}, self.quit)
        self.overlay.set_audio_level.assert_not_called()

    def test_quit_runs_the_hook(self):
        apply_command(self.overlay, {"cmd": CMD_QUIT}, self.quit)
        self.quit.assert_called_once_with()

    def test_unknown_command_is_ignored(self):
        apply_command(self.overlay, {"cmd": "reboot"}, self.quit)
        self.quit.assert_not_called()


class TestReadCommands(unittest.TestCase):
    """The helper's stdin loop: dispatch every line, always signal EOF."""

    def test_dispatches_lines_and_signals_eof(self):
        stream = io.StringIO('{"cmd":"enabled","enabled":true}\n\njunk\n{"cmd":"quit"}\n')
        handled: list[str] = []
        on_eof = MagicMock()
        read_commands(stream, handled.append, on_eof)
        self.assertEqual(len(handled), 3)
        on_eof.assert_called_once_with()

    def test_empty_stream_still_signals_eof(self):
        on_eof = MagicMock()
        read_commands(io.StringIO(""), MagicMock(), on_eof)
        on_eof.assert_called_once_with()


class TestRemoteDictationOverlay(unittest.TestCase):
    """Parent-side proxy: forwarding, handshake, fallback and teardown."""

    def test_forwards_state_enabled_and_level_over_stdin(self):
        proc = _fake_proc()
        proxy = _proxy(proc, enabled=True)
        proxy.on_recognition_state(RecognitionState.LISTENING)
        proxy.set_audio_level(55.0)
        proxy.set_enabled(False)
        self.assertFalse(proxy.prewarm())

        lines = [json.loads(call.args[0]) for call in proc.stdin.write.call_args_list]
        self.assertEqual(
            lines,
            [
                {"cmd": CMD_ENABLED, "enabled": True},  # initial sync after handshake
                {"cmd": CMD_STATE, "state": "LISTENING"},
                {"cmd": CMD_LEVEL, "level": 55.0},
                {"cmd": CMD_ENABLED, "enabled": False},
            ],
        )
        proxy.destroy()

    def test_missing_ready_handshake_raises(self):
        proc = _fake_proc(events="")
        with self.assertRaises(RuntimeError):
            _proxy(proc)

    def test_close_request_is_marshaled_to_the_main_thread(self):
        proc = _fake_proc('{"event":"ready"}\n{"event":"close_request"}\n')
        done = threading.Event()
        proxy = _proxy(proc, enabled=False, on_close_request=done.set)
        try:
            self.assertTrue(done.wait(2.0))
        finally:
            proxy.destroy()

    def test_dead_helper_falls_back_to_the_in_process_overlay(self):
        proc = _fake_proc()
        proxy = _proxy(proc, enabled=True)
        with patch("vocalinux.ui.overlay_helper.DictationOverlay") as in_process:
            proc.poll.return_value = 1
            proxy.on_recognition_state(RecognitionState.LISTENING)

            fallback = in_process.return_value
            in_process.assert_called_once_with(enabled=True, on_close_request=None)
            # The controller already advanced, so the replay and the fresh
            # command both apply LISTENING: state and window stay consistent.
            self.assertEqual(
                fallback.on_recognition_state.call_args_list,
                [call(RecognitionState.LISTENING), call(RecognitionState.LISTENING)],
            )
            proxy.set_enabled(False)
            fallback.set_enabled.assert_called_once_with(False)
        proxy.destroy()

    def test_destroy_sends_quit_and_is_idempotent(self):
        proc = _fake_proc()
        proxy = _proxy(proc, enabled=False)
        proc.stdin.write.reset_mock()

        proxy.destroy()
        proc.stdin.write.assert_called_once_with(encode_message({"cmd": CMD_QUIT}) + "\n")
        proc.stdin.close.assert_called_once()
        proc.wait.assert_called_once()

        proxy.set_enabled(True)  # after destroy: no writes, no resurrection
        proxy.on_recognition_state(RecognitionState.LISTENING)
        proc.stdin.write.assert_called_once()
        proxy.destroy()  # second teardown is a no-op
        proc.stdin.write.assert_called_once()


class TestHelperMain(unittest.TestCase):
    """The helper entry point: report readiness, then run the GLib loop."""

    def test_running_the_module_as_main_actually_runs_helper(self):
        """`python -m vocalinux.ui.overlay_helper` must invoke main().

        The parent spawns exactly this way; a missing ``__main__`` guard makes
        the module define main() and exit 0 without a byte of output, which the
        parent sees only as a handshake timeout (caught on a real machine).
        """
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_REPO_ROOT / "src")
        env["GDK_BACKEND"] = "x11"
        result = subprocess.run(
            [sys.executable, "-m", "vocalinux.ui.overlay_helper"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=8,
            env=env,
        )
        self.assertIn(
            '"event"',
            result.stdout,
            f"main() never ran (exit={result.returncode}); stderr: {result.stderr}",
        )
        self.assertIn(result.returncode, (0, 1), result.stderr)

    def test_emits_ready_and_runs_the_loop(self):
        with (
            patch("vocalinux.ui.overlay_helper.DictationOverlay") as overlay_cls,
            patch("vocalinux.ui.overlay_helper._emit") as emit,
            patch("vocalinux.ui.overlay_helper.read_commands"),
        ):
            result = main()

        self.assertEqual(result, 0)
        kwargs = overlay_cls.call_args.kwargs
        self.assertEqual(kwargs["enabled"], False)
        self.assertTrue(callable(kwargs["on_close_request"]))
        self.assertTrue(kwargs["raise_on_show"])
        emit.assert_called_once_with({"event": EVT_READY})
        overlay_cls.return_value.destroy.assert_called_once_with()

    def test_reports_error_when_the_window_cannot_be_created(self):
        with (
            patch("vocalinux.ui.overlay_helper.DictationOverlay") as overlay_cls,
            patch("vocalinux.ui.overlay_helper._emit") as emit,
        ):
            overlay_cls.return_value.gtk_ready = False
            result = main()

        self.assertEqual(result, 1)
        self.assertEqual(emit.call_args.args[0]["event"], EVT_ERROR)


class TestHelperBackendAvailable(unittest.TestCase):
    """Backend selection: only a display-less-free Wayland session without
    layer shell should ever spawn the helper."""

    @staticmethod
    def _wayland_env(**extra) -> dict:
        env = {"XDG_SESSION_TYPE": "wayland", "WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        env.update(extra)
        return env

    def test_x11_session_stays_in_process(self):
        self.assertFalse(helper_backend_available({"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}))

    def test_wayland_without_x_display_stays_in_process(self):
        self.assertFalse(helper_backend_available(self._wayland_env(DISPLAY="")))

    def test_wayland_with_x_display_uses_helper(self):
        with patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=False):
            self.assertTrue(helper_backend_available(self._wayland_env()))

    def test_layer_shell_session_stays_in_process(self):
        with patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=True):
            self.assertFalse(helper_backend_available(self._wayland_env()))

    def test_gdk_backend_x11_forces_in_process(self):
        with patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=False):
            self.assertFalse(helper_backend_available(self._wayland_env(GDK_BACKEND="x11")))
            self.assertFalse(helper_backend_available(self._wayland_env(GDK_BACKEND="x11,wayland")))
            self.assertTrue(helper_backend_available(self._wayland_env(GDK_BACKEND="wayland,x11")))

    def test_pytest_never_spawns_a_helper(self):
        with patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=False):
            self.assertFalse(helper_backend_available(self._wayland_env(PYTEST_RUNNING="1")))

    def test_live_wayland_socket_alone_counts_as_wayland(self):
        # Issue #752: Plasma can keep XDG_SESSION_TYPE=x11 while WAYLAND_DISPLAY is live.
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        with patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=False):
            self.assertTrue(helper_backend_available(env))

    def test_defaults_to_the_process_environment(self):
        env = {
            "XDG_SESSION_TYPE": "wayland",
            "WAYLAND_DISPLAY": "wayland-0",
            "DISPLAY": ":0",
            "PYTEST_RUNNING": "",
        }
        with (
            patch.dict(os.environ, env),
            patch("vocalinux.ui.dictation_overlay._layer_shell_supported", return_value=False),
        ):
            self.assertTrue(helper_backend_available())


class TestCreateDictationOverlay(unittest.TestCase):
    """The factory: proxy when the helper fits, in-process everywhere else."""

    def test_helper_backend_returns_the_proxy(self):
        sentinel = MagicMock()
        with (
            patch("vocalinux.ui.dictation_overlay.helper_backend_available", return_value=True),
            patch(
                "vocalinux.ui.overlay_helper.RemoteDictationOverlay", return_value=sentinel
            ) as remote,
        ):
            result = create_dictation_overlay(enabled=False)

        self.assertIs(result, sentinel)
        remote.assert_called_once_with(enabled=False, on_close_request=None)

    def test_helper_failure_falls_back_in_process(self):
        with (
            patch("vocalinux.ui.dictation_overlay.helper_backend_available", return_value=True),
            patch(
                "vocalinux.ui.overlay_helper.RemoteDictationOverlay",
                side_effect=RuntimeError("boom"),
            ),
            patch("vocalinux.ui.dictation_overlay.DictationOverlay") as in_process,
        ):
            result = create_dictation_overlay(enabled=True)

        in_process.assert_called_once_with(enabled=True, on_close_request=None)
        self.assertIs(result, in_process.return_value)

    def test_unavailable_backend_returns_the_in_process_overlay(self):
        with (
            patch("vocalinux.ui.dictation_overlay.helper_backend_available", return_value=False),
            patch("vocalinux.ui.dictation_overlay.DictationOverlay") as in_process,
        ):
            result = create_dictation_overlay()

        in_process.assert_called_once_with(enabled=True, on_close_request=None)
        self.assertIs(result, in_process.return_value)


class TestTrayWiring(unittest.TestCase):
    """The tray must build its overlay through the factory (source level)."""

    def test_tray_calls_the_factory_and_never_the_class_directly(self):
        source = (_REPO_ROOT / "src" / "vocalinux" / "ui" / "tray_indicator.py").read_text(
            encoding="utf-8"
        )
        called = {
            node.func.id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("create_dictation_overlay", called)
        self.assertNotIn("DictationOverlay", called)
