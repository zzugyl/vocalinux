"""
XWayland helper process for the dictation overlay.

On a Wayland session without gtk-layer-shell (GNOME/Mutter) the overlay is a
plain xdg_toplevel: the compositor ignores ``move()`` and ``keep_above``, so a
maximized editor covers the orb while dictation itself keeps working. This
module runs the existing :class:`~vocalinux.ui.dictation_overlay.DictationOverlay`
inside a small helper process started with ``GDK_BACKEND=x11``, where the
window can be placed bottom-center and raised above the desktop's toplevels.

The main process keeps a :class:`RemoteDictationOverlay` with the same surface
the tray drives. Commands travel over the helper's stdin as JSON lines and
events come back on stdout; if the helper cannot start or dies mid-session the
proxy falls back to the in-process overlay, so the worst case is today's
behavior rather than no indicator at all.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from typing import IO, Any, Callable, Optional

from ..common_types import RecognitionState
from .dictation_overlay import DictationOverlay, DictationOverlayController

logger = logging.getLogger(__name__)

# Commands: main process → helper, one JSON object per line on stdin.
CMD_STATE = "state"
CMD_ENABLED = "enabled"
CMD_LEVEL = "level"
CMD_QUIT = "quit"

# Events: helper → main process, one JSON object per line on stdout.
EVT_READY = "ready"
EVT_ERROR = "error"
EVT_CLOSE_REQUEST = "close_request"

#: How long the proxy waits for the helper's ready handshake.
READY_TIMEOUT_S = 5.0
#: How long destroy() waits for a polite exit before terminating the helper.
QUIT_TIMEOUT_S = 2.0


def encode_message(message: dict[str, Any]) -> str:
    """Serialize one protocol message as a compact JSON object (no newline)."""
    return json.dumps(message, separators=(",", ":"))


def decode_message(line: str) -> Optional[dict[str, Any]]:
    """Parse one protocol line; return None for garbage instead of raising."""
    try:
        message = json.loads(line)
    except (TypeError, ValueError):
        return None
    return message if isinstance(message, dict) else None


def apply_command(
    overlay: DictationOverlay,
    message: dict[str, Any],
    request_quit: Callable[[], None],
) -> None:
    """Apply one decoded command to an overlay.

    Runs inside the helper's GLib main loop, or directly against the proxy's
    in-process fallback. Malformed input is logged and skipped, never raised.
    """
    cmd = message.get("cmd")
    if cmd == CMD_STATE:
        raw = message.get("state")
        try:
            state = RecognitionState[str(raw)]
        except (KeyError, TypeError):
            logger.warning("Overlay helper: unknown state %r", raw)
            return
        overlay.on_recognition_state(state)
    elif cmd == CMD_ENABLED:
        overlay.set_enabled(bool(message.get("enabled", False)))
    elif cmd == CMD_LEVEL:
        try:
            level = float(message.get("level", 0.0))
        except (TypeError, ValueError):
            return
        overlay.set_audio_level(level)
    elif cmd == CMD_QUIT:
        request_quit()
    else:
        logger.debug("Overlay helper: ignoring unknown command %r", cmd)


def read_commands(
    stream: IO[str],
    handle: Callable[[str], None],
    on_eof: Callable[[], None],
) -> None:
    """Read command lines from the helper's stdin until EOF.

    Runs on a daemon thread inside the helper. EOF also covers the main
    process dying, which is how the helper avoids becoming an orphan: the
    kernel closes the pipe and this loop ends.
    """
    try:
        for line in stream:
            line = line.strip()
            if line:
                handle(line)
    except (OSError, ValueError) as e:
        logger.debug("Overlay helper stdin closed: %s", e)
    on_eof()


def _emit(message: dict[str, Any]) -> None:
    """Write one event to stdout for the parent process (best effort)."""
    try:
        sys.stdout.write(encode_message(message) + "\n")
        sys.stdout.flush()
    except (OSError, ValueError) as e:
        logger.debug("Overlay helper could not report to the parent: %s", e)


def _forward_close_request() -> None:
    """Tell the parent the dock asked to close the overlay (#1028)."""
    _emit({"event": EVT_CLOSE_REQUEST})


def _apply_line(overlay: DictationOverlay, line: str, request_quit: Callable[[], None]) -> None:
    """Decode one stdin line and apply it (helper side, GLib main loop)."""
    message = decode_message(line)
    if message is None:
        logger.warning("Overlay helper: ignoring unparsable command: %r", line[:200])
        return
    apply_command(overlay, message, request_quit)


def _noop() -> None:
    """Quit hook for apply_command against the in-process fallback overlay."""


def main() -> int:
    """Entry point for ``python -m vocalinux.ui.overlay_helper``.

    The parent starts this module with ``GDK_BACKEND=x11`` so the overlay
    window is an X window the compositor lets us position and raise.
    """
    overlay = DictationOverlay(
        enabled=False,
        on_close_request=_forward_close_request,
        # Unmanaged X window: raise it ourselves so a focused editor cannot
        # bury the orb (the in-process path keeps its WM-managed behavior).
        raise_on_show=True,
    )
    if not overlay.gtk_ready:
        _emit({"event": EVT_ERROR, "error": "GTK window unavailable (no X display?)"})
        return 1

    from gi.repository import GLib  # after GTK init so the display backend has settled

    loop = GLib.MainLoop()

    def quit_main() -> None:
        # GLib.MainLoop.quit is documented thread-safe, unlike GTK calls.
        loop.quit()

    def handle_line(line: str) -> None:
        # Commands arrive on the reader thread; GTK work must hop to the loop.
        # idle_add passes the extra arguments straight to the callback.
        GLib.idle_add(_apply_line, overlay, line, quit_main)

    _emit({"event": EVT_READY})
    threading.Thread(
        target=read_commands,
        args=(sys.stdin, handle_line, quit_main),
        name="overlay-helper-stdin",
        daemon=True,
    ).start()
    loop.run()
    overlay.destroy()
    return 0


def _spawn_helper() -> subprocess.Popen[str]:
    """Start the helper with the X11 GDK backend.

    This is our own interpreter, so the environment passes through untouched
    (an AppImage child needs those paths); ``GDK_BACKEND=x11`` is the one
    addition that makes the helper's window an X window.
    """
    env = os.environ.copy()
    env["GDK_BACKEND"] = "x11"
    return subprocess.Popen(
        [sys.executable, "-m", "vocalinux.ui.overlay_helper"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )


def _post_via_glib(callback: Callable[[], None]) -> None:
    """Queue a callback on the GLib main loop the tray already runs."""
    try:
        from gi.repository import GLib

        GLib.idle_add(callback)
    except Exception as e:
        logger.warning("Could not queue overlay event on the main loop: %s", e)


class RemoteDictationOverlay:
    """Proxy that drives a :class:`DictationOverlay` in the XWayland helper.

    Exposes the same surface the tray uses (``set_enabled``,
    ``on_recognition_state``, ``set_audio_level``, ``destroy``), is safe to
    call from the audio thread, and degrades to the in-process overlay when
    the helper is unavailable.
    """

    def __init__(
        self,
        enabled: bool = True,
        on_close_request: Optional[Callable[[], None]] = None,
        spawn: Optional[Callable[[], subprocess.Popen[str]]] = None,
        post_to_main: Optional[Callable[[Callable[[], None]], Any]] = None,
    ) -> None:
        self.controller = DictationOverlayController(enabled=enabled)
        self._on_close_request = on_close_request
        self._post_to_main = post_to_main if post_to_main is not None else _post_via_glib
        self._lock = threading.Lock()
        self._destroyed = False
        self._fallback: Optional[DictationOverlay] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._threads: list[threading.Thread] = []
        self._handshake = threading.Event()
        self._ready = False

        proc = (spawn or _spawn_helper)()
        self._proc = proc
        self._start_thread(self._drain_stderr, "overlay-helper-stderr")
        self._start_thread(self._read_events, "overlay-helper-stdout")
        if not self._handshake.wait(READY_TIMEOUT_S) or not self._ready:
            self._reap_helper()
            raise RuntimeError("XWayland overlay helper did not become ready")
        # Sync the starting state; set_enabled(True) prewarms the helper window.
        self._send({"cmd": CMD_ENABLED, "enabled": bool(enabled)})

    def _start_thread(self, target: Callable[[], None], name: str) -> None:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        self._threads.append(thread)

    def _drain_stderr(self) -> None:
        """Log helper stderr so a crash there is visible and never blocks it."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                logger.debug("overlay helper: %s", line.rstrip())
        except (OSError, ValueError):
            pass

    def _read_events(self) -> None:
        """Consume helper stdout: handshake, forwarded dock-close requests, EOF."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._handshake.set()
            return
        try:
            for line in proc.stdout:
                message = decode_message(line)
                if message is None:
                    continue
                event = message.get("event")
                if event == EVT_READY:
                    self._ready = True
                    self._handshake.set()
                elif event == EVT_ERROR:
                    logger.warning("Overlay helper reported an error: %s", message.get("error"))
                    self._handshake.set()
                elif event == EVT_CLOSE_REQUEST:
                    # tray _quit ends in Gtk.main_quit(): never run it here.
                    self._post_to_main(self._run_close_request)
                else:
                    logger.debug("Overlay helper: unknown event %r", event)
        except (OSError, ValueError) as e:
            logger.debug("Overlay helper stdout closed: %s", e)
        finally:
            self._handshake.set()

    def _run_close_request(self) -> None:
        """Forward the helper's dock-close request to the tray's Quit (#1028)."""
        if self._destroyed:
            return
        callback = self._on_close_request
        if callback is not None:
            callback()

    def set_enabled(self, enabled: bool) -> None:
        """Enable/disable the overlay feature and forward it to the helper."""
        self.controller.set_enabled(enabled)
        self._send({"cmd": CMD_ENABLED, "enabled": bool(enabled)})

    def on_recognition_state(self, state: RecognitionState) -> None:
        """Forward a recognition state change (same path as the tray icon)."""
        self.controller.set_state(state)
        self._send({"cmd": CMD_STATE, "state": state.name})

    def set_audio_level(self, level: float) -> None:
        """Accept microphone loudness (0–100); safe from the audio thread."""
        self._send({"cmd": CMD_LEVEL, "level": float(level)})

    def prewarm(self) -> bool:
        """The helper prewarms its own window; nothing to do in the parent."""
        return False

    def _send(self, message: dict[str, Any]) -> None:
        """Queue one command for the helper; deliver locally if it is gone."""
        with self._lock:
            if self._destroyed:
                return
            proc = self._proc
            if proc is not None and proc.stdin is not None and proc.poll() is None:
                try:
                    proc.stdin.write(encode_message(message) + "\n")
                    proc.stdin.flush()
                    return
                except (BrokenPipeError, OSError, ValueError) as e:
                    logger.warning(
                        "Overlay helper pipe failed (%s); using the in-process overlay", e
                    )
                    self._reap_helper()
            self._deliver_locally(message)

    def _deliver_locally(self, message: dict[str, Any]) -> None:
        """Run a command against the in-process overlay (helper unavailable)."""
        if self._fallback is None:
            logger.warning("XWayland overlay helper unavailable; using the in-process overlay")
            self._fallback = DictationOverlay(
                enabled=self.controller.enabled,
                on_close_request=self._on_close_request,
            )
            # The fresh window never saw the earlier commands: replay the state.
            self._fallback.on_recognition_state(self.controller.state)
        apply_command(self._fallback, message, _noop)

    def destroy(self) -> None:
        """Ask the helper to quit, reap it, and drop any in-process fallback."""
        with self._lock:
            if self._destroyed:
                return
            proc = self._proc
            if proc is not None and proc.stdin is not None and proc.poll() is None:
                try:
                    proc.stdin.write(encode_message({"cmd": CMD_QUIT}) + "\n")
                    proc.stdin.flush()
                except (BrokenPipeError, OSError, ValueError):
                    pass
            self._destroyed = True
            self._reap_helper()
        fallback, self._fallback = self._fallback, None
        if fallback is not None:
            fallback.destroy()

    def _reap_helper(self) -> None:
        """Close stdin (EOF is the helper's normal exit), then reap with a deadline.

        Caller holds ``self._lock``.
        """
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            try:
                proc.wait(timeout=QUIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            except OSError:
                pass
        threads, self._threads = self._threads, []
        for thread in threads:
            thread.join(timeout=1.0)


if __name__ == "__main__":
    # `python -m vocalinux.ui.overlay_helper` must actually run the helper;
    # without this guard the module only defines main() and exits 0 silently,
    # which the parent sees as a handshake timeout.
    sys.exit(main())
