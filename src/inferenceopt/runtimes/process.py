# SPDX-License-Identifier: Apache-2.0
"""Run a ``LaunchSpec`` as a supervised child process.

Guarantees:
* No shell: ``argv`` is executed directly.
* The whole process tree dies with the supervisor (Windows Job Object with
  KILL_ON_JOB_CLOSE; a new process group on POSIX), including on crashes.
* Readiness means the health endpoint returned 200; an early exit or a timeout becomes a
  classified ``FailureRecord`` with a log excerpt.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx

from inferenceopt.runtimes.base import LaunchSpec, RuntimeAdapterError
from inferenceopt.schemas import FailureRecord
from inferenceopt.schemas.trial import FailureKind

log = logging.getLogger(__name__)

_OOM = re.compile(
    r"out of memory|cudaMalloc failed|CUDA error: out of memory|failed to allocate|"
    r"OutOfMemoryError|insufficient memory|unable to allocate|not enough memory|"
    r"No available memory for the cache blocks",
    re.I,
)
_CONFIG = re.compile(
    r"error loading model|failed to load model|invalid argument|unrecognized arguments|"
    r"unknown argument|error: invalid|No such file or directory|does not exist",
    re.I,
)
_PORT = re.compile(r"address already in use|couldn't bind|only one usage of each socket", re.I)


# Windows "code integrity" denial (Smart App Control / App Control for Business).
_CODE_INTEGRITY_STATUS = 0xC0E90002
CODE_INTEGRITY_HINT = (
    "Windows blocked this unsigned executable (Smart App Control or an App Control policy). "
    "Use docker mode, run inside WSL2, or ask your administrator to allow it."
)


def is_code_integrity_block(returncode: int) -> bool:
    return sys.platform == "win32" and (returncode & 0xFFFFFFFF) == _CODE_INTEGRITY_STATUS


class LaunchError(RuntimeAdapterError):
    def __init__(self, failure: FailureRecord) -> None:
        super().__init__(f"{failure.kind}: {failure.message}")
        self.failure = failure


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def classify_failure(log_text: str, *, timed_out: bool = False) -> FailureKind:
    if _OOM.search(log_text):
        return "oom"
    if timed_out:
        return "timeout"
    if _PORT.search(log_text) or _CONFIG.search(log_text):
        return "config"
    return "crash"


# --- Windows job object ----------------------------------------------------------------


if sys.platform == "win32":

    class _WinJob:
        """Kill-on-close Job Object; also reports the tree's peak memory."""

        def __init__(self) -> None:
            import ctypes
            from ctypes import wintypes

            self._ctypes = ctypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._k32 = k32

            class IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
                )]  # fmt: skip

            class BasicLimits(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_longlong),
                    ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BasicLimits),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            self._Extended = ExtendedLimits
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.OpenProcess.restype = wintypes.HANDLE
            self.handle = k32.CreateJobObjectW(None, None)
            if not self.handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
            info = ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(
                self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)
            ):  # 9 = JobObjectExtendedLimitInformation
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

        def assign(self, pid: int) -> None:
            proc = self._k32.OpenProcess(0x0100 | 0x0001, False, pid)  # SET_QUOTA | TERMINATE
            if not proc:
                raise OSError(self._ctypes.get_last_error(), "OpenProcess failed")
            try:
                if not self._k32.AssignProcessToJobObject(self.handle, proc):
                    raise OSError(self._ctypes.get_last_error(), "AssignProcessToJobObject failed")
            finally:
                self._k32.CloseHandle(proc)

        def peak_memory(self) -> int | None:
            info = self._Extended()
            ok = self._k32.QueryInformationJobObject(
                self.handle, 9, self._ctypes.byref(info), self._ctypes.sizeof(info), None
            )
            return int(info.PeakJobMemoryUsed) if ok else None

        def close(self) -> None:
            if self.handle:
                self._k32.CloseHandle(self.handle)  # kills every process in the job
                self.handle = None


# --- supervised server ------------------------------------------------------------------

_running: set[RunningServer] = set()
_lock = threading.Lock()


@dataclass(eq=False)
class RunningServer:
    spec: LaunchSpec
    process: subprocess.Popen[bytes]
    log_path: Path
    started_at: float
    ready_at: float | None = None
    _log_file: IO[bytes] | None = field(default=None, repr=False)
    _job: Any = field(default=None, repr=False)

    @property
    def load_time_s(self) -> float | None:
        return None if self.ready_at is None else self.ready_at - self.started_at

    @property
    def health_url(self) -> str:
        host = "127.0.0.1" if self.spec.host in ("0.0.0.0", "") else self.spec.host  # noqa: S104
        return f"http://{host}:{self.spec.port}{self.spec.health_path}"

    def log_tail(self, lines: int = 60) -> str:
        try:
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def peak_host_memory(self) -> int | None:
        """Peak memory of the whole process tree (Windows job accounting), if available."""
        return self._job.peak_memory() if self._job is not None else None

    def alive(self) -> bool:
        return self.process.poll() is None

    def stop(self, timeout: float = 15) -> None:
        try:
            if self.spec.container_name and (docker := shutil.which("docker")):
                subprocess.run(  # noqa: S603
                    [docker, "stop", "-t", str(int(timeout)), self.spec.container_name],
                    capture_output=True,
                    timeout=timeout + 10,
                    check=False,
                )
            if self.alive():
                if sys.platform == "win32":
                    self.process.terminate()
                else:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout)
                except subprocess.TimeoutExpired:
                    if sys.platform != "win32":
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.kill()
                    self.process.wait(5)
        finally:
            if self._job is not None:
                self._job.close()  # takes any surviving grandchildren with it
            if self._log_file is not None:
                self._log_file.close()
            with _lock:
                _running.discard(self)


@atexit.register
def _stop_all() -> None:
    with _lock:
        servers = list(_running)
    for server in servers:
        with contextlib.suppress(Exception):
            server.stop(timeout=5)


def start_server(spec: LaunchSpec, log_path: Path) -> RunningServer:
    """Start the process and wait until its health endpoint answers 200."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("wb")
    env = {**os.environ, **spec.env}
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(  # noqa: S603 - structured argv, never a shell
            spec.argv,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            **kwargs,
        )
    except OSError as exc:
        log_file.close()
        raise LaunchError(
            FailureRecord(stage="start", kind="config", message=f"cannot execute: {exc}")
        ) from exc

    job = None
    if sys.platform == "win32":
        try:
            job = _WinJob()
            job.assign(process.pid)
        except OSError as exc:  # still supervised, just without tree-kill guarantees
            log.warning("could not attach job object: %s", exc)
            job = None

    server = RunningServer(
        spec=spec,
        process=process,
        log_path=log_path,
        started_at=time.monotonic(),
        _log_file=log_file,
        _job=job,
    )
    with _lock:
        _running.add(server)
    try:
        _wait_ready(server)
    except BaseException:
        server.stop(timeout=5)
        raise
    return server


def _wait_ready(server: RunningServer) -> None:
    deadline = server.started_at + server.spec.startup_timeout_s
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while True:
            code = server.process.poll()
            if code is not None and is_code_integrity_block(code):
                raise LaunchError(
                    FailureRecord(stage="start", kind="config", message=CODE_INTEGRITY_HINT)
                )
            if code is not None:
                tail = server.log_tail()
                raise LaunchError(
                    FailureRecord(
                        stage="start",
                        kind=classify_failure(tail),
                        message=f"process exited with code {code} before becoming ready",
                        log_excerpt=tail[-4000:],
                    )
                )
            try:
                if client.get(server.health_url).status_code == 200:
                    server.ready_at = time.monotonic()
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                tail = server.log_tail()
                raise LaunchError(
                    FailureRecord(
                        stage="start",
                        kind=classify_failure(tail, timed_out=True),
                        message=f"not ready after {server.spec.startup_timeout_s:.0f}s",
                        log_excerpt=tail[-4000:],
                    )
                )
            time.sleep(0.25)
