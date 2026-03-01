import logging
import time
import os
import sys
from datetime import datetime, timedelta


class _OffsetFormatter(logging.Formatter):
    def __init__(self, *args, offset_hours: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self._offset = offset_hours

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created) + __import__("datetime").timedelta(hours=self._offset)
        return ct.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def start_log(log_dir: str, run_name: str) -> str:
    """
    Initialise logging to file + stdout. Call once at the top of your main script.
    Returns the path to the log file.
    """
    global _logger, _start_time
    _start_time = datetime.now() + timedelta(hours=1)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = _start_time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{timestamp}_{run_name}.log")

    _logger = logging.getLogger(run_name)
    _logger.setLevel(logging.DEBUG)
    _logger.propagate = False  # don't double-log if root logger has handlers

    fmt = _OffsetFormatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        offset_hours=1,   # <-- change this as needed
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    _logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(fmt)
    _logger.addHandler(sh)

    sys.stderr = _TeeWriter(sys.stderr, _logger)

    log(f"{'─' * 60}")
    log(f"Run: {run_name}   started at {_start_time:%Y-%m-%d %H:%M:%S}")
    log(f"Log file: {log_path}")
    log(f"{'─' * 60}")

    return log_path


def log(message: str, level: str = "info"):
    """
    Drop-in replacement for print(). Writes a formatted line to file and stdout.

    Levels: 'info' | 'warning' | 'error' | 'debug'
    """
    if _logger is None:
        raise RuntimeError("Call start_log() before logging.")

    elapsed = _elapsed()
    full_msg = f"[+{elapsed}] {message}"
    getattr(_logger, level.lower(), _logger.info)(full_msg)


def log_section(title: str):
    """
    Print a clearly visible section divider.
    """
    pad = "─" * ((58 - len(title)) // 2)
    log(f"{pad}  {title}  {pad}")

def log_warning(message: str):
    log(message, level="warning")

def log_error(message: str):
    log(message, level="error")

def log_debug(message: str):
    log(message, level="debug")


def _elapsed() -> str:
    """
    Returns elapsed time since start_log() as HH:MM:SS.
    """
    if _start_time is None:
        return "??:??:??"
    delta = datetime.now() + timedelta(hours=1) - _start_time
    total_seconds = int(delta.total_seconds())
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


class _TeeWriter:
    """
    Writes to both the original stderr and the logger (as ERROR).
    """
    def __init__(self, original, logger: logging.Logger):
        self._original = original
        self._logger = logger
        self._buffer = ""

    def write(self, message: str):
        self._original.write(message)
        self._buffer += message
        if "\n" in self._buffer:
            lines = self._buffer.split("\n")
            for line in lines[:-1]:
                if line.strip():
                    self._logger.error(f"[stderr] {line}")
            self._buffer = lines[-1]

    def flush(self):
        self._original.flush()