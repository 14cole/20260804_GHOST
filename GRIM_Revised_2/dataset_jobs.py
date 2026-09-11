"""Background dataset workers and bounded parallel loading.

Publication transactions live in grim_backend.io.batch; catalog/UI ownership
remains with DatasetOpsMixin."""
from __future__ import annotations

import ctypes
import math
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import QObject, QThread, Signal
from grim_backend.datasets.grid import RcsGrid
from grim_backend.io.loaders import is_supported_path, load_dataset as load_dataset_headless
from grim_backend.io.batch import _stage_and_publish_csv_batch


def _is_supported_dataset_path(path: str) -> bool:
    return is_supported_path(path)

def _available_memory_bytes() -> int | None:
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        # Optional telemetry must not prevent loading a dataset. Use the OS
        # probe or a conservative unknown-memory budget if psutil fails.
        pass

    # GRIM is commonly copied to a clean workstation where psutil is not yet
    # installed.  Retain a real memory budget on the two primary deployment
    # families instead of falling back immediately to CPU-count concurrency.
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = (
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                )

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if pages >= 0 and page_size > 0:
            return pages * page_size
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return None

_LOADER_MIN_WORKING_BYTES = 64 * 1024**2

_LOADER_UNKNOWN_MEMORY_BUDGET_BYTES = 512 * 1024**2

_GRIM_LOAD_PEAK_FACTOR = 3.5

_TEXT_LOADER_EXTENSIONS = (".csv", ".txt", ".cst_data", ".out", ".ss")

def _grim_archive_uncompressed_bytes(path: str) -> int | None:
    """Return declared uncompressed NPZ bytes without extracting the archive."""

    if not str(path).casefold().endswith(".grim"):
        return None
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if not members:
                return 0
            return sum(max(0, int(member.file_size)) for member in members)
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile):
        # The authoritative loader will report a malformed archive.  Keep a
        # conservative file-size fallback here so planning itself remains
        # non-destructive and does not mask that parse error.
        return None

def _dataset_load_memory_estimate(path: str) -> tuple[int, int]:
    """Return conservative ``(retained, per-load peak)`` byte estimates."""

    try:
        stored_bytes = max(0, int(os.path.getsize(path)))
    except OSError:
        stored_bytes = 0
    expanded_bytes = _grim_archive_uncompressed_bytes(path)
    if expanded_bytes is not None:
        retained = max(stored_bytes, int(expanded_bytes))
        # RcsGrid validates and cleans power/phase into new arrays while the
        # NPZ members are still live.  Include those transient copies instead
        # of treating a highly compressed archive as its on-disk byte count.
        peak = max(
            _LOADER_MIN_WORKING_BYTES,
            int(math.ceil(_GRIM_LOAD_PEAK_FACTOR * retained)),
        )
        return retained, peak

    lower = str(path).casefold()
    # Delimited readers create Python strings/dicts, coordinate-key maps, dense
    # output arrays, and duplicate-validation state concurrently. Real SENTRi
    # imports have measured around 77x their file size at peak, so the old 4x
    # rule was unsafe by more than an order of magnitude.
    if lower.endswith((".csv", ".txt")):
        retained = max(stored_bytes, 16 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 96 * stored_bytes)
        return retained, peak
    if lower.endswith((".cst_data", ".out", ".ss")):
        retained = max(stored_bytes, 8 * stored_bytes)
        peak = max(_LOADER_MIN_WORKING_BYTES, 40 * stored_bytes)
        return retained, peak

    # Binary single-cut formats still materialize complex, power, and phase
    # arrays but avoid the large per-cell Python parser overhead.
    retained = max(stored_bytes, 2 * stored_bytes)
    peak = max(_LOADER_MIN_WORKING_BYTES, 6 * stored_bytes)
    return retained, peak

def _recommended_loader_workers(tasks) -> int:
    if isinstance(tasks, int):
        task_count = int(tasks)
        paths = []
    else:
        paths = [str(task[1]) for task in tasks]
        task_count = len(paths)
    cpu_total = os.cpu_count() or 1
    if cpu_total <= 2:
        target = cpu_total
    else:
        target = cpu_total - 1
    target = max(1, min(int(task_count), int(target)))
    if not paths:
        return target
    estimates = [_dataset_load_memory_estimate(path) for path in paths]
    retained_total = sum(retained for retained, _peak in estimates)
    transient_extras = sorted(
        (max(0, peak - retained) for retained, peak in estimates),
        reverse=True,
    )
    available = _available_memory_bytes()
    budget = (
        int(available * 0.5)
        if available is not None
        else _LOADER_UNKNOWN_MEMORY_BUDGET_BYTES
    )

    # Every successfully loaded grid remains in the result batch.  Reserve
    # that final retained footprint, then admit only as many simultaneous
    # parse/clean workspaces as fit in the remaining budget.  This prevents a
    # tiny, highly compressed .grim file from spawning CPU-count workers that
    # each expand into a large in-memory grid.
    safe_workers = 0
    for worker_count in range(1, target + 1):
        planned_peak = retained_total + sum(transient_extras[:worker_count])
        if planned_peak <= budget:
            safe_workers = worker_count
        else:
            break
    if safe_workers < 1:
        budget_mib = budget / 1024**2
        required_mib = (
            retained_total + (transient_extras[0] if transient_extras else 0)
        ) / 1024**2
        source = (
            "available memory"
            if available is not None
            else "the conservative fallback budget"
        )
        raise MemoryError(
            f"This dataset batch needs an estimated {required_mib:.0f} MiB "
            f"but only {budget_mib:.0f} MiB of {source} is reserved for "
            "loading. Load fewer or smaller dataset files at a time."
        )
    # Python's delimited parsers are memory-heavy and mostly GIL-bound. Running
    # several simultaneously increases peak memory without a reliable speedup.
    if any(path.casefold().endswith(_TEXT_LOADER_EXTENSIONS) for path in paths):
        return 1
    return safe_workers

def _load_dataset_path_task(task: tuple[int, str]) -> dict[str, object]:
    index, path = task
    file_name = os.path.basename(path)
    dataset_name = os.path.splitext(file_name)[0]
    lower = path.lower()
    try:
        if not _is_supported_dataset_path(path):
            return {
                "status": "ignored",
                "index": index,
                "path": path,
                "file_name": file_name,
                "error": "Unsupported file extension",
            }
        dataset = load_dataset_headless(path)
        history = str(getattr(dataset, "history", "") or path)
    except Exception as exc:
        return {
            "status": "error",
            "index": index,
            "path": path,
            "file_name": file_name,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "index": index,
        "path": path,
        "file_name": file_name,
        "name": dataset_name,
        "history": history,
        "dataset": dataset,
    }

def _join_many_with_progress(
    grids: list[RcsGrid],
    *,
    tol: float = 1e-6,
    progress_cb=None,
) -> RcsGrid:
    checked = RcsGrid._ensure_grids(grids)
    total = len(checked)
    available = _available_memory_bytes()
    limit = int(available * 0.5) if available is not None else None
    result = RcsGrid.join_many(
        *checked,
        tol=tol,
        overlap="error",
        max_output_bytes=limit,
    )
    if progress_cb is not None:
        progress_cb(total, total)
    return result

class _DatasetLoadWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, tasks: list[tuple[int, str]], ignored_count: int = 0, parent=None) -> None:
        super().__init__(parent)
        self._tasks = list(tasks)
        self._ignored_count = int(ignored_count)

    def run(self) -> None:
        total = len(self._tasks)
        loaded: list[dict[str, object]] = []
        failed: list[str] = []
        used_parallel = False

        def _consume(result: dict[str, object], done_count: int) -> None:
            status = str(result.get("status", "error"))
            file_name = str(result.get("file_name", "dataset"))
            if status == "ok":
                loaded.append(result)
                self.progress.emit(done_count, total, f"Loaded {file_name}")
                return
            error_text = str(result.get("error", "Unknown error"))
            failed.append(f"{file_name} ({error_text})")
            self.progress.emit(done_count, total, f"Failed {file_name}")

        try:
            worker_count = _recommended_loader_workers(self._tasks) if total else 1
            if total == 1:
                _consume(_load_dataset_path_task(self._tasks[0]), 1)
            elif total > 1:
                with ThreadPoolExecutor(max_workers=worker_count) as pool:
                    futures = {
                        pool.submit(_load_dataset_path_task, task): task
                        for task in self._tasks
                    }
                    done_count = 0
                    for future in as_completed(futures):
                        result = future.result()
                        done_count += 1
                        _consume(result, done_count)
                used_parallel = worker_count > 1
        except Exception as exc:
            # Individual parse failures normally arrive as result mappings.
            # This catches pool setup/submission/future faults so the owning
            # QThread still receives exactly one terminal signal and can quit.
            failed.append(f"Dataset loader ({type(exc).__name__}: {exc})")
        finally:
            self.finished.emit(
                {
                    "loaded": loaded,
                    "failed": failed,
                    "ignored": self._ignored_count,
                    "used_parallel": used_parallel,
                    "total_supported": total,
                }
            )

class _CsvExportWorker(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        entries: list[tuple[RcsGrid, str]],
        *,
        scale: str,
        include_phase: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._entries = list(entries)
        self._scale = str(scale)
        self._include_phase = bool(include_phase)

    def run(self) -> None:
        total = len(self._entries)
        try:
            self.progress.emit(0, total, "Writing staged CSV files")
            paths = _stage_and_publish_csv_batch(
                self._entries,
                scale=self._scale,
                include_phase=self._include_phase,
            )
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.progress.emit(total, total, "Published CSV files")
        self.finished.emit({"ok": True, "paths": paths, "total": total})

class _BackgroundJobCancelled(RuntimeError):
    """Internal cooperative-cancellation sentinel."""

class _BackgroundCallableWorker(QObject):
    """Run one pure-Python/NumPy callable away from Qt's GUI thread."""

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, function, *, reports_progress: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._function = function
        self._reports_progress = bool(reports_progress)
        self.supports_cancellation = self._reports_progress

    def run(self) -> None:
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
            if self._reports_progress:
                def report_progress(done_count, total_count, detail=""):
                    if QThread.currentThread().isInterruptionRequested():
                        raise _BackgroundJobCancelled()
                    self.progress.emit(done_count, total_count, detail)

                result = self._function(report_progress)
            else:
                result = self._function()
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True})
            return
        except Exception as exc:
            self.finished.emit(
                {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
            )
            return
        self.finished.emit({"ok": True, "result": result})

class _JoinDatasetsWorker(QObject):
    supports_cancellation = False
    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(self, grids: list[RcsGrid], tol: float = 1e-6, parent=None) -> None:
        super().__init__(parent)
        self._grids = list(grids)
        self._tol = float(tol)

    def run(self) -> None:
        total = max(1, len(self._grids))
        try:
            if QThread.currentThread().isInterruptionRequested():
                raise _BackgroundJobCancelled()

            def _emit_progress(done_count: int, total_count: int) -> None:
                if QThread.currentThread().isInterruptionRequested():
                    raise _BackgroundJobCancelled()
                self.progress.emit(done_count, total_count, "Joining datasets")

            merged = _join_many_with_progress(self._grids, tol=self._tol, progress_cb=_emit_progress)
        except _BackgroundJobCancelled:
            self.finished.emit({"ok": False, "cancelled": True, "total": total})
            return
        except Exception as exc:
            self.finished.emit({"ok": False, "error": str(exc), "total": total})
            return
        self.finished.emit({"ok": True, "merged": merged, "total": total})

class _RangeCalibrationWorker(QObject):
    """Apply one calibration definition to DUT grids off the GUI thread."""

    supports_cancellation = True

    progress = Signal(int, int, str)
    finished = Signal(object)

    def __init__(
        self,
        targets: list[tuple[str, RcsGrid]],
        measured_entry: tuple[str, RcsGrid],
        exact_entry: tuple[str, RcsGrid],
        params: dict[str, object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._targets = list(targets)
        self._measured_name, self._measured = measured_entry
        self._exact_name, self._exact = exact_entry
        self._params = dict(params)

    def run(self) -> None:
        total = len(self._targets)
        results: list[dict[str, object]] = []
        failed: list[str] = []
        try:
            offset_m = float(self._params["range_offset_m"])
        except (KeyError, TypeError, ValueError) as exc:
            self.finished.emit(
                {
                    "results": results,
                    "failed": [f"invalid range-calibration parameters ({exc})"],
                    "total": total,
                }
            )
            return
        exact_display = str(self._exact_name)
        if len(exact_display) > 48:
            exact_display = exact_display[:45] + "..."

        for index, (target_name, target) in enumerate(self._targets, start=1):
            if QThread.currentThread().isInterruptionRequested():
                failed.append("cancelled by user")
                break
            try:
                calibrated = target.range_calibrate(
                    self._measured,
                    self._exact,
                    offset_m,
                    allow_singleton_angular_broadcast=bool(
                        self._params.get(
                            "allow_singleton_angular_broadcast", False
                        )
                    ),
                    convention_attested=False,
                    measured_label=self._measured_name,
                    exact_label=self._exact_name,
                    maximum_correction_gain_db=self._params.get(
                        "maximum_correction_gain_db", 60.0
                    ),
                )
            except Exception as exc:
                failed.append(f"{target_name} ({exc})")
                self.progress.emit(index, total, f"Skipped {target_name}")
                continue

            results.append(
                {
                    "dataset": calibrated,
                    "source_dataset": target,
                    "name": (
                        f"{target_name} [Range Cal: {exact_display}; "
                        f"ΔR {offset_m:+.6g} m]"
                    ),
                    "history": (
                        f"Range Cal: {target_name}; measured={self._measured_name}; "
                        f"exact={self._exact_name}; ΔR={offset_m:+.12g} m "
                        "(positive away)"
                    ),
                }
            )
            self.progress.emit(index, total, f"Calibrated {target_name}")

        self.finished.emit(
            {
                "results": results,
                "failed": failed,
                "total": total,
            }
        )
