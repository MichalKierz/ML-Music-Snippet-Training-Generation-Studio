from threading import Event

from PySide6.QtCore import QObject, Signal, Slot

from src.core.task_cancelled import TaskCancelledError


class TaskWorker(QObject):
    progress = Signal(str, float, float, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal(str)
    finished = Signal()

    def __init__(self, target, kwargs: dict | None = None):
        super().__init__()
        self.target = target
        self.kwargs = kwargs or {}
        self._cancel_event = Event()

    @Slot()
    def run(self):
        try:
            result = self.target(
                **self.kwargs,
                progress_callback=self._emit_progress,
                is_cancelled=self.is_cancelled,
            )
            self.succeeded.emit(result)
        except TaskCancelledError as exc:
            self.cancelled.emit(str(exc) or "Task cancelled.")
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    @Slot()
    def cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def _emit_progress(self, stage: str, current, total, status: str):
        self.progress.emit(str(stage), float(current), float(total), str(status))