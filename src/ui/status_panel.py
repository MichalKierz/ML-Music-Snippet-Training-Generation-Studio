import math
import re
import time
from collections import deque

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import QGridLayout, QLabel, QProgressBar, QPushButton, QWidget

from src.ui.error_dialog import ErrorDialog


def format_seconds(value: float, round_up: bool = False) -> str:
    if round_up:
        total = max(0, int(math.ceil(value)))
    else:
        total = max(0, int(value))
    hours = total // 3600
    minutes = (total % 3600) // 60
    seconds = total % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StatusPanel(QWidget):
    cancel_requested = Signal()

    def __init__(self, defaults: dict):
        super().__init__()
        self.start_timestamp = None
        self.stage_start_timestamp = None
        self.error_details = ""
        self.progress_fraction = None
        self.progress_samples = deque(maxlen=300)
        self.current_stage_name = str(defaults["stage"])
        self.generation_token_current = None
        self.generation_token_total = None
        self.generation_token_samples = deque(maxlen=600)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(6)

        self.stage_title = QLabel("Stage")
        self.stage_value = QLabel(defaults["stage"])
        self.stage_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.stage_value.setToolTip(defaults["stage"])

        self.progress_title = QLabel("Progress")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")

        self.cancel_button = QPushButton("Cancel Process")
        self.cancel_button.clicked.connect(self._emit_cancel_requested)
        self.cancel_button.setVisible(False)
        self.cancel_button.setEnabled(False)

        self.elapsed_title = QLabel("Elapsed Time")
        self.elapsed_value = QLabel(defaults["elapsed"])

        self.eta_title = QLabel("Time Left")
        self.eta_value = QLabel(defaults["eta"])

        self.status_title = QLabel("Status")
        self.status_value = QLabel(defaults["status"])
        self.status_value.setWordWrap(False)
        self.status_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.status_value.setToolTip(defaults["status"])

        self.details_button = QPushButton("Details")
        self.details_button.clicked.connect(self._show_details)
        self.details_button.setVisible(False)

        layout.addWidget(self.stage_title, 0, 0)
        layout.addWidget(self.stage_value, 0, 1)
        layout.addWidget(self.progress_title, 0, 2)
        layout.addWidget(self.progress_bar, 0, 3, 1, 3)
        layout.addWidget(self.cancel_button, 0, 6)

        layout.addWidget(self.elapsed_title, 1, 0)
        layout.addWidget(self.elapsed_value, 1, 1)
        layout.addWidget(self.eta_title, 1, 2)
        layout.addWidget(self.eta_value, 1, 3)
        layout.addWidget(self.status_title, 1, 4)
        layout.addWidget(self.status_value, 1, 5)
        layout.addWidget(self.details_button, 1, 6)

        layout.setColumnStretch(3, 1)
        layout.setColumnStretch(5, 3)

    def _emit_cancel_requested(self):
        self.cancel_button.setEnabled(False)
        self.cancel_requested.emit()

    def set_cancel_state(self, visible: bool, enabled: bool):
        self.cancel_button.setVisible(bool(visible))
        self.cancel_button.setEnabled(bool(enabled))

    def mark_cancelling(self):
        self._set_stage_text("Cancelling")
        self._set_status_text("Waiting for the current file or batch to finish")

    def _set_stage_text(self, text: str):
        value = str(text).strip() or "Unknown"
        self.current_stage_name = value
        self.stage_value.setText(value)
        self.stage_value.setToolTip(value)

    def _set_status_text(self, text: str):
        value = str(text).strip() or "Ready"
        self.status_value.setText(value)
        self.status_value.setToolTip(value)

    def _reset_stage_speed_tracking(self):
        self.stage_start_timestamp = time.time()
        self.progress_samples.clear()
        self.generation_token_current = None
        self.generation_token_total = None
        self.generation_token_samples.clear()

    def set_idle(self):
        self.start_timestamp = None
        self.stage_start_timestamp = None
        self.error_details = ""
        self.progress_fraction = None
        self.progress_samples.clear()
        self.generation_token_current = None
        self.generation_token_total = None
        self.generation_token_samples.clear()
        self.timer.stop()
        self._set_stage_text("Idle")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.elapsed_value.setText("00:00:00")
        self.eta_value.setText("--")
        self._set_status_text("Ready")
        self.details_button.setVisible(False)
        self.set_cancel_state(False, False)

    def start(self, stage: str, status: str):
        self._set_stage_text(stage)
        self._set_status_text(status)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.elapsed_value.setText("00:00:00")
        self.eta_value.setText("--")
        self.error_details = ""
        self.progress_fraction = None
        self.progress_samples.clear()
        self.generation_token_current = None
        self.generation_token_total = None
        self.generation_token_samples.clear()
        self.details_button.setVisible(False)
        self.start_timestamp = time.time()
        self.stage_start_timestamp = self.start_timestamp
        self.timer.start(1000)
        self._tick()

    def update_progress(self, stage: str, current, total, status: str):
        if self.start_timestamp is None:
            self.start(stage, status)

        stage_changed = str(stage).strip() != self.current_stage_name
        if stage_changed:
            self._set_stage_text(stage)
            self._reset_stage_speed_tracking()
        else:
            self._set_stage_text(stage)

        self._set_status_text(status)

        total = float(total)

        if total <= 0:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("")
            self.progress_fraction = None
            self._update_generation_token_tracking(stage, status)
            self.eta_value.setText("--")
            return

        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")

        current = max(0.0, float(current))
        total = max(1.0, total)
        progress = max(0.0, min(1.0, current / total))

        previous = self.progress_fraction
        if previous is not None and progress + 1e-9 < previous:
            self._reset_stage_speed_tracking()

        self.progress_fraction = progress
        self.progress_bar.setValue(int(progress * 100))

        now = time.time()
        if previous is None or progress > previous:
            self.progress_samples.append((now, progress))

        self._update_generation_token_tracking(stage, status)
        self._update_eta()

    def complete(self, status: str):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.progress_bar.setFormat("%p%")
        self._set_stage_text("Completed")
        self._set_status_text(status)
        self.progress_fraction = 1.0
        self.eta_value.setText("00:00:00")
        self.error_details = ""
        self.details_button.setVisible(False)
        self.set_cancel_state(False, False)
        self.timer.stop()
        self._tick()

    def fail(self, summary: str, details: str | None = None):
        self._set_stage_text("Error")
        self._set_status_text(summary)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setFormat("%p%")
        self.progress_fraction = None
        self.progress_samples.clear()
        self.generation_token_current = None
        self.generation_token_total = None
        self.generation_token_samples.clear()
        self.eta_value.setText("--")
        self.set_cancel_state(False, False)
        self.timer.stop()

        if details and str(details).strip():
            self.error_details = str(details)
            self.details_button.setVisible(True)
        else:
            self.error_details = ""
            self.details_button.setVisible(False)

        self._tick()

    def _parse_generation_token_status(self, status: str):
        match = re.search(r"Generated token\s+(\d+)\s*/\s*(\d+)", str(status), re.IGNORECASE)
        if not match:
            return None

        current = int(match.group(1))
        total = int(match.group(2))

        if total <= 0:
            return None

        current = max(0, min(current, total))
        return current, total

    def _update_generation_token_tracking(self, stage: str, status: str):
        if "generating tokens" not in str(stage).strip().lower():
            self.generation_token_current = None
            self.generation_token_total = None
            self.generation_token_samples.clear()
            return

        parsed = self._parse_generation_token_status(status)
        if parsed is None:
            return

        current, total = parsed
        now = time.time()

        previous_current = self.generation_token_current
        previous_total = self.generation_token_total

        if previous_total is not None and total != previous_total:
            self.generation_token_samples.clear()

        self.generation_token_current = current
        self.generation_token_total = total

        if previous_current is None:
            self.generation_token_samples.append((now, current))
            return

        if current > previous_current:
            self.generation_token_samples.append((now, current))

    def _estimate_progress_speed(self):
        if len(self.progress_samples) < 4:
            return None

        now = time.time()
        recent = [sample for sample in self.progress_samples if now - sample[0] <= 20.0]

        if len(recent) < 4:
            recent = list(self.progress_samples)

        if len(recent) < 4:
            return None

        start_time, start_progress = recent[0]
        end_time, end_progress = recent[-1]

        delta_time = end_time - start_time
        delta_progress = end_progress - start_progress

        if delta_time < 3.0:
            return None

        if delta_progress <= 0:
            return None

        return delta_progress / delta_time

    def _estimate_generation_token_rate(self):
        if self.generation_token_current is None or self.generation_token_total is None:
            return None

        if len(self.generation_token_samples) < 3:
            return None

        now = time.time()
        recent = [sample for sample in self.generation_token_samples if now - sample[0] <= 15.0]

        if len(recent) < 3:
            recent = list(self.generation_token_samples)

        if len(recent) < 3:
            return None

        start_time, start_token = recent[0]
        end_time, end_token = recent[-1]

        delta_time = end_time - start_time
        delta_tokens = end_token - start_token

        if delta_time <= 0:
            return None

        if delta_tokens <= 0:
            return None

        return delta_tokens / delta_time

    def _is_training_stage(self) -> bool:
        stage = self.current_stage_name.lower()
        return "training" in stage

    def _is_generation_stage(self) -> bool:
        stage = self.current_stage_name.lower()
        return "generating" in stage

    def _update_eta(self):
        if self.start_timestamp is None:
            self.eta_value.setText("--")
            return

        if self._is_generation_stage() and self.generation_token_current is not None and self.generation_token_total is not None:
            current = int(self.generation_token_current)
            total = int(self.generation_token_total)

            if current <= 0 or total <= 0 or current >= total:
                self.eta_value.setText("00:00:00" if current >= total and total > 0 else "--")
                return

            stage_elapsed = 0.0 if self.stage_start_timestamp is None else time.time() - self.stage_start_timestamp
            rate = self._estimate_generation_token_rate()

            if rate is None:
                if current < 3 or stage_elapsed <= 0:
                    self.eta_value.setText("--")
                    return
                rate = current / stage_elapsed

            if rate <= 0:
                self.eta_value.setText("--")
                return

            remaining_tokens = total - current
            remaining_seconds = remaining_tokens / rate
            self.eta_value.setText(format_seconds(remaining_seconds, round_up=True))
            return

        if self.progress_fraction is None:
            self.eta_value.setText("--")
            return

        if self.progress_fraction <= 0.0:
            self.eta_value.setText("--")
            return

        if self.progress_fraction >= 1.0:
            self.eta_value.setText("00:00:00")
            return

        elapsed = time.time() - self.start_timestamp
        stage_elapsed = 0.0 if self.stage_start_timestamp is None else time.time() - self.stage_start_timestamp
        speed = self._estimate_progress_speed()

        if self._is_training_stage():
            if len(self.progress_samples) < 4 and stage_elapsed < 8.0:
                self.eta_value.setText("--")
                return
        else:
            if len(self.progress_samples) < 3 and stage_elapsed < 3.0:
                self.eta_value.setText("--")
                return

        if speed is None:
            if elapsed < 8.0:
                self.eta_value.setText("--")
                return
            speed = self.progress_fraction / elapsed

        if speed <= 0:
            self.eta_value.setText("--")
            return

        remaining = (1.0 - self.progress_fraction) / speed

        if remaining < 1.0:
            self.eta_value.setText("00:00:01")
            return

        self.eta_value.setText(format_seconds(remaining, round_up=True))

    def _show_details(self):
        if not self.error_details:
            return
        dialog = ErrorDialog("Error Details", self.error_details, self)
        dialog.exec()

    def _tick(self):
        if self.start_timestamp is None:
            return
        elapsed = time.time() - self.start_timestamp
        self.elapsed_value.setText(format_seconds(elapsed))
        self._update_eta()