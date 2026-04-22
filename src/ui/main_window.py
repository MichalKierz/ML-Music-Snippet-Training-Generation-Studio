from pathlib import Path
import gc

import torch
from PySide6.QtCore import QThread
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget, QVBoxLayout, QWidget

from src.core.logger import get_logger
from src.core.runtime_context import RuntimeContext
from src.generation.token_inference_service import TokenInferenceService
from src.preprocessing.token_preprocess_service import TokenPreprocessService
from src.training.token_training_service import TokenTrainingService
from src.ui.error_dialog import ErrorDialog, summarize_error_text
from src.ui.generate_tab import GenerateTab
from src.ui.status_panel import StatusPanel
from src.ui.training_tab import TrainingTab
from src.ui.workers import TaskWorker


def release_process_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self, context: RuntimeContext):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self.logger = get_logger(__name__)

        app_config = context.configs["app"]
        window_settings = self.settings.get("window")

        self.preprocess_service = TokenPreprocessService(
            context.configs["preprocess"],
            context.configs["token_model"],
        )
        self.training_service = TokenTrainingService(
            context.configs["token_model"],
            context.configs["training"],
        )
        self.inference_service = TokenInferenceService(context.configs["generate"])

        self.active_thread = None
        self.active_worker = None
        self.active_task_name = None

        self.setWindowTitle(app_config["app_name"])
        self.resize(window_settings["width"], window_settings["height"])

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.training_tab = TrainingTab(context)
        self.generate_tab = GenerateTab(context)
        self.status_panel = StatusPanel(app_config["status_defaults"])

        self.tabs.addTab(self.training_tab, "Training")
        self.tabs.addTab(self.generate_tab, "Generate")
        self.tabs.setCurrentIndex(0)

        layout.addWidget(self.tabs)
        layout.addWidget(self.status_panel)

        self.training_tab.preprocess_requested.connect(self.handle_preprocess_requested)
        self.training_tab.train_requested.connect(self.handle_train_requested)
        self.generate_tab.generate_requested.connect(self.handle_generate_requested)
        self.status_panel.cancel_requested.connect(self.handle_cancel_requested)

        self.logger.info("Main window initialized")

    def _set_busy(self, busy: bool):
        self.training_tab.set_busy(busy)
        self.generate_tab.set_busy(busy)

    def _task_supports_cancel(self, task_name: str | None) -> bool:
        return task_name in {"Preprocessing", "Training", "Generating"}

    def _show_error(self, message: str):
        summary = summarize_error_text(message)
        self.logger.error(message)
        self.status_panel.fail(summary, message)
        dialog = ErrorDialog("Error", message, self)
        dialog.exec()

    def _start_worker(self, task_name: str, target, kwargs: dict):
        if self.active_thread is not None:
            self._show_error("Another task is already running.")
            return

        self.logger.info(f"Starting task: {task_name}")
        self.active_task_name = task_name
        self._set_busy(True)
        self.status_panel.start(task_name, f"{task_name} started")
        self.status_panel.set_cancel_state(self._task_supports_cancel(task_name), self._task_supports_cancel(task_name))

        thread = QThread(self)
        worker = TaskWorker(target=target, kwargs=kwargs)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progress.connect(self._handle_worker_progress)
        worker.succeeded.connect(self._handle_worker_success)
        worker.failed.connect(self._handle_worker_failure)
        worker.cancelled.connect(self._handle_worker_cancelled)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._handle_thread_finished)

        self.active_thread = thread
        self.active_worker = worker

        thread.start()

    def _handle_worker_progress(self, stage: str, current: float, total: float, status: str):
        self.status_panel.update_progress(stage, current, total, status)

    def _handle_worker_success(self, result):
        task_name = self.active_task_name or "Task"
        self.logger.info(f"Task completed successfully: {task_name}")

        if task_name == "Preprocessing":
            audio_file_count = int(result.get("audio_file_count", 0))
            chunk_count = int(result.get("chunk_count", 0))
            self.status_panel.complete(
                f"Preprocessing completed | files={audio_file_count} | chunks={chunk_count}"
            )
            return

        if task_name == "Training":
            final_model_path = str(result.get("final_model_path", ""))
            if final_model_path:
                self.generate_tab.set_model_file(final_model_path)
            self.status_panel.complete(
                f"Training completed | model={Path(final_model_path).name if final_model_path else 'model.pt'}"
            )
            return

        if task_name == "Generating":
            output_path = str(result.get("output_path", ""))
            song_mode = bool(result.get("song_mode", False))
            snippet_count = int(result.get("generated_snippet_count", 1))

            if song_mode:
                self.status_panel.complete(
                    f"Song generation completed | snippets={snippet_count} | saved={Path(output_path).name if output_path else 'song.mp3'}"
                )
            else:
                self.status_panel.complete(
                    f"Generation completed | saved={Path(output_path).name if output_path else 'snippet.mp3'}"
                )
            return

        self.status_panel.complete(f"{task_name} completed")

    def _handle_worker_failure(self, message: str):
        self.logger.error(f"Task failed: {self.active_task_name or 'Unknown'} | {message}")
        self._show_error(message)

    def _handle_worker_cancelled(self, message: str):
        task_name = self.active_task_name or "Task"
        self.logger.info(f"Task cancelled: {task_name} | {message}")
        self.status_panel.complete(f"{task_name} cancelled")

    def _handle_thread_finished(self):
        self.logger.info(f"Task finished: {self.active_task_name or 'Unknown'}")
        self.active_thread = None
        self.active_worker = None
        self.active_task_name = None
        self._set_busy(False)
        self.status_panel.set_cancel_state(False, False)
        release_process_memory()

    def handle_cancel_requested(self):
        if self.active_worker is None:
            return
        if not self._task_supports_cancel(self.active_task_name):
            return
        self.logger.info(f"Cancellation requested for task: {self.active_task_name}")
        self.status_panel.mark_cancelling()
        self.active_worker.cancel()
        self.status_panel.set_cancel_state(True, False)

    def handle_preprocess_requested(self, payload: dict):
        raw_music_folder = payload["raw_music_folder"]
        processed_training_data_folder = payload["processed_training_data_folder"]

        if not raw_music_folder:
            self._show_error("Raw Music Folder is required.")
            return

        if not processed_training_data_folder:
            self._show_error("Processed Training Data Folder is required.")
            return

        if not Path(raw_music_folder).exists():
            self._show_error("Raw Music Folder does not exist.")
            return

        Path(processed_training_data_folder).mkdir(parents=True, exist_ok=True)

        self._start_worker(
            task_name="Preprocessing",
            target=self.preprocess_service.run,
            kwargs=payload,
        )

    def handle_train_requested(self, payload: dict):
        processed_training_data_folder = Path(payload["processed_training_data_folder"])
        model_output_folder = payload["model_output_folder"]

        token_manifest_path = processed_training_data_folder / "manifests" / "token_chunk_manifest.csv"
        codec_info_path = processed_training_data_folder / "codec_info.json"
        metadata_vocab_path = processed_training_data_folder / "vocab" / "metadata_vocab.json"

        if not processed_training_data_folder.exists():
            self._show_error("Processed Training Data Folder does not exist.")
            return

        if not token_manifest_path.exists():
            self._show_error("Token manifest was not found. Run Preprocess Data first.")
            return

        if not codec_info_path.exists():
            self._show_error("Codec info file was not found. Run Preprocess Data first.")
            return

        if not model_output_folder:
            self._show_error("Model Output Folder is required.")
            return

        Path(model_output_folder).mkdir(parents=True, exist_ok=True)

        train_payload = dict(payload)
        train_payload["token_manifest_path"] = str(token_manifest_path)
        train_payload["codec_info_path"] = str(codec_info_path)
        train_payload["metadata_vocab_path"] = str(metadata_vocab_path)

        self._start_worker(
            task_name="Training",
            target=self.training_service.run,
            kwargs=train_payload,
        )

    def handle_generate_requested(self, payload: dict):
        model_file = payload["model_file"]
        output_folder = payload["output_folder"]
        generate_song = bool(payload.get("generate_song", False))
        song_snippet_count = int(payload.get("song_snippet_count", 1))

        if not model_file:
            self._show_error("Model File is required.")
            return

        if not output_folder:
            self._show_error("Output Folder is required.")
            return

        if not Path(model_file).exists():
            self._show_error("Selected model file does not exist.")
            return

        if generate_song and song_snippet_count < 1:
            self._show_error("Snippet Count must be at least 1.")
            return

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        self._start_worker(
            task_name="Generating",
            target=self.inference_service.run,
            kwargs=payload,
        )

    def closeEvent(self, event: QCloseEvent):
        self.settings.update_section(
            "window",
            {
                "width": int(self.width()),
                "height": int(self.height()),
                "last_tab_index": 0,
            },
        )
        release_process_memory()
        self.logger.info("Main window closed")
        super().closeEvent(event)