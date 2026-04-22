from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from src.core.runtime_context import RuntimeContext
from src.ui.advanced_model_settings_dialog import AdvancedModelSettingsDialog
from src.ui.training_tab_sections import TrainingTabSections


class TrainingTab(QWidget):
    preprocess_requested = Signal(dict)
    train_requested = Signal(dict)

    def __init__(self, context: RuntimeContext):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self.configs = context.configs
        self.advanced_dialog = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(0)

        self.ui = TrainingTabSections(context)
        scroll_layout.addWidget(self.ui)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self._connect_signals()

    def _connect_signals(self):
        self.ui.raw_music_button.clicked.connect(
            lambda: self.ui.select_folder(self.ui.raw_music_edit, "paths", "raw_music_folder")
        )
        self.ui.processed_training_data_button.clicked.connect(
            lambda: self.ui.select_folder(
                self.ui.processed_training_data_edit,
                "paths",
                "processed_training_data_folder",
            )
        )
        self.ui.model_output_button.clicked.connect(
            lambda: self.ui.select_folder(self.ui.model_output_edit, "paths", "model_output_folder")
        )

        self.ui.raw_music_edit.textChanged.connect(
            lambda value: self.settings.set("paths", "raw_music_folder", value)
        )
        self.ui.processed_training_data_edit.textChanged.connect(
            lambda value: self.settings.set("paths", "processed_training_data_folder", value)
        )
        self.ui.model_output_edit.textChanged.connect(
            lambda value: self.settings.set("paths", "model_output_folder", value)
        )

        self.ui.chunk_length_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "chunk_length_sec", float(value))
        )
        self.ui.chunk_stride_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "chunk_stride_sec", float(value))
        )
        self.ui.preprocess_gpu_combo.currentIndexChanged.connect(
            self.ui.persist_preprocess_gpu_selection
        )

        self.ui.batch_size_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "batch_size", int(value))
        )
        self.ui.epochs_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "epochs", int(value))
        )
        self.ui.learning_rate_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "learning_rate", float(value))
        )
        self.ui.validation_split_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "validation_split", float(value))
        )
        self.ui.weight_decay_spin.valueChanged.connect(
            lambda value: self.settings.set("training", "weight_decay", float(value))
        )
        self.ui.mixed_precision_check.toggled.connect(
            lambda value: self.settings.set("training", "mixed_precision", bool(value))
        )
        self.ui.pin_memory_check.toggled.connect(
            lambda value: self.settings.set("training", "pin_memory", bool(value))
        )
        self.ui.use_genre_metadata_check.toggled.connect(
            lambda value: self.settings.set("training", "use_genre_metadata", bool(value))
        )
        self.ui.use_year_metadata_check.toggled.connect(
            lambda value: self.settings.set("training", "use_year_metadata", bool(value))
        )
        self.ui.use_mood_metadata_check.toggled.connect(
            lambda value: self.settings.set("training", "use_mood_metadata", bool(value))
        )
        self.ui.use_key_metadata_check.toggled.connect(
            lambda value: self.settings.set("training", "use_key_metadata", bool(value))
        )
        self.ui.use_bpm_metadata_check.toggled.connect(
            lambda value: self.settings.set("training", "use_bpm_metadata", bool(value))
        )

        self.ui.training_gpu_toggle.toggled.connect(self.ui.handle_training_gpu_toggle)
        self.ui.preprocess_button.clicked.connect(self._emit_preprocess_requested)
        self.ui.train_button.clicked.connect(self._emit_train_requested)
        self.ui.reset_defaults_button.clicked.connect(self.reset_to_default_settings)
        self.ui.advanced_model_settings_button.clicked.connect(self.open_advanced_model_settings)

    def _default_values(self) -> dict:
        preprocess_defaults = self.configs["preprocess"]
        training_defaults = self.configs["training"]

        return {
            "chunk_length_sec": float(preprocess_defaults["chunk_length_sec"]),
            "chunk_stride_sec": float(preprocess_defaults["chunk_stride_sec"]),
            "batch_size": int(training_defaults["batch_size"]),
            "epochs": int(training_defaults["epochs"]),
            "learning_rate": float(training_defaults["learning_rate"]),
            "validation_split": float(training_defaults["validation_split"]),
            "weight_decay": float(training_defaults["weight_decay"]),
            "mixed_precision": bool(training_defaults["mixed_precision"]),
            "pin_memory": bool(training_defaults.get("pin_memory", False)),
            "use_genre_metadata": bool(training_defaults.get("use_genre_metadata", True)),
            "use_year_metadata": bool(training_defaults.get("use_year_metadata", True)),
            "use_mood_metadata": bool(training_defaults.get("use_mood_metadata", False)),
            "use_key_metadata": bool(training_defaults.get("use_key_metadata", False)),
            "use_bpm_metadata": bool(training_defaults.get("use_bpm_metadata", False)),
        }

    def _apply_training_values(self, values: dict):
        self.ui.set_spin_value(self.ui.chunk_length_spin, float(values["chunk_length_sec"]))
        self.ui.set_spin_value(self.ui.chunk_stride_spin, float(values["chunk_stride_sec"]))
        self.ui.set_spin_value(self.ui.batch_size_spin, int(values["batch_size"]))
        self.ui.set_spin_value(self.ui.epochs_spin, int(values["epochs"]))
        self.ui.set_spin_value(self.ui.learning_rate_spin, float(values["learning_rate"]))
        self.ui.set_spin_value(self.ui.validation_split_spin, float(values["validation_split"]))
        self.ui.set_spin_value(self.ui.weight_decay_spin, float(values["weight_decay"]))
        self.ui.set_check_value(self.ui.mixed_precision_check, bool(values["mixed_precision"]))
        self.ui.set_check_value(self.ui.pin_memory_check, bool(values["pin_memory"]))
        self.ui.set_check_value(self.ui.use_genre_metadata_check, bool(values["use_genre_metadata"]))
        self.ui.set_check_value(self.ui.use_year_metadata_check, bool(values["use_year_metadata"]))
        self.ui.set_check_value(self.ui.use_mood_metadata_check, bool(values["use_mood_metadata"]))
        self.ui.set_check_value(self.ui.use_key_metadata_check, bool(values["use_key_metadata"]))
        self.ui.set_check_value(self.ui.use_bpm_metadata_check, bool(values["use_bpm_metadata"]))
        self.settings.update_section("training", values)

    def reset_to_default_settings(self):
        self._apply_training_values(self._default_values())

    def open_advanced_model_settings(self):
        dialog = AdvancedModelSettingsDialog(self.context, self)
        dialog.exec()

    def _emit_preprocess_requested(self):
        self.preprocess_requested.emit(self.get_preprocess_payload())

    def _emit_train_requested(self):
        self.train_requested.emit(self.get_train_payload())

    def get_preprocess_payload(self) -> dict:
        return {
            "raw_music_folder": self.ui.raw_music_edit.text().strip(),
            "processed_training_data_folder": self.ui.processed_training_data_edit.text().strip(),
            "chunk_length_sec": float(self.ui.chunk_length_spin.value()),
            "chunk_stride_sec": float(self.ui.chunk_stride_spin.value()),
            "preprocess_device": str(self.ui.preprocess_gpu_combo.currentData()),
        }

    def get_train_payload(self) -> dict:
        token_model = dict(self.settings.get("token_model"))

        return {
            "processed_training_data_folder": self.ui.processed_training_data_edit.text().strip(),
            "model_output_folder": self.ui.model_output_edit.text().strip(),
            "batch_size": int(self.ui.batch_size_spin.value()),
            "epochs": int(self.ui.epochs_spin.value()),
            "learning_rate": float(self.ui.learning_rate_spin.value()),
            "validation_split": float(self.ui.validation_split_spin.value()),
            "weight_decay": float(self.ui.weight_decay_spin.value()),
            "mixed_precision": bool(self.ui.mixed_precision_check.isChecked()),
            "pin_memory": bool(self.ui.pin_memory_check.isChecked()),
            "use_genre_metadata": bool(self.ui.use_genre_metadata_check.isChecked()),
            "use_year_metadata": bool(self.ui.use_year_metadata_check.isChecked()),
            "use_mood_metadata": bool(self.ui.use_mood_metadata_check.isChecked()),
            "use_key_metadata": bool(self.ui.use_key_metadata_check.isChecked()),
            "use_bpm_metadata": bool(self.ui.use_bpm_metadata_check.isChecked()),
            "training_gpu_ids": self.ui.selected_training_gpu_ids(),
            "d_model": int(token_model["d_model"]),
            "n_heads": int(token_model["n_heads"]),
            "n_kv_heads": int(token_model.get("n_kv_heads", token_model["n_heads"])),
            "n_layers": int(token_model["n_layers"]),
            "ff_mult": int(token_model["ff_mult"]),
            "model_dropout": float(token_model["dropout"]),
            "metadata_prefix_tokens": int(max(8, token_model["metadata_prefix_tokens"])),
            "reference_track_duration_sec": float(token_model["reference_track_duration_sec"]),
            "rope_base": float(token_model.get("rope_base", 10000.0)),
        }

    def set_busy(self, busy: bool):
        for widget in self.ui.busy_widgets:
            widget.setEnabled(not busy)