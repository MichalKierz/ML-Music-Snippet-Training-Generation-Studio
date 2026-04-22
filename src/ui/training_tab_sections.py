from PySide6.QtWidgets import QFileDialog, QVBoxLayout, QWidget

from src.core.device_utils import normalize_training_gpu_ids
from src.core.runtime_context import RuntimeContext
from src.ui.training_gpu_selector import TrainingGpuSelector
from src.ui.training_tab_forms import TrainingTabForms


class TrainingTabSections(QWidget):
    def __init__(self, context: RuntimeContext):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self.configs = context.configs

        self._build_ui()
        self._build_busy_widgets()

    def _build_ui(self):
        paths = self.settings.get("paths")
        hardware = self.settings.get("hardware")
        training = self.settings.get("training")

        processed_training_data_folder = (
            paths.get("processed_training_data_folder")
            or str(self.context.runtime_dirs["processed_training_data"])
        )

        self.training_gpu_selector = TrainingGpuSelector(
            context=self.context,
            selected_gpu_ids=normalize_training_gpu_ids(hardware.get("training_gpu_ids", [])),
        )

        forms = TrainingTabForms(self.context, self.training_gpu_selector)

        input_group, input_widgets = forms.build_input_group(
            paths=paths,
            processed_training_data_folder=processed_training_data_folder,
        )
        preprocess_group, preprocess_widgets = forms.build_preprocess_group(
            training=training,
            preprocess_device=hardware.get("preprocess_device", "cpu"),
        )
        training_group, training_widgets = forms.build_training_group(training=training)

        self.raw_music_edit = input_widgets["raw_music_edit"]
        self.raw_music_button = input_widgets["raw_music_button"]
        self.processed_training_data_edit = input_widgets["processed_training_data_edit"]
        self.processed_training_data_button = input_widgets["processed_training_data_button"]
        self.model_output_edit = input_widgets["model_output_edit"]
        self.model_output_button = input_widgets["model_output_button"]

        self.chunk_length_spin = preprocess_widgets["chunk_length_spin"]
        self.chunk_stride_spin = preprocess_widgets["chunk_stride_spin"]
        self.preprocess_gpu_combo = preprocess_widgets["preprocess_gpu_combo"]
        self.preprocess_button = preprocess_widgets["preprocess_button"]

        self.use_genre_metadata_check = training_widgets["use_genre_metadata_check"]
        self.use_year_metadata_check = training_widgets["use_year_metadata_check"]
        self.use_mood_metadata_check = training_widgets["use_mood_metadata_check"]
        self.use_key_metadata_check = training_widgets["use_key_metadata_check"]
        self.use_bpm_metadata_check = training_widgets["use_bpm_metadata_check"]
        self.training_gpu_toggle = self.training_gpu_selector.toggle
        self.batch_size_spin = training_widgets["batch_size_spin"]
        self.epochs_spin = training_widgets["epochs_spin"]
        self.learning_rate_spin = training_widgets["learning_rate_spin"]
        self.validation_split_spin = training_widgets["validation_split_spin"]
        self.weight_decay_spin = training_widgets["weight_decay_spin"]
        self.mixed_precision_check = training_widgets["mixed_precision_check"]
        self.pin_memory_check = training_widgets["pin_memory_check"]
        self.advanced_model_settings_button = training_widgets["advanced_model_settings_button"]
        self.reset_defaults_button = training_widgets["reset_defaults_button"]
        self.train_button = training_widgets["train_button"]

        layout = QVBoxLayout(self)
        layout.addWidget(input_group)
        layout.addWidget(preprocess_group)
        layout.addWidget(training_group)
        layout.addStretch(1)

    def _build_busy_widgets(self):
        self.busy_widgets = [
            self.raw_music_edit,
            self.raw_music_button,
            self.processed_training_data_edit,
            self.processed_training_data_button,
            self.model_output_edit,
            self.model_output_button,
            self.chunk_length_spin,
            self.chunk_stride_spin,
            self.preprocess_gpu_combo,
            self.preprocess_button,
            self.use_genre_metadata_check,
            self.use_year_metadata_check,
            self.use_mood_metadata_check,
            self.use_key_metadata_check,
            self.use_bpm_metadata_check,
            self.batch_size_spin,
            self.epochs_spin,
            self.learning_rate_spin,
            self.validation_split_spin,
            self.weight_decay_spin,
            self.mixed_precision_check,
            self.pin_memory_check,
            self.advanced_model_settings_button,
            self.reset_defaults_button,
            self.train_button,
            *self.training_gpu_selector.busy_widgets,
        ]

    def select_folder(self, target, section: str, key: str):
        current = target.text().strip() or str(self.context.base_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select Folder", current)
        if selected:
            target.setText(selected)
            self.settings.set(section, key, selected)

    def set_spin_value(self, widget, value):
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def set_check_value(self, widget, value):
        widget.blockSignals(True)
        widget.setChecked(value)
        widget.blockSignals(False)

    def selected_training_gpu_ids(self) -> list[int]:
        return self.training_gpu_selector.selected_gpu_ids()

    def persist_preprocess_gpu_selection(self):
        device_value = str(self.preprocess_gpu_combo.currentData())
        self.settings.set("hardware", "preprocess_device", device_value)

    def handle_training_gpu_toggle(self, expanded: bool):
        self.training_gpu_selector.handle_toggle(expanded)