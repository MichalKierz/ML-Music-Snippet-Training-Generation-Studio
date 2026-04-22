from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.device_utils import format_gpu_label, get_compatible_cuda_gpus, normalize_preprocess_device
from src.core.runtime_context import RuntimeContext


class TrainingTabForms:
    def __init__(self, context: RuntimeContext, training_gpu_selector: QWidget):
        self.context = context
        self.training_gpu_selector = training_gpu_selector
        self.gpu_infos = get_compatible_cuda_gpus()
        self.label_width = 220

    def build_input_group(self, paths: dict, processed_training_data_folder: str):
        group = QGroupBox("Input")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)

        raw_music_edit = QLineEdit(paths["raw_music_folder"])
        raw_music_button = QPushButton("Browse")
        layout.addWidget(self._labeled_row("Raw Music Folder", self._with_button(raw_music_edit, raw_music_button)))

        processed_training_data_edit = QLineEdit(processed_training_data_folder)
        processed_training_data_button = QPushButton("Browse")
        layout.addWidget(
            self._labeled_row(
                "Processed Training Data Folder",
                self._with_button(processed_training_data_edit, processed_training_data_button),
            )
        )

        model_output_edit = QLineEdit(paths["model_output_folder"])
        model_output_button = QPushButton("Browse")
        layout.addWidget(self._labeled_row("Model Output Folder", self._with_button(model_output_edit, model_output_button)))

        raw_music_edit.setToolTip("Folder with original MP3 files used as training source.")
        raw_music_button.setToolTip("Select the folder with original music files.")
        processed_training_data_edit.setToolTip(
            "Folder where token manifests, token cache and vocab files will be stored."
        )
        processed_training_data_button.setToolTip("Select the folder for processed training data.")
        model_output_edit.setToolTip("Folder where trained token model files will be saved.")
        model_output_button.setToolTip("Select the output folder for trained models.")

        return group, {
            "raw_music_edit": raw_music_edit,
            "raw_music_button": raw_music_button,
            "processed_training_data_edit": processed_training_data_edit,
            "processed_training_data_button": processed_training_data_button,
            "model_output_edit": model_output_edit,
            "model_output_button": model_output_button,
        }

    def build_preprocess_group(self, training: dict, preprocess_device: str):
        group = QGroupBox("Preprocessing")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)

        chunk_length_spin = QDoubleSpinBox()
        chunk_length_spin.setRange(1.0, 60.0)
        chunk_length_spin.setSingleStep(1.0)
        chunk_length_spin.setValue(float(training["chunk_length_sec"]))
        layout.addWidget(self._labeled_row("Chunk Length", chunk_length_spin))

        chunk_stride_spin = QDoubleSpinBox()
        chunk_stride_spin.setRange(1.0, 60.0)
        chunk_stride_spin.setSingleStep(1.0)
        chunk_stride_spin.setValue(float(training["chunk_stride_sec"]))
        layout.addWidget(self._labeled_row("Chunk Stride", chunk_stride_spin))

        preprocess_gpu_combo = QComboBox()
        self._populate_preprocess_gpu_combo(preprocess_gpu_combo, preprocess_device)
        layout.addWidget(self._labeled_row("Preprocessing GPU", preprocess_gpu_combo))

        preprocess_button = QPushButton("Preprocess Data")
        preprocess_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(preprocess_button)

        chunk_length_spin.setToolTip(
            "Length of one training snippet in seconds. The trained model will generate exactly this duration."
        )
        chunk_stride_spin.setToolTip(
            "Step between consecutive snippets. Lower values create more overlap, more data and much longer preprocessing and training."
        )
        preprocess_gpu_combo.setToolTip("Select one device for token preprocessing.")
        preprocess_button.setToolTip("Build token manifests and token cache from the raw music folder.")

        return group, {
            "chunk_length_spin": chunk_length_spin,
            "chunk_stride_spin": chunk_stride_spin,
            "preprocess_gpu_combo": preprocess_gpu_combo,
            "preprocess_button": preprocess_button,
        }

    def build_training_group(self, training: dict):
        group = QGroupBox("Training")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 20, 12, 12)
        layout.setSpacing(10)

        metadata_group = QGroupBox("Metadata Conditioning")
        metadata_layout = QVBoxLayout(metadata_group)
        metadata_layout.setContentsMargins(12, 20, 12, 12)
        metadata_info = QLabel("Artist, Title, and Position are always used. Select additional metadata below.")
        metadata_info.setWordWrap(True)
        metadata_layout.addWidget(metadata_info)

        metadata_grid = QGridLayout()
        metadata_grid.setHorizontalSpacing(24)
        metadata_grid.setVerticalSpacing(8)

        use_genre_metadata_check = QCheckBox("Use Genre")
        use_genre_metadata_check.setChecked(bool(training.get("use_genre_metadata", True)))

        use_year_metadata_check = QCheckBox("Use Year")
        use_year_metadata_check.setChecked(bool(training.get("use_year_metadata", True)))

        use_mood_metadata_check = QCheckBox("Use Mood")
        use_mood_metadata_check.setChecked(bool(training.get("use_mood_metadata", False)))

        use_key_metadata_check = QCheckBox("Use Initial Key")
        use_key_metadata_check.setChecked(bool(training.get("use_key_metadata", False)))

        use_bpm_metadata_check = QCheckBox("Use BPM")
        use_bpm_metadata_check.setChecked(bool(training.get("use_bpm_metadata", False)))

        metadata_grid.addWidget(use_genre_metadata_check, 0, 0)
        metadata_grid.addWidget(use_year_metadata_check, 0, 1)
        metadata_grid.addWidget(use_mood_metadata_check, 1, 0)
        metadata_grid.addWidget(use_key_metadata_check, 1, 1)
        metadata_grid.addWidget(use_bpm_metadata_check, 2, 0)

        metadata_layout.addLayout(metadata_grid)
        layout.addWidget(metadata_group)
        layout.addWidget(self.training_gpu_selector)

        batch_size_spin = QSpinBox()
        batch_size_spin.setRange(1, 256)
        batch_size_spin.setValue(int(training["batch_size"]))
        layout.addWidget(self._labeled_row("Batch Size", batch_size_spin))

        epochs_spin = QSpinBox()
        epochs_spin.setRange(1, 100000)
        epochs_spin.setValue(int(training["epochs"]))
        layout.addWidget(self._labeled_row("Epochs", epochs_spin))

        learning_rate_spin = QDoubleSpinBox()
        learning_rate_spin.setDecimals(6)
        learning_rate_spin.setRange(0.000001, 1.0)
        learning_rate_spin.setSingleStep(0.0001)
        learning_rate_spin.setValue(float(training["learning_rate"]))
        layout.addWidget(self._labeled_row("Learning Rate", learning_rate_spin))

        validation_split_spin = QDoubleSpinBox()
        validation_split_spin.setDecimals(2)
        validation_split_spin.setRange(0.0, 0.9)
        validation_split_spin.setSingleStep(0.05)
        validation_split_spin.setValue(float(training["validation_split"]))
        layout.addWidget(self._labeled_row("Validation Split", validation_split_spin))

        weight_decay_spin = QDoubleSpinBox()
        weight_decay_spin.setDecimals(6)
        weight_decay_spin.setRange(0.0, 1.0)
        weight_decay_spin.setSingleStep(0.0001)
        weight_decay_spin.setValue(float(training["weight_decay"]))
        layout.addWidget(self._labeled_row("Weight Decay", weight_decay_spin))

        mixed_precision_check = QCheckBox("Use Mixed Precision")
        mixed_precision_check.setChecked(bool(training["mixed_precision"]))

        pin_memory_check = QCheckBox("Use Pin Memory")
        pin_memory_check.setChecked(bool(training.get("pin_memory", False)))

        runtime_flags_row = QWidget()
        runtime_flags_layout = QHBoxLayout(runtime_flags_row)
        runtime_flags_layout.setContentsMargins(0, 0, 0, 0)
        runtime_flags_layout.setSpacing(16)
        runtime_flags_layout.addWidget(mixed_precision_check)
        runtime_flags_layout.addWidget(pin_memory_check)
        runtime_flags_layout.addStretch(1)
        layout.addWidget(runtime_flags_row)

        advanced_model_settings_button = QPushButton("Advanced Model Settings")
        advanced_model_settings_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        layout.addWidget(advanced_model_settings_button)

        buttons_row = QWidget()
        buttons_layout = QHBoxLayout(buttons_row)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(12)

        reset_defaults_button = QPushButton("Reset to Default Settings")
        train_button = QPushButton("Train Model")
        reset_defaults_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        train_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        buttons_layout.addWidget(reset_defaults_button)
        buttons_layout.addWidget(train_button)
        layout.addWidget(buttons_row)

        use_genre_metadata_check.setToolTip("If enabled, the model learns to condition generation on the Genre field.")
        use_year_metadata_check.setToolTip("If enabled, the model learns to condition generation on the Year field.")
        use_mood_metadata_check.setToolTip("If enabled, the model learns to condition generation on the Mood field.")
        use_key_metadata_check.setToolTip("If enabled, the model learns to condition generation on the Initial Key field.")
        use_bpm_metadata_check.setToolTip("If enabled, the model learns to condition generation on the BPM field.")

        batch_size_spin.setToolTip(
            "Base reference batch size. Real training batches are built dynamically with token-budget batching, so the actual number of examples per step can vary."
        )
        epochs_spin.setToolTip("How many full passes over the token dataset to train.")
        learning_rate_spin.setToolTip(
            "Peak learning rate used by the optimizer. Training uses warmup first and then cosine decay."
        )
        validation_split_spin.setToolTip("Fraction of data reserved for validation.")
        weight_decay_spin.setToolTip(
            "Regularization strength. Higher values reduce overfitting but can make memorization harder."
        )
        mixed_precision_check.setToolTip(
            "Uses mixed precision on supported GPUs. Usually speeds up training and lowers VRAM usage."
        )
        pin_memory_check.setToolTip(
            "Pins CPU memory for faster transfer to GPU. Can improve throughput but uses more RAM."
        )
        advanced_model_settings_button.setToolTip("Open advanced decoder architecture settings.")
        reset_defaults_button.setToolTip("Restore recommended default training settings.")
        train_button.setToolTip("Start token model training using the current settings.")

        return group, {
            "use_genre_metadata_check": use_genre_metadata_check,
            "use_year_metadata_check": use_year_metadata_check,
            "use_mood_metadata_check": use_mood_metadata_check,
            "use_key_metadata_check": use_key_metadata_check,
            "use_bpm_metadata_check": use_bpm_metadata_check,
            "batch_size_spin": batch_size_spin,
            "epochs_spin": epochs_spin,
            "learning_rate_spin": learning_rate_spin,
            "validation_split_spin": validation_split_spin,
            "weight_decay_spin": weight_decay_spin,
            "mixed_precision_check": mixed_precision_check,
            "pin_memory_check": pin_memory_check,
            "advanced_model_settings_button": advanced_model_settings_button,
            "reset_defaults_button": reset_defaults_button,
            "train_button": train_button,
        }

    def _populate_preprocess_gpu_combo(self, combo: QComboBox, selected_device: str):
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("CPU", "cpu")

        for gpu in self.gpu_infos:
            combo.addItem(format_gpu_label(gpu), f"cuda:{gpu.id}")

        normalized_selected = normalize_preprocess_device(selected_device)

        index = 0
        for combo_index in range(combo.count()):
            if combo.itemData(combo_index) == normalized_selected:
                index = combo_index
                break

        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _labeled_row(self, label_text: str, field_widget: QWidget):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        label = QLabel(label_text)
        label.setMinimumWidth(self.label_width)
        label.setMaximumWidth(self.label_width)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        field_widget.setSizePolicy(QSizePolicy.Expanding, field_widget.sizePolicy().verticalPolicy())

        layout.addWidget(label)
        layout.addWidget(field_widget, 1)

        return row

    def _with_button(self, line_edit: QLineEdit, button: QPushButton):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        button.setMinimumWidth(92)

        layout.addWidget(line_edit, 1)
        layout.addWidget(button)

        return container