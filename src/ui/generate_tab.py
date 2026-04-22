from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from src.core.runtime_context import RuntimeContext
from src.generation.token_model_loader import load_token_model_summary
from src.ui.generate_tab_helpers import format_duration_hms, is_valid_seed, make_seed
from src.ui.generate_tab_sections import GenerateTabSections


class GenerateTab(QWidget):
    generate_requested = Signal(dict)

    def __init__(self, context: RuntimeContext):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self.model_uses_genre_metadata = True
        self.model_uses_year_metadata = True
        self.model_uses_mood_metadata = False
        self.model_uses_key_metadata = False
        self.model_uses_bpm_metadata = False
        self.model_clip_length_sec = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll_area.viewport().setAutoFillBackground(False)

        scroll_content = QWidget()
        scroll_content.setAutoFillBackground(False)

        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(8, 8, 8, 8)
        scroll_layout.setSpacing(0)

        self.ui = GenerateTabSections(context)
        scroll_layout.addWidget(self.ui)

        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self._connect_signals()
        self._sync_position_stack()
        self._ensure_seed()
        self._sync_model_info_from_current_path()

    def _connect_signals(self):
        self.ui.model_file_button.clicked.connect(self._select_model_file)
        self.ui.output_folder_button.clicked.connect(
            lambda: self._select_folder(
                self.ui.output_folder_edit,
                "paths",
                "generate_output_folder",
            )
        )
        self.ui.refresh_seed_button.clicked.connect(self.refresh_seed)

        self.ui.model_file_edit.textChanged.connect(
            lambda value: self.settings.set("paths", "selected_model_file", value)
        )
        self.ui.model_file_edit.editingFinished.connect(
            self._sync_model_info_from_current_path
        )

        self.ui.output_folder_edit.textChanged.connect(
            lambda value: self.settings.set("paths", "generate_output_folder", value)
        )
        self.ui.artist_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "artist", value)
        )
        self.ui.title_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "title", value)
        )
        self.ui.year_spin.valueChanged.connect(
            lambda value: self.settings.set("generate", "year", int(value))
        )
        self.ui.genre_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "genre", value)
        )
        self.ui.mood_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "mood", value)
        )
        self.ui.initial_key_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "initial_key", value)
        )
        self.ui.bpm_spin.valueChanged.connect(
            lambda value: self.settings.set("generate", "bpm", float(value))
        )
        self.ui.position_mode_combo.currentTextChanged.connect(
            self._handle_position_mode_changed
        )
        self.ui.start_time_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "start_time", value)
        )
        self.ui.relative_position_combo.editTextChanged.connect(
            self._handle_relative_position_changed
        )
        self.ui.section_combo.currentTextChanged.connect(
            lambda value: self.settings.set("generate", "section", value)
        )
        self.ui.temperature_spin.valueChanged.connect(
            lambda value: self.settings.set("generate", "temperature", float(value))
        )
        self.ui.top_k_spin.valueChanged.connect(
            lambda value: self.settings.set("generate", "top_k", int(value))
        )
        self.ui.seed_edit.textChanged.connect(
            lambda value: self.settings.set("generate", "seed", value)
        )
        self.ui.song_snippet_count_spin.valueChanged.connect(
            self._handle_song_snippet_count_changed
        )
        self.ui.generate_snippet_button.clicked.connect(
            self._emit_generate_snippet_requested
        )
        self.ui.generate_song_button.clicked.connect(
            self._emit_generate_song_requested
        )

    def _select_folder(self, target, section: str, key: str):
        self.ui.select_folder(target, section, key)

    def _select_model_file(self):
        self.ui.select_model_file()

    def _handle_position_mode_changed(self, value: str):
        self.settings.set("generate", "position_mode", value)
        self._sync_position_stack()

    def _sync_position_stack(self):
        mode = self.ui.position_mode_combo.currentText()
        mapping = {
            "Start Time": 0,
            "Relative Position": 1,
            "Section": 2,
        }
        self.ui.position_stack.setCurrentIndex(mapping.get(mode, 0))

    def _handle_relative_position_changed(self, value: str):
        try:
            numeric = float(value)
        except Exception:
            return
        self.settings.set("generate", "relative_position", float(numeric))

    def _ensure_seed(self):
        current = self.ui.seed_edit.text().strip()
        if not is_valid_seed(current):
            self.refresh_seed()

    def refresh_seed(self):
        seed = make_seed()
        self.ui.seed_edit.setText(seed)
        self.settings.set("generate", "seed", seed)

    def _handle_song_snippet_count_changed(self, value: int):
        self.settings.set("generate", "song_snippet_count", int(value))
        self._update_song_length_summary()

    def _emit_generate_snippet_requested(self):
        if not is_valid_seed(self.ui.seed_edit.text().strip()):
            self.refresh_seed()
        self.generate_requested.emit(self.get_payload(generate_song=False))

    def _emit_generate_song_requested(self):
        if not is_valid_seed(self.ui.seed_edit.text().strip()):
            self.refresh_seed()
        self.generate_requested.emit(self.get_payload(generate_song=True))

    def _apply_metadata_field_state(self):
        self.ui.genre_edit.setEnabled(self.model_uses_genre_metadata)
        self.ui.year_spin.setEnabled(self.model_uses_year_metadata)
        self.ui.mood_edit.setEnabled(self.model_uses_mood_metadata)
        self.ui.initial_key_edit.setEnabled(self.model_uses_key_metadata)
        self.ui.bpm_spin.setEnabled(self.model_uses_bpm_metadata)

        self.ui.genre_edit.setToolTip(
            "Genre value used as a conditioning input."
            if self.model_uses_genre_metadata
            else "This model was trained without genre metadata."
        )
        self.ui.year_spin.setToolTip(
            "Year value used as a conditioning input."
            if self.model_uses_year_metadata
            else "This model was trained without year metadata."
        )
        self.ui.mood_edit.setToolTip(
            "Mood value used as a conditioning input."
            if self.model_uses_mood_metadata
            else "This model was trained without mood metadata."
        )
        self.ui.initial_key_edit.setToolTip(
            "Initial key value used as a conditioning input."
            if self.model_uses_key_metadata
            else "This model was trained without initial key metadata."
        )
        self.ui.bpm_spin.setToolTip(
            "BPM value used as a conditioning input."
            if self.model_uses_bpm_metadata
            else "This model was trained without BPM metadata."
        )

    def _clear_model_info(self):
        self.model_clip_length_sec = 0.0
        self.ui.model_architecture_value.setText("Unknown")
        self.ui.model_architecture_version_value.setText("Unknown")
        self.ui.model_clip_length_value.setText("Unknown")
        self.ui.model_sample_rate_value.setText("Unknown")
        self.ui.model_dimension_value.setText("Unknown")
        self.ui.model_heads_value.setText("Unknown")
        self.ui.model_kv_heads_value.setText("Unknown")
        self.ui.model_gqa_groups_value.setText("Unknown")
        self.ui.model_layers_value.setText("Unknown")
        self.ui.model_rope_base_value.setText("Unknown")
        self.ui.model_dataset_size_value.setText("Unknown")
        self.ui.model_token_length_value.setText("Unknown")
        self.ui.model_use_genre_value.setText("Unknown")
        self.ui.model_use_year_value.setText("Unknown")
        self.ui.model_use_mood_value.setText("Unknown")
        self.ui.model_use_key_value.setText("Unknown")
        self.ui.model_use_bpm_value.setText("Unknown")
        self.ui.model_run_dir_value.setText("Unknown")

        self.model_uses_genre_metadata = True
        self.model_uses_year_metadata = True
        self.model_uses_mood_metadata = False
        self.model_uses_key_metadata = False
        self.model_uses_bpm_metadata = False

        self._apply_metadata_field_state()
        self._update_song_length_summary()

    def _sync_model_info_from_current_path(self):
        self._load_model_summary(self.ui.model_file_edit.text().strip())

    def _load_model_summary(self, model_file: str):
        if not model_file:
            self._clear_model_info()
            return

        try:
            summary = load_token_model_summary(model_file)
            clip_length_sec = float(summary.get("clip_length_sec", 0.0))
            architecture_name = str(summary.get("architecture_name", ""))
            architecture_version = int(summary.get("architecture_version", 0))
            sample_rate = int(summary.get("sample_rate", 0))
            d_model = int(summary.get("d_model", 0))
            n_heads = int(summary.get("n_heads", 0))
            n_kv_heads = int(summary.get("n_kv_heads", 0))
            gqa_groups = int(summary.get("gqa_groups", 0))
            n_layers = int(summary.get("n_layers", 0))
            rope_base = float(summary.get("rope_base", 0.0))
            dataset_size = int(summary.get("dataset_size", 0))
            target_token_length = int(summary.get("target_token_length", 0))
            run_dir = str(summary.get("run_dir", ""))
            use_genre_metadata = bool(summary.get("use_genre_metadata", True))
            use_year_metadata = bool(summary.get("use_year_metadata", True))
            use_mood_metadata = bool(summary.get("use_mood_metadata", False))
            use_key_metadata = bool(summary.get("use_key_metadata", False))
            use_bpm_metadata = bool(summary.get("use_bpm_metadata", False))

            self.model_clip_length_sec = clip_length_sec
            self.ui.model_architecture_value.setText(architecture_name or "Unknown")
            self.ui.model_architecture_version_value.setText(str(architecture_version) if architecture_version > 0 else "Unknown")
            self.ui.model_clip_length_value.setText(
                f"{clip_length_sec:.2f} s" if clip_length_sec > 0 else "Unknown"
            )
            self.ui.model_sample_rate_value.setText(
                str(sample_rate) if sample_rate > 0 else "Unknown"
            )
            self.ui.model_dimension_value.setText(
                str(d_model) if d_model > 0 else "Unknown"
            )
            self.ui.model_heads_value.setText(str(n_heads) if n_heads > 0 else "Unknown")
            self.ui.model_kv_heads_value.setText(str(n_kv_heads) if n_kv_heads > 0 else "Unknown")
            self.ui.model_gqa_groups_value.setText(str(gqa_groups) if gqa_groups > 0 else "Unknown")
            self.ui.model_layers_value.setText(str(n_layers) if n_layers > 0 else "Unknown")
            self.ui.model_rope_base_value.setText(f"{rope_base:.1f}" if rope_base > 0 else "Unknown")
            self.ui.model_dataset_size_value.setText(
                str(dataset_size) if dataset_size > 0 else "Unknown"
            )
            self.ui.model_token_length_value.setText(
                str(target_token_length) if target_token_length > 0 else "Unknown"
            )
            self.ui.model_use_genre_value.setText("Yes" if use_genre_metadata else "No")
            self.ui.model_use_year_value.setText("Yes" if use_year_metadata else "No")
            self.ui.model_use_mood_value.setText("Yes" if use_mood_metadata else "No")
            self.ui.model_use_key_value.setText("Yes" if use_key_metadata else "No")
            self.ui.model_use_bpm_value.setText("Yes" if use_bpm_metadata else "No")
            self.ui.model_run_dir_value.setText(run_dir or "Unknown")

            self.model_uses_genre_metadata = use_genre_metadata
            self.model_uses_year_metadata = use_year_metadata
            self.model_uses_mood_metadata = use_mood_metadata
            self.model_uses_key_metadata = use_key_metadata
            self.model_uses_bpm_metadata = use_bpm_metadata

            self._apply_metadata_field_state()
            self._update_song_length_summary()
        except Exception:
            self._clear_model_info()

    def _update_song_length_summary(self):
        snippet_count = int(self.ui.song_snippet_count_spin.value())

        if self.model_clip_length_sec <= 0:
            self.ui.song_length_value.setText("Unknown")
            return

        total_duration = float(self.model_clip_length_sec) * float(snippet_count)
        self.ui.song_length_value.setText(
            f"{format_duration_hms(total_duration)} ({total_duration:.2f} s)"
        )

    def get_payload(self, generate_song: bool) -> dict:
        try:
            relative_position = float(
                self.ui.relative_position_combo.currentText().strip()
            )
        except Exception:
            relative_position = 0.5

        return {
            "model_file": self.ui.model_file_edit.text().strip(),
            "output_folder": self.ui.output_folder_edit.text().strip(),
            "artist": self.ui.artist_edit.text().strip(),
            "title": self.ui.title_edit.text().strip(),
            "year": int(self.ui.year_spin.value()),
            "genre": self.ui.genre_edit.text().strip(),
            "mood": self.ui.mood_edit.text().strip(),
            "initial_key": self.ui.initial_key_edit.text().strip(),
            "bpm": float(self.ui.bpm_spin.value()),
            "position_mode": self.ui.position_mode_combo.currentText(),
            "start_time": self.ui.start_time_edit.text().strip(),
            "relative_position": float(relative_position),
            "section": self.ui.section_combo.currentText(),
            "temperature": float(self.ui.temperature_spin.value()),
            "top_k": int(self.ui.top_k_spin.value()),
            "seed": self.ui.seed_edit.text().strip(),
            "generate_song": bool(generate_song),
            "song_snippet_count": int(self.ui.song_snippet_count_spin.value()),
        }

    def set_busy(self, busy: bool):
        for widget in self.ui.busy_widgets:
            widget.setEnabled(not busy)

    def set_model_file(self, model_file: str):
        self.ui.model_file_edit.setText(model_file)
        self.settings.set("paths", "selected_model_file", model_file)
        self._load_model_summary(model_file)