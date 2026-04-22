from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.runtime_context import RuntimeContext


class GenerateTabSections(QWidget):
    def __init__(self, context: RuntimeContext):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self._build_ui()
        self._apply_tooltips()
        self._build_busy_widgets()

    def _build_ui(self):
        paths = self.settings.get("paths")
        generate = self.settings.get("generate")

        stored_song_snippet_count = int(generate.get("song_snippet_count", 50))
        initial_song_snippet_count = 50 if stored_song_snippet_count == 8 else max(1, stored_song_snippet_count)
        if initial_song_snippet_count != stored_song_snippet_count:
            self.settings.set("generate", "song_snippet_count", initial_song_snippet_count)

        main_layout = QVBoxLayout(self)

        source_group = QGroupBox("Source")
        source_layout = QVBoxLayout(source_group)

        source_paths_layout = QFormLayout()

        self.model_file_edit = QLineEdit(paths["selected_model_file"])
        self.model_file_button = QPushButton("Browse")
        source_paths_layout.addRow(
            "Model File",
            self._with_button(self.model_file_edit, self.model_file_button),
        )

        self.output_folder_edit = QLineEdit(paths["generate_output_folder"])
        self.output_folder_button = QPushButton("Browse")
        source_paths_layout.addRow(
            "Output Folder",
            self._with_button(self.output_folder_edit, self.output_folder_button),
        )

        source_layout.addLayout(source_paths_layout)

        model_info_grid = QGridLayout()
        model_info_grid.setHorizontalSpacing(18)
        model_info_grid.setVerticalSpacing(6)

        self.model_architecture_value = QLabel("Unknown")
        self.model_architecture_version_value = QLabel("Unknown")
        self.model_clip_length_value = QLabel("Unknown")
        self.model_sample_rate_value = QLabel("Unknown")
        self.model_dimension_value = QLabel("Unknown")
        self.model_heads_value = QLabel("Unknown")
        self.model_kv_heads_value = QLabel("Unknown")
        self.model_gqa_groups_value = QLabel("Unknown")
        self.model_layers_value = QLabel("Unknown")
        self.model_rope_base_value = QLabel("Unknown")
        self.model_dataset_size_value = QLabel("Unknown")
        self.model_token_length_value = QLabel("Unknown")
        self.model_use_genre_value = QLabel("Unknown")
        self.model_use_year_value = QLabel("Unknown")
        self.model_use_mood_value = QLabel("Unknown")
        self.model_use_key_value = QLabel("Unknown")
        self.model_use_bpm_value = QLabel("Unknown")
        self.model_run_dir_value = QLabel("Unknown")
        self.model_run_dir_value.setWordWrap(True)

        info_items = [
            ("Architecture", self.model_architecture_value),
            ("Architecture Version", self.model_architecture_version_value),
            ("Model Clip Length", self.model_clip_length_value),
            ("Codec Sample Rate", self.model_sample_rate_value),
            ("Transformer Dimension", self.model_dimension_value),
            ("Attention Heads", self.model_heads_value),
            ("KV Heads", self.model_kv_heads_value),
            ("GQA Groups", self.model_gqa_groups_value),
            ("Layers", self.model_layers_value),
            ("RoPE Base", self.model_rope_base_value),
            ("Training Samples", self.model_dataset_size_value),
            ("Target Token Length", self.model_token_length_value),
            ("Uses Genre Metadata", self.model_use_genre_value),
            ("Uses Year Metadata", self.model_use_year_value),
            ("Uses Mood Metadata", self.model_use_mood_value),
            ("Uses Initial Key Metadata", self.model_use_key_value),
            ("Uses BPM Metadata", self.model_use_bpm_value),
            ("Run Folder", self.model_run_dir_value),
        ]

        for index, (label_text, value_widget) in enumerate(info_items):
            row = index // 3
            column = index % 3
            model_info_grid.addWidget(self._make_info_item(label_text, value_widget), row, column)

        source_layout.addLayout(model_info_grid)

        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        top_prompt_grid = QGridLayout()
        top_prompt_grid.setHorizontalSpacing(12)
        top_prompt_grid.setVerticalSpacing(8)

        self.artist_edit = QLineEdit(generate["artist"])
        self.title_edit = QLineEdit(generate["title"])
        self.year_spin = QSpinBox()
        self.year_spin.setRange(0, 3000)
        self.year_spin.setValue(int(generate["year"]))

        top_prompt_grid.addWidget(QLabel("Artist"), 0, 0)
        top_prompt_grid.addWidget(self.artist_edit, 0, 1)
        top_prompt_grid.addWidget(QLabel("Title"), 0, 2)
        top_prompt_grid.addWidget(self.title_edit, 0, 3)
        top_prompt_grid.addWidget(QLabel("Year"), 0, 4)
        top_prompt_grid.addWidget(self.year_spin, 0, 5)

        self.mood_edit = QLineEdit(generate.get("mood", ""))
        self.initial_key_edit = QLineEdit(generate.get("initial_key", ""))
        self.bpm_spin = QDoubleSpinBox()
        self.bpm_spin.setDecimals(2)
        self.bpm_spin.setRange(0.0, 400.0)
        self.bpm_spin.setSingleStep(1.0)
        self.bpm_spin.setValue(float(generate.get("bpm", 0.0)))

        top_prompt_grid.addWidget(QLabel("Mood"), 1, 0)
        top_prompt_grid.addWidget(self.mood_edit, 1, 1)
        top_prompt_grid.addWidget(QLabel("Initial Key"), 1, 2)
        top_prompt_grid.addWidget(self.initial_key_edit, 1, 3)
        top_prompt_grid.addWidget(QLabel("BPM"), 1, 4)
        top_prompt_grid.addWidget(self.bpm_spin, 1, 5)

        prompt_layout.addLayout(top_prompt_grid)

        middle_prompt_grid = QGridLayout()
        middle_prompt_grid.setHorizontalSpacing(12)
        middle_prompt_grid.setVerticalSpacing(8)

        self.genre_edit = QLineEdit(generate["genre"])
        self.position_mode_combo = QComboBox()
        self.position_mode_combo.addItems(
            ["Start Time", "Relative Position", "Section"]
        )
        self.position_mode_combo.setCurrentText(generate["position_mode"])

        middle_prompt_grid.addWidget(QLabel("Genre"), 0, 0)
        middle_prompt_grid.addWidget(self.genre_edit, 0, 1)
        middle_prompt_grid.addWidget(QLabel("Position Mode"), 0, 2)
        middle_prompt_grid.addWidget(self.position_mode_combo, 0, 3)

        prompt_layout.addLayout(middle_prompt_grid)

        self.position_stack = QStackedWidget()

        self.start_time_page = QWidget()
        start_time_layout = QFormLayout(self.start_time_page)
        self.start_time_edit = QLineEdit(generate["start_time"])
        start_time_layout.addRow("Start Time", self.start_time_edit)

        self.relative_position_page = QWidget()
        relative_position_layout = QFormLayout(self.relative_position_page)
        self.relative_position_combo = QComboBox()
        self.relative_position_combo.setEditable(True)
        self.relative_position_combo.addItems(
            ["0.00", "0.10", "0.25", "0.50", "0.75", "0.90", "1.00"]
        )
        self.relative_position_combo.setCurrentText(
            str(generate["relative_position"])
        )
        relative_position_layout.addRow(
            "Relative Position",
            self.relative_position_combo,
        )

        self.section_page = QWidget()
        section_layout = QFormLayout(self.section_page)
        self.section_combo = QComboBox()
        self.section_combo.addItems(["Intro", "Early", "Middle", "Late", "Outro"])
        self.section_combo.setCurrentText(generate["section"])
        section_layout.addRow("Section", self.section_combo)

        self.position_stack.addWidget(self.start_time_page)
        self.position_stack.addWidget(self.relative_position_page)
        self.position_stack.addWidget(self.section_page)

        prompt_layout.addWidget(self.position_stack)

        output_group = QGroupBox("Output")
        output_layout = QVBoxLayout(output_group)

        output_grid = QGridLayout()
        output_grid.setHorizontalSpacing(12)
        output_grid.setVerticalSpacing(8)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setRange(0.0, 5.0)
        self.temperature_spin.setSingleStep(0.05)
        self.temperature_spin.setValue(float(generate.get("temperature", 1.0)))

        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(0, 5000)
        self.top_k_spin.setValue(int(generate.get("top_k", 50)))

        self.seed_edit = QLineEdit(generate["seed"])
        self.refresh_seed_button = QPushButton("Refresh Seed")

        output_grid.addWidget(QLabel("Temperature"), 0, 0)
        output_grid.addWidget(self.temperature_spin, 0, 1)
        output_grid.addWidget(QLabel("Top K"), 0, 2)
        output_grid.addWidget(self.top_k_spin, 0, 3)
        output_grid.addWidget(QLabel("Seed"), 1, 0)
        output_grid.addWidget(self._with_button(self.seed_edit, self.refresh_seed_button), 1, 1, 1, 3)

        output_layout.addLayout(output_grid)

        self.generate_snippet_button = QPushButton("Generate Snippet")
        output_layout.addWidget(self.generate_snippet_button)

        song_group = QGroupBox("Song Generation")
        song_layout = QVBoxLayout(song_group)

        song_form_layout = QFormLayout()
        self.song_snippet_count_spin = QSpinBox()
        self.song_snippet_count_spin.setRange(1, 10000)
        self.song_snippet_count_spin.setValue(initial_song_snippet_count)
        song_form_layout.addRow("Snippet Count", self.song_snippet_count_spin)

        self.song_length_value = QLabel("Unknown")
        song_form_layout.addRow("Estimated Song Length", self.song_length_value)

        song_layout.addLayout(song_form_layout)

        self.generate_song_button = QPushButton("Generate Song")
        song_layout.addWidget(self.generate_song_button)

        main_layout.addWidget(source_group)
        main_layout.addWidget(prompt_group)
        main_layout.addWidget(output_group)
        main_layout.addWidget(song_group)
        main_layout.addStretch(1)

    def _make_info_item(self, label_text: str, value_widget: QLabel):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(f"{label_text}:")
        layout.addWidget(label)
        layout.addWidget(value_widget, 1)
        return container

    def _apply_tooltips(self):
        self.model_file_edit.setToolTip("Select the trained token model file.")
        self.model_file_button.setToolTip(
            "Browse for a trained token model file."
        )
        self.output_folder_edit.setToolTip(
            "Folder where generated MP3 files will be saved."
        )
        self.output_folder_button.setToolTip(
            "Select the output folder for generated snippets."
        )

        self.artist_edit.setToolTip("Artist value used as a mandatory conditioning input.")
        self.title_edit.setToolTip("Title value used as a mandatory conditioning input.")
        self.year_spin.setToolTip("Year value used as a conditioning input.")
        self.genre_edit.setToolTip("Genre value used as a conditioning input.")
        self.mood_edit.setToolTip("Mood value used as a conditioning input.")
        self.initial_key_edit.setToolTip("Initial key value used as a conditioning input.")
        self.bpm_spin.setToolTip("BPM value used as a conditioning input.")
        self.position_mode_combo.setToolTip(
            "Choose how the target position inside a song is described."
        )
        self.start_time_edit.setToolTip(
            "Exact time position in format MM:SS."
        )
        self.relative_position_combo.setToolTip(
            "Relative position from 0.0 to 1.0 inside the estimated source track."
        )
        self.section_combo.setToolTip(
            "Named section mapped to an internal relative position."
        )

        self.temperature_spin.setToolTip(
            "Controls randomness during token sampling. Lower values are more conservative, higher values are more varied."
        )
        self.top_k_spin.setToolTip(
            "Limits sampling to the K most likely next tokens. Lower values are safer, higher values are more diverse. Set 0 to disable top-k filtering."
        )

        self.seed_edit.setToolTip("Random seed for token generation.")
        self.refresh_seed_button.setToolTip("Generate a new random seed.")

        self.song_snippet_count_spin.setToolTip(
            "Number of snippets to generate and stitch into one song."
        )
        self.song_length_value.setToolTip(
            "Estimated total song length based on snippet count and the clip length supported by the selected model."
        )
        self.generate_snippet_button.setToolTip(
            "Generate a single snippet using the current position settings."
        )
        self.generate_song_button.setToolTip(
            "Generate a full song by stitching sequential snippets into one MP3 file."
        )

    def _build_busy_widgets(self):
        self.busy_widgets = [
            self.model_file_edit,
            self.model_file_button,
            self.output_folder_edit,
            self.output_folder_button,
            self.artist_edit,
            self.title_edit,
            self.year_spin,
            self.genre_edit,
            self.mood_edit,
            self.initial_key_edit,
            self.bpm_spin,
            self.position_mode_combo,
            self.start_time_edit,
            self.relative_position_combo,
            self.section_combo,
            self.temperature_spin,
            self.top_k_spin,
            self.seed_edit,
            self.refresh_seed_button,
            self.song_snippet_count_spin,
            self.generate_snippet_button,
            self.generate_song_button,
        ]

    def _with_button(self, line_edit: QLineEdit, button: QPushButton):
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(line_edit)
        layout.addWidget(button)
        return container

    def select_folder(self, target: QLineEdit, section: str, key: str):
        current = target.text().strip() or str(self.context.base_dir)
        selected = QFileDialog.getExistingDirectory(self, "Select Folder", current)
        if selected:
            target.setText(selected)
            self.settings.set(section, key, selected)

    def select_model_file(self):
        start_dir = self.settings.get(
            "paths",
            "model_output_folder",
            str(self.context.runtime_dirs["models"]),
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            start_dir,
            "Model Files (*.pt *.pth *.ckpt);;All Files (*)",
        )
        if selected:
            self.model_file_edit.setText(selected)
            self.settings.set("paths", "selected_model_file", selected)