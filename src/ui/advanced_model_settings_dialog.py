from PySide6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.runtime_context import RuntimeContext


def _largest_divisor_not_greater_than(value: int, maximum: int) -> int:
    value = max(1, int(value))
    maximum = max(1, int(maximum))
    for candidate in range(min(value, maximum), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


class AdvancedModelSettingsDialog(QDialog):
    def __init__(self, context: RuntimeContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.settings = context.settings
        self.configs = context.configs

        self.setWindowTitle("Advanced Model Settings")
        self.resize(560, 360)

        token_model_settings = self.settings.get("token_model")

        layout = QVBoxLayout(self)

        form_widget = QWidget()
        form = QFormLayout(form_widget)

        self.d_model_spin = QSpinBox()
        self.d_model_spin.setRange(64, 8192)
        self.d_model_spin.setSingleStep(64)
        self.d_model_spin.setValue(int(token_model_settings["d_model"]))
        form.addRow("Model Dimension", self.d_model_spin)

        self.n_heads_spin = QSpinBox()
        self.n_heads_spin.setRange(1, 128)
        self.n_heads_spin.setValue(int(token_model_settings["n_heads"]))
        form.addRow("Attention Heads", self.n_heads_spin)

        self.n_kv_heads_spin = QSpinBox()
        self.n_kv_heads_spin.setRange(1, int(token_model_settings["n_heads"]))
        self.n_kv_heads_spin.setValue(
            _largest_divisor_not_greater_than(
                int(token_model_settings["n_heads"]),
                int(token_model_settings.get("n_kv_heads", token_model_settings["n_heads"])),
            )
        )
        form.addRow("KV Heads", self.n_kv_heads_spin)

        self.n_layers_spin = QSpinBox()
        self.n_layers_spin.setRange(1, 128)
        self.n_layers_spin.setValue(int(token_model_settings["n_layers"]))
        form.addRow("Transformer Layers", self.n_layers_spin)

        self.ff_mult_spin = QSpinBox()
        self.ff_mult_spin.setRange(1, 32)
        self.ff_mult_spin.setValue(int(token_model_settings["ff_mult"]))
        form.addRow("Feedforward Multiplier", self.ff_mult_spin)

        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setRange(0.0, 0.95)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setValue(float(token_model_settings["dropout"]))
        form.addRow("Model Dropout", self.dropout_spin)

        self.metadata_prefix_tokens_spin = QSpinBox()
        self.metadata_prefix_tokens_spin.setRange(8, 64)
        self.metadata_prefix_tokens_spin.setValue(int(max(8, token_model_settings["metadata_prefix_tokens"])))
        form.addRow("Metadata Prefix Tokens", self.metadata_prefix_tokens_spin)

        self.reference_track_duration_spin = QDoubleSpinBox()
        self.reference_track_duration_spin.setDecimals(1)
        self.reference_track_duration_spin.setRange(10.0, 3600.0)
        self.reference_track_duration_spin.setSingleStep(10.0)
        self.reference_track_duration_spin.setValue(float(token_model_settings["reference_track_duration_sec"]))
        form.addRow("Reference Track Duration", self.reference_track_duration_spin)

        self.rope_base_spin = QDoubleSpinBox()
        self.rope_base_spin.setDecimals(1)
        self.rope_base_spin.setRange(1000.0, 1000000.0)
        self.rope_base_spin.setSingleStep(1000.0)
        self.rope_base_spin.setValue(float(token_model_settings.get("rope_base", 10000.0)))
        form.addRow("RoPE Base", self.rope_base_spin)

        layout.addWidget(form_widget)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)

        self.reset_button = QPushButton("Reset to Default Settings")
        self.close_button = QPushButton("Close")

        button_layout.addWidget(self.reset_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.close_button)

        layout.addWidget(button_row)

        self._apply_tooltips()

        self.d_model_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "d_model", int(value)))
        self.n_heads_spin.valueChanged.connect(self._handle_n_heads_changed)
        self.n_kv_heads_spin.valueChanged.connect(self._handle_n_kv_heads_changed)
        self.n_layers_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "n_layers", int(value)))
        self.ff_mult_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "ff_mult", int(value)))
        self.dropout_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "dropout", float(value)))
        self.metadata_prefix_tokens_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "metadata_prefix_tokens", int(max(8, value))))
        self.reference_track_duration_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "reference_track_duration_sec", float(value)))
        self.rope_base_spin.valueChanged.connect(lambda value: self.settings.set("token_model", "rope_base", float(value)))

        self.reset_button.clicked.connect(self.reset_to_defaults)
        self.close_button.clicked.connect(self.accept)

    def _handle_n_heads_changed(self, value: int):
        value = int(value)
        self.settings.set("token_model", "n_heads", value)
        self.n_kv_heads_spin.blockSignals(True)
        self.n_kv_heads_spin.setMaximum(value)
        adjusted_kv_heads = _largest_divisor_not_greater_than(value, int(self.n_kv_heads_spin.value()))
        self.n_kv_heads_spin.setValue(adjusted_kv_heads)
        self.n_kv_heads_spin.blockSignals(False)
        self.settings.set("token_model", "n_kv_heads", adjusted_kv_heads)

    def _handle_n_kv_heads_changed(self, value: int):
        n_heads = int(self.n_heads_spin.value())
        adjusted_value = _largest_divisor_not_greater_than(n_heads, int(value))
        if adjusted_value != int(value):
            self.n_kv_heads_spin.blockSignals(True)
            self.n_kv_heads_spin.setValue(adjusted_value)
            self.n_kv_heads_spin.blockSignals(False)
        self.settings.set("token_model", "n_kv_heads", adjusted_value)

    def _apply_tooltips(self):
        self.d_model_spin.setToolTip("Main hidden size of the decoder. Higher values increase capacity, VRAM usage and training time.")
        self.n_heads_spin.setToolTip("Number of query attention heads. Model Dimension must be divisible by this value.")
        self.n_kv_heads_spin.setToolTip("Number of key/value heads. It must divide Attention Heads. Lower values enable grouped-query attention and reduce KV cache memory.")
        self.n_layers_spin.setToolTip("Number of decoder layers. More layers increase capacity and training time.")
        self.ff_mult_spin.setToolTip("Multiplier for the SwiGLU feedforward block width.")
        self.dropout_spin.setToolTip("Dropout used during training.")
        self.metadata_prefix_tokens_spin.setToolTip("Reserved metadata prefix positions before audio tokens. Minimum is 8 because the model uses 8 conditioning slots.")
        self.reference_track_duration_spin.setToolTip("Reference song duration used to map relative positions and section prompts into time ranges.")
        self.rope_base_spin.setToolTip("RoPE frequency base. Leave at 10000 unless you know you need a different positional scaling.")

        self.reset_button.setToolTip("Restore default advanced model settings from token_model_defaults.json.")
        self.close_button.setToolTip("Close the advanced model settings window.")

    def _set_spin_value(self, widget, value):
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def reset_to_defaults(self):
        defaults = self.context.configs["token_model"]
        default_n_heads = int(defaults["n_heads"])
        default_n_kv_heads = _largest_divisor_not_greater_than(
            default_n_heads,
            int(defaults.get("n_kv_heads", default_n_heads)),
        )

        values = {
            "d_model": int(defaults["d_model"]),
            "n_heads": default_n_heads,
            "n_kv_heads": default_n_kv_heads,
            "n_layers": int(defaults["n_layers"]),
            "ff_mult": int(defaults["ff_mult"]),
            "dropout": float(defaults["dropout"]),
            "metadata_prefix_tokens": int(max(8, defaults["metadata_prefix_tokens"])),
            "reference_track_duration_sec": float(defaults["reference_track_duration_sec"]),
            "rope_base": float(defaults.get("rope_base", 10000.0)),
        }

        self._set_spin_value(self.d_model_spin, values["d_model"])
        self._set_spin_value(self.n_heads_spin, values["n_heads"])
        self.n_kv_heads_spin.blockSignals(True)
        self.n_kv_heads_spin.setMaximum(values["n_heads"])
        self.n_kv_heads_spin.setValue(values["n_kv_heads"])
        self.n_kv_heads_spin.blockSignals(False)
        self._set_spin_value(self.n_layers_spin, values["n_layers"])
        self._set_spin_value(self.ff_mult_spin, values["ff_mult"])
        self._set_spin_value(self.dropout_spin, values["dropout"])
        self._set_spin_value(self.metadata_prefix_tokens_spin, values["metadata_prefix_tokens"])
        self._set_spin_value(self.reference_track_duration_spin, values["reference_track_duration_sec"])
        self._set_spin_value(self.rope_base_spin, values["rope_base"])

        self.settings.update_section("token_model", values)