import platform

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QLabel,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.core.device_utils import (
    format_gpu_label,
    get_compatible_cuda_gpus,
    normalize_training_gpu_ids,
)
from src.core.runtime_context import RuntimeContext


class TrainingGpuSelector(QWidget):
    def __init__(self, context: RuntimeContext, selected_gpu_ids):
        super().__init__()
        self.context = context
        self.settings = context.settings
        self.gpu_infos = get_compatible_cuda_gpus()
        self.training_gpu_checkboxes = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.toggle = QToolButton()
        self.toggle.setText("Training GPUs")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(False)
        self.toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle.setToolTip("Expand the list of GPUs used for training.")
        layout.addWidget(self.toggle)

        self.panel = QWidget()
        self.panel.setVisible(False)
        self.panel_layout = QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(6)

        self.windows_note = QLabel("Multi-GPU on Windows is experimental.")
        self.windows_note.setVisible(platform.system().lower() == "windows")
        self.panel_layout.addWidget(self.windows_note)

        self.empty_label = QLabel("No compatible CUDA GPUs detected. Training will run on CPU.")
        self.panel_layout.addWidget(self.empty_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(4)
        self.scroll.setWidget(self.scroll_content)

        self.panel_layout.addWidget(self.scroll)
        layout.addWidget(self.panel)

        self.build_checkboxes(selected_gpu_ids)

    @property
    def busy_widgets(self):
        return [self.toggle, *self.training_gpu_checkboxes]

    def build_checkboxes(self, selected_gpu_ids):
        normalized_selected = normalize_training_gpu_ids(selected_gpu_ids)

        if not normalized_selected and self.gpu_infos:
            normalized_selected = [self.gpu_infos[0].id]
            self.settings.set("hardware", "training_gpu_ids", normalized_selected)

        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self.training_gpu_checkboxes = []

        if not self.gpu_infos:
            self.empty_label.setVisible(True)
            self.scroll.setVisible(False)
            return

        self.empty_label.setVisible(False)
        self.scroll.setVisible(True)

        for gpu in self.gpu_infos:
            checkbox = QCheckBox(format_gpu_label(gpu))
            checkbox.setChecked(gpu.id in normalized_selected)
            checkbox.setToolTip("Use this GPU for training.")
            checkbox.toggled.connect(self.persist_selection)
            self.scroll_layout.addWidget(checkbox)
            self.training_gpu_checkboxes.append(checkbox)

        self.scroll_layout.addStretch(1)

    def selected_gpu_ids(self) -> list[int]:
        selected = []
        for gpu, checkbox in zip(self.gpu_infos, self.training_gpu_checkboxes):
            if checkbox.isChecked():
                selected.append(int(gpu.id))
        return selected

    def persist_selection(self):
        self.settings.set("hardware", "training_gpu_ids", self.selected_gpu_ids())

    def handle_toggle(self, expanded: bool):
        self.panel.setVisible(bool(expanded))
        self.toggle.setArrowType(
            Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        )