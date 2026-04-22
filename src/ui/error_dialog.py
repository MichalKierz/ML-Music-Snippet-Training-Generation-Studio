from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
)


def summarize_error_text(message: str, max_length: int = 120) -> str:
    text = str(message).strip().replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if not lines:
        return "Unknown error"

    first_line = lines[0]

    if ":" in first_line:
        first_part = first_line.split(":", 1)[0].strip()
        if first_part and len(first_part) <= max_length:
            return first_part

    if "." in first_line:
        first_sentence = first_line.split(".", 1)[0].strip() + "."
        if len(first_sentence) <= max_length:
            return first_sentence

    if len(first_line) <= max_length:
        return first_line

    return first_line[: max_length - 3].rstrip() + "..."


class ErrorDialog(QDialog):
    def __init__(self, title: str, message: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(820, 560)

        layout = QVBoxLayout(self)

        summary_label = QLabel(summarize_error_text(message, max_length=220))
        summary_label.setWordWrap(True)

        details_box = QTextEdit()
        details_box.setReadOnly(True)
        details_box.setPlainText(str(message))

        buttons_row = QHBoxLayout()
        buttons_row.addStretch(1)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        buttons_row.addWidget(close_button)

        layout.addWidget(summary_label)
        layout.addWidget(details_box)
        layout.addLayout(buttons_row)