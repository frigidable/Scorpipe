from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


def apply_theme(app: QtWidgets.QApplication, *, mode: str = "dark") -> None:
    """Apply a modern theme (Fusion + palette + QSS).

    The goal is a *commercial* feel: predictable spacing, readable typography,
    and low visual noise.

    Parameters
    ----------
    mode:
        "dark" or "light".
    """

    mode = (mode or "dark").strip().lower()
    if mode not in {"dark", "light"}:
        mode = "dark"

    app.setStyle("Fusion")

    pal = QtGui.QPalette()

    if mode == "dark":
        bg = QtGui.QColor(30, 30, 30)
        base = QtGui.QColor(22, 22, 22)
        alt = QtGui.QColor(35, 35, 35)
        text = QtGui.QColor(225, 225, 225)
        disabled = QtGui.QColor(140, 140, 140)
        accent = QtGui.QColor(61, 174, 233)
        hl_text = QtGui.QColor(0, 0, 0)
        border = "#3a3a3a"
        panel = "#242424"
        input_bg = "#151515"
        editor_bg = "#101010"
        btn_bg = "#2a2a2a"
        btn_hover = "#323232"
        btn_pressed = "#1f1f1f"
    else:
        bg = QtGui.QColor(245, 245, 247)
        base = QtGui.QColor(255, 255, 255)
        alt = QtGui.QColor(250, 250, 250)
        text = QtGui.QColor(25, 25, 25)
        disabled = QtGui.QColor(150, 150, 150)
        accent = QtGui.QColor(47, 111, 237)
        hl_text = QtGui.QColor(255, 255, 255)
        border = "#d7d7db"
        panel = "#ffffff"
        input_bg = "#ffffff"
        editor_bg = "#ffffff"
        btn_bg = "#ffffff"
        btn_hover = "#f0f0f3"
        btn_pressed = "#e8e8ec"

    pal.setColor(QtGui.QPalette.ColorRole.Window, bg)
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, text)
    pal.setColor(QtGui.QPalette.ColorRole.Base, base)
    pal.setColor(QtGui.QPalette.ColorRole.AlternateBase, alt)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipBase, base)
    pal.setColor(QtGui.QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QtGui.QPalette.ColorRole.Text, text)
    pal.setColor(QtGui.QPalette.ColorRole.Button, alt)
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, text)
    pal.setColor(QtGui.QPalette.ColorRole.Highlight, accent)
    pal.setColor(QtGui.QPalette.ColorRole.HighlightedText, hl_text)
    pal.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 0, 0))

    pal.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text, disabled)
    pal.setColor(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.ButtonText, disabled)
    app.setPalette(pal)

    # Global font: keep defaults, but encourage consistent sizing.
    f = app.font()
    if f.pointSize() < 10:
        f.setPointSize(10)
        app.setFont(f)

    # QSS: gentle rounding, consistent padding, no "thick" borders.
    qss = f"""
    QWidget {{ font-size: 10pt; }}
    QMainWindow::separator {{ background: {border}; width: 1px; height: 1px; }}

    QGroupBox {{
        margin-top: 10px;
        padding: 10px;
        border: 1px solid {border};
        border-radius: 10px;
        background: {panel};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        padding: 6px 8px;
        border-radius: 8px;
        border: 1px solid {border};
        background: {input_bg};
        selection-background-color: {accent.name()};
    }}
    QComboBox::drop-down {{ border: 0px; width: 28px; }}
    QComboBox::down-arrow {{
        width: 10px;
        height: 10px;
        image: url(data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10' viewBox='0 0 10 10'><path d='M2 3l3 3 3-3' fill='none' stroke='%23{('ECECEC' if mode=="dark" else '666666')}' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/></svg>);
    }}

    QPushButton, QToolButton {{
        padding: 7px 12px;
        border-radius: 10px;
        border: 1px solid {border};
        background: {btn_bg};
        color: {text};
    }}
    QPushButton:hover, QToolButton:hover {{ background: {btn_hover}; }}
    QPushButton:pressed, QToolButton:pressed {{ background: {btn_pressed}; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {disabled.name()}; }}

    QPushButton[primary="true"], QToolButton[primary="true"] {{
        background: {accent.name()};
        border-color: {accent.name()};
        color: {hl_text.name() if mode=="dark" else "#ffffff"};
        font-weight: 600;
    }}
    QPushButton[primary="true"]:hover, QToolButton[primary="true"]:hover {{
        background: {accent.lighter(112).name()};
        border-color: {accent.lighter(112).name()};
    }}
    QPushButton[primary="true"]:pressed, QToolButton[primary="true"]:pressed {{
        background: {accent.darker(112).name()};
        border-color: {accent.darker(112).name()};
    }}

    QToolButton[compact="true"] {{ padding: 6px 10px; border-radius: 10px; }}

    QPlainTextEdit {{
        border-radius: 10px;
        border: 1px solid {border};
        background: {editor_bg};
    }}

    QTabWidget::pane {{
        border: 1px solid {border};
        border-radius: 12px;
        padding: 6px;
        top: -1px;
        background: {panel};
    }}
    QTabBar::tab {{
        padding: 8px 14px;
        border: 1px solid {border};
        border-bottom: 0px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        background: {btn_bg};
        margin-right: 4px;
    }}
    QTabBar::tab:selected {{ background: {panel}; }}

    QListWidget {{
        border-radius: 12px;
        border: 1px solid {border};
        background: {panel};
        padding: 6px;
    }}
    QListWidget::item {{ padding: 10px 10px; border-radius: 10px; }}
    QListWidget::item:selected {{ background: {btn_hover}; border: 1px solid {border}; }}

    QDockWidget {{ border: 1px solid {border}; border-radius: 12px; }}
    QDockWidget::title {{
        padding: 8px;
        background: {panel};
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
    }}
    """
    app.setStyleSheet(qss)


def load_ui_settings() -> QtCore.QSettings:
    # QSettings stores per-user config automatically.
    return QtCore.QSettings("ScorpioPipe", "ScorpioPipe")
