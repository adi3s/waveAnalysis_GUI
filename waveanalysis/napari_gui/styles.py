"""
Global styles for the Wave Analysis GUI.

This module contains reusable style definitions that can be imported
and applied to widgets throughout the application.

Usage:
    from .styles import BUTTON_STYLE
    my_button.setStyleSheet(BUTTON_STYLE)
"""

# Professional button style - gray background with black bold text
BUTTON_STYLE = """
    QPushButton {
        background-color: #d8d8d8;
        border: 1px solid #b0b0b0;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 11px;
        font-weight: bold;
        color: #000000;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #c8c8c8;
        border-color: #909090;
        color: #000000;
    }
    QPushButton:pressed {
        background-color: #b8b8b8;
        color: #000000;
    }
    QPushButton:disabled {
        background-color: #e8e8e8;
        color: #808080;
        border-color: #c0c0c0;
    }
"""

# Primary action button style - slightly blue tinted for important actions
BUTTON_STYLE_PRIMARY = """
    QPushButton {
        background-color: #4a90d9;
        border: 1px solid #3a7bc8;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 11px;
        font-weight: bold;
        color: #ffffff;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #5a9fe9;
        border-color: #4a8bd8;
        color: #ffffff;
    }
    QPushButton:pressed {
        background-color: #3a80c9;
        color: #ffffff;
    }
    QPushButton:disabled {
        background-color: #a0c0e0;
        color: #d0d0d0;
        border-color: #90b0d0;
    }
"""

# Danger/warning button style - for destructive actions like delete/clear
BUTTON_STYLE_DANGER = """
    QPushButton {
        background-color: #e05050;
        border: 1px solid #c04040;
        border-radius: 4px;
        padding: 6px 12px;
        font-size: 11px;
        font-weight: bold;
        color: #ffffff;
        min-width: 60px;
    }
    QPushButton:hover {
        background-color: #f06060;
        border-color: #d05050;
        color: #ffffff;
    }
    QPushButton:pressed {
        background-color: #d04040;
        color: #ffffff;
    }
    QPushButton:disabled {
        background-color: #e0a0a0;
        color: #d0d0d0;
        border-color: #d09090;
    }
"""
