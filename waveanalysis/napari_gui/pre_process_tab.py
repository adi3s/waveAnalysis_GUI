from qtpy.QtWidgets import *
from magicgui import magicgui
from qtpy.QtCore import Signal

class PreProcessingTab(QWidget):
    #analyze = Signal()

    def __init__(self, parent):
        """Initialize the PreProcessingTab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the PreProcessingTab"""
        layout = QVBoxLayout()
        
        # Threshold controls
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setValue(0.5)
        self.threshold.setSingleStep(0.01)
        
        # Smoothing controls
        self.smooth_window = QSpinBox()
        self.smooth_window.setRange(1, 21)
        self.smooth_window.setValue(11)
        
        self.smooth_order = QSpinBox()
        self.smooth_order.setRange(1, 5)
        self.smooth_order.setValue(2)
        
        # Analyze button
        self.analyze = QPushButton("Analyze")
        
        form = QFormLayout()
        form.addRow("ACF Peak Threshold:", self.threshold)
        form.addRow("Smoothing Window:", self.smooth_window)
        form.addRow("Smoothing Order:", self.smooth_order)
        
        layout.addLayout(form)
        layout.addWidget(self.analyze)
        self.setLayout(layout)

    def get_params(self):
        """Return the current parameters for analysis"""
        return {
            "threshold": self.threshold.value(),
            "smooth_window": self.smooth_window.value(),
            "smooth_order": self.smooth_order.value()
        }