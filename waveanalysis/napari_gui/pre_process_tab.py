from qtpy.QtWidgets import *
from magicgui import magicgui
from qtpy.QtCore import Signal, Qt

class PreProcessingTab(QWidget):
    """Tab for preprocessing options"""
    # Signals to indicate selected options
    acf_selected = Signal(bool)
    ccf_selected = Signal(bool)
    peaks_selected = Signal(bool)
    indv_acf_selected = Signal(bool)
    indv_ccf_selected = Signal(bool)
    indv_peaks_selected = Signal(bool)

    def __init__(self, parent):
        """Initialize the PreProcessingTab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the PreProcessingTab"""
        # Create scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create content widget
        content_widget = QWidget()
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
        
        # Group box for plot options
        plots_group = QGroupBox("Plot Generation Options")
        plots_layout = QVBoxLayout()
        
        # Summary plots subgroup
        summary_plots_group = QGroupBox("Summary Plots")
        summary_plots_layout = QVBoxLayout()
        
        self.acf_checkbox = QCheckBox("ACF Plots")
        self.acf_checkbox.stateChanged.connect(self.on_acf_selected)
        summary_plots_layout.addWidget(self.acf_checkbox)
        
        self.ccf_checkbox = QCheckBox("CCF Plots")
        self.ccf_checkbox.stateChanged.connect(self.on_ccf_selected)
        summary_plots_layout.addWidget(self.ccf_checkbox)
        
        self.peaks_checkbox = QCheckBox("Peaks Plots")
        self.peaks_checkbox.stateChanged.connect(self.on_peaks_selected)
        summary_plots_layout.addWidget(self.peaks_checkbox)
        
        summary_plots_group.setLayout(summary_plots_layout)
        plots_layout.addWidget(summary_plots_group)
        
        # Individual bin plots subgroup
        individual_plots_group = QGroupBox("Individual Bin Plots (Optional)")
        individual_plots_layout = QVBoxLayout()
        
        # Warning label
        warning_label = QLabel("Note: Generating individual bin plots may significantly increase processing time and memory usage.")
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet("color: #666;")
        individual_plots_layout.addWidget(warning_label)
        
        self.indv_acf_checkbox = QCheckBox("Individual ACF Plots")
        self.indv_acf_checkbox.stateChanged.connect(self.on_indv_acf_selected)
        individual_plots_layout.addWidget(self.indv_acf_checkbox)
        
        self.indv_ccf_checkbox = QCheckBox("Individual CCF Plots")
        self.indv_ccf_checkbox.stateChanged.connect(self.on_indv_ccf_selected)
        individual_plots_layout.addWidget(self.indv_ccf_checkbox)
        
        self.indv_peaks_checkbox = QCheckBox("Individual Peaks Plots")
        self.indv_peaks_checkbox.stateChanged.connect(self.on_indv_peaks_selected)
        individual_plots_layout.addWidget(self.indv_peaks_checkbox)
        
        individual_plots_group.setLayout(individual_plots_layout)
        plots_layout.addWidget(individual_plots_group)
        
        plots_group.setLayout(plots_layout)
        
        # Form layout for other controls
        form = QFormLayout()
        form.addRow("ACF Peak Threshold:", self.threshold)
        form.addRow("Smoothing Window:", self.smooth_window)
        form.addRow("Smoothing Order:", self.smooth_order)
        
        layout.addLayout(form)
        layout.addWidget(self.analyze)
        layout.addWidget(plots_group)
        
        # Set the layout for the content widget
        content_widget.setLayout(layout)
        
        # Set the content widget in the scroll area
        scroll.setWidget(content_widget)
        
        # Create the final layout
        final_layout = QVBoxLayout()
        final_layout.addWidget(scroll)
        final_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(final_layout)

    def on_acf_selected(self, state):
        """Emit signal when ACF checkbox is toggled"""
        self.acf_selected.emit(state == Qt.Checked)

    def on_ccf_selected(self, state):
        """Emit signal when CCF checkbox is toggled"""
        self.ccf_selected.emit(state == Qt.Checked)

    def on_peaks_selected(self, state):
        """Emit signal when Peaks checkbox is toggled"""
        self.peaks_selected.emit(state == Qt.Checked)

    def on_indv_acf_selected(self, state):
        """Emit signal when individual ACF checkbox is toggled"""
        self.indv_acf_selected.emit(state == Qt.Checked)

    def on_indv_ccf_selected(self, state):
        """Emit signal when individual CCF checkbox is toggled"""
        self.indv_ccf_selected.emit(state == Qt.Checked)

    def on_indv_peaks_selected(self, state):
        """Emit signal when individual Peaks checkbox is toggled"""
        self.indv_peaks_selected.emit(state == Qt.Checked)

    def get_params(self):
        """Return the current parameters for analysis"""
        return {
            "threshold": self.threshold.value(),
            "smooth_window": self.smooth_window.value(),
            "smooth_order": self.smooth_order.value(),
            "plot_indv_acfs": self.indv_acf_checkbox.isChecked(),
            "plot_indv_ccfs": self.indv_ccf_checkbox.isChecked(),
            "plot_indv_peaks": self.indv_peaks_checkbox.isChecked()
        }