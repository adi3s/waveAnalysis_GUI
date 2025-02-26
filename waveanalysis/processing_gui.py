import napari
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari_roi_manager import QRoiManager
import os
import tifffile as tiff

import waveanalysis.signal_processing as sp
import waveanalysis.plotting as plt

class WaveAnalysisWidget(QWidget):
    def __init__(self, viewer, folder_path, group_names):
        super().__init__()
        self.viewer = viewer
        self.folder_path = folder_path
        self.group_names = group_names
        self.image_files = []
        self.combined_results = None
        self.threshold_value = None
        self.smooth_value = None
        self.results = {}
        self.current_image_index = 0
        self.init_ui()
        self.load_image()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.roi_manager_tab = QWidget()
        self.pre_processing_tab = QWidget()
        self.workflow_tab = QWidget()
        self.post_processing_tab = QWidget()

        self.tabs.addTab(self.roi_manager_tab, "ROI Manager")
        self.tabs.addTab(self.pre_processing_tab, "Pre-Processing")
        self.tabs.addTab(self.workflow_tab, "Data Workflow")
        self.tabs.addTab(self.post_processing_tab, "Post-Processing")

        self.init_roi_manager_tab()
        self.init_pre_processing_tab()
        self.init_post_processing_tab()
        self.init_workflow_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_pre_processing_tab(self):
        """Initialize the pre-processing tab."""
        pre_processing_layout = QVBoxLayout()
        self.pre_processing_tab.setLayout(pre_processing_layout)

        self.threshold_widget = magicgui(
            self.threshold,
            call_button="Set Threshold",
            threshold_value={"widget_type": "FloatSlider", "min": 0, "max": 1, "step": 0.01}
        )
        self.smooth_widget = magicgui(
            self.smooth,
            call_button="Set Smoothing",
            smooth_value={"widget_type": "FloatSlider", "min": 0, "max": 10, "step": 0.1}
        )
        self.correlation_widget = magicgui(self.correlation, call_button="Calculate Correlation")
        self.peak_widget = magicgui(self.peak, call_button="Detect Peaks")
        self.wave_speed_widget = magicgui(self.wave_speed, call_button="Calculate Wave Speed")
        self.analyze_widget = magicgui(self.analyze, call_button="Analyze")

        threshold_smoothing_group_box = QGroupBox("Apply")
        threshold_smoothing_layout = QVBoxLayout()
        threshold_smoothing_layout.addWidget(self.threshold_widget.native)
        threshold_smoothing_layout.addWidget(self.smooth_widget.native)
        threshold_smoothing_group_box.setLayout(threshold_smoothing_layout)

        correlation_peak_wave_speed_group_box = QGroupBox("Calculate")
        correlation_peak_wave_speed_layout = QVBoxLayout()
        correlation_peak_wave_speed_layout.addWidget(self.correlation_widget.native)
        correlation_peak_wave_speed_layout.addWidget(self.peak_widget.native)
        correlation_peak_wave_speed_layout.addWidget(self.wave_speed_widget.native)
        correlation_peak_wave_speed_group_box.setLayout(correlation_peak_wave_speed_layout)

        pre_processing_layout.addWidget(threshold_smoothing_group_box)
        pre_processing_layout.addWidget(correlation_peak_wave_speed_group_box)
        pre_processing_layout.addWidget(self.analyze_widget.native)

    def init_post_processing_tab(self):
        """Initialize the post-processing tab."""
        post_processing_layout = QVBoxLayout()
        self.post_processing_tab.setLayout(post_processing_layout)
        self.output_table = QTableWidget(10, 4)
        self.output_table.setHorizontalHeaderLabels([
            'Image', 'Wave Speed', 'Correlation', 'Peak'
        ])
        post_processing_layout.addWidget(QLabel("Analysis Results:"))
        post_processing_layout.addWidget(self.output_table)

    def init_roi_manager_tab(self):
        """Initialize the ROI manager tab."""
        roi_manager_layout = QVBoxLayout()
        self.roi_manager_tab.setLayout(roi_manager_layout)
        self.roi_manager = QRoiManager(self.viewer)
        roi_manager_layout.addWidget(self.roi_manager)

    def init_workflow_tab(self):
        """Initialize the workflow tab."""
        workflow_layout = QVBoxLayout()
        self.workflow_tab.setLayout(workflow_layout)
        workflow_group_box = QGroupBox("Select Workflow")
        workflow_layout_inner = QVBoxLayout()
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(["Standard", "Rolling", "Kymograph"])
        self.workflow_combo.currentIndexChanged.connect(self.update_workflow_parameters)
        workflow_layout_inner.addWidget(self.workflow_combo)
        workflow_group_box.setLayout(workflow_layout_inner)
        workflow_layout.addWidget(workflow_group_box)
        self.workflow_parameters_layout = QVBoxLayout()
        workflow_layout.addLayout(self.workflow_parameters_layout)
        self.update_workflow_parameters(0)

    def update_workflow_parameters(self, index):
        """Update the workflow parameters based on the selected workflow."""
        for i in reversed(range(self.workflow_parameters_layout.count())):
            widget = self.workflow_parameters_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        workflow_name = self.workflow_combo.currentText()
        self.workflow_parameters_layout.addWidget(QLabel(f"{workflow_name} Workflow Parameters"))
        if index == 0:
            self.add_parameter_checkboxes(["Summary ACFs", "Summary CCFs", "Summary peaks", "Individual ACFs", "Individual CCFs", "Individual peaks"])
        elif index == 1:
            self.add_parameter_checkboxes(["Period", "Amplitude", "Maximum", "Minimum", "Width", "Shift"])
        elif index == 2:
            self.add_parameter_checkboxes(["Summary ACFs", "Summary CCFs", "Summary peaks", "Individual ACFs", "Individual CCFs", "Individual peaks"])

    def add_parameter_checkboxes(self, parameters):
        """Add checkboxes for the given parameters."""
        for parameter in parameters:
            checkbox = QCheckBox(parameter)
            checkbox.stateChanged.connect(lambda state, param=parameter: self.update_plot_visibility(param, state))
            self.workflow_parameters_layout.addWidget(checkbox)

    def update_plot_visibility(self, parameter, state):
        """Update the plot visibility based on the selected parameter."""
        if state == Qt.Checked:
            plot_dialog = QDialog(self)
            plot_dialog.setWindowTitle(f"{self.workflow_combo.currentText()} Workflow - {parameter} Plot")
            plot_layout = QVBoxLayout(plot_dialog)
            plot_canvas = FigureCanvas(plt.figure())
            plot_layout.addWidget(plot_canvas)
            ax = plot_canvas.figure.add_subplot(111)
            ax.set_title(f"{self.workflow_combo.currentText()} Workflow - {parameter} Plot")

            if parameter in self.results:
                ax.plot(self.results[parameter], label=parameter)

            ax.legend()
            plot_canvas.draw()
            button_box = QDialogButtonBox(QDialogButtonBox.Ok)
            button_box.accepted.connect(plot_dialog.accept)
            plot_layout.addWidget(button_box)
            plot_dialog.show()

    def threshold(self, threshold_value: float):
        """Set the threshold value."""
        print(f"Threshold button clicked with value: {threshold_value}")
        self.threshold_value = threshold_value

    def smooth(self, smooth_value: float):
        """Set the smoothing value."""
        print(f"Smooth button clicked with value: {smooth_value}")
        self.smooth_value = smooth_value

    def load_image(self):
        """Load the entire TIFF stack and clear previous image layers while retaining and clearing ROI shapes."""
        self.image_files = [
            os.path.join(self.folder_path, f) 
            for f in sorted(os.listdir(self.folder_path))
            if f.lower().endswith((".tif", ".tiff")) 
            and any(group in f for group in self.group_names)
        ]

        if self.current_image_index < len(self.image_files):
            img_path = self.image_files[self.current_image_index]
            image_data = tiff.imread(img_path)
            
            # Remove previous image layers
            for layer in list(self.viewer.layers):
                if isinstance(layer, napari.layers.Image):
                    self.viewer.layers.remove(layer)
            
            # Clear existing ROI shapes from all shapes layers
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Shapes):
                    layer.data = []

            # Add new image layer
            image_layer = self.viewer.add_image(image_data, name=os.path.basename(img_path))
            self.viewer.reset_view()
            self.viewer.layers.move(self.viewer.layers.index(image_layer), 0)
        else:
            print("Analysis complete.")
            self.viewer.close()

    def analyze(self):
        """Analyze images and load next TIFF."""
        print("Analyze button clicked")
        if self.current_image_index < len(self.image_files):
            self.current_image_index += 1
            self.load_image()  # Loads next image and clears previous image/ROI shapes
        else:
            print("No more images.")
            self.viewer.close()

    def correlation(self):
        """Calculate correlation parameters."""
        print("Correlation button clicked")
        current_data = self.viewer.layers[0].data
        self.results['indv_ACF'] = sp.calc_indv_ACF_workflow(current_data, self.threshold_value, self.smooth_value)
        self.results['indv_period'] = sp.calc_indv_period_workflow(current_data, self.threshold_value, self.smooth_value)
        self.results['indv_CCF'] = sp.calc_indv_CCF_workflow(current_data, self.threshold_value, self.smooth_value)
        self.results['indv_shift'] = sp.calc_indv_shift_workflow(current_data, self.threshold_value, self.smooth_value)
        self.update_post_processing_tab()

    def peak(self):
        """Calculate peak parameters."""
        print("Peak button clicked")
        current_data = self.viewer.layers[0].data
        self.results['indv_peak'] = sp.calc_indv_peak_props_workflow(current_data, self.threshold_value, self.smooth_value)
        self.results['indv_peak_rolling'] = sp.calc_indv_peak_props_rolling(current_data)
        self.update_post_processing_tab()

    def wave_speed(self):
        """Calculate wave speed parameters."""
        print("Wave Speed button clicked")
        current_data = self.viewer.layers[0].data
        self.results['wave_speed'] = sp.calculate_wave_speed(current_data, self.threshold_value, self.smooth_value)
        self.update_post_processing_tab()

    def update_post_processing_tab(self):
        """Update the post-processing tab with results."""
        self.output_table.setRowCount(len(self.results))
        for row, (key, value) in enumerate(self.results.items()):
            self.output_table.setItem(row, 0, QTableWidgetItem(key))
            self.output_table.setItem(row, 1, QTableWidgetItem(str(value)))