import napari
import numpy as np
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt
from magicgui import magicgui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from napari_roi_manager import QRoiManager
import os
import tifffile as tiff
import pandas as pd

# Imports the files
from waveanalysis.signal_processing import correlation_functions, peak_properties, wave_speed
from waveanalysis.plotting import group_plotting, indv_plot_creation, mean_plot_creation, rolling_plot_creation
from waveanalysis.data_workflows import combined_workflow, rolling_workflow
from waveanalysis.image_props import image_bin_calc, image_properties, image_to_np_arrays

class WaveAnalysisWidget(QWidget):
    def __init__(self, viewer, folder_path, analysis_mode, *args, plot_params):
        """
        Parameters:
          viewer        - a Napari viewer instance
          folder_path   - the directory containing TIFF images
          analysis_mode - a string: "standard", "rolling", or "kymograph"
          *args         - the remaining parameters, which differ by mode:
                          • standard: (group_names, box_size, bin_shift, acf_peak_thresh)
                          • rolling: (box_size, bin_shift, subframe_size, subframe_roll)
                          • kymograph: (group_names, line_width, bin_shift, acf_peak_thresh)
          plot_params   - a dictionary of booleans indicating which plots to generate.
        """
        super().__init__()
        self.viewer = viewer
        self.folder_path = folder_path
        self.analysis_mode = analysis_mode.lower()
        self.plot_params = plot_params
        self.smooth_value = 0.0  # Default smoothing value

        if self.analysis_mode == "standard":
            self.group_names = args[0]
            self.box_size = args[1]
            self.bin_shift = args[2]
            self.acf_peak_thresh = args[3]
        elif self.analysis_mode == "rolling":
            self.box_size = args[0]
            self.bin_shift = args[1]
            self.subframe_size = args[2]
            self.subframe_roll = args[3]
            self.group_names = []  # not used in rolling mode 
        elif self.analysis_mode == "kymograph":
            self.group_names = args[0]
            self.line_width = args[1]
            self.bin_shift = args[2]
            self.acf_peak_thresh = args[3]
        else:
            raise ValueError("Invalid analysis mode specified")

        self.image_files = []
        # Initialize results dictionary with empty values
        self.results = {
            "auto_correlation": {},
            "cross_correlation": {},
            "period": {},
            "peak": {},
            "wave_speed": None,
            "rolling_plots": None
        }
        self.current_image_index = 0
        self.image_props = {}
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

    def init_roi_manager_tab(self):
        """Initialize the ROI Manager tab."""
        roi_layout = QVBoxLayout()
        self.roi_manager_tab.setLayout(roi_layout)
        self.roi_manager = QRoiManager(self.viewer)
        roi_layout.addWidget(self.roi_manager)

    def init_pre_processing_tab(self):
        """Initialize the Pre-Processing tab."""
        pre_proc_layout = QVBoxLayout()
        self.pre_processing_tab.setLayout(pre_proc_layout)

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

        group_box_apply = QGroupBox("Apply")
        layout_apply = QVBoxLayout()
        layout_apply.addWidget(self.threshold_widget.native)
        layout_apply.addWidget(self.smooth_widget.native)
        group_box_apply.setLayout(layout_apply)

        group_box_calc = QGroupBox("Calculate")
        layout_calc = QVBoxLayout()
        layout_calc.addWidget(self.correlation_widget.native)
        layout_calc.addWidget(self.peak_widget.native)
        layout_calc.addWidget(self.wave_speed_widget.native)
        group_box_calc.setLayout(layout_calc)

        pre_proc_layout.addWidget(group_box_apply)
        pre_proc_layout.addWidget(group_box_calc)
        pre_proc_layout.addWidget(self.analyze_widget.native)

    def init_post_processing_tab(self):
        """Initialize the Post-Processing tab."""
        post_proc_layout = QVBoxLayout()
        self.post_processing_tab.setLayout(post_proc_layout)
        self.output_table = QTableWidget(10, 4)
        self.output_table.setHorizontalHeaderLabels(['Frame', 'Wave Speed', 'Correlation', 'Peak'])
        post_proc_layout.addWidget(QLabel("Analysis Results:"))
        post_proc_layout.addWidget(self.output_table)

    def init_workflow_tab(self):
        """Initialize the Workflow tab."""
        workflow_layout = QVBoxLayout()
        self.workflow_tab.setLayout(workflow_layout)
        group_box = QGroupBox("Select Workflow")
        layout_inner = QVBoxLayout()
        self.workflow_combo = QComboBox()
        self.workflow_combo.addItems(["Standard", "Rolling", "Kymograph"])
        self.workflow_combo.currentIndexChanged.connect(self.update_workflow_parameters)
        layout_inner.addWidget(self.workflow_combo)
        group_box.setLayout(layout_inner)
        workflow_layout.addWidget(group_box)
        self.workflow_parameters_layout = QVBoxLayout()
        workflow_layout.addLayout(self.workflow_parameters_layout)
        self.update_workflow_parameters(0)

    def update_workflow_parameters(self, index):
        """Update the workflow parameters based on the selected workflow."""
        for i in reversed(range(self.workflow_parameters_layout.count())):
            widget = self.workflow_parameters_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.workflow_parameters_layout.addWidget(QLabel(f"{self.workflow_combo.currentText()} Workflow Parameters"))
        if index == 0:
            self.add_parameter_checkboxes(["Individual ACFs", "Individual CCFs", "Individual peaks", "Wave Speeds"])
        elif index == 1:
            self.add_parameter_checkboxes(["Period", "Amplitude", "Maximum", "Minimum", "Width", "Shift"])
        elif index == 2:
            self.add_parameter_checkboxes(["Individual ACFs", "Individual CCFs", "Individual peaks", "Wave Speeds"])

    def add_parameter_checkboxes(self, parameters):
        """Add checkboxes for the given parameters to the workflow parameters layout."""
        for parameter in parameters:
            checkbox = QCheckBox(parameter)
            checkbox.stateChanged.connect(lambda state, param=parameter: self.update_plot_visibility(param, state))
            self.workflow_parameters_layout.addWidget(checkbox)

    def update_plot_visibility(self, parameter, state):
        """Show a QDialog with the corresponding plot when a checkbox is checked."""
        param_key = "calc_wave_speeds" if parameter.lower() == "wave speeds" else f"plot_{parameter.lower().replace(' ', '_')}"
        self.plot_params[param_key] = (state == Qt.Checked)
        if state == Qt.Checked:
            mapping = {
                "plot_indv_acfs": "auto_correlation",
                "plot_indv_ccfs": "cross_correlation",
                "plot_indv_peaks": "peak",
                "calc_wave_speeds": "wave_speed"
            }
            if param_key in mapping:
                result_key = mapping[param_key]
                if result_key in self.results and self.results[result_key]:
                    fig = self.results[result_key]
                    self.show_plot_dialog(fig, title=parameter)
                else:
                    self.show_plot_dialog_message(f"No plot available for {parameter}.\nRun analysis first.", title=parameter)

    def show_plot_dialog(self, fig, title):
        """Show a dialog with the given plot."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout()
        if isinstance(fig, dict):
            # If fig is a dictionary, show the first plot
            if fig:
                first_key = next(iter(fig))
                canvas = FigureCanvas(fig[first_key])
                layout.addWidget(canvas)
            else:
                layout.addWidget(QLabel("No plot available"))
        elif isinstance(fig, plt.Figure):
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
        else:
            layout.addWidget(QLabel(str(fig)))
        dialog.setLayout(layout)
        dialog.exec_()

    def show_plot_dialog_message(self, message, title):
        """Show a dialog with the given message."""
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout()
        layout.addWidget(QLabel(message))
        dialog.setLayout(layout)
        dialog.exec_()

    def threshold(self, threshold_value: float):
        """Set the threshold value for ACF peak detection."""
        print(f"Threshold button clicked with value: {threshold_value}")
        self.acf_peak_thresh = threshold_value

    def smooth(self, smooth_value: float):
        """Set the smoothing value for the analysis."""
        print(f"Smooth button clicked with value: {smooth_value}")
        self.smooth_value = smooth_value

    def load_image(self):
        """Load the image based on the current analysis mode and update the viewer."""
        if self.analysis_mode in ("standard", "kymograph"):
            self.image_files = [os.path.join(self.folder_path, f)
                                for f in sorted(os.listdir(self.folder_path))
                                if f.lower().endswith((".tif", ".tiff"))
                                and any(group in f for group in self.group_names)]
        elif self.analysis_mode == "rolling":
            self.image_files = [os.path.join(self.folder_path, f)
                                for f in sorted(os.listdir(self.folder_path))
                                if f.lower().endswith((".tif", ".tiff"))]
        if self.current_image_index < len(self.image_files):
            img_path = self.image_files[self.current_image_index]
            with tiff.TiffFile(img_path) as tif:
                num_frames = len(tif.pages)
                # If single-frame, force kymograph mode
                if num_frames > 1:
                    self.image_props = image_properties.get_multi_frame_properties(img_path)
                    # Set extra keys needed for binning:
                    self.image_props['step'] = self.bin_shift
                    if self.analysis_mode == "standard":
                        self.image_props['box_size'] = self.box_size
                    elif self.analysis_mode == "kymograph":
                        self.image_props['line_width'] = self.line_width
                    self.image_props['analysis_type'] = self.analysis_mode
                    image_data = image_to_np_arrays.tiff_to_np_array_multi_frame(img_path)
                else:
                    self.image_props = image_properties.get_single_frame_properties(img_path)
                    self.image_props['analysis_type'] = 'kymograph'
                    # Force switching to kymograph mode if only one frame
                    self.analysis_mode = "kymograph"
                    # For kymograph analysis, you may also need to set 'line_width'
                    self.image_props['line_width'] = self.line_width
                    self.image_props['step'] = self.bin_shift
                    image_data = image_to_np_arrays.tiff_to_np_array_single_frame(img_path)
            for layer in list(self.viewer.layers):
                if isinstance(layer, napari.layers.Image):
                    self.viewer.layers.remove(layer)
            for layer in self.viewer.layers:
                if isinstance(layer, napari.layers.Shapes):
                    layer.data = []
            image_layer = self.viewer.add_image(image_data, name=os.path.basename(img_path))
            self.viewer.reset_view()
            self.viewer.layers.move(self.viewer.layers.index(image_layer), 0)
        else:
            self.viewer.close()

    def get_image_properties(self):
        """Return the properties of the current image."""
        return self.image_props if self.image_props else {}

    def analyze(self):
        """Analyze the current image and update the results."""
        print("Analyze button clicked")
        if self.current_image_index < len(self.image_files):
            img_path = self.image_files[self.current_image_index]
            print(f"Analyzing image: {img_path}")
            
            if self.analysis_mode == "rolling":
                self.run_rolling_workflow()
            else:
                self.run_combined_workflow(self.analysis_mode)
                
            self.current_image_index += 1
            self.load_image()
        else:
            self.viewer.close()

    def run_combined_workflow(self, analysis_type):
        """Run the combined workflow for the given analysis type."""
        log_params = {
            'Pixel Size': [],
            'Frame Interval': [],
            'Errors': [],
            'Files Not Processed': []
        }
        
        # Set appropriate parameters based on analysis type
        kwargs = {
            'folder_path': self.folder_path,
            'group_names': self.group_names,
            'log_params': log_params,
            'analysis_type': analysis_type,
            'acf_peak_thresh': self.acf_peak_thresh,
            'plot_indv_ACFs': self.plot_params.get("plot_indv_acfs", False),
            'plot_indv_CCFs': self.plot_params.get("plot_indv_ccfs", False),
            'plot_indv_peaks': self.plot_params.get("plot_indv_peaks", False),
            'calc_wave_speeds': self.plot_params.get("calc_wave_speeds", False),
            'plot_summary_ACFs': self.plot_params.get("plot_summary_acfs", False),
            'plot_summary_CCFs': self.plot_params.get("plot_summary_ccfs", False),
            'plot_summary_peaks': self.plot_params.get("plot_summary_peaks", False),
            'plot_wave_speeds': False,
            'bin_shift': self.bin_shift
        }
        
        # Add appropriate parameters based on analysis mode
        if analysis_type == "standard":
            kwargs['box_size'] = self.box_size
        elif analysis_type == "kymograph":
            kwargs['line_width'] = self.line_width
        
        # Run the workflow
        results_df = combined_workflow(**kwargs)
        
        if results_df is not None and not results_df.empty:
            # Update image properties with binning parameters
            image_props = self.get_image_properties()
            image_props['step'] = self.bin_shift
            
            if analysis_type == "standard":
                image_props['box_size'] = self.box_size
            elif analysis_type == "kymograph":
                image_props['line_width'] = self.line_width
                
            # Compute num_bins using the appropriate function
            try:
                if analysis_type == "kymograph":
                    _, num_bins = image_bin_calc.create_kymo_bin_array(results_df, image_props)
                else:
                    _, num_bins, _, _ = image_bin_calc.create_multi_frame_bin_array(results_df, image_props)
                image_props['num_bins'] = num_bins
            except Exception as e:
                print(f"Error calculating num_bins: {e}")
                image_props['num_bins'] = 10  # Default value
            
            # Get pixel size and frame interval for wave speed calculations
            pixel_size = image_props.get("pixel_size", [1.0])[0]  # Default to 1.0 if not found
            frame_interval = image_props.get("frame_interval", 1.0)  # Default to 1.0 if not found
            
            try:
                # Process results
                self.results["auto_correlation"] = indv_plot_creation.plot_indv_acf_workflow(
                    results_df, image_props, self.results["auto_correlation"])
                
                self.results["cross_correlation"] = indv_plot_creation.plot_indv_ccf_workflow(
                    results_df, image_props, self.results["cross_correlation"])
                
                self.results["period"] = correlation_functions.calc_indv_period_workflow(
                    results_df, image_props, self.results["period"])
                
                self.results["peak"] = indv_plot_creation.plot_indv_peak_workflow(
                    results_df, image_props, self.results["peak"])
                
                if self.plot_params.get("calc_wave_speeds", False):
                    self.results["wave_speed"] = wave_speed.calc_wave_speeds(
                        results_df, pixel_size, frame_interval)
                    
            except Exception as e:
                print(f"Error processing results: {e}")
            
            # Update the table with the results
            self.update_post_processing_tab(results_df)
        else:
            print("No results returned from combined_workflow")

    def run_rolling_workflow(self):
        """Run the rolling workflow for the current analysis."""
        log_params = {
            'Box Size(px)': self.box_size,
            'Box Shift(px)': self.bin_shift,
            'Base Directory': self.folder_path,
            'ACF Peak Prominence': self.acf_peak_thresh,
            'Plot sub-movie ACFs': self.plot_params.get("plot_sf_acfs", False),
            'Plot movie CCFs': self.plot_params.get("plot_sf_ccfs", False),
            'Plot movie Peaks': self.plot_params.get("plot_sf_peaks", False),
            'Files Processed': [],
            'Files Not Processed': [],
            'Plotting errors': [],
            'Submovies Used': [],
            'Errors': [],
            'Frame Interval': [],
            'Pixel Size': []
        }
        
        results_df = rolling_workflow(
            folder_path=self.folder_path,
            log_params=log_params,
            box_size=self.box_size,
            box_shift=self.bin_shift,
            roll_size=self.subframe_size,
            roll_by=self.subframe_roll,
            acf_peak_thresh=self.acf_peak_thresh
        )
        
        if results_df is not None and not results_df.empty:
            image_props = self.get_image_properties()
            try:
                self.results["auto_correlation"] = indv_plot_creation.plot_indv_acf_workflow(
                    results_df, image_props, self.results["auto_correlation"])
                
                self.results["cross_correlation"] = indv_plot_creation.plot_indv_ccf_workflow(
                    results_df, image_props, self.results["cross_correlation"])
                
                self.results["period"] = correlation_functions.calc_indv_period_workflow(
                    results_df, image_props, self.results["period"])
                
                self.results["peak"] = peak_properties.calc_indv_peak_props_workflow(
                    results_df, image_props, self.results["peak"])
                
                # For rolling analysis, we need to get the number of channels
                num_channels = self.get_num_channels()
                self.results["rolling_plots"] = rolling_plot_creation.plot_rolling_summary(
                    num_channels, results_df, [])
                
            except Exception as e:
                print(f"Error processing rolling results: {e}")
            
            # Update the table with the results
            self.update_post_processing_tab(results_df)
        else:
            print("No results returned from rolling_workflow")

    def correlation(self):
        """Calculate the correlation for the current analysis."""
        print("Correlation button clicked")
        try:
            if self.analysis_mode == "rolling":
                self.run_rolling_workflow()
            else:
                self.run_combined_workflow(self.analysis_mode)
        except Exception as e:
            print(f"Error calculating correlation: {e}")

    def peak(self):
        """Detect peaks in the current analysis."""
        print("Peak detection button clicked")
        try:
            if self.analysis_mode == "rolling":
                self.run_rolling_workflow()
            else:
                self.run_combined_workflow(self.analysis_mode)
        except Exception as e:
            print(f"Error detecting peaks: {e}")

    def wave_speed(self):
        """Calculate the wave speed for the current analysis."""
        print("Wave speed button clicked")
        try:
            # Ensure wave speed calculation is enabled
            self.plot_params["calc_wave_speeds"] = True
            
            if self.analysis_mode == "rolling":
                self.run_rolling_workflow()
            else:
                self.run_combined_workflow(self.analysis_mode)
        except Exception as e:
            print(f"Error calculating wave speed: {e}")

    def update_post_processing_tab(self, results_df=None):
        """Update the post-processing tab with the results of the analysis."""
        self.output_table.clearContents()
        row_count = 0
        
        # If results_df is provided, use it to populate the table
        if results_df is not None and not results_df.empty:
            for idx, row in results_df.iterrows():
                if row_count >= self.output_table.rowCount():
                    self.output_table.setRowCount(row_count + 1)
                
                # Extract frame/image name
                if 'Image' in row:
                    self.output_table.setItem(row_count, 0, QTableWidgetItem(str(row['Image'])))
                elif 'frame' in row:
                    self.output_table.setItem(row_count, 0, QTableWidgetItem(str(row['frame'])))
                else:
                    self.output_table.setItem(row_count, 0, QTableWidgetItem(f"Frame {idx}"))
                
                # Extract wave speed if available
                if self.results["wave_speed"] is not None:
                    if isinstance(self.results["wave_speed"], pd.DataFrame):
                        if idx < len(self.results["wave_speed"]):
                            wave_speed_val = self.results["wave_speed"].iloc[idx].get('Wave Speed', 'N/A')
                            self.output_table.setItem(row_count, 1, QTableWidgetItem(str(wave_speed_val)))
                    else:
                        self.output_table.setItem(row_count, 1, QTableWidgetItem("N/A"))
                
                # Extract correlation and peak data if available
                if 'ACF_Peak_Height' in row:
                    self.output_table.setItem(row_count, 2, QTableWidgetItem(str(row['ACF_Peak_Height'])))
                if 'ACF_Peak_Pos' in row:
                    self.output_table.setItem(row_count, 3, QTableWidgetItem(str(row['ACF_Peak_Pos'])))
                
                row_count += 1
        
        # Ensure the table has at least one row
        if row_count == 0:
            self.output_table.setRowCount(1)
            for col in range(4):
                self.output_table.setItem(0, col, QTableWidgetItem("No data"))
        else:
            self.output_table.setRowCount(row_count)

    def get_num_channels(self):
        """Return the number of channels in the current image."""
        return self.image_props.get("num_channels", 1)