import os
import datetime
import napari
import pandas as pd
from qtpy.QtWidgets import QWidget, QScrollArea, QTabWidget, QVBoxLayout, QMessageBox
from qtpy.QtCore import Qt
from napari_gui.values_tab import ValuesTab
from napari_gui.ROI_tab import ROITab
from napari_gui.pre_process_tab import PreProcessingTab
from napari_gui.post_process_tab import PostProcessingTab
from waveanalysis.image_props import image_bin_calc, image_properties, image_to_np_arrays
from waveanalysis.data_workflows import combined_workflow, rolling_workflow
from waveanalysis.signal_processing import correlation_functions, peak_properties, wave_speed

class WaveAnalysisWidget(QWidget):
    """Main widget for wave analysis GUI, integrating all tabs and handling workflow"""
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_image = None
        self.current_image_path = None
        self.crops = []
        self.results = None
        
        # Create a scroll area to contain the entire widget
        self.scroll = QScrollArea()
        self.log_params = {
            "Box Size(px)": None,
            "Bin Shift(px)": None,
            "Base Directory": "",
            "ACF Peak": None,
            "CCF Peak": None,
            "Group Names": [],
            "Files Processed": [],
            "Files Not Processed": [],
            "Errors": [],
            "Time Elapsed": "",
            "Pixel Size": [],
            "Frame Interval": [],
            "Submovies Used": [],
            "Miscellaneous": ""
        }

        # Initialize tabs
        self.tabs = QTabWidget()
        self.values_tab = ValuesTab(self)
        self.roi_tab = ROITab(self)
        self.pre_process_tab = PreProcessingTab(self)
        self.post_process_tab = PostProcessingTab(self)

        # Add tabs
        self.tabs.addTab(self.values_tab, "Values")
        self.tabs.addTab(self.roi_tab, "ROI")
        self.tabs.addTab(self.pre_process_tab, "Pre Processing")
        self.tabs.addTab(self.post_process_tab, "Post Processing")

        # Create main content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.addWidget(self.tabs)
        content_widget.setLayout(content_layout)

        # Set up scroll area
        self.scroll.setWidget(content_widget)
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Main layout for the widget
        layout = QVBoxLayout()
        layout.addWidget(self.scroll)
        layout.setContentsMargins(0, 0, 0, 0)  # Remove margins to maximize space
        self.setLayout(layout)

        # Connect signals
        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.roi_tab.measurements_ready.connect(self.post_process_tab.set_roi_results)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def handle_new_image(self, image_path):
        """Handle new image loaded in Values tab"""
        self.current_image_path = image_path
        self.current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
        img_props = image_properties.get_multi_frame_properties(image_path)

        # Add image to viewer
        self.viewer.add_image(self.current_image, name=os.path.basename(image_path))
        
        # Notify the ROI tab about the new image
        self.roi_tab.set_current_image(image_path)
        
        # Update log parameters with just the selected file
        self.log_params.update({
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"],
            "Base Directory": os.path.dirname(image_path),
            "Files Processed": [os.path.basename(image_path)]
        })

    def handle_new_roi(self, rois):
        """Process the ROIs using the parameters"""
        self.crops = rois
        QMessageBox.information(self, "ROI Saved", f"ROI saved successfully. Count: {len(self.crops)}")

    def get_active_rois(self):
        """Get all active ROIs for analysis."""
        if self.crops and len(self.crops) > 0:
            return self.crops
        return None

    def run_analysis(self):
        """Run the analysis workflow."""
        try:
            validation_errors = self.values_tab.validate_inputs()
            if validation_errors:
                error_message = "Parameter errors:\n- " + "\n- ".join(validation_errors)
                QMessageBox.warning(self, "Input Error", error_message)
                return
                
            if self.current_image is None:
                QMessageBox.warning(self, "Input Error", "No image loaded.")
                return

            # Get parameters from tabs and ROIs
            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            active_rois = self.get_active_rois()
            
            # Update log parameters
            self.log_params.update({
                "Box Size(px)": params.get("box_size"),
                "Bin Shift(px)": params.get("bin_shift"),
                "ACF Peak": pre_params.get("threshold"),
                "ROI Used": f"Yes ({len(self.crops)} ROIs)" if active_rois is not None else "No"
            })
            
            # Determine which workflow to use and run it
            if params.get("type") == "rolling":
                results_dir = self.run_rolling_workflow(params, pre_params)
            else:
                results_dir = self.run_combined_workflow(params, pre_params)

            # Set results directory and show results
            self.post_process_tab.set_results_directory(results_dir)
            self.post_process_tab.show_results(self.results, params)
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            import traceback
            error_message = f"Analysis Error: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Analysis Error", error_message)
            self.log_params["Errors"].append(str(e))

    def run_rolling_workflow(self, params, pre_params):
        """Run the rolling analysis workflow"""
        folder_path = os.path.dirname(self.current_image_path)
        now = datetime.datetime.now()
        results_dir = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
        os.makedirs(results_dir, exist_ok=True)

        # Set results directory in post-processing tab
        self.post_process_tab.set_results_directory(results_dir)

        # Get active ROIs
        active_rois = self.get_active_rois()

        # Create a list to store results from each ROI
        all_results = []
        
        # Process each ROI or the whole image if no ROIs
        for i, roi in enumerate(active_rois or [None]):
            roi_results = rolling_workflow(
                folder_path=folder_path,
                log_params=self.log_params,
                box_size=params.get("box_size"),
                box_shift=params.get("bin_shift"),
                roll_size=params.get("subframe_size"),
                roll_by=params.get("subframe_shift"),
                acf_peak_thresh=pre_params.get("threshold"),
                test=False,
                roi=roi,  # Pass the ROI to the workflow
                image_path=self.current_image_path  # Pass current image path
            )
            all_results.append(roi_results)

        # Combine results from all ROIs if any
        if all_results:
            self.results = all_results[0] if len(all_results) == 1 else pd.concat(all_results, keys=[f'ROI_{i+1}' for i in range(len(all_results))])
        return results_dir

    def run_combined_workflow(self, params, pre_params):
        """Run combined workflow"""
        folder_path = os.path.dirname(self.current_image_path)
        now = datetime.datetime.now()
        results_dir = os.path.join(folder_path, f"0_signalProcessing-{now.strftime('%Y%m%d%H%M')}")
        os.makedirs(results_dir, exist_ok=True)

        # Get active ROIs
        active_rois = self.get_active_rois()

        # Set results directory in post-processing tab
        self.post_process_tab.set_results_directory(results_dir)

        # Create a list to store results from each ROI
        all_results = []
        
        # Process each ROI
        for i, roi in enumerate(active_rois or [None]):
            roi_results = combined_workflow(
                folder_path=folder_path,
                group_names=params.get("group_names", [""]),
                log_params=self.log_params,
                analysis_type=params.get("type", "standard"),
                acf_peak_thresh=pre_params.get("threshold"),
                plot_summary_ACFs=True,
                plot_summary_CCFs=True,
                plot_summary_peaks=True,
                plot_indv_ACFs=pre_params.get("plot_indv_acfs", False),
                plot_indv_CCFs=pre_params.get("plot_indv_ccfs", False),
                plot_indv_peaks=pre_params.get("plot_indv_peaks", False),
                calc_wave_speeds=params.get("calc_wave_speeds", False),
                plot_wave_speeds=params.get("calc_wave_speeds", False),
                box_size=params.get("box_size"),
                bin_shift=params.get("bin_shift"),
                line_width=params.get("line_width"),
                roi=roi,
                test=False,
                image_path=self.current_image_path  # Pass current image path
            )
            all_results.append(roi_results)
            
        # Combine results from all ROIs
        if all_results:
            self.results = all_results[0] if len(all_results) == 1 else pd.concat(all_results, keys=[f'ROI_{i+1}' for i in range(len(all_results))])
        
        return results_dir

    def update_params(self, params):
        """Update parameters from ValuesTab"""
        if "group_names" in params:
            self.log_params["Group Names"] = params["group_names"]