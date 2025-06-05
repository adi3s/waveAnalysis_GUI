import os
import napari
import numpy as np
from qtpy.QtWidgets import *
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
        """Initialize the WaveAnalysisWidget with the Napari viewer"""
        super().__init__()
        self.viewer = viewer
        self.current_image = None
        self.current_image_path = None
        self.crops = []
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

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        self.setLayout(layout)

        # Connect signals
        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def handle_new_image(self, image_path):
        """Handle new image loaded in Values tab"""
        self.current_image_path = image_path
        self.current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
        img_props = image_properties.get_multi_frame_properties(image_path)

        # Add image to viewer
        self.viewer.add_image(self.current_image, name=os.path.basename(image_path))
        
        # Update log parameters with image properties
        self.log_params.update({
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"],
            "Base Directory": os.path.dirname(image_path),
            "Files Processed": [os.path.basename(image_path)]
        })

    def handle_new_roi(self):
        """Handle saving a new ROI."""
        # Get ROIs from ROI tab
        self.crops = self.roi_tab.get_rois()
        if self.crops:
            QMessageBox.information(self, "ROI Saved", f"{len(self.crops)} ROI(s) saved successfully.")
            # Update log with ROI info
            self.log_params["Miscellaneous"] += f"Added {len(self.crops)} ROI(s). "

    def run_analysis(self):
        """Run the analysis workflow."""
        try:
            # Check for validation errors from ValuesTab
            validation_errors = self.values_tab.validate_inputs()
            if validation_errors:
                error_message = "Parameter errors:\n- " + "\n- ".join(validation_errors)
                QMessageBox.warning(self, "Input Error", error_message)
                return
                
            if self.current_image is None:
                QMessageBox.warning(self, "Input Error", "No image loaded.")
                return

            # Get parameters from tabs
            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            
            # Update log parameters
            self.log_params.update({
                "Box Size(px)": params.get("box_size") if "box_size" in params else params.get("line_width"),
                "Bin Shift(px)": params.get("bin_shift"),
                "ACF Peak": pre_params.get("threshold")
            })
            
            # Check if ROIs should be used
            use_rois = len(self.crops) > 0
            if use_rois:
                QMessageBox.information(self, "Analysis Info", 
                                       f"Analysis will use {len(self.crops)} ROI(s).")
            
            # Determine which workflow to use based on parameters
            if params.get("type") == "rolling":
                result = self.run_rolling_workflow(params, pre_params)
            else:
                result = self.run_combined_workflow(params, pre_params)

            # Show results and switch to post-processing tab
            self.post_process_tab.show_results([result], params)
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            import traceback
            error_message = f"Analysis Error: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Analysis Error", error_message)
            # Add error to log
            self.log_params["Errors"].append(str(e))

    def run_rolling_workflow(self, params, pre_params):
        """Run the rolling analysis workflow for time-series analysis"""
        folder_path = os.path.dirname(self.current_image_path)
        
        # Extract parameters for rolling workflow
        box_size = params.get("box_size")
        box_shift = params.get("bin_shift")
        roll_size = params.get("subframe_size")
        roll_by = params.get("subframe_shift")
        acf_peak_thresh = pre_params.get("threshold")
        
        # Create temporary log params for the workflow
        workflow_log_params = self.log_params.copy()
        
        # Run rolling workflow on the current file only
        result_df = rolling_workflow(
            folder_path=folder_path,
            log_params=workflow_log_params,
            box_size=box_size,
            box_shift=box_shift,
            roll_size=roll_size,
            roll_by=roll_by,
            acf_peak_thresh=acf_peak_thresh,
            rois=self.crops if len(self.crops) > 0 else None,
            test=False
        )
        
        # Update our log params with the results from the workflow
        self.log_params.update({k: v for k, v in workflow_log_params.items() 
                              if k in self.log_params})
        
        return result_df

    def run_combined_workflow(self, params, pre_params):
        """Run combined workflow for kymograph or standard analysis"""
        file_path = self.current_image_path
        folder_path = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        
        # Extract parameters for combined workflow
        analysis_type = params.get("type", "standard" or "kymograph")
        box_size = params.get("box_size") if analysis_type == "standard" else None
        line_width = params.get("line_width") if analysis_type == "kymograph" else None
        bin_shift = params.get("bin_shift")
        acf_peak_thresh = pre_params.get("threshold")
        calc_wave_speeds = params.get("calc_wave_speeds", False)
        
        # Plotting preferences
        plot_prefs = {
            "plot_summary_ACFs": True,
            "plot_summary_CCFs": True,
            "plot_summary_peaks": True,
            "plot_indv_ACFs": False, 
            "plot_indv_CCFs": False,
            "plot_indv_peaks": False,
            "plot_wave_speeds": calc_wave_speeds
        }
        
        # Get group names directly from parameters
        group_names = params.get("group_names", [''])
        if not group_names:
            group_names = ['']
        
        # Run combined workflow on the current file only
        workflow_log_params = self.log_params.copy()
        
        # Check if we have ROIs
        if len(self.crops) > 0:
            # Log ROI information
            workflow_log_params["Miscellaneous"] += f"Using {len(self.crops)} ROI(s) for analysis. "
            # TODO: Apply ROI pre-processing if needed
        
        # Remove the rois parameter and others that might cause issues
        result_df = combined_workflow(
            folder_path=folder_path,
            group_names=group_names,
            log_params=workflow_log_params,
            analysis_type=analysis_type,
            acf_peak_thresh=acf_peak_thresh,
            calc_wave_speeds=calc_wave_speeds,
            box_size=box_size,
            bin_shift=bin_shift,
            line_width=line_width,
            test=False,
            plot_summary_ACFs=plot_prefs["plot_summary_ACFs"],
            plot_summary_CCFs=plot_prefs["plot_summary_CCFs"],
            plot_summary_peaks=plot_prefs["plot_summary_peaks"],
            plot_indv_ACFs=plot_prefs["plot_indv_ACFs"],
            plot_indv_CCFs=plot_prefs["plot_indv_CCFs"],
            plot_indv_peaks=plot_prefs["plot_indv_peaks"],
            plot_wave_speeds=plot_prefs["plot_wave_speeds"]
        )
        
        # Update our log params with the results from the workflow
        self.log_params.update({k: v for k, v in workflow_log_params.items() 
                            if k in self.log_params})
        
        return result_df

    def update_params(self, params):
        """Update parameters from ValuesTab"""
        if "group_names" in params:
            self.log_params["Group Names"] = params["group_names"]