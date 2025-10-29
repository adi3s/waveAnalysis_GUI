import os
import datetime
import numpy as np
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
        self.roi_images = []  # Store ROI image data
        self.roi_bins = {}  # Store binned data for each ROI
        
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
            "Plotting errors": [],
            "Time Elapsed": "",
            "Pixel Size": [],
            "Frame Interval": [],
            "Submovies Used": [],
            "Miscellaneous": ""
        }

        # Initialize tabs
        self.tabs = QTabWidget()
        
        # Create tabs with scroll areas for content that might be tall
        self.values_tab = ValuesTab(self)
        self.roi_tab = ROITab(self)
        self.pre_process_tab = PreProcessingTab(self)
        self.post_process_tab = PostProcessingTab(self)
        
        # Wrap tabs that might have tall content in scroll areas
        self.values_scroll = self._create_scrollable_tab(self.values_tab)
        self.roi_scroll = self._create_scrollable_tab(self.roi_tab)
        self.pre_process_scroll = self._create_scrollable_tab(self.pre_process_tab)

        # Add tabs
        self.tabs.addTab(self.values_scroll, "Values")
        self.tabs.addTab(self.roi_scroll, "ROI")
        self.tabs.addTab(self.pre_process_scroll, "Pre Processing")
        self.tabs.addTab(self.post_process_tab, "Post Processing")  # This one has its own scroll area

        # Set up main layout - simplified without nested scroll areas
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.setContentsMargins(5, 5, 5, 5)  # Small margins for better appearance
        self.setLayout(layout)
        
        # Set minimum size to ensure proper display
        self.setMinimumWidth(350)
        self.setMinimumHeight(400)

        # Connect signals
        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.values_tab.images_updated.connect(self.handle_images_updated)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.roi_tab.roi_updated.connect(self.process_roi)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def _create_scrollable_tab(self, tab_widget):
        """Create a scrollable wrapper for a tab widget"""
        from qtpy.QtWidgets import QScrollArea
        scroll = QScrollArea()
        scroll.setWidget(tab_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

        # Connect signals
        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.roi_tab.roi_updated.connect(self.process_roi)
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

    def handle_images_updated(self, image_list):
        """Handle when the list of images in values_tab is updated"""
        # Update the post_process_tab to know about all loaded images
        if hasattr(self, 'post_process_tab'):
            self.post_process_tab.set_loaded_images(image_list)
        
        # If we have images, set the first one as current if no current image exists
        if image_list and not self.current_image_path:
            self.handle_new_image(image_list[0])

    def handle_new_roi(self, rois):
        """Process the ROIs using the parameters
        
        Args:
            rois: List of numpy arrays, each array is (N, 2) with ROI vertices
        """
        # Store ROIs as numpy arrays (already converted in ROI_tab)
        self.crops = []
        for roi in rois:
            # Ensure it's a numpy array
            if not isinstance(roi, np.ndarray):
                roi = np.array(roi)
            self.crops.append(roi)

    def get_active_rois(self):
        """Get all active ROIs for analysis.
        
        Returns:
            List of numpy arrays with ROI vertices, or None if no ROIs
        """
        if self.crops and len(self.crops) > 0:
            return self.crops
        return None

    def run_analysis(self):
        """Run the analysis workflow for all loaded images."""
        try:
            validation_errors = self.values_tab.validate_inputs()
            if validation_errors:
                error_message = "Parameter errors:\n- " + "\n- ".join(validation_errors)
                QMessageBox.warning(self, "Input Error", error_message)
                return
                
            # Get all loaded images from values tab
            loaded_images = self.values_tab.get_loaded_images()
            if not loaded_images:
                QMessageBox.warning(self, "Input Error", "No images loaded.")
                return

            # Get parameters from tabs and ROIs
            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            active_rois = self.get_active_rois()
            
            # Create main results directory (same structure as before)
            first_image_dir = os.path.dirname(loaded_images[0])
            now = datetime.datetime.now()
            timestamp = now.strftime('%Y%m%d%H%M')
            main_results_dir = os.path.join(first_image_dir, f"0_signalProcessing-{timestamp}")
            os.makedirs(main_results_dir, exist_ok=True)
            
            # Process each loaded image and put results in the same directory structure
            all_image_results = []
            all_files_processed = []
            
            for image_idx, image_path in enumerate(loaded_images):
                # Load current image
                current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
                img_props = image_properties.get_multi_frame_properties(image_path)
                
                # Update log parameters for this image
                current_log_params = self.log_params.copy()
                current_log_params.update({
                    "Box Size(px)": params.get("box_size"),
                    "Bin Shift(px)": params.get("bin_shift"),
                    "ACF Peak": pre_params.get("threshold"),
                    "ROI Used": f"Yes ({len(self.crops)} ROIs)" if active_rois is not None else "No",
                    "pixel_size": img_props["pixel_size"],
                    "frame_interval": img_props["frame_interval"],
                    "Base Directory": os.path.dirname(image_path),
                    "Files Processed": [os.path.basename(image_path)]
                })
                
                # Determine which workflow to use and run it - put results in main directory
                try:
                    if params.get("type") == "rolling":
                        image_results = self.run_rolling_workflow_for_image(
                            image_path, current_image, main_results_dir, params, pre_params, current_log_params)
                    else:
                        image_results = self.run_combined_workflow_for_image(
                            image_path, current_image, main_results_dir, params, pre_params, current_log_params)
                except Exception as e:
                    error_msg = f"Error processing {os.path.basename(image_path)}: {str(e)}"
                    print(error_msg)
                    self.log_params["Errors"].append(error_msg)
                    current_log_params["Errors"].append(error_msg)
                    continue
                
                # Add image identifier to results
                if image_results is not None:
                    if isinstance(image_results, pd.DataFrame):
                        image_results['Image_Name'] = os.path.splitext(os.path.basename(image_path))[0]
                        image_results['Image_Path'] = image_path
                    all_image_results.append(image_results)
                
                all_files_processed.append(os.path.basename(image_path))

            # Update log params with all processed files
            self.log_params["Files Processed"] = all_files_processed

            # Combine all results
            if all_image_results:
                try:
                    self.results = pd.concat(all_image_results, ignore_index=True)
                except:
                    self.results = all_image_results[0] if len(all_image_results) == 1 else all_image_results

            # Set results directory and show results
            self.post_process_tab.set_results_directory(main_results_dir)
            # Pass the loaded image names (not paths) to post_process_tab
            loaded_image_names = [os.path.splitext(os.path.basename(img))[0] for img in loaded_images]
            self.post_process_tab.set_loaded_image_names(loaded_image_names)
            self.post_process_tab.show_results(self.results, params)
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            import traceback
            error_message = f"Analysis Error: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Analysis Error", error_message)
            self.log_params["Errors"].append(str(e))

    def run_rolling_workflow_for_image(self, image_path, image_data, results_dir, params, pre_params, log_params):
        """Run the rolling analysis workflow for a single image"""
        # Get active ROIs
        active_rois = self.get_active_rois()
        
        # Get analysis type selections from pre-process tab
        analyze_whole_image = pre_params.get("analyze_whole_image", True)
        analyze_roi_data = pre_params.get("analyze_roi_data", False)
        
        # Get image name for file naming
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create a list to store results from each ROI
        all_results = []
        
        # Process ROIs if user selected ROI analysis and ROIs exist
        if active_rois and analyze_roi_data:
            for i, roi in enumerate(active_rois):
                # Update log params to include image name in file naming
                roi_log_params = log_params.copy()
                roi_log_params["Files Processed"] = [f"{image_name}_ROI_{i+1}"]
                    
                roi_results = rolling_workflow(
                    folder_path=results_dir,  # Use main results directory
                    log_params=roi_log_params,
                    box_size=params.get("box_size"),
                    box_shift=params.get("bin_shift"),
                    roll_size=params.get("subframe_size"),
                    roll_by=params.get("subframe_shift"),
                    acf_peak_thresh=pre_params.get("threshold"),
                    test=False,
                    roi=roi,  # Pass the ROI to the workflow
                    image_path=image_path  # Pass current image path
                )
                all_results.append(roi_results)
        
        # Process the whole image if user selected whole image analysis
        if analyze_whole_image:
            whole_image_log_params = log_params.copy()
            whole_image_log_params["Files Processed"] = [image_name]
            
            whole_image_results = rolling_workflow(
                folder_path=results_dir,
                log_params=whole_image_log_params,
                box_size=params.get("box_size"),
                box_shift=params.get("bin_shift"),
                roll_size=params.get("subframe_size"),
                roll_by=params.get("subframe_shift"),
                acf_peak_thresh=pre_params.get("threshold"),
                test=False,
                roi=None,  # No ROI - process whole image
                image_path=image_path
            )
            all_results.append(whole_image_results)

        # Combine results from all ROIs if any
        if all_results:
            return all_results[0] if len(all_results) == 1 else pd.concat(all_results, keys=[f'ROI_{i+1}' for i in range(len(all_results))])
        return None

    def run_combined_workflow_for_image(self, image_path, image_data, results_dir, params, pre_params, log_params):
        """Run combined workflow for a single image"""
        # Get active ROIs
        active_rois = self.get_active_rois()
        
        # Get analysis type selections from pre-process tab
        analyze_whole_image = pre_params.get("analyze_whole_image", True)
        analyze_roi_data = pre_params.get("analyze_roi_data", False)
        
        # Get image name for file naming
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        # Create a list to store results from each ROI
        all_results = []
        
        # Process ROIs if user selected ROI analysis and ROIs exist
        if active_rois and analyze_roi_data:
            # Process each ROI separately but save to main results directory
            for i, roi in enumerate(active_rois):
                # Ensure ROI is a numpy array with correct shape
                if not isinstance(roi, np.ndarray):
                    roi = np.array(roi)
                
                # Update log params to include image name in file naming
                roi_log_params = log_params.copy()
                roi_log_params["Files Processed"] = [f"{image_name}_ROI_{i+1}"]
                
                # Process ROI - pass the ROI vertices to the workflow
                # The workflow will handle extracting the image data using the ROI mask
                roi_results = combined_workflow(
                    folder_path=results_dir,  # Use main results directory
                    group_names=[f"{image_name}_ROI_{i+1}"],  # Use descriptive but flat naming
                    log_params=roi_log_params,
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
                    roi=roi,  # Pass ROI vertices (numpy array) - workflow will create mask
                    test=False,
                    image_path=image_path
                )
                # Add ROI identifier to results
                if isinstance(roi_results, pd.DataFrame):
                    roi_results['ROI_ID'] = f'ROI_{i+1}'
                all_results.append(roi_results)
                
                # Move the results into ROI-specific subdirectory
                # The workflow creates a subdirectory with the image name
                image_results_subdir = os.path.join(results_dir, image_name)
                if os.path.exists(image_results_subdir):
                    # Create ROI subdirectory
                    roi_subdir = os.path.join(results_dir, f"ROI_{i+1}")
                    os.makedirs(roi_subdir, exist_ok=True)
                    
                    # Move the image results directory into ROI subdirectory
                    import shutil
                    dest_dir = os.path.join(roi_subdir, image_name)
                    if os.path.exists(dest_dir):
                        shutil.rmtree(dest_dir)
                    shutil.move(image_results_subdir, dest_dir)
        
        # Process the whole image if user selected whole image analysis
        if analyze_whole_image:
            whole_image_log_params = log_params.copy()
            whole_image_log_params["Files Processed"] = [image_name]
            
            whole_image_results = combined_workflow(
                folder_path=results_dir,
                group_names=[f"{image_name}"],  # Use image name directly
                log_params=whole_image_log_params,
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
                roi=None,  # No ROI - process whole image
                test=False,
                image_path=image_path
            )
            if isinstance(whole_image_results, pd.DataFrame):
                whole_image_results['ROI_ID'] = 'Whole_Image'
            all_results.append(whole_image_results)
            
        # Combine results from all ROIs or whole image
        if all_results:
            return pd.concat(all_results, ignore_index=True)
        return None

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

        # Get active ROIs
        active_rois = self.get_active_rois()

        # Create a list to store results from each ROI
        all_results = []
        results_dir = None
        
        if active_rois:
            # Process each ROI separately - workflow will create the main results directory
            for i, roi in enumerate(active_rois):
                # Ensure ROI is a numpy array with correct shape
                if not isinstance(roi, np.ndarray):
                    roi = np.array(roi)
                
                # Get image name without extension for the subdirectory name
                image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                
                # Process ROI - workflow will create timestamped directory and then subdirectory for each ROI
                roi_results = combined_workflow(
                    folder_path=folder_path,  # Workflow will create 0_signalProcessing-TIMESTAMP
                    group_names=[f"{image_name}_ROI_{i+1}"],  # Use descriptive name
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
                    roi=roi,  # Pass ROI vertices (numpy array) - workflow will create mask
                    test=False,
                    image_path=self.current_image_path
                )
                # Add ROI identifier to results
                roi_results['ROI_ID'] = f'ROI_{i+1}'
                all_results.append(roi_results)
                
                # Capture and organize results directory from the first ROI
                if results_dir is None:
                    # The workflow creates 0_signalProcessing-TIMESTAMP in folder_path
                    # Find the most recent one
                    signal_dirs = [d for d in os.listdir(folder_path) if d.startswith('0_signalProcessing-')]
                    if signal_dirs:
                        signal_dirs.sort(reverse=True)  # Most recent first
                        results_dir = os.path.join(folder_path, signal_dirs[0])
                
                # Move the results into ROI-specific subdirectory
                if results_dir:
                    image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                    image_results_dir = os.path.join(results_dir, image_name)
                    
                    if os.path.exists(image_results_dir):
                        # Create ROI subdirectory
                        roi_subdir = os.path.join(results_dir, f"ROI_{i+1}")
                        os.makedirs(roi_subdir, exist_ok=True)
                        
                        # Move the image results directory into ROI subdirectory
                        import shutil
                        dest_dir = os.path.join(roi_subdir, image_name)
                        if os.path.exists(dest_dir):
                            shutil.rmtree(dest_dir)
                        shutil.move(image_results_dir, dest_dir)
        else:
            # Process whole image if no ROIs
            results = combined_workflow(
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
                test=False,
                image_path=self.current_image_path
            )
            all_results.append(results)
            
            # Find the results directory
            signal_dirs = [d for d in os.listdir(folder_path) if d.startswith('0_signalProcessing-')]
            if signal_dirs:
                signal_dirs.sort(reverse=True)  # Most recent first
                results_dir = os.path.join(folder_path, signal_dirs[0])
            
        # Combine results from all ROIs or whole image
        if all_results:
            self.results = pd.concat(all_results, ignore_index=True)
        
        # Set results directory in post-processing tab
        if results_dir:
            self.post_process_tab.set_results_directory(results_dir)
        
        return results_dir

    def update_params(self, params):
        """Update parameters from ValuesTab"""
        if "group_names" in params:
            self.log_params["Group Names"] = params["group_names"]
            
    def process_roi(self, roi_image, original_layer):
        """Process a single ROI for ACF, CCF, Peak and Summary calculations"""
        # Store the ROI image
        self.roi_images.append(roi_image)
        roi_idx = len(self.roi_images) - 1
        
        # Get parameters from values tab
        params = self.values_tab.get_params()
        box_size = params.get('box_size', 50)  # Default box size
        bin_shift = params.get('bin_shift', 25)  # Default bin shift
        
        # Calculate bins for this ROI
        roi_height, roi_width = roi_image.shape
        num_bins_x = (roi_width - box_size) // bin_shift + 1
        num_bins_y = (roi_height - box_size) // bin_shift + 1
        
        roi_bins = []
        for y in range(num_bins_y):
            for x in range(num_bins_x):
                start_x = x * bin_shift
                start_y = y * bin_shift
                bin_data = roi_image[start_y:start_y+box_size, start_x:start_x+box_size]
                roi_bins.append(bin_data)
        
        self.roi_bins[roi_idx] = roi_bins
        
        # Analyze the whole ROI for summary plots
        summary_result = combined_workflow(
            folder_path=os.path.dirname(self.current_image_path),
            group_names=params.get("group_names", [""]),
            log_params=self.log_params,
            analysis_type="standard",
            acf_peak_thresh=self.pre_process_tab.threshold.value(),
            plot_summary_ACFs=True,
            plot_summary_CCFs=True,
            plot_summary_peaks=True,
            plot_indv_ACFs=False,
            plot_indv_CCFs=False,
            plot_indv_peaks=False,
            calc_wave_speeds=False,
            box_size=None,  # Use whole ROI
            bin_shift=None,  # Use whole ROI
            roi_data=roi_image,  # Pass ROI image directly
            test=False
        )
        
        # If individual plots are requested, analyze each bin
        if (self.pre_process_tab.indv_acf_checkbox.isChecked() or 
            self.pre_process_tab.indv_ccf_checkbox.isChecked() or 
            self.pre_process_tab.indv_peaks_checkbox.isChecked()):
            
            for bin_idx, bin_data in enumerate(roi_bins):
                bin_result = combined_workflow(
                    folder_path=os.path.dirname(self.current_image_path),
                    group_names=params.get("group_names", [""]),
                    log_params=self.log_params,
                    analysis_type="standard",
                    acf_peak_thresh=self.pre_process_tab.threshold.value(),
                    plot_summary_ACFs=False,
                    plot_summary_CCFs=False,
                    plot_summary_peaks=False,
                    plot_indv_ACFs=self.pre_process_tab.indv_acf_checkbox.isChecked(),
                    plot_indv_CCFs=self.pre_process_tab.indv_ccf_checkbox.isChecked(),
                    plot_indv_peaks=self.pre_process_tab.indv_peaks_checkbox.isChecked(),
                    calc_wave_speeds=False,
                    box_size=None,  # Use whole bin
                    bin_shift=None,  # Use whole bin
                    roi_data=bin_data,  # Pass bin data directly
                    bin_index=bin_idx,  # Pass bin index for naming
                    test=False
                )