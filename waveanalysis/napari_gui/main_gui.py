import os
import datetime
import numpy as np
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
        self.log_params = self._initialize_log_params()
        
        # Install event filter to catch napari errors
        self._install_napari_error_handler()
        
        self._init_ui()
        self._connect_signals()
    
    def _install_napari_error_handler(self):
        """Install error handler to suppress harmless napari internal errors."""
        import sys
        import warnings
        
        # Suppress specific napari warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='napari')
        
        # Store original excepthook
        self._original_excepthook = sys.excepthook
        
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            """Custom exception handler to suppress certain napari errors."""
            # Check if it's the shapes layer index error we want to suppress
            if exc_type == IndexError:
                error_str = str(exc_value)
                if 'list index out of range' in error_str:
                    # Check if it's from napari shapes layer
                    if exc_traceback:
                        frame = exc_traceback.tb_frame
                        while frame:
                            if 'napari' in str(frame.f_code.co_filename) and 'shapes' in str(frame.f_code.co_filename):
                                # Suppress this error - it's a harmless napari internal issue
                                print("Suppressed harmless napari shapes layer error")
                                return
                            frame = frame.f_back if hasattr(frame, 'f_back') else None
            
            # For all other errors, use the original handler
            self._original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = custom_excepthook

    def _initialize_log_params(self):
        """Initialize logging parameters dictionary."""
        return {
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

    def _init_ui(self):
        """Initialize the user interface."""
        self.tabs = QTabWidget()
        
        # Create tabs
        self.values_tab = ValuesTab(self)
        self.roi_tab = ROITab(self)
        self.pre_process_tab = PreProcessingTab(self)
        self.post_process_tab = PostProcessingTab(self)
        
        # Wrap tabs in scroll areas
        self.values_scroll = self._create_scrollable_tab(self.values_tab)
        self.roi_scroll = self._create_scrollable_tab(self.roi_tab)
        self.pre_process_scroll = self._create_scrollable_tab(self.pre_process_tab)

        # Add tabs
        self.tabs.addTab(self.values_scroll, "Values")
        self.tabs.addTab(self.roi_scroll, "ROI")
        self.tabs.addTab(self.pre_process_scroll, "Pre Processing")
        self.tabs.addTab(self.post_process_tab, "Post Processing")

        # Set up main layout
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)
        layout.setContentsMargins(5, 5, 5, 5)
        self.setLayout(layout)
        
        self.setMinimumWidth(350)
        self.setMinimumHeight(400)

    def _connect_signals(self):
        """Connect tab signals to handlers."""
        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.values_tab.images_updated.connect(self.handle_images_updated)
        self.values_tab.analysis_type_changed.connect(self.pre_process_tab.set_analysis_type)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.roi_tab.roi_updated.connect(self.process_roi)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def _create_scrollable_tab(self, tab_widget):
        """Create a scrollable wrapper for a tab widget."""
        scroll = QScrollArea()
        scroll.setWidget(tab_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        return scroll

    def handle_new_image(self, image_path):
        """Handle new image loaded in Values tab."""
        self.current_image_path = image_path
        self.current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
        img_props = image_properties.get_multi_frame_properties(image_path)

        self._clear_previous_image_layers()
        image_layer = self.viewer.add_image(self.current_image, name=os.path.basename(image_path))
        
        self.roi_tab.set_current_image(image_path)
        self._organize_layer_order(image_layer)
        self._update_log_params(img_props, image_path)

    def _clear_previous_image_layers(self):
        """Remove previous image layers from viewer."""
        layers_to_remove = [
            layer for layer in self.viewer.layers
            if hasattr(layer, 'data') and 
               layer.__class__.__name__ == 'Image' and 
               not layer.name.startswith('ROI_')
        ]
        for layer in layers_to_remove:
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass

    def _organize_layer_order(self, image_layer):
        """Organize viewer layers to keep ROIs on top."""
        try:
            image_layer_index = self.viewer.layers.index(image_layer)
            if image_layer_index > 0:
                self.viewer.layers.move(image_layer_index, 0)
        except (ValueError, IndexError):
            pass
        
        roi_layer = self._find_roi_layer()
        if roi_layer is not None:
            self._manage_roi_layer_visibility(roi_layer, image_layer)

    def _find_roi_layer(self):
        """Find the active ROI layer in the viewer."""
        for layer in self.viewer.layers:
            if type(layer).__name__ in ['Shapes', 'RoiManagerLayer']:
                if hasattr(layer, 'name') and 'roi' in layer.name.lower():
                    return layer
        return None

    def _manage_roi_layer_visibility(self, roi_layer, image_layer):
        """Manage ROI layer visibility and selection."""
        try:
            roi_layer.visible = True
            
            if hasattr(roi_layer, 'data') and len(roi_layer.data) > 0:
                try:
                    self.viewer.layers.selection.active = roi_layer
                except (IndexError, RuntimeError):
                    pass
            else:
                try:
                    self.viewer.layers.selection.active = image_layer
                except (IndexError, RuntimeError):
                    pass
        except Exception:
            pass

    def _update_log_params(self, img_props, image_path):
        """Update log parameters with image properties."""
        self.log_params.update({
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"],
            "Base Directory": os.path.dirname(image_path),
            "Files Processed": [os.path.basename(image_path)]
        })

    def handle_images_updated(self, image_list):
        """Handle when the list of images in values_tab is updated."""
        if hasattr(self, 'post_process_tab'):
            self.post_process_tab.set_loaded_images(image_list)
        
        if hasattr(self, 'roi_tab'):
            self.roi_tab.set_loaded_images(image_list)
        
        if image_list and not self.current_image_path:
            self.handle_new_image(image_list[0])

    def handle_new_roi(self, rois):
        """Process the ROIs using the parameters."""
        self.crops = []
        for roi in rois:
            if not isinstance(roi, np.ndarray):
                roi = np.array(roi, dtype=np.float64)
            self.crops.append(roi)

    def get_active_rois(self, image_path=None):
        """Get active ROIs for analysis for a specific image."""
        if image_path:
            return self.roi_tab.get_rois_for_image(image_path)
        
        if self.crops and len(self.crops) > 0:
            return self.crops
        elif self.current_image_path:
            return self.roi_tab.get_rois_for_image(self.current_image_path)
        return None

    def run_analysis(self):
        """Run the analysis workflow for all loaded images."""
        try:
            if not self._validate_analysis_inputs():
                return

            loaded_images = self.values_tab.get_loaded_images()
            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            
            main_results_dir = self._create_results_directory(loaded_images[0])
            all_image_results, all_files_processed = self._process_all_images(
                loaded_images, params, pre_params, main_results_dir
            )

            self.log_params["Files Processed"] = all_files_processed
            self._combine_and_display_results(all_image_results, main_results_dir, loaded_images, params, pre_params)

        except Exception as e:
            self._handle_analysis_error(e)

    def _validate_analysis_inputs(self):
        """Validate analysis inputs."""
        validation_errors = self.values_tab.validate_inputs()
        if validation_errors:
            error_message = "Parameter errors:\n- " + "\n- ".join(validation_errors)
            QMessageBox.warning(self, "Input Error", error_message)
            return False
                
        loaded_images = self.values_tab.get_loaded_images()
        if not loaded_images:
            QMessageBox.warning(self, "Input Error", "No images loaded.")
            return False
        
        return True

    def _create_results_directory(self, first_image_path):
        """Create main results directory."""
        first_image_dir = os.path.dirname(first_image_path)
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
        main_results_dir = os.path.join(first_image_dir, f"0_signalProcessing-{timestamp}")
        os.makedirs(main_results_dir, exist_ok=True)
        return main_results_dir

    def _process_all_images(self, loaded_images, params, pre_params, main_results_dir):
        """Process all loaded images and return results."""
        all_image_results = []
        all_files_processed = []
        
        for image_path in loaded_images:
            image_rois = self.get_active_rois(image_path)
            
            if not self._check_roi_availability(image_path, image_rois, pre_params):
                continue
            
            try:
                image_results = self._process_single_image(
                    image_path, params, pre_params, main_results_dir, image_rois
                )
                
                if image_results is not None:
                    self._add_image_identifier(image_results, image_path)
                    all_image_results.append(image_results)
                
                all_files_processed.append(os.path.basename(image_path))
                
            except Exception as e:
                self._handle_image_processing_error(e, image_path)
        
        return all_image_results, all_files_processed

    def _check_roi_availability(self, image_path, image_rois, pre_params):
        """Check if ROIs are available when needed and prompt user."""
        analyze_roi_data = pre_params.get("analyze_roi_data", False)
        analyze_whole_image = pre_params.get("analyze_whole_image", False)
        
        if analyze_roi_data and not analyze_whole_image and not image_rois:
            reply = QMessageBox.question(
                self, 
                "No ROIs Found", 
                f"No ROIs found for {os.path.basename(image_path)}.\n\n"
                f"Do you want to continue processing this image with whole image analysis?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.No:
                return False
            else:
                pre_params["analyze_whole_image"] = True
        
        return True

    def _process_single_image(self, image_path, params, pre_params, main_results_dir, image_rois):
        """Process a single image and return results."""
        current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
        img_props = image_properties.get_multi_frame_properties(image_path)
        
        current_log_params = self._create_image_log_params(
            image_path, params, pre_params, img_props, image_rois
        )
        
        if params.get("type") == "rolling":
            return self._run_rolling_workflow_for_image(
                image_path, current_image, main_results_dir, params, pre_params, 
                current_log_params, image_rois
            )
        else:
            return self._run_combined_workflow_for_image(
                image_path, current_image, main_results_dir, params, pre_params, 
                current_log_params, image_rois
            )

    def _create_image_log_params(self, image_path, params, pre_params, img_props, image_rois):
        """Create log parameters for a specific image."""
        current_log_params = self.log_params.copy()
        num_rois = len(image_rois) if image_rois else 0
        current_log_params.update({
            "Box Size(px)": params.get("box_size"),
            "Bin Shift(px)": params.get("bin_shift"),
            "ACF Peak": pre_params.get("threshold"),
            "ROI Used": f"Yes ({num_rois} ROIs)" if image_rois else "No",
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"],
            "Base Directory": os.path.dirname(image_path),
            "Files Processed": [os.path.basename(image_path)]
        })
        return current_log_params

    def _add_image_identifier(self, image_results, image_path):
        """Add image identifier to results."""
        if isinstance(image_results, pd.DataFrame):
            image_results['Image_Name'] = os.path.splitext(os.path.basename(image_path))[0]
            image_results['Image_Path'] = image_path

    def _handle_image_processing_error(self, error, image_path):
        """Handle errors during image processing."""
        error_msg = f"Error processing {os.path.basename(image_path)}: {str(error)}"
        print(error_msg)
        self.log_params["Errors"].append(error_msg)

    def _combine_and_display_results(self, all_image_results, main_results_dir, loaded_images, params, pre_params):
        """Combine results and display in post-processing tab."""
        if all_image_results:
            try:
                self.results = pd.concat(all_image_results, ignore_index=True)
            except Exception:
                self.results = all_image_results[0] if len(all_image_results) == 1 else all_image_results

        self.post_process_tab.set_results_directory(main_results_dir)
        loaded_image_names = [os.path.splitext(os.path.basename(img))[0] for img in loaded_images]
        self.post_process_tab.set_loaded_image_names(loaded_image_names)
        self.post_process_tab.set_plot_preferences(pre_params)
        self.post_process_tab.show_results(self.results, params)
        self.tabs.setCurrentIndex(3)

    def _handle_analysis_error(self, error):
        """Handle errors during analysis."""
        import traceback
        error_message = f"Analysis Error: {str(error)}\n\n{traceback.format_exc()}"
        QMessageBox.critical(self, "Analysis Error", error_message)
        self.log_params["Errors"].append(str(error))

    def _run_rolling_workflow_for_image(self, image_path, image_data, results_dir, params, pre_params, log_params, image_rois=None):
        """Run the rolling analysis workflow for a single image."""
        active_rois = image_rois if image_rois is not None else self.get_active_rois(image_path)
        analyze_whole_image, analyze_roi_data = self._get_analysis_types(pre_params)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        all_results = []
        
        if active_rois and analyze_roi_data:
            all_results.extend(self._process_rois_rolling(
                active_rois, results_dir, image_name, params, pre_params, log_params, image_path
            ))
        
        if analyze_whole_image:
            whole_image_result = self._process_whole_image_rolling(
                results_dir, image_name, params, pre_params, log_params, image_path
            )
            all_results.append(whole_image_result)

        return self._combine_results(all_results)

    def _run_combined_workflow_for_image(self, image_path, image_data, results_dir, params, pre_params, log_params, image_rois=None):
        """Run combined workflow for a single image."""
        active_rois = image_rois if image_rois is not None else self.get_active_rois(image_path)
        analyze_whole_image, analyze_roi_data = self._get_analysis_types(pre_params)
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        all_results = []
        
        if active_rois and analyze_roi_data:
            all_results.extend(self._process_rois_combined(
                active_rois, results_dir, image_name, params, pre_params, log_params, image_path
            ))
        
        if analyze_whole_image:
            whole_image_result = self._process_whole_image_combined(
                results_dir, image_name, params, pre_params, log_params, image_path
            )
            all_results.append(whole_image_result)

        return pd.concat(all_results, ignore_index=True) if all_results else None

    def _get_analysis_types(self, pre_params):
        """Get analysis type settings."""
        analyze_whole_image = pre_params.get("analyze_whole_image", False)
        analyze_roi_data = pre_params.get("analyze_roi_data", False)
        
        if not analyze_whole_image and not analyze_roi_data:
            analyze_whole_image = True
        
        return analyze_whole_image, analyze_roi_data

    def _process_rois_rolling(self, active_rois, results_dir, image_name, params, pre_params, log_params, image_path):
        """Process ROIs using rolling workflow."""
        results = []
        for i, roi in enumerate(active_rois):
            # Ensure ROI is a float numpy array for consistency
            if not isinstance(roi, np.ndarray):
                roi = np.array(roi, dtype=np.float64)
            else:
                roi = roi.astype(np.float64)
                
            roi_subdir = os.path.join(results_dir, f"ROI_{i+1}", image_name)
            os.makedirs(roi_subdir, exist_ok=True)
            
            roi_log_params = log_params.copy()
            roi_log_params["Files Processed"] = [f"{image_name}_ROI_{i+1}"]
                
            roi_results = rolling_workflow(
                folder_path=roi_subdir,
                log_params=roi_log_params,
                box_size=params.get("box_size"),
                box_shift=params.get("bin_shift"),
                roll_size=params.get("subframe_size"),
                roll_by=params.get("subframe_shift"),
                acf_peak_thresh=pre_params.get("threshold"),
                test=False,
                roi=roi,
                image_path=image_path
            )
            results.append(roi_results)
        return results

    def _process_whole_image_rolling(self, results_dir, image_name, params, pre_params, log_params, image_path):
        """Process whole image using rolling workflow."""
        whole_image_subdir = os.path.join(results_dir, "Whole_Image", image_name)
        os.makedirs(whole_image_subdir, exist_ok=True)
        
        whole_image_log_params = log_params.copy()
        whole_image_log_params["Files Processed"] = [image_name]
        
        return rolling_workflow(
            folder_path=whole_image_subdir,
            log_params=whole_image_log_params,
            box_size=params.get("box_size"),
            box_shift=params.get("bin_shift"),
            roll_size=params.get("subframe_size"),
            roll_by=params.get("subframe_shift"),
            acf_peak_thresh=pre_params.get("threshold"),
            test=False,
            roi=None,
            image_path=image_path
        )

    def _process_rois_combined(self, active_rois, results_dir, image_name, params, pre_params, log_params, image_path):
        """Process ROIs using combined workflow."""
        results = []
        for i, roi in enumerate(active_rois):
            if not isinstance(roi, np.ndarray):
                roi = np.array(roi, dtype=np.float64)
            else:
                # Ensure it's float type for consistency
                roi = roi.astype(np.float64)
            
            print("\n" + "="*60)
            print(f"Processing ROI {i+1}/{len(active_rois)} for {image_name}")
            print(f"ROI {i+1} shape: {roi.shape}")
            print(f"ROI {i+1} bounds: Y=[{roi[:, 0].min():.1f}, {roi[:, 0].max():.1f}], X=[{roi[:, 1].min():.1f}, {roi[:, 1].max():.1f}]")
            print("="*60 + "\n")
            
            roi_log_params = log_params.copy()
            roi_log_params["Files Processed"] = [f"{image_name}_ROI_{i+1}"]
            
            roi_results = combined_workflow(
                folder_path=results_dir,
                group_names=[f"{image_name}_ROI_{i+1}"],
                log_params=roi_log_params,
                analysis_type=params.get("type", "standard"),
                acf_peak_thresh=pre_params.get("threshold"),
                plot_summary_ACFs=pre_params.get("plot_summary_acfs", True),
                plot_summary_CCFs=pre_params.get("plot_summary_ccfs", True),
                plot_summary_peaks=pre_params.get("plot_summary_peaks", True),
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
                image_path=image_path
            )
            
            if isinstance(roi_results, pd.DataFrame):
                roi_results['ROI_ID'] = f'ROI_{i+1}'
            results.append(roi_results)
            
            self._organize_roi_results(results_dir, image_name, i)
        
        return results

    def _process_whole_image_combined(self, results_dir, image_name, params, pre_params, log_params, image_path):
        """Process whole image using combined workflow."""
        whole_image_log_params = log_params.copy()
        whole_image_log_params["Files Processed"] = [image_name]
        
        whole_image_results = combined_workflow(
            folder_path=results_dir,
            group_names=[f"{image_name}"],
            log_params=whole_image_log_params,
            analysis_type=params.get("type", "standard"),
            acf_peak_thresh=pre_params.get("threshold"),
            plot_summary_ACFs=pre_params.get("plot_summary_acfs", True),
            plot_summary_CCFs=pre_params.get("plot_summary_ccfs", True),
            plot_summary_peaks=pre_params.get("plot_summary_peaks", True),
            plot_indv_ACFs=pre_params.get("plot_indv_acfs", False),
            plot_indv_CCFs=pre_params.get("plot_indv_ccfs", False),
            plot_indv_peaks=pre_params.get("plot_indv_peaks", False),
            calc_wave_speeds=params.get("calc_wave_speeds", False),
            plot_wave_speeds=params.get("calc_wave_speeds", False),
            box_size=params.get("box_size"),
            bin_shift=params.get("bin_shift"),
            line_width=params.get("line_width"),
            roi=None,
            test=False,
            image_path=image_path
        )
        
        if isinstance(whole_image_results, pd.DataFrame):
            whole_image_results['ROI_ID'] = 'Whole_Image'
        
        self._organize_whole_image_results(results_dir, image_name)
        
        return whole_image_results

    def _organize_roi_results(self, results_dir, image_name, roi_index):
        """Organize ROI results into subdirectories."""
        image_results_subdir = os.path.join(results_dir, image_name)
        if os.path.exists(image_results_subdir):
            roi_subdir = os.path.join(results_dir, f"ROI_{roi_index+1}")
            os.makedirs(roi_subdir, exist_ok=True)
            
            import shutil
            dest_dir = os.path.join(roi_subdir, image_name)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.move(image_results_subdir, dest_dir)

    def _organize_whole_image_results(self, results_dir, image_name):
        """Organize whole image results into subdirectory."""
        image_results_subdir = os.path.join(results_dir, image_name)
        if os.path.exists(image_results_subdir):
            whole_image_subdir = os.path.join(results_dir, "Whole_Image")
            os.makedirs(whole_image_subdir, exist_ok=True)
            
            import shutil
            dest_dir = os.path.join(whole_image_subdir, image_name)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            shutil.move(image_results_subdir, dest_dir)

    def _combine_results(self, all_results):
        """Combine multiple results into single output."""
        if not all_results:
            return None
        return all_results[0] if len(all_results) == 1 else pd.concat(
            all_results, keys=[f'ROI_{i+1}' for i in range(len(all_results))]
        )

    def update_params(self, params):
        """Update parameters from ValuesTab."""
        if "group_names" in params:
            self.log_params["Group Names"] = params["group_names"]

    def process_roi(self, roi_image, original_layer):
        """Process a single ROI (legacy method for compatibility)."""
        pass
