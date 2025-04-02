import os
import napari
import numpy as np
import tifffile as tiff
import scipy.signal as sig
from qtpy.QtWidgets import *
from napari_gui.values_tab import ValuesTab
from napari_gui.ROI_tab import ROITab
from napari_gui.pre_process_tab import PreProcessingTab
from napari_gui.post_process_tab import PostProcessingTab
from waveanalysis.image_props import image_bin_calc, image_properties, image_to_np_arrays
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
            "Time Elapsed": ""
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
        self.roi_tab.image_loaded.connect(self.handle_new_image)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def handle_new_image(self, image_path):
        """Handle new image loaded in ROI tab"""
        self.current_image_path = image_path
        if 'kymograph' in self.values_tab.get_params()["type"]:
            self.current_image = image_to_np_arrays.tiff_to_np_array_single_frame(image_path)
            img_props = image_properties.get_single_frame_properties(image_path)
        else:
            self.current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
            img_props = image_properties.get_multi_frame_properties(image_path)
        
        print(f"Loaded image shape: {self.current_image.shape}")
        self.viewer.add_image(self.current_image, name=os.path.basename(image_path))
        self.log_params.update({
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"]
        })

    def handle_new_roi(self, roi_data):
        """Store new ROI coordinates"""
        print(f"ROI data: {roi_data}")  # Debug the ROI coordinates
        self.crops.append(roi_data)
        self.process_roi(roi_data)

    def create_box(self, data):
        """Convert ROI data to numpy array and calculate bounding box"""
        data_array = np.array(data)  # Convert to numpy array
        min_val = data_array.min(axis=0)
        max_val = data_array.max(axis=0)
        tl = np.array([min_val[0], min_val[1]])  # (x_min, y_min)
        br = np.array([max_val[0], max_val[1]])  # (x_max, y_max)
        print(f"Bounding box: top-left={tl}, bottom-right={br}")
        return np.round(np.array([tl, br])).astype(int)

    def crop(self, image, rectangle):
        """Crop image using ROI coordinates (handles multi-dim images)"""
        print(f"Image shape before crop: {image.shape}")
        min_val, max_val = self.create_box(rectangle)
        
        # Extract coordinates (x=column, y=row)
        x_min, y_min = min_val[0], min_val[1]
        x_max, y_max = max_val[0], max_val[1]

        print(f"Crop coordinates: x={x_min}:{x_max}, y={y_min}:{y_max}")

        # Ensure coordinates are within image bounds
        y_min = max(0, int(y_min))
        y_max = min(image.shape[-2], int(y_max))
        x_min = max(0, int(x_min))
        x_max = min(image.shape[-1], int(x_max))

        print(f"Adjusted crop: x={x_min}:{x_max}, y={y_min}:{y_max}")
        
        # Original cropping (y, x order for numpy)
        cropped = image[..., y_min:y_max, x_min:x_max]
        print(f"Cropped shape: {cropped.shape}")
        
        # Alternative cropping to test if coordinate order is wrong
        # Uncomment to test this alternative
        # cropped_alt = image[..., x_min:x_max, y_min:y_max]
        # print(f"Alternative cropped shape: {cropped_alt.shape}")
        
        return cropped

    def process_roi(self, roi_data):
        """Process ROI, save it, and remove other layers from the viewer"""
        cropped_image = self.crop(self.current_image, roi_data)

        current_image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
        cropped_layer_name = f"{current_image_name}_crop"

        self.viewer.add_image(
            cropped_image,
            name=cropped_layer_name,
            scale=self.viewer.layers[os.path.basename(self.current_image_path)].scale  # Inherit scale
        )
        save_filename = os.path.join(self.roi_tab.save_path, f'{cropped_layer_name}.tif')
        tiff.imwrite(save_filename, cropped_image)
        print(f"Cropped image saved to: {save_filename}")

        # Remove all other layers except the cropped ROI
        for layer in list(self.viewer.layers):
            if layer.name != cropped_layer_name:
                self.viewer.layers.remove(layer)

        # Update the current image to the cropped ROI
        self.current_image = cropped_image
        self.current_image_path = save_filename
        print(f"Updated current image to cropped ROI: {cropped_layer_name}")
                
    def run_analysis(self):
        """Execute complete analysis workflow"""
        try:
            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            
            print(f"Analysis parameters: {params}")
            print(f"Pre-processing parameters: {pre_params}")
            
            if not self.validate_parameters(params):
                return

            results = []
            for idx, roi in enumerate(self.crops):
                print(f"Processing ROI {idx}...")
                cropped = self.crop(self.current_image, roi)
                processed = self.pre_process(cropped, pre_params)
                print(f"Shape after pre-processing: {processed.shape}")
                
                result = self.analyze(processed, params)
                results.append(result)
                
                # Save cropped image
                save_path = f"{self.roi_tab.save_path}/crop_{idx}.tif"
                tiff.imwrite(save_path, processed)
                print(f"Saved processed image to {save_path}")

            self.post_process_tab.show_results(results, params)
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            import traceback
            error_message = f"Analysis Error: {str(e)}\n\n{traceback.format_exc()}"
            print(error_message)
            QMessageBox.critical(self, "Analysis Error", error_message)

    def pre_process(self, image, params):
        """Apply preprocessing steps"""
        print(f"Pre-processing image of shape: {image.shape}")
        if params['smooth_window'] > 1:
            print(f"Applying Savitzky-Golay filter with window={params['smooth_window']}, order={params['smooth_order']}")
            try:
                filtered = sig.savgol_filter(
                    image, 
                    window_length=params['smooth_window'],
                    polyorder=params['smooth_order']
                )
                print(f"Filtered image shape: {filtered.shape}")
                return filtered
            except Exception as e:
                print(f"Error in pre-processing: {str(e)}")
                raise
        return image

    def analyze(self, image, params):
        """Run analysis workflow"""
        print(f"Starting analysis on image of shape: {image.shape}")
        analysis_type = params["type"]
        
        # Setup image properties based on shape and analysis type
        if analysis_type == "kymograph":
            num_channels = image.shape[0] if len(image.shape) >= 3 else 1
            num_frames = image.shape[1] if len(image.shape) >= 3 else image.shape[0]
        else:
            if len(image.shape) == 4:  # t, c, y, x
                num_channels = image.shape[1]
                num_frames = image.shape[0]
            elif len(image.shape) == 3:  # t, y, x or c, y, x
                num_channels = 1  # Assume single channel if not specified
                num_frames = image.shape[0]
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        
        img_props = {
            "num_channels": num_channels,
            "num_frames": num_frames,
            "frame_interval": self.log_params["frame_interval"],
            "pixel_size": self.log_params["pixel_size"],
            "peak_thresh": params["threshold"]
        }
        
        print(f"Image properties: {img_props}")

        # Determine parameters based on analysis type
        if analysis_type == "kymograph":
            img_props.update({
                "line_width": params["line_width"],
                "step": params["bin_shift"],
                "num_columns": image.shape[-1]  # Last dimension should be width
            })
            print(f"Kymograph analysis with line_width={params['line_width']}, step={params['bin_shift']}")
            bin_values, num_bins = image_bin_calc.create_kymo_bin_array(image, img_props)
        elif analysis_type == "rolling":
            # Use subframe parameters for rolling analysis
            img_props.update({
                "box_size": params["subframe_size"],
                "step": params["subframe_shift"]
            })
            print(f"Rolling analysis with box_size={params['subframe_size']}, step={params['subframe_shift']}")
            bin_values, num_bins, *_ = image_bin_calc.create_multi_frame_bin_array(image, img_props)
        else:  # Standard analysis
            img_props.update({
                "box_size": params["box_size"],
                "step": params["bin_shift"]
            })
            print(f"Standard analysis with box_size={params['box_size']}, step={params['bin_shift']}")
            bin_values, num_bins, *_ = image_bin_calc.create_multi_frame_bin_array(image, img_props)

        print(f"Generated {num_bins} bins")

        # Signal processing
        print("Calculating ACF...")
        acfs = correlation_functions.calc_indv_ACF_workflow(bin_values, img_props)
        print("Calculating CCF...")
        ccfs = correlation_functions.calc_indv_CCF_workflow(bin_values, img_props)
        print("Calculating periods...")
        periods = peak_properties.calc_indv_period_workflow(acfs, img_props)
        print("Calculating peak properties...")
        peak_props = peak_properties.calc_indv_peak_props_workflow(bin_values, img_props)
        
        result = {
            "acf": acfs,
            "ccf": ccfs,
            "period": periods,
            "peak_props": peak_props
        }

        if analysis_type == "kymograph" and params.get("calc_wave_speeds", False):
            print("Calculating wave speeds...")
            wave_tracks = wave_speed.define_wave_tracks(self.current_image_path)
            result["wave_speed"] = wave_speed.calc_wave_speeds(
                wave_tracks,
                self.log_params["pixel_size"],
                self.log_params["frame_interval"]
            )

        print("Analysis complete!")
        return result
    
    def update_params(self, params):
        """Update the parameters from the ValuesTab"""
        print(f"Updating parameters: {params}")
        self.log_params.update(params)

    def validate_parameters(self, params):
        """Validate analysis parameters"""
        errors = self.values_tab.validate_inputs()
        if not self.crops:
            errors.append("No ROIs saved - create at least one ROI")
        if params["type"] == "kymograph" and params["line_width"] < 1:
            errors.append("Line width must be ≥1 (from create_kymo_bin_array)")

        if errors:
            error_msg = "\n".join(errors)
            print(f"Validation errors: {error_msg}")
            QMessageBox.critical(self, "Validation Error", error_msg)
            return False
        return True