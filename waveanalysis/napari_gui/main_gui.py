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

        self.values_tab.image_loaded.connect(self.handle_new_image)
        self.roi_tab.roi_saved.connect(self.handle_new_roi)
        self.pre_process_tab.analyze.clicked.connect(self.run_analysis)

    def handle_new_image(self, image_path):
        """Handle new image loaded in ROI tab"""
        self.current_image_path = image_path
        self.current_image = image_to_np_arrays.tiff_to_np_array_multi_frame(image_path)
        img_props = image_properties.get_multi_frame_properties(image_path)

        self.viewer.add_image(self.current_image, name=os.path.basename(image_path))
        self.log_params.update({
            "pixel_size": img_props["pixel_size"],
            "frame_interval": img_props["frame_interval"]
        })

    def handle_new_roi(self, roi_data):
        """Handle saving a new ROI."""
        pass

    def run_analysis(self):
        """Run the analysis workflow."""
        try:
            if not self.validate_inputs():
                return

            params = self.values_tab.get_params()
            pre_params = self.pre_process_tab.get_params()
            processed = self.pre_process(self.current_image, pre_params)
            result = self.analyze(processed, params, pre_params)

            self.post_process_tab.show_results([result], params)
            self.tabs.setCurrentIndex(3)

        except Exception as e:
            import traceback
            error_message = f"Analysis Error: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Analysis Error", error_message)

    def validate_inputs(self):
        """Validate that all required inputs are provided"""
        if self.current_image is None:
            QMessageBox.warning(self, "Input Error", "No image loaded.")
            return False
        return self.validate_parameters(self.values_tab.get_params())

    def pre_process(self, image, params):
        """Apply preprocessing steps"""
        print(f"Pre-processing image of shape: {image.shape}")
        if params['smooth_window'] > 1:
            try:
                return sig.savgol_filter(
                    image,
                    window_length=params['smooth_window'],
                    polyorder=params['smooth_order']
                )
            except Exception as e:
                raise RuntimeError(f"Preprocessing failed: {e}")
        return image

    def analyze(self, image, params, pre_params):
        """Run analysis workflow"""
        analysis_type = params["type"]

        # Ensure image shape is always 5D (frames, z, channels, y, x)
        if image.ndim == 5:
            pass
        elif image.ndim == 4:
            image = image[:, np.newaxis, :, :, :]
        elif image.ndim == 3:
            image = image[:, np.newaxis, np.newaxis, :, :]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        rois = self.roi_tab.get_rois()
        if rois:
            roi = rois[0]
            bbox = roi.bounding_box
            y0, y1 = bbox[0]
            x0, x1 = bbox[1]
            image = image[:, :, :, y0:y1, x0:x1]

        num_frames = image.shape[0]
        num_channels = image.shape[2] if image.shape[2] > 0 else 1

        img_props = {
            "frame_interval": self.log_params["frame_interval"],
            "pixel_size": self.log_params["pixel_size"],
            "peak_thresh": pre_params["threshold"],
            "num_channels": num_channels,
            "num_frames": num_frames,
            "analysis_type": analysis_type,
            "channel_combos": [[0, 0]],
            "num_combos": 1
        }

        if analysis_type == "kymograph":
            img_props.update({
                "line_width": params["line_width"],
                "step": params["bin_shift"],
                "num_columns": image.shape[-1]
            })
            bin_values, num_bins = image_bin_calc.create_kymo_bin_array(image, img_props)
            img_props["num_bins"] = num_bins
        else:
            img_props.update({
                "box_size": params.get("box_size", 20),
                "step": params["bin_shift"],
                "num_columns": image.shape[-1]
            })
            bin_values, num_bins, num_x_bins, num_y_bins = image_bin_calc.create_multi_frame_bin_array(image, img_props)
            img_props["num_bins"] = num_bins

            if bin_values.shape[0] == num_frames:
                bin_values = np.transpose(bin_values, (1, 2, 0))

        if num_channels > 1:
            channel_combos = []
            for i in range(num_channels):
                for j in range(i, num_channels):
                    channel_combos.append([i, j])
            img_props["channel_combos"] = channel_combos
            img_props["num_combos"] = len(channel_combos)

        acfs = correlation_functions.calc_indv_ACF_workflow(bin_values, img_props)
        ccfs = correlation_functions.calc_indv_CCF_workflow(bin_values, img_props)
        periods = correlation_functions.calc_indv_period_workflow(acfs, img_props)
        peak_props_result = peak_properties.calc_indv_peak_props_workflow(bin_values, img_props)
        peak_widths, peak_maxs, peak_mins, peak_offsets, detailed_peak_props = peak_props_result

        result = {
            "acf": acfs,
            "ccf": ccfs,
            "period": periods,
            "peak_props": [detailed_peak_props],
            "peak_widths": peak_widths,
            "peak_maxs": peak_maxs,
            "peak_mins": peak_mins,
            "peak_offsets": peak_offsets
        }

        if analysis_type == "kymograph" and params.get("calc_wave_speeds", False):
            wave_tracks = wave_speed.define_wave_tracks(self.current_image_path)
            result["wave_speed"] = wave_speed.calc_wave_speeds(
                wave_tracks,
                self.log_params["pixel_size"],
                self.log_params["frame_interval"]
            )
        return result

    def validate_parameters(self, params):
        """Validate analysis parameters"""
        errors = self.values_tab.validate_inputs()
        if params["type"] == "kymograph" and params["line_width"] < 1:
            errors.append("Line width must be ≥1 (from create_kymo_bin_array)")

        if errors:
            QMessageBox.critical(self, "Validation Error", "\n".join(errors))
            return False
        return True