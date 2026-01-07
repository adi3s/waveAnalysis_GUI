import os
import json
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QDoubleSpinBox, QSpinBox, 
    QPushButton, QGroupBox, QCheckBox, QLabel, QFormLayout,
    QHBoxLayout, QComboBox
)
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
        self.current_analysis_type = "standard"  # Track current analysis type
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the PreProcessingTab"""
        layout = QVBoxLayout()
        
        # Add Analysis Type Selection group
        analysis_type_group = QGroupBox("Analysis Type")
        analysis_type_layout = QVBoxLayout()
        
        analysis_help = QLabel("Select which data to analyze:")
        analysis_help.setStyleSheet("font-size: 10px; color: gray;")
        analysis_type_layout.addWidget(analysis_help)
        
        checkbox_layout = QHBoxLayout()
        self.whole_image_checkbox = QCheckBox("Whole Image")
        self.whole_image_checkbox.setChecked(True)
        self.whole_image_checkbox.setToolTip("Analyze the entire image")
        
        self.roi_data_checkbox = QCheckBox("ROI Data")
        self.roi_data_checkbox.setChecked(False)
        self.roi_data_checkbox.setToolTip("Analyze individual ROIs (ROIs must be created in the ROI tab)")
        self.roi_data_checkbox.stateChanged.connect(self.on_roi_data_checkbox_changed)
        
        checkbox_layout.addWidget(self.whole_image_checkbox)
        checkbox_layout.addWidget(self.roi_data_checkbox)
        analysis_type_layout.addLayout(checkbox_layout)
        
        analysis_type_group.setLayout(analysis_type_layout)
        layout.addWidget(analysis_type_group)
        
        # Add ROI Processing group
        roi_group = QGroupBox("ROI Processing")
        roi_layout = QVBoxLayout()
        
        # Channel selection for ROIs
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("ROI Channel:"))
        self.roi_channel_combo = QComboBox()
        self.roi_channel_combo.addItems(["First", "Max Projection", "Mean Projection"])
        channel_layout.addWidget(self.roi_channel_combo)
        roi_layout.addLayout(channel_layout)
        
        # Frame options for ROI time series
        roi_frame_layout = QHBoxLayout()
        self.process_all_frames = QCheckBox("Process all frames")
        self.process_all_frames.setChecked(True)
        roi_frame_layout.addWidget(self.process_all_frames)
        
        roi_frame_layout.addWidget(QLabel("Frame interval:"))
        self.frame_interval = QSpinBox()
        self.frame_interval.setMinimum(1)
        self.frame_interval.setMaximum(1000)
        self.frame_interval.setValue(1)
        roi_frame_layout.addWidget(self.frame_interval)
        roi_layout.addLayout(roi_frame_layout)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
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
        
        # Standard/Kymograph plot options
        self.acf_checkbox = QCheckBox("ACF Plots")
        self.acf_checkbox.setChecked(True)
        self.acf_checkbox.stateChanged.connect(self.on_acf_selected)
        summary_plots_layout.addWidget(self.acf_checkbox)
        
        self.ccf_checkbox = QCheckBox("CCF Plots")
        self.ccf_checkbox.setChecked(True)
        self.ccf_checkbox.stateChanged.connect(self.on_ccf_selected)
        summary_plots_layout.addWidget(self.ccf_checkbox)
        
        self.peaks_checkbox = QCheckBox("Peak Properties")
        self.peaks_checkbox.setChecked(True)
        self.peaks_checkbox.stateChanged.connect(self.on_peaks_selected)
        summary_plots_layout.addWidget(self.peaks_checkbox)
        
        # Rolling-specific plot options (initially hidden)
        self.period_checkbox = QCheckBox("Period Plots")
        self.period_checkbox.setChecked(True)
        self.period_checkbox.stateChanged.connect(self.on_acf_selected)
        self.period_checkbox.setVisible(False)
        summary_plots_layout.addWidget(self.period_checkbox)
        
        self.shift_checkbox = QCheckBox("Shift Plots")
        self.shift_checkbox.setChecked(True)
        self.shift_checkbox.stateChanged.connect(self.on_ccf_selected)
        self.shift_checkbox.setVisible(False)
        summary_plots_layout.addWidget(self.shift_checkbox)
        
        self.rolling_peaks_checkbox = QCheckBox("Peak Properties")
        self.rolling_peaks_checkbox.setChecked(True)
        self.rolling_peaks_checkbox.stateChanged.connect(self.on_peaks_selected)
        self.rolling_peaks_checkbox.setVisible(False)
        summary_plots_layout.addWidget(self.rolling_peaks_checkbox)
        
        summary_plots_group.setLayout(summary_plots_layout)
        plots_layout.addWidget(summary_plots_group)
        
        # Individual bin plots subgroup (only for Standard/Kymograph)
        self.individual_plots_group = QGroupBox("Individual Bin Plots (Optional)")
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
        
        self.individual_plots_group.setLayout(individual_plots_layout)
        plots_layout.addWidget(self.individual_plots_group)
        
        plots_group.setLayout(plots_layout)
        
        # Form layout for other controls
        form = QFormLayout()
        form.addRow("ACF Peak Threshold:", self.threshold)
        form.addRow("Smoothing Window:", self.smooth_window)
        form.addRow("Smoothing Order:", self.smooth_order)
        
        layout.addLayout(form)
        layout.addWidget(self.analyze)
        layout.addWidget(plots_group)
        
        # Set the layout directly
        self.setLayout(layout)

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
    
    def on_roi_data_checkbox_changed(self, state):
        """Handle ROI Data checkbox state change - populate ROIs when enabled"""
        if state == Qt.Checked:
            self.populate_roi_data_from_files()
    
    def populate_roi_data_from_files(self):
        """Populate ROI data for all loaded images by reading from saved ROI data files"""
        # Access the parent's values_tab to get loaded images
        if not hasattr(self.parent, 'values_tab'):
            print("Warning: Cannot access values_tab")
            return
        
        loaded_images = self.parent.values_tab.get_loaded_images()
        if not loaded_images:
            print("No images loaded")
            return
        
        # Access the ROI tab
        if not hasattr(self.parent, 'roi_tab'):
            print("Warning: Cannot access roi_tab")
            return
        
        roi_tab = self.parent.roi_tab
        
        print("\n" + "="*60)
        print("ROI DATA POPULATION REPORT")
        print("="*60)
        
        images_with_rois = 0
        images_without_rois = 0
        total_rois_loaded = 0
        
        for image_path in loaded_images:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            roi_file = self._get_roi_file_path(image_path)
            
            if not os.path.exists(roi_file):
                print(f"✗ {image_name}: No ROI data file found")
                images_without_rois += 1
                # Clear ROIs for this image if it was previously loaded
                if image_path in roi_tab.per_image_rois:
                    del roi_tab.per_image_rois[image_path]
                continue
            
            try:
                with open(roi_file, 'r') as f:
                    roi_data = json.load(f)
                
                rois = roi_data.get('rois', [])
                
                if not rois:
                    print(f"✗ {image_name}: ROI file exists but contains no ROIs")
                    images_without_rois += 1
                    continue
                
                # Populate the ROI data in roi_tab.per_image_rois
                roi_tab.per_image_rois[image_path] = []
                for roi_info in rois:
                    if 'vertices' in roi_info:
                        vertices = np.array(roi_info['vertices'], dtype=np.float64)
                        roi_tab.per_image_rois[image_path].append(vertices)
                
                num_rois = len(roi_tab.per_image_rois[image_path])
                print(f"✓ {image_name}: Loaded {num_rois} ROI(s)")
                images_with_rois += 1
                total_rois_loaded += num_rois
                
            except Exception as e:
                print(f"✗ {image_name}: Error loading ROI data - {str(e)}")
                images_without_rois += 1
        
        print("="*60)
        print(f"Summary:")
        print(f"  Total images: {len(loaded_images)}")
        print(f"  Images with ROIs: {images_with_rois}")
        print(f"  Images without ROIs: {images_without_rois}")
        print(f"  Total ROIs loaded: {total_rois_loaded}")
        print("="*60 + "\n")
        
        # Update the ROI scope label if available
        if hasattr(roi_tab, 'update_roi_scope_label'):
            roi_tab.update_roi_scope_label()
    
    def _get_roi_file_path(self, image_path):
        """Get the ROI file path for an image (matches ROI_tab implementation)"""
        roi_dir = os.path.join(os.path.dirname(image_path), 'ROI_management')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(roi_dir, f"{image_name}_ROIs.json")

    def set_analysis_type(self, analysis_type):
        """Update the UI based on the analysis type (standard, rolling, kymograph)"""
        self.current_analysis_type = analysis_type.lower()
        
        if self.current_analysis_type == "rolling":
            # Hide standard/kymograph options
            self.acf_checkbox.setVisible(False)
            self.ccf_checkbox.setVisible(False)
            self.peaks_checkbox.setVisible(False)
            self.individual_plots_group.setVisible(False)
            
            # Show rolling-specific options
            self.period_checkbox.setVisible(True)
            self.shift_checkbox.setVisible(True)
            self.rolling_peaks_checkbox.setVisible(True)
        else:
            # Show standard/kymograph options
            self.acf_checkbox.setVisible(True)
            self.ccf_checkbox.setVisible(True)
            self.peaks_checkbox.setVisible(True)
            self.individual_plots_group.setVisible(True)
            
            # Hide rolling-specific options
            self.period_checkbox.setVisible(False)
            self.shift_checkbox.setVisible(False)
            self.rolling_peaks_checkbox.setVisible(False)
    
    def get_params(self):
        """Return the current parameters for analysis"""
        # Determine which checkboxes to use based on analysis type
        if self.current_analysis_type == "rolling":
            plot_summary_acfs = self.period_checkbox.isChecked()
            plot_summary_ccfs = self.shift_checkbox.isChecked()
            plot_summary_peaks = self.rolling_peaks_checkbox.isChecked()
            # Rolling doesn't have individual plots
            plot_indv_acfs = False
            plot_indv_ccfs = False
            plot_indv_peaks = False
        else:
            plot_summary_acfs = self.acf_checkbox.isChecked()
            plot_summary_ccfs = self.ccf_checkbox.isChecked()
            plot_summary_peaks = self.peaks_checkbox.isChecked()
            plot_indv_acfs = self.indv_acf_checkbox.isChecked()
            plot_indv_ccfs = self.indv_ccf_checkbox.isChecked()
            plot_indv_peaks = self.indv_peaks_checkbox.isChecked()
        
        return {
            "threshold": self.threshold.value(),
            "smooth_window": self.smooth_window.value(),
            "smooth_order": self.smooth_order.value(),
            # Summary plot options
            "plot_summary_acfs": plot_summary_acfs,
            "plot_summary_ccfs": plot_summary_ccfs,
            "plot_summary_peaks": plot_summary_peaks,
            # Individual plot options
            "plot_indv_acfs": plot_indv_acfs,
            "plot_indv_ccfs": plot_indv_ccfs,
            "plot_indv_peaks": plot_indv_peaks,
            # ROI processing parameters
            "roi_channel": self.roi_channel_combo.currentText(),
            "process_all_frames": self.process_all_frames.isChecked(),
            "frame_interval": self.frame_interval.value(),
            # Analysis type selection
            "analyze_whole_image": self.whole_image_checkbox.isChecked(),
            "analyze_roi_data": self.roi_data_checkbox.isChecked()
        }