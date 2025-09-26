import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional
import json

import napari
from napari.types import LayerDataTuple
from magicgui import magicgui
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                           QListWidget, QPushButton, QTableWidget, 
                           QTableWidgetItem, QLabel, QMessageBox, QFileDialog,
                           QCheckBox, QSpinBox, QProgressBar, QComboBox,
                           QGroupBox, QListWidget, QLineEdit, QSplitter)
from qtpy.QtCore import Qt, QThread, Signal
import tifffile

class MeasurementWorker(QThread):
    """Worker thread for processing measurements to avoid UI freezing"""
    progress = Signal(int)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, image_data, roi_layer, process_all_frames=False, channel_mode="First", process_all_channels=False):
        super().__init__()
        self.image_data = image_data
        self.roi_layer = roi_layer
        self.process_all_frames = process_all_frames
        self.channel_mode = channel_mode
        self.process_all_channels = process_all_channels
    
    def run(self):
        try:
            results = []
            
            # Determine if it's a movie (3D or 4D data)
            is_movie = self.image_data.ndim >= 3
            
            if self.process_all_channels and self.image_data.ndim >= 4:
                # Process all channels individually
                num_channels = self.image_data.shape[1] if self.image_data.ndim == 4 else 1
                total_operations = num_channels
                if self.process_all_frames:
                    total_operations *= self.image_data.shape[0]
                
                operation_count = 0
                
                for channel in range(num_channels):
                    channel_results = []
                    
                    if self.process_all_frames and is_movie:
                        # Process all frames for this channel
                        num_frames = self.image_data.shape[0]
                        for frame in range(num_frames):
                            frame_results = self.process_frame(frame, channel)
                            channel_results.extend(frame_results)
                            operation_count += 1
                            self.progress.emit(int(operation_count / total_operations * 100))
                    else:
                        # Process only current/first frame for this channel
                        frame = 0 if is_movie else None
                        channel_results = self.process_frame(frame, channel)
                        operation_count += 1
                        self.progress.emit(int(operation_count / total_operations * 100))
                    
                    results.extend(channel_results)
            else:
                # Process using the selected channel mode
                if is_movie and self.process_all_frames:
                    # Process all frames in the movie
                    num_frames = self.image_data.shape[0]
                    for frame in range(num_frames):
                        frame_results = self.process_frame(frame)
                        results.extend(frame_results)
                        self.progress.emit(int((frame + 1) / num_frames * 100))
                else:
                    # Process only current frame (for 2D) or first frame (for movie)
                    frame = 0 if is_movie else None
                    results = self.process_frame(frame)
                    self.progress.emit(100)
            
            self.finished.emit(results)
            
        except Exception as e:
            self.error.emit(str(e))
    
    def extract_2d_frame(self, frame_index=None, channel_index=None):
        """Extract a 2D frame from multi-dimensional data, handling channels properly"""
        if frame_index is not None and self.image_data.ndim >= 3:
            # For movies, get the specific frame
            if self.image_data.ndim == 3:  # (time, height, width)
                frame_data = self.image_data[frame_index]
            elif self.image_data.ndim == 4:  # (time, channels, height, width)
                if channel_index is not None:
                    # Use specific channel
                    frame_data = self.image_data[frame_index, channel_index]
                elif self.channel_mode == "First":
                    frame_data = self.image_data[frame_index, 0]  # First channel
                elif self.channel_mode == "Max Projection":
                    frame_data = np.max(self.image_data[frame_index], axis=0)
                elif self.channel_mode == "Mean Projection":
                    frame_data = np.mean(self.image_data[frame_index], axis=0)
                else:  # Specific channel
                    channel_idx = int(self.channel_mode) if self.channel_mode.isdigit() else 0
                    frame_data = self.image_data[frame_index, min(channel_idx, self.image_data.shape[1]-1)]
            else:  # Higher dimensions
                # Try to extract 2D data by taking first elements of extra dimensions
                frame_data = self.image_data[frame_index]
                while frame_data.ndim > 2:
                    frame_data = frame_data[0]  # Take first element of leading dimension
        else:
            # For 2D images, use the entire data
            frame_data = self.image_data
        
        # Ensure we have 2D data for processing
        if frame_data.ndim > 2:
            # For multi-channel data, apply channel mode
            if frame_data.ndim == 3 and frame_data.shape[0] in [1, 2, 3, 4]:  # Likely channels
                if channel_index is not None and channel_index < frame_data.shape[0]:
                    frame_data = frame_data[channel_index]
                elif self.channel_mode == "First":
                    frame_data = frame_data[0]
                elif self.channel_mode == "Max Projection":
                    frame_data = np.max(frame_data, axis=0)
                elif self.channel_mode == "Mean Projection":
                    frame_data = np.mean(frame_data, axis=0)
                else:  # Specific channel
                    channel_idx = int(self.channel_mode) if self.channel_mode.isdigit() else 0
                    frame_data = frame_data[min(channel_idx, frame_data.shape[0]-1)]
            else:
                # For other multi-dimensional data, take max projection
                frame_data = np.max(frame_data, axis=0)
        
        return frame_data
    
    def process_frame(self, frame_index=None, channel_index=None):
        """Process ROIs for a specific frame and channel"""
        frame_results = []
        
        # Extract 2D frame data
        frame_data = self.extract_2d_frame(frame_index, channel_index)
        
        # Ensure we have 2D data
        if frame_data.ndim != 2:
            print(f"Warning: Frame data has {frame_data.ndim} dimensions, expected 2. Shape: {frame_data.shape}")
            # Try to flatten to 2D
            if frame_data.ndim > 2:
                frame_data = frame_data.reshape(frame_data.shape[-2], frame_data.shape[-1])
            else:
                return frame_results
        
        for i, shape in enumerate(self.roi_layer.data):
            try:
                # Convert to numpy array
                vertices = np.array(shape)
                
                # Process rectangle shapes
                if vertices.shape == (4, 2):  # Rectangle with 4 corners
                    # Get bounding box coordinates (napari uses y, x)
                    y_coords = vertices[:, 0]
                    x_coords = vertices[:, 1]
                    
                    y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
                    x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
                    
                    # Ensure within image bounds
                    y_min = max(0, y_min)
                    x_min = max(0, x_min)
                    y_max = min(frame_data.shape[0], y_max)
                    x_max = min(frame_data.shape[1], x_max)
                    
                    if y_min < y_max and x_min < x_max:
                        # Extract ROI region
                        roi_region = frame_data[y_min:y_max, x_min:x_max]
                        
                        # Calculate metrics
                        max_density = np.max(roi_region)
                        mean_density = np.mean(roi_region)
                        min_density = np.min(roi_region)
                        std_density = np.std(roi_region)
                        area = (y_max - y_min) * (x_max - x_min)
                        
                        # Determine channel info
                        if channel_index is not None:
                            channel_info = f"Channel {channel_index}"
                        elif self.process_all_channels:
                            channel_info = "All Channels"
                        else:
                            channel_info = self.channel_mode
                        
                        frame_results.append({
                            'frame': frame_index if frame_index is not None else 0,
                            'channel': channel_index if channel_index is not None else -1,
                            'image': self.roi_layer.name,
                            'roi_id': i,
                            'shape_type': 'rectangle',
                            'max_density': max_density,
                            'mean_density': mean_density,
                            'min_density': min_density,
                            'std_density': std_density,
                            'area': area,
                            'channel_mode': channel_info
                        })
                
            except Exception as e:
                print(f"Error processing ROI {i} in frame {frame_index}, channel {channel_index}: {str(e)}")
                continue
        
        return frame_results

class ROIManagerWidget(QWidget):
    """Main widget with three tabs: Load, ROI Manager, and Measurements"""
    
    def __init__(self, viewer: napari.Viewer):
        super().__init__()
        self.viewer = viewer
        self.image_files = []
        self.current_image_path = None
        self.roi_data = {}  # Store ROI data per image
        self.measurement_worker = None
        self.current_roi_layer = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface with three tabs"""
        layout = QVBoxLayout()
        
        # Create tab widget
        self.tabs = QTabWidget()
        
        # Create tabs
        self.load_tab = self.create_load_tab()
        self.roi_tab = self.create_roi_manager_tab()  # Updated to use enhanced ROI manager
        self.measure_tab = self.create_measurements_tab()
        
        # Add tabs to widget
        self.tabs.addTab(self.load_tab, "Load")
        self.tabs.addTab(self.roi_tab, "ROI Manager")
        self.tabs.addTab(self.measure_tab, "Measurements")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
        
    def create_load_tab(self):
        """Create the Load tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Image list
        self.image_list = QListWidget()
        layout.addWidget(QLabel("Loaded Images/Movies:"))
        layout.addWidget(self.image_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Image/Movie")
        self.load_btn.clicked.connect(self.load_image)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_image)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.remove_btn)
        layout.addLayout(button_layout)
        
        # Movie info label
        self.movie_info_label = QLabel("Supported formats: TIFF, PNG, JPG, LSM (2D movies as TIFF stacks)")
        self.movie_info_label.setWordWrap(True)
        layout.addWidget(self.movie_info_label)
        
        widget.setLayout(layout)
        return widget
    
    def create_roi_manager_tab(self):
        """Create an enhanced ROI Manager tab with napari-roi-manager functionality"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Create a splitter for better layout
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: ROI list and operations
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        
        # ROI list group
        roi_list_group = QGroupBox("ROI List")
        roi_list_layout = QVBoxLayout()
        
        self.roi_list_widget = QListWidget()
        self.roi_list_widget.itemSelectionChanged.connect(self.on_roi_selection_changed)
        roi_list_layout.addWidget(self.roi_list_widget)
        
        # ROI operations
        roi_ops_layout = QHBoxLayout()
        self.add_roi_btn = QPushButton("Add ROI")
        self.add_roi_btn.clicked.connect(self.add_roi)
        self.delete_roi_btn = QPushButton("Delete Selected")
        self.delete_roi_btn.clicked.connect(self.delete_selected_roi)
        self.clear_rois_btn = QPushButton("Clear All")
        self.clear_rois_btn.clicked.connect(self.clear_all_rois)
        
        roi_ops_layout.addWidget(self.add_roi_btn)
        roi_ops_layout.addWidget(self.delete_roi_btn)
        roi_ops_layout.addWidget(self.clear_rois_btn)
        roi_list_layout.addLayout(roi_ops_layout)
        
        roi_list_group.setLayout(roi_list_layout)
        left_layout.addWidget(roi_list_group)
        
        # ROI properties group
        props_group = QGroupBox("ROI Properties")
        props_layout = QVBoxLayout()
        
        props_layout.addWidget(QLabel("ROI Name:"))
        self.roi_name_edit = QLineEdit()
        self.roi_name_edit.textChanged.connect(self.update_roi_name)
        props_layout.addWidget(self.roi_name_edit)
        
        props_group.setLayout(props_layout)
        left_layout.addWidget(props_group)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # Right panel: Import/Export operations
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        
        # Import/Export group
        io_group = QGroupBox("Import/Export")
        io_layout = QVBoxLayout()
        
        # Save/Load operations
        save_load_layout = QHBoxLayout()
        self.save_rois_btn = QPushButton("Save ROIs")
        self.save_rois_btn.clicked.connect(self.save_rois)
        self.load_rois_btn = QPushButton("Load ROIs")
        self.load_rois_btn.clicked.connect(self.load_rois)
        
        save_load_layout.addWidget(self.save_rois_btn)
        save_load_layout.addWidget(self.load_rois_btn)
        io_layout.addLayout(save_load_layout)
        
        # Auto-save option
        self.auto_save_check = QCheckBox("Auto-save ROIs on changes")
        self.auto_save_check.setChecked(True)
        io_layout.addWidget(self.auto_save_check)
        
        io_group.setLayout(io_layout)
        right_layout.addWidget(io_group)
        
        # ROI layer management
        layer_group = QGroupBox("ROI Layer Management")
        layer_layout = QVBoxLayout()
        
        layer_ops_layout = QHBoxLayout()
        self.create_layer_btn = QPushButton("Create ROI Layer")
        self.create_layer_btn.clicked.connect(self.create_roi_layer)
        self.link_layer_btn = QPushButton("Link to Existing")
        self.link_layer_btn.clicked.connect(self.link_to_existing_layer)
        
        layer_ops_layout.addWidget(self.create_layer_btn)
        layer_ops_layout.addWidget(self.link_layer_btn)
        layer_layout.addLayout(layer_ops_layout)
        
        # Current layer info
        self.layer_info_label = QLabel("No ROI layer linked")
        layer_layout.addWidget(self.layer_info_label)
        
        layer_group.setLayout(layer_layout)
        right_layout.addWidget(layer_group)
        
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # Set splitter proportions
        splitter.setSizes([300, 200])
        
        layout.addWidget(splitter)
        
        # Instructions
        instructions = QLabel(
            "1. Create or link an ROI layer\n"
            "2. Use rectangle tool to draw ROIs or click 'Add ROI'\n"
            "3. Select ROIs in the list to edit properties\n"
            "4. ROIs are automatically saved when auto-save is enabled"
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        widget.setLayout(layout)
        return widget
    
    def create_measurements_tab(self):
        """Create the Measurements tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Channel processing options
        channel_layout = QVBoxLayout()
        
        # First row: Channel mode selection
        channel_mode_layout = QHBoxLayout()
        channel_mode_layout.addWidget(QLabel("Channel Processing:"))
        
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["First", "Max Projection", "Mean Projection", "Channel 0", "Channel 1", "Channel 2"])
        channel_mode_layout.addWidget(self.channel_combo)
        channel_mode_layout.addStretch()
        
        channel_layout.addLayout(channel_mode_layout)
        
        # Second row: Process all channels checkbox
        self.process_all_channels_check = QCheckBox("Process all channels individually")
        self.process_all_channels_check.setChecked(False)
        self.process_all_channels_check.stateChanged.connect(self.on_all_channels_changed)
        channel_layout.addWidget(self.process_all_channels_check)
        
        layout.addLayout(channel_layout)
        
        # Frame processing options
        frame_layout = QHBoxLayout()
        self.process_all_frames_check = QCheckBox("Process all frames (for movies)")
        self.process_all_frames_check.setChecked(True)
        frame_layout.addWidget(self.process_all_frames_check)
        
        self.frame_interval_spin = QSpinBox()
        self.frame_interval_spin.setMinimum(1)
        self.frame_interval_spin.setMaximum(1000)
        self.frame_interval_spin.setValue(1)
        self.frame_interval_spin.setToolTip("Process every N frames")
        frame_layout.addWidget(QLabel("Frame interval:"))
        frame_layout.addWidget(self.frame_interval_spin)
        
        layout.addLayout(frame_layout)
        
        # Measurement buttons
        btn_layout = QHBoxLayout()
        self.measure_btn = QPushButton("Calculate Measurements")
        self.measure_btn.clicked.connect(self.calculate_measurements)
        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        
        btn_layout.addWidget(self.measure_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(11)
        self.results_table.setHorizontalHeaderLabels([
            "Frame", "Channel", "Image", "ROI ID", "Shape Type", "Max Density", "Mean Density", 
            "Min Density", "Std Density", "Area", "Channel Mode"
        ])
        layout.addWidget(self.results_table)
        
        widget.setLayout(layout)
        return widget
    
    def on_all_channels_changed(self, state):
        """Enable/disable channel combo based on all channels checkbox"""
        if state == Qt.Checked:
            self.channel_combo.setEnabled(False)
        else:
            self.channel_combo.setEnabled(True)
    
    # Enhanced ROI Manager Methods
    def create_roi_layer(self):
        """Create a new ROI layer"""
        if not self.viewer.layers:
            QMessageBox.warning(self, "Warning", "Please load an image first")
            return
        
        # Remove existing ROI layer if it exists
        for layer in self.viewer.layers:
            if layer.name == "ROIs":
                self.viewer.layers.remove(layer)
                break
        
        # Create new shapes layer for ROIs
        self.current_roi_layer = self.viewer.add_shapes(
            name="ROIs", 
            shape_type='rectangle',
            edge_color='red', 
            face_color='red',
            opacity=0.3
        )
        
        # Connect to layer events
        self.current_roi_layer.events.data.connect(self.on_roi_layer_changed)
        self.current_roi_layer.events.name.connect(self.on_roi_layer_changed)
        
        # Enable editing
        self.current_roi_layer.mode = 'add_rectangle'
        
        self.update_layer_info()
        self.update_roi_list()
        
        QMessageBox.information(self, "ROI Layer Created", 
                              "ROI layer created. Use rectangle tool to draw ROIs.")
    
    def link_to_existing_layer(self):
        """Link to an existing ROI layer"""
        roi_layers = []
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes):
                roi_layers.append(layer)
        
        if not roi_layers:
            QMessageBox.warning(self, "Warning", "No ROI layers found in the viewer")
            return
        
        if len(roi_layers) == 1:
            self.current_roi_layer = roi_layers[0]
        else:
            # Let user choose which layer to link
            layer_names = [layer.name for layer in roi_layers]
            layer_name, ok = QInputDialog.getItem(self, "Select ROI Layer", 
                                                 "Choose ROI layer:", layer_names, 0, False)
            if ok:
                for layer in roi_layers:
                    if layer.name == layer_name:
                        self.current_roi_layer = layer
                        break
        
        if self.current_roi_layer:
            # Connect to layer events
            self.current_roi_layer.events.data.connect(self.on_roi_layer_changed)
            self.current_roi_layer.events.name.connect(self.on_roi_layer_changed)
            
            self.update_layer_info()
            self.update_roi_list()
            
            QMessageBox.information(self, "Success", f"Linked to ROI layer: {self.current_roi_layer.name}")
    
    def on_roi_layer_changed(self, event=None):
        """Handle changes to the ROI layer"""
        self.update_roi_list()
        
        # Auto-save if enabled
        if self.auto_save_check.isChecked() and self.current_image_path:
            self.auto_save_rois()
    
    def update_layer_info(self):
        """Update the layer information display"""
        if self.current_roi_layer:
            roi_count = len(self.current_roi_layer.data) if hasattr(self.current_roi_layer, 'data') else 0
            self.layer_info_label.setText(f"Linked: {self.current_roi_layer.name} ({roi_count} ROIs)")
        else:
            self.layer_info_label.setText("No ROI layer linked")
    
    def update_roi_list(self):
        """Update the ROI list widget"""
        self.roi_list_widget.clear()
        
        if self.current_roi_layer and hasattr(self.current_roi_layer, 'data'):
            for i, shape in enumerate(self.current_roi_layer.data):
                item_text = f"ROI {i}"
                # Try to get custom name if available
                if hasattr(self.current_roi_layer, 'properties') and 'name' in self.current_roi_layer.properties:
                    if i < len(self.current_roi_layer.properties['name']):
                        custom_name = self.current_roi_layer.properties['name'][i]
                        if custom_name:
                            item_text = custom_name
                
                self.roi_list_widget.addItem(item_text)
    
    def on_roi_selection_changed(self):
        """Handle ROI selection changes"""
        selected_items = self.roi_list_widget.selectedItems()
        if selected_items and self.current_roi_layer:
            index = self.roi_list_widget.row(selected_items[0])
            # Update the name editor
            if hasattr(self.current_roi_layer, 'properties') and 'name' in self.current_roi_layer.properties:
                if index < len(self.current_roi_layer.properties['name']):
                    self.roi_name_edit.setText(self.current_roi_layer.properties['name'][index])
                else:
                    self.roi_name_edit.setText(f"ROI {index}")
            else:
                self.roi_name_edit.setText(f"ROI {index}")
    
    def update_roi_name(self):
        """Update the name of the selected ROI"""
        selected_items = self.roi_list_widget.selectedItems()
        if selected_items and self.current_roi_layer:
            index = self.roi_list_widget.row(selected_items[0])
            new_name = self.roi_name_edit.text()
            
            # Ensure properties exist
            if not hasattr(self.current_roi_layer, 'properties'):
                self.current_roi_layer.properties = {}
            
            if 'name' not in self.current_roi_layer.properties:
                self.current_roi_layer.properties['name'] = [f"ROI {i}" for i in range(len(self.current_roi_layer.data))]
            
            # Update the name
            if index < len(self.current_roi_layer.properties['name']):
                self.current_roi_layer.properties['name'][index] = new_name
            
            # Update the list display
            self.update_roi_list()
            # Reselect the item
            self.roi_list_widget.setCurrentRow(index)
    
    def add_roi(self):
        """Add a new ROI at a default position"""
        if not self.current_roi_layer:
            QMessageBox.warning(self, "Warning", "Please create or link an ROI layer first")
            return
        
        # Add a default rectangle ROI (100x100 at position 50,50)
        default_roi = np.array([[50, 50], [50, 150], [150, 150], [150, 50]])
        self.current_roi_layer.add([default_roi], shape_type='rectangle')
        
        # Add default name to properties
        if not hasattr(self.current_roi_layer, 'properties'):
            self.current_roi_layer.properties = {}
        
        if 'name' not in self.current_roi_layer.properties:
            self.current_roi_layer.properties['name'] = []
        
        self.current_roi_layer.properties['name'].append(f"ROI {len(self.current_roi_layer.data) - 1}")
    
    def delete_selected_roi(self):
        """Delete the selected ROI"""
        selected_items = self.roi_list_widget.selectedItems()
        if selected_items and self.current_roi_layer:
            index = self.roi_list_widget.row(selected_items[0])
            if index < len(self.current_roi_layer.data):
                # Remove the ROI
                new_data = list(self.current_roi_layer.data)
                new_data.pop(index)
                self.current_roi_layer.data = new_data
                
                # Remove from properties
                if hasattr(self.current_roi_layer, 'properties') and 'name' in self.current_roi_layer.properties:
                    if index < len(self.current_roi_layer.properties['name']):
                        self.current_roi_layer.properties['name'].pop(index)
    
    def clear_all_rois(self):
        """Clear all ROIs"""
        if self.current_roi_layer:
            reply = QMessageBox.question(self, "Clear ROIs", 
                                       "Are you sure you want to clear all ROIs?",
                                       QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.current_roi_layer.data = []
                if hasattr(self.current_roi_layer, 'properties'):
                    self.current_roi_layer.properties = {}
    
    def auto_save_rois(self):
        """Auto-save ROIs when changes are detected"""
        if not self.current_image_path or not self.current_roi_layer:
            return
        
        try:
            # Create ROI directory
            image_path = Path(self.current_image_path)
            roi_dir = image_path.parent / f"{image_path.stem}_ROIs"
            roi_dir.mkdir(exist_ok=True)
            
            # Save ROI data
            roi_data = {
                'image_path': str(self.current_image_path),
                'rois': [],
                'properties': {}
            }
            
            for i, shape in enumerate(self.current_roi_layer.data):
                roi_data['rois'].append({
                    'id': i,
                    'vertices': shape.tolist() if hasattr(shape, 'tolist') else shape,
                    'type': 'rectangle'
                })
            
            # Save properties if available
            if hasattr(self.current_roi_layer, 'properties'):
                roi_data['properties'] = self.current_roi_layer.properties
            
            # Save as JSON
            roi_file = roi_dir / "rois_auto_save.json"
            with open(roi_file, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            print(f"Auto-saved ROIs to {roi_file}")
            
        except Exception as e:
            print(f"Error auto-saving ROIs: {str(e)}")
    
    # Existing methods (load_image, remove_image, save_rois, load_rois, calculate_measurements, etc.)
    # ... [Previous implementation of these methods remains the same]
    
    def load_image(self):
        """Load an image or movie file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image/Movie", "", 
            "Image/Movie files (*.tif *.tiff *.png *.jpg *.jpeg *.lsm)"
        )
        
        if file_path:
            try:
                # Load image/movie using tifffile (handles TIFF stacks)
                image_data = tifffile.imread(file_path)
                
                # Add to napari viewer
                image_name = Path(file_path).stem
                
                # Check if it's a movie (3D or more dimensions)
                if image_data.ndim >= 3:
                    self.viewer.add_image(image_data, name=image_name)
                    print(f"Loaded movie: {image_data.shape} (frames: {image_data.shape[0]})")
                else:
                    self.viewer.add_image(image_data, name=image_name)
                    print(f"Loaded image: {image_data.shape}")
                
                # Add to image list
                self.image_files.append(file_path)
                item_text = f"{Path(file_path).name}"
                if image_data.ndim >= 3:
                    item_text += f" ({image_data.shape[0]} frames)"
                    if image_data.ndim >= 4:
                        item_text += f", {image_data.shape[1]} channels"
                self.image_list.addItem(item_text)
                
                # Set as current image
                self.current_image_path = file_path
                
                # Try to auto-link to ROI layer if one exists
                self.auto_link_roi_layer()
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load image/movie: {str(e)}")
    
    def auto_link_roi_layer(self):
        """Try to automatically link to an ROI layer"""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Shapes) and layer.name == "ROIs":
                self.current_roi_layer = layer
                # Connect to layer events
                self.current_roi_layer.events.data.connect(self.on_roi_layer_changed)
                self.current_roi_layer.events.name.connect(self.on_roi_layer_changed)
                self.update_layer_info()
                self.update_roi_list()
                print(f"Auto-linked to ROI layer: {layer.name}")
                break
    
    def remove_image(self):
        """Remove selected image from list and viewer"""
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            # Remove from viewer
            image_name = Path(self.image_files[current_row]).stem
            for layer in self.viewer.layers:
                if layer.name == image_name:
                    self.viewer.layers.remove(layer)
                    break
            
            # Remove from list
            self.image_files.pop(current_row)
            self.image_list.takeItem(current_row)
    
    def save_rois(self):
        """Save ROIs to a subfolder of the original image folder"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        if not self.current_roi_layer or len(self.current_roi_layer.data) == 0:
            QMessageBox.warning(self, "Warning", "No ROIs to save")
            return
        
        try:
            # Create ROI directory
            image_path = Path(self.current_image_path)
            roi_dir = image_path.parent / f"{image_path.stem}_ROIs"
            roi_dir.mkdir(exist_ok=True)
            
            # Save ROI data
            roi_data = {
                'image_path': str(self.current_image_path),
                'rois': [],
                'properties': {}
            }
            
            for i, shape in enumerate(self.current_roi_layer.data):
                roi_data['rois'].append({
                    'id': i,
                    'vertices': shape.tolist() if hasattr(shape, 'tolist') else shape,
                    'type': 'rectangle'
                })
            
            # Save properties if available
            if hasattr(self.current_roi_layer, 'properties'):
                roi_data['properties'] = self.current_roi_layer.properties
            
            # Save as JSON
            roi_file = roi_dir / "rois.json"
            with open(roi_file, 'w') as f:
                json.dump(roi_data, f, indent=2)
            
            # Store in memory
            self.roi_data[image_path.stem] = roi_data
            
            QMessageBox.information(self, "Success", f"ROIs saved to {roi_file}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not save ROIs: {str(e)}")
    
    def load_rois(self):
        """Load ROIs from file"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load ROIs", "", "JSON files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    roi_data = json.load(f)
                
                # Ensure we have an ROI layer
                if not self.current_roi_layer:
                    self.create_roi_layer()
                
                # Clear existing data
                self.current_roi_layer.data = []
                if hasattr(self.current_roi_layer, 'properties'):
                    self.current_roi_layer.properties = {}
                
                # Add ROIs to layer
                for roi in roi_data['rois']:
                    self.current_roi_layer.add_rectangles([roi['vertices']])
                
                # Load properties if available
                if 'properties' in roi_data:
                    self.current_roi_layer.properties = roi_data['properties']
                
                # Store in memory
                image_name = Path(self.current_image_path).stem
                self.roi_data[image_name] = roi_data
                
                self.update_roi_list()
                
                QMessageBox.information(self, "Success", "ROIs loaded successfully")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load ROIs: {str(e)}")
    
    def calculate_measurements(self):
        """Calculate measurements for all ROIs, handling both images and movies"""
        # Use the linked ROI layer
        roi_layer = self.current_roi_layer
        
        if not roi_layer or not hasattr(roi_layer, 'data') or len(roi_layer.data) == 0:
            QMessageBox.warning(self, "Warning", "No ROI layer found or ROIs are empty")
            return
        
        # Find image layer
        image_layer = None
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and hasattr(layer, 'name') and layer.name != "ROIs" and not isinstance(layer, napari.layers.Shapes):
                image_layer = layer
                break
        
        if not image_layer:
            QMessageBox.warning(self, "Warning", "No image layer found")
            return
        
        image_data = image_layer.data
        
        # Check if it's a movie
        is_movie = image_data.ndim >= 3
        process_all_frames = self.process_all_frames_check.isChecked() and is_movie
        process_all_channels = self.process_all_channels_check.isChecked()
        
        # Get channel mode
        channel_mode = self.channel_combo.currentText()
        
        # Build status message
        message_parts = []
        if is_movie:
            num_frames = image_data.shape[0]
            if process_all_frames:
                message_parts.append(f"Processing {num_frames} frames")
            else:
                message_parts.append(f"Processing first frame of {num_frames} total frames")
        
        message_parts.append(f"with {len(roi_layer.data)} ROIs")
        
        if process_all_channels and image_data.ndim >= 4:
            num_channels = image_data.shape[1]
            message_parts.append(f"across {num_channels} channels")
        else:
            message_parts.append(f"(Channel mode: {channel_mode})")
        
        message = " ".join(message_parts) + "..."
        print(message)
        
        # Disable button during processing
        self.measure_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create and start worker thread
        self.measurement_worker = MeasurementWorker(
            image_data, roi_layer, process_all_frames, channel_mode, process_all_channels
        )
        self.measurement_worker.progress.connect(self.progress_bar.setValue)
        self.measurement_worker.finished.connect(self.on_measurements_finished)
        self.measurement_worker.error.connect(self.on_measurements_error)
        self.measurement_worker.start()
    
    def on_measurements_finished(self, results):
        """Handle completion of measurement calculations"""
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if not results:
            QMessageBox.warning(self, "Warning", "No valid ROIs could be processed")
            return
        
        # Display results in table
        self.display_results(results)
        
        # Show success message
        total_measurements = len(results)
        QMessageBox.information(self, "Success", f"Processed {total_measurements} measurements")
    
    def on_measurements_error(self, error_message):
        """Handle errors during measurement calculations"""
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "Error", f"Measurement error: {error_message}")
    
    def display_results(self, results):
        """Display measurement results in table"""
        self.results_table.setRowCount(len(results))
        
        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(result['frame'])))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(result['channel']) if result['channel'] != -1 else "N/A"))
            self.results_table.setItem(row, 2, QTableWidgetItem(result['image']))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(result['roi_id'])))
            self.results_table.setItem(row, 4, QTableWidgetItem(result['shape_type']))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{result['max_density']:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{result['mean_density']:.2f}"))
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{result['min_density']:.2f}"))
            self.results_table.setItem(row, 8, QTableWidgetItem(f"{result['std_density']:.2f}"))
            self.results_table.setItem(row, 9, QTableWidgetItem(f"{result['area']}"))
            self.results_table.setItem(row, 10, QTableWidgetItem(result['channel_mode']))
        
        # Resize columns to content
        self.results_table.resizeColumnsToContents()
    
    def clear_results(self):
        """Clear the results table"""
        self.results_table.setRowCount(0)
    
    def export_results(self):
        """Export results to CSV file"""
        if self.results_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV files (*.csv)"
        )
        
        if file_path:
            try:
                # Collect data from table
                data = []
                for row in range(self.results_table.rowCount()):
                    row_data = [
                        self.results_table.item(row, 0).text(),  # Frame
                        self.results_table.item(row, 1).text(),  # Channel
                        self.results_table.item(row, 2).text(),  # Image
                        self.results_table.item(row, 3).text(),  # ROI ID
                        self.results_table.item(row, 4).text(),  # Shape Type
                        self.results_table.item(row, 5).text(),  # Max Density
                        self.results_table.item(row, 6).text(),  # Mean Density
                        self.results_table.item(row, 7).text(),  # Min Density
                        self.results_table.item(row, 8).text(),  # Std Density
                        self.results_table.item(row, 9).text(),  # Area
                        self.results_table.item(row, 10).text()  # Channel Mode
                    ]
                    data.append(row_data)
                
                # Create DataFrame and save
                df = pd.DataFrame(data, columns=[
                    'Frame', 'Channel', 'Image', 'ROI_ID', 'Shape_Type', 'Max_Density', 
                    'Mean_Density', 'Min_Density', 'Std_Density', 'Area', 'Channel_Mode'
                ])
                df.to_csv(file_path, index=False)
                
                QMessageBox.information(self, "Success", f"Results exported to {file_path}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not export results: {str(e)}")

# Napari plugin entry point
def roi_manager_widget_factory(viewer: napari.Viewer) -> ROIManagerWidget:
    """Factory function to create the widget"""
    return ROIManagerWidget(viewer)

# For testing and direct usage
if __name__ == "__main__":
    # Create viewer and widget for testing
    viewer = napari.Viewer()
    widget = ROIManagerWidget(viewer)
    
    # Add widget to viewer
    viewer.window.add_dock_widget(widget, name="ROI Manager", area="right")
    
    # Start napari
    napari.run()