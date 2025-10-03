import os
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QLabel, QFileDialog, QMessageBox, QComboBox, QCheckBox,
    QSpinBox, QListWidget, QLineEdit, QSplitter, QProgressBar
)
from qtpy.QtCore import Signal, Qt, QThread
import numpy as np
import pandas as pd
import json
import os
import numpy as np
import pandas as pd
import json
import os

class MeasurementProcess(QThread):
    """Worker thread for processing measurements to avoid UI freezing"""
    progress = Signal(int)
    finished = Signal(list)
    error = Signal(str)
    
    def __init__(self, image_data, roi_layer, process_all_frames=False, channel_mode="First", frame_interval=1):
        super().__init__()
        self.image_data = image_data
        self.roi_layer = roi_layer
        self.process_all_frames = process_all_frames
        self.channel_mode = channel_mode
        self.frame_interval = frame_interval
    
    def run(self):
        try:
            results = []
            if self.process_all_frames and self.image_data.ndim >= 3:
                total_frames = self.image_data.shape[0]
                frame_indices = range(0, total_frames, self.frame_interval)
            else:
                frame_indices = [None]
            
            # Process each frame
            for i, frame_idx in enumerate(frame_indices):
                frame_results = self.process_frame(frame_idx)
                if frame_results:
                    results.extend(frame_results)
                self.progress.emit(int((i + 1) / len(frame_indices) * 100))
            
            if results:
                self.finished.emit(results)
            else:
                self.error.emit("No measurements were generated")
                
        except Exception as e:
            import traceback
            error_msg = f"Error in measurement worker: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)

    def process_frame(self, frame_idx):
        """Process all ROIs for a specific frame"""
        frame_results = []
        
        try:
            if frame_idx is not None and self.image_data.ndim >= 3:
                frame_data = self.image_data[frame_idx]
            else:
                frame_data = self.image_data

            if frame_data.ndim > 2:
                # Ensure we're working with the right dimensions
                if len(frame_data.shape) == 4:  # If we have (time, channel, height, width)
                    frame_data = frame_data[0]  # Take first timepoint if it exists
                
                if self.channel_mode == "Max Projection":
                    frame_data = np.max(frame_data, axis=0)
                elif self.channel_mode == "Mean Projection":
                    frame_data = np.mean(frame_data, axis=0)
                else:  # "First" or specific channel
                    frame_data = frame_data[0] if frame_data.ndim > 2 else frame_data
                
                if frame_data.ndim != 2:  # Ensure we have a 2D image
                    print(f"[MeasurementWorker] Unexpected data shape after processing: {frame_data.shape}")
                    frame_data = frame_data.squeeze()
                    if frame_data.ndim > 2:
                        frame_data = frame_data[0]
            for i, roi in enumerate(self.roi_layer.data):
                vertices = np.array(roi)
                if vertices.shape[0] > 0:
                    # Get bounding box
                    x_min, y_min = np.min(vertices, axis=0)
                    x_max, y_max = np.max(vertices, axis=0)
                    
                    # Convert to integers and ensure within image bounds
                    h, w = frame_data.shape[-2:]
                    x_min = max(0, int(x_min))
                    x_max = min(w, int(x_max))
                    y_min = max(0, int(y_min))
                    y_max = min(h, int(y_max))
                    
                    if x_min < x_max and y_min < y_max:  # Valid region
                        # Extract ROI region
                        roi_data = frame_data[y_min:y_max, x_min:x_max]
                        
                        # Create mask for non-rectangular ROIs
                        if vertices.shape != (4, 2):  # Non-rectangular ROI
                            from skimage import draw
                            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
                            roi_vertices = vertices - np.array([x_min, y_min])
                            rr, cc = draw.polygon(roi_vertices[:, 1], roi_vertices[:, 0])
                            valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                            mask[rr[valid], cc[valid]] = True
                            roi_data = roi_data[mask]
                        
                        if roi_data.size > 0:
                            result = {
                                'Frame': frame_idx if frame_idx is not None else 0,
                                'ROI_ID': i + 1,
                                'Mean_Intensity': float(np.mean(roi_data)),
                                'Max_Intensity': float(np.max(roi_data)),
                                'Min_Intensity': float(np.min(roi_data)),
                                'Std_Intensity': float(np.std(roi_data)),
                                'Area': roi_data.size,
                                'X_min': x_min,
                                'X_max': x_max,
                                'Y_min': y_min,
                                'Y_max': y_max
                            }
                            frame_results.append(result)
            
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            
        return frame_results


class ROITab(QWidget):
    """Tab for loading images and saving ROIs in Napari viewer"""
    measurements_ready = Signal(pd.DataFrame)  # Signal when ROI measurements are ready
    roi_saved = Signal(list)  # Signal when ROIs are saved

    def __init__(self, parent):
        """Initialize the ROITab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.viewer = parent.viewer
        self.current_image_path = None
        self.saved_rois = {}  # Store ROIs per image
        self.roi_layer = None  # Current ROI layer
        self.roi_list = QListWidget()  # List to show ROIs
        self.measurement_worker = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the ROITab"""
        layout = QVBoxLayout()

        # Create a splitter for better layout
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: ROI list and operations
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # ROI management group
        roi_group = QGroupBox("ROI Management")
        roi_layout = QVBoxLayout()
        
        # ROI list
        roi_layout.addWidget(QLabel("ROIs:"))
        self.roi_list.itemSelectionChanged.connect(self.on_roi_selection_changed)
        roi_layout.addWidget(self.roi_list)
        
        # ROI operations
        roi_ops_layout = QHBoxLayout()
        self.create_roi_btn = QPushButton("Create ROI Layer")
        self.create_roi_btn.clicked.connect(self.create_roi_layer)
        self.delete_roi_btn = QPushButton("Delete Selected")
        self.delete_roi_btn.clicked.connect(self.delete_selected_roi)
        self.delete_roi_btn.setEnabled(False)
        
        roi_ops_layout.addWidget(self.create_roi_btn)
        roi_ops_layout.addWidget(self.delete_roi_btn)
        roi_layout.addLayout(roi_ops_layout)
        
        # Save operations
        save_layout = QHBoxLayout()
        self.save_rois_btn = QPushButton("Save ROIs")
        self.save_rois_btn.clicked.connect(self.save_rois)
        save_layout.addWidget(self.save_rois_btn)
        roi_layout.addLayout(save_layout)
        
        roi_group.setLayout(roi_layout)
        left_layout.addWidget(roi_group)

        # Measurement options group
        measure_group = QGroupBox("Measurement Options")
        measure_layout = QVBoxLayout()
        
        # Measurement options
        # Channel options
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("Channel:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(["First", "Max Projection", "Mean Projection"])
        channel_layout.addWidget(self.channel_combo)
        measure_layout.addLayout(channel_layout)
        
        # Frame options for time series
        frame_layout = QHBoxLayout()
        self.process_all_frames = QCheckBox("Process all frames")
        self.process_all_frames.setChecked(True)
        frame_layout.addWidget(self.process_all_frames)
        
        frame_layout.addWidget(QLabel("Frame interval:"))
        self.frame_interval = QSpinBox()
        self.frame_interval.setMinimum(1)
        self.frame_interval.setMaximum(1000)
        self.frame_interval.setValue(1)
        frame_layout.addWidget(self.frame_interval)
        measure_layout.addLayout(frame_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        measure_layout.addWidget(self.progress_bar)
        
        # Measure button
        self.measure_btn = QPushButton("Calculate Measurements")
        self.measure_btn.clicked.connect(self.calculate_measurements)
        measure_layout.addWidget(self.measure_btn)
        
        measure_group.setLayout(measure_layout)
        left_layout.addWidget(measure_group)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        layout.addWidget(splitter)
        self.setLayout(layout)

    def calculate_measurements(self):
        """Calculate measurements for all ROIs"""
        if self.measurement_worker is not None and self.measurement_worker.isRunning():
            return
            
        if self.roi_layer is None or len(self.roi_layer.data) == 0:
            QMessageBox.warning(self, "No ROIs", "Please create some ROIs first.")
            return

        # First check if we have a valid image path
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first in the Values tab.")
            return

        # Get the current image data
        image_layer = None
        for layer in self.viewer.layers:
            # Look for layer with name matching our image
            if (layer.__class__.__name__ == 'Image' and 
                layer.name == os.path.basename(self.current_image_path)):
                image_layer = layer
                break

        # If not found, try getting the first image layer
        if image_layer is None:
            for layer in self.viewer.layers:
                if layer.__class__.__name__ == 'Image':
                    image_layer = layer
                    break

        if image_layer is None:
            QMessageBox.warning(self, "No Image", "No image found to measure. Please load an image in the Values tab.")
            return

        try:
            self.measure_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            self.measurement_worker = MeasurementProcess(
                image_data=image_layer.data,
                roi_layer=self.roi_layer,
                process_all_frames=self.process_all_frames.isChecked(),
                channel_mode=self.channel_combo.currentText(),
                frame_interval=self.frame_interval.value()
            )
            
            # Connect signals
            self.measurement_worker.progress.connect(self.progress_bar.setValue)
            self.measurement_worker.finished.connect(self.on_measurements_finished)
            self.measurement_worker.error.connect(self.on_measurements_error)
            
            # Start processing
            self.measurement_worker.start()
            
        except Exception as e:
            self.measure_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Failed to start measurements: {str(e)}")

    def _process_frame(self, image_data, frame_idx, rois, channel_mode):
        """Process ROIs for a specific frame"""
        frame_results = []
        
        try:
            # Extract frame data
            if frame_idx is not None and image_data.ndim >= 3:
                print(f"Extracting frame {frame_idx} from {image_data.shape} data")
                frame_data = image_data[frame_idx]
            else:
                frame_data = image_data

            # Handle multi-channel data
            if frame_data.ndim > 2:
                print(f"Processing multi-channel data with shape {frame_data.shape}")
                # Ensure we're working with the right dimensions
                if len(frame_data.shape) == 4:  # If we have (time, channel, height, width)
                    frame_data = frame_data[0]  # Take first timepoint if it exists
                
                if channel_mode == "Max Projection":
                    frame_data = np.max(frame_data, axis=0)
                elif channel_mode == "Mean Projection":
                    frame_data = np.mean(frame_data, axis=0)
                else:  # "First" or specific channel
                    frame_data = frame_data[0] if frame_data.ndim > 2 else frame_data  # Take first channel
                
                print(f"After channel processing, shape: {frame_data.shape}")
                if frame_data.ndim != 2:  # Ensure we have a 2D image
                    print(f"Warning: Unexpected data shape after processing: {frame_data.shape}")
                    frame_data = frame_data.squeeze()  # Remove any singleton dimensions
                    if frame_data.ndim > 2:
                        frame_data = frame_data[0]  # Take first slice if still multi-dimensional
                print(f"Final frame shape: {frame_data.shape}")

            # Process each ROI
            for i, roi in enumerate(rois):
                print(f"Processing ROI {i+1}")
                # Extract ROI coordinates
                vertices = np.array(roi)
                print(f"ROI shape: {vertices.shape}")
                
                # Handle different ROI shapes
                if vertices.shape[0] > 0:  # Check if ROI has any points
                    # Get bounding box
                    x_min, y_min = np.min(vertices, axis=0)
                    x_max, y_max = np.max(vertices, axis=0)
                    
                    # Convert to integers and ensure within image bounds
                    h, w = frame_data.shape[-2:]  # Get image dimensions
                    x_min = max(0, int(x_min))
                    x_max = min(w, int(x_max))
                    y_min = max(0, int(y_min))
                    y_max = min(h, int(y_max))
                    
                    if x_min < x_max and y_min < y_max:  # Valid region
                        # Extract ROI region
                        roi_data = frame_data[y_min:y_max, x_min:x_max]
                        
                        # Create mask for non-rectangular ROIs
                        if vertices.shape != (4, 2):  # Non-rectangular ROI
                            print("Creating mask for non-rectangular ROI")
                            from skimage import draw
                            mask = np.zeros((y_max - y_min, x_max - x_min), dtype=bool)
                            roi_vertices = vertices - np.array([x_min, y_min])  # Adjust vertices
                            rr, cc = draw.polygon(roi_vertices[:, 1], roi_vertices[:, 0])
                            valid = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
                            mask[rr[valid], cc[valid]] = True
                            roi_data = roi_data[mask]
                        
                        if roi_data.size > 0:  # Check if we have valid data
                            # Calculate statistics
                            result = {
                                'Frame': frame_idx if frame_idx is not None else 0,
                                'ROI_ID': i + 1,
                                'Mean_Intensity': float(np.mean(roi_data)),
                                'Max_Intensity': float(np.max(roi_data)),
                                'Min_Intensity': float(np.min(roi_data)),
                                'Std_Intensity': float(np.std(roi_data)),
                                'Area': roi_data.size,  # Use actual pixel count
                                'X_min': x_min,
                                'X_max': x_max,
                                'Y_min': y_min,
                                'Y_max': y_max
                            }
                            frame_results.append(result)
                            print(f"ROI {i+1} measurements calculated:")
                            print(f"- Shape: {roi_data.shape}")
                            print(f"- Mean: {result['Mean_Intensity']}")
                            print(f"- Min: {result['Min_Intensity']}")
                            print(f"- Max: {result['Max_Intensity']}")
                            print(f"- Std: {result['Std_Intensity']}")
                            print(f"- Area: {result['Area']}")
                    else:
                        print(f"ROI {i+1} has invalid dimensions")
                else:
                    print(f"ROI {i+1} has no points")
            
        except Exception as e:
            import traceback
            print(f"Error processing frame {frame_idx}:")
            print(traceback.format_exc())
            raise
            
        return frame_results

    def auto_save_rois(self):
        """Automatically save ROIs when changes occur"""
        if not self.current_image_path:
            return
            
        try:
            self.save_rois(auto=True)
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")

    def save_rois(self, auto=False):
        """Save ROIs to a file"""
        if self.roi_layer is None or len(self.roi_layer.data) == 0:
            if not auto:
                QMessageBox.warning(self, "No ROIs", "No ROIs to save.")
            return

        if not self.current_image_path:
            if not auto:
                QMessageBox.warning(self, "No Image", "No image loaded to associate ROIs with.")
            return

        try:
            # Create ROI_management folder if it doesn't exist
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            os.makedirs(roi_dir, exist_ok=True)

            # Create filename based on image name
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            roi_file = os.path.join(roi_dir, f"{image_name}_ROIs.json")

            # Convert ROIs to serializable format
            roi_data = []
            for roi in self.roi_layer.data:
                roi_data.append(roi.tolist())

            # Save to JSON
            data = {
                'image_path': self.current_image_path,
                'rois': roi_data,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(roi_file, 'w') as f:
                json.dump(data, f, indent=2)

            # Store in memory
            self.saved_rois[str(self.current_image_path)] = roi_data

            if not auto:
                QMessageBox.information(self, "Success", f"ROIs saved to {roi_file}")
                
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs: {str(e)}")

        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs: {str(e)}")

    def load_rois(self):
        """Load ROIs from a file"""
        try:
            if not self.current_image_path:
                QMessageBox.warning(self, "No Image", "Please load an image first.")
                return

            # Look for ROI file in ROI_management folder
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            roi_file = os.path.join(roi_dir, f"{image_name}_ROIs.json")

            if not os.path.exists(roi_file):
                QMessageBox.warning(self, "No ROIs", f"No saved ROIs found for {image_name}")
                return

            # Load ROIs from file
            with open(roi_file, 'r') as f:
                data = json.load(f)

            # Verify image path matches
            if data['image_path'] != self.current_image_path:
                response = QMessageBox.question(
                    self, "Different Image",
                    "These ROIs were saved for a different image. Load anyway?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.No:
                    return

            # Create ROI layer if it doesn't exist
            if self.roi_layer is None:
                self.create_roi_layer()

            # Convert loaded data back to numpy arrays
            rois = [np.array(roi) for roi in data['rois']]
            
            # Update ROI layer with loaded ROIs
            self.roi_layer.data = rois

            # Update the list
            self.update_roi_list()
            QMessageBox.information(self, "Success", f"Loaded {len(rois)} ROIs")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load ROIs: {str(e)}")

    def set_current_image(self, image_path):
        """Set the current image path"""
        self.current_image_path = image_path
        
        if not image_path:
            return
            
        if self.roi_layer is not None:
            try:
                self.viewer.layers.remove(self.roi_layer)
            except ValueError:
                pass
            self.roi_layer = None

    def on_roi_data_changed(self, event=None):
        """Handle changes to ROI data"""
        self.update_roi_list()
        if self.auto_save_check.isChecked() and self.current_image_path:
            self.auto_save_rois()

    def update_roi_list(self):
        """Update the ROI list widget"""
        self.roi_list.clear()
        rois = self.get_rois()
        if rois:
            for i, roi in enumerate(rois):
                self.roi_list.addItem(f"ROI_{i+1}")

    def on_roi_selection_changed(self):
        """Handle ROI selection changes"""
        selected_items = self.roi_list.selectedItems()
        self.delete_roi_btn.setEnabled(len(selected_items) > 0)
        
        # Highlight selected ROI in viewer if possible
        if hasattr(self.roi_manager, 'layer'):
            layer = self.roi_manager.layer
        elif hasattr(self.roi_manager, 'shapes_layer'):
            layer = self.roi_manager.shapes_layer
        elif hasattr(self.roi_manager, 'roi_layer'):
            layer = self.roi_manager.roi_layer
        else:
            return
            
        if selected_items:
            roi_idx = int(selected_items[0].text().split('_')[1]) - 1
            layer.selected_data = {roi_idx}
        else:
            layer.selected_data = set()

    def delete_selected_roi(self):
        """Delete the selected ROI"""
        selected_items = self.roi_list.selectedItems()
        if not selected_items:
            return
            
        roi_idx = int(selected_items[0].text().split('_')[1]) - 1
        rois = self.get_rois()
        if roi_idx < len(rois):
            rois.pop(roi_idx)
            if hasattr(self.roi_manager, 'layer'):
                self.roi_manager.layer.data = rois
            elif hasattr(self.roi_manager, 'shapes_layer'):
                self.roi_manager.shapes_layer.data = rois
            elif hasattr(self.roi_manager, 'roi_layer'):
                self.roi_manager.roi_layer.data = rois

    def _on_roi_save(self):
        """Handler for when the Save button in ROI manager is clicked"""
        rois = self.get_rois()
        if rois:
            # Save ROIs
            self.save_rois()
            # Emit our signal with the list of ROIs
            self.roi_saved.emit(rois)

    def get_rois(self):
        """Retrieve the list of ROIs from the shapes layer"""
        if self.roi_layer is not None:
            return list(self.roi_layer.data)
        return []
        
    def create_roi_layer(self):
        """Create a new ROI layer for the current image"""
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        # Look for the matching image layer
        image_layer = None
        for layer in self.viewer.layers:
            if (layer.__class__.__name__ == 'Image' and 
                layer.name == os.path.basename(self.current_image_path)):
                image_layer = layer
                break

        # If not found, try getting any image layer
        if image_layer is None:
            for layer in self.viewer.layers:
                if layer.__class__.__name__ == 'Image':
                    image_layer = layer
                    break

        if image_layer is None:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        if self.roi_layer is not None:
            self.viewer.layers.remove(self.roi_layer)
            
        self.roi_layer = self.viewer.add_shapes(
            name='ROIs',
            shape_type='rectangle',
            edge_width=2,
            edge_color='red',
            face_color='transparent'
        )
        
        self.roi_layer.events.data.connect(self.on_roi_data_changed)

        # After creating the layer, try to load existing ROIs
        try:
            # Look for ROI file
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            roi_file = os.path.join(roi_dir, f"{image_name}_ROIs.json")

            if os.path.exists(roi_file):
                # Ask user if they want to load existing ROIs
                response = QMessageBox.question(
                    self, 
                    "Existing ROIs Found", 
                    "ROIs were found for this image. Would you like to load them?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if response == QMessageBox.Yes:
                    self.load_rois()
                    QMessageBox.information(self, "ROI Layer Created", 
                                      "ROI layer created and existing ROIs loaded.")
                    return
        except Exception:
            pass  # If anything goes wrong, just create empty layer

        self.update_roi_list()
        QMessageBox.information(self, "ROI Layer Created", 
                              "ROI layer created. Use rectangle tool to draw ROIs.")
                              
    def delete_selected_roi(self):
        """Delete the selected ROI"""
        if self.roi_layer is None:
            return
            
        selected_items = self.roi_list.selectedItems()
        if not selected_items:
            return
            
        roi_idx = int(selected_items[0].text().split('_')[1]) - 1
        if roi_idx < len(self.roi_layer.data):
            data = list(self.roi_layer.data)
            data.pop(roi_idx)
            self.roi_layer.data = data
            self.update_roi_list()
            
    def update_roi_list(self):
        """Update the ROI list widget"""
        self.roi_list.clear()
        if self.roi_layer is not None:
            for i, _ in enumerate(self.roi_layer.data):
                self.roi_list.addItem(f"ROI_{i+1}")
                
    def on_roi_selection_changed(self):
        """Handle ROI selection changes"""
        selected_items = self.roi_list.selectedItems()
        self.delete_roi_btn.setEnabled(len(selected_items) > 0)
        
        if self.roi_layer is not None and selected_items:
            roi_idx = int(selected_items[0].text().split('_')[1]) - 1
            self.roi_layer.selected_data = {roi_idx}
        
    def on_measurements_finished(self, results):
        """Handle completed measurements"""
        try:
            df = pd.DataFrame(results)
            df = df.sort_values(['Frame', 'ROI_ID'])
            
            self.measure_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.progress_bar.setValue(0)
            
            self.measurements_ready.emit(df)
            
        except Exception as e:
            self.on_measurements_error(str(e))
            
    def on_measurements_error(self, error_msg):
        """Handle measurement errors"""
        self.measure_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Error during measurements: {error_msg}")

    def on_roi_data_changed(self, event=None):
        """Handle changes to ROI data"""
        self.update_roi_list()
        if hasattr(self, 'save_rois'):
            self.save_rois(auto=True)