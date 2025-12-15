import os
import json
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QLabel, QMessageBox, QCheckBox, QScrollArea
)
from qtpy.QtCore import Signal, Qt
from qtpy.QtCore import QTimer
from napari_roi_manager import QRoiManager


class ROITab(QWidget):
    """Tab for managing ROIs in Napari viewer."""
    
    measurements_ready = Signal(pd.DataFrame)
    roi_saved = Signal(list)
    roi_updated = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.viewer = parent.viewer
        self.current_image_path = None
        self.per_image_rois = {}  # Map of {image_path: [roi_arrays]}
        self.roi_manager = None
        self.roi_manager_initialized = False
        self.roi_layer = None
        self.loaded_images = []
        self.prompted_images = set()  # Track prompted images
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.NoFrame)
        
        scroll_content = QWidget()
        layout = QVBoxLayout()
        
        # Setup section
        setup_group = QGroupBox("ROI Manager Setup")
        setup_layout = QVBoxLayout()
        
        self.setup_instructions = QLabel(
            "1. Load an image\n"
            "2. Click 'Initialize ROI Manager'\n"
            "3. Draw ROIs using napari tools\n"
            "4. ROIs are saved per image"
        )
        self.setup_instructions.setWordWrap(True)
        setup_layout.addWidget(self.setup_instructions)
        
        self.roi_scope_label = QLabel("ROI Scope: No images loaded")
        self.roi_scope_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        self.roi_scope_label.setWordWrap(True)
        setup_layout.addWidget(self.roi_scope_label)
        
        self.init_roi_btn = QPushButton("Initialize ROI Manager")
        self.init_roi_btn.clicked.connect(self.initialize_roi_manager)
        setup_layout.addWidget(self.init_roi_btn)
        
        self.status_label = QLabel("Status: Not initialized")
        setup_layout.addWidget(self.status_label)
        
        setup_group.setLayout(setup_layout)
        layout.addWidget(setup_group)
        
        # ROI Manager container
        self.roi_manager_container = QScrollArea()
        self.roi_manager_container.setWidgetResizable(True)
        self.roi_manager_container.setMaximumHeight(300)
        self.roi_manager_container.setVisible(False)
        layout.addWidget(self.roi_manager_container)
        
        # Operations group
        roi_ops_group = QGroupBox("ROI Operations")
        roi_ops_layout = QVBoxLayout()
        
        button_row = QHBoxLayout()
        self.save_rois_btn = QPushButton("Save ROIs")
        self.save_rois_btn.clicked.connect(self.save_rois)
        self.save_rois_btn.setEnabled(False)
        
        self.refresh_rois_btn = QPushButton("Refresh ROIs")
        self.refresh_rois_btn.clicked.connect(self.refresh_rois)
        self.refresh_rois_btn.setEnabled(False)
        
        button_row.addWidget(self.save_rois_btn)
        button_row.addWidget(self.refresh_rois_btn)
        roi_ops_layout.addLayout(button_row)
        
        # Add snapshot button
        snapshot_row = QHBoxLayout()
        self.save_snapshot_btn = QPushButton("Save ROI Snapshot (PNG)")
        self.save_snapshot_btn.clicked.connect(self.save_roi_snapshot)
        self.save_snapshot_btn.setEnabled(False)
        self.save_snapshot_btn.setToolTip("Save a PNG image showing ROIs with labels")
        snapshot_row.addWidget(self.save_snapshot_btn)
        roi_ops_layout.addLayout(snapshot_row)
        
        # Add crop button
        crop_row = QHBoxLayout()
        self.save_crops_btn = QPushButton("Save ROI Crops (TIF)")
        self.save_crops_btn.clicked.connect(self.save_roi_crops)
        self.save_crops_btn.setEnabled(False)
        self.save_crops_btn.setToolTip("Save cropped TIF files for each ROI with all dimensions")
        crop_row.addWidget(self.save_crops_btn)
        roi_ops_layout.addLayout(crop_row)
        
        checkbox_row = QHBoxLayout()
        self.auto_save_check = QCheckBox("Auto-save ROIs")
        self.auto_save_check.setChecked(False)
        self.auto_save_check.setEnabled(False)
        
        self.show_labels_check = QCheckBox("Show ROI Labels")
        self.show_labels_check.setChecked(False)
        self.show_labels_check.setEnabled(False)
        self.show_labels_check.stateChanged.connect(self.toggle_roi_labels)
        
        checkbox_row.addWidget(self.auto_save_check)
        checkbox_row.addWidget(self.show_labels_check)
        roi_ops_layout.addLayout(checkbox_row)
        
        roi_ops_group.setLayout(roi_ops_layout)
        layout.addWidget(roi_ops_group)
        
        # Fallback container
        self.fallback_container = QWidget()
        fallback_layout = QVBoxLayout()
        fallback_instructions = QLabel(
            "Using fallback ROI tools.\n"
            "1. Create an ROI layer\n"
            "2. Use napari's rectangle tool to draw ROIs"
        )
        fallback_instructions.setWordWrap(True)
        fallback_layout.addWidget(fallback_instructions)
        
        self.create_roi_btn = QPushButton("Create ROI Layer")
        self.create_roi_btn.clicked.connect(self.create_fallback_roi_layer)
        fallback_layout.addWidget(self.create_roi_btn)
        
        self.fallback_container.setLayout(fallback_layout)
        self.fallback_container.setVisible(False)
        layout.addWidget(self.fallback_container)
        
        layout.addStretch()
        scroll_content.setLayout(layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        self.setMinimumSize(400, 300)
        self.setMaximumSize(800, 600)
        self.resize(500, 450)

    def initialize_roi_manager(self):
        """Initialize the napari-roi-manager."""
        if self.roi_manager_initialized:
            return
        
        if not self._has_image_loaded():
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        try:
            self._cleanup_existing_roi_layers()
            self.roi_manager = QRoiManager(self.viewer)
            
            container_widget = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(self.roi_manager)
            container_widget.setLayout(container_layout)
            
            self.roi_manager_container.setWidget(container_widget)
            self.roi_manager_container.setVisible(True)
            self.fallback_container.setVisible(False)
            
            self._enable_operations(True)
            self.status_label.setText("Status: ROI Manager initialized")
            self.init_roi_btn.setText("ROI Manager Initialized")
            self.init_roi_btn.setEnabled(False)
            self.roi_manager_initialized = True
            
            if self.current_image_path:
                self.load_image_rois(self.current_image_path)
            
            QMessageBox.information(self, "Success", "ROI Manager initialized successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            self.fallback_container.setVisible(True)
            self.roi_manager_container.setVisible(False)
            QMessageBox.warning(self, "ROI Manager Error", 
                              f"Failed to initialize: {str(e)}\n\nUsing fallback mode.")

    def create_fallback_roi_layer(self):
        """Create a fallback ROI layer."""
        self._cleanup_existing_roi_layers()
        
        text_properties = self._get_text_properties() if self.show_labels_check.isChecked() else None
        
        self.roi_layer = self.viewer.add_shapes(
            name="ROIs",
            shape_type='rectangle',
            edge_color='red',
            face_color='red',
            opacity=0.3,
            text=text_properties
        )
        
        self.roi_layer.events.data.connect(self.on_shapes_layer_changed)
        
        self._enable_operations(True)
        self.status_label.setText("Status: Fallback ROI layer created")
        self.init_roi_btn.setText("ROI Layer Created")
        self.init_roi_btn.setEnabled(False)
        self.roi_manager_initialized = True
        
        QMessageBox.information(self, "Success", "Fallback ROI layer created!")

    def on_shapes_layer_changed(self, event=None):
        """Handle changes to shapes layer data."""
        try:
            QTimer.singleShot(100, self.delayed_roi_check)
            # Always update labels when shapes change to ensure proper naming
            QTimer.singleShot(110, self.update_roi_labels)
        except Exception:
            pass

    def delayed_roi_check(self):
        """Delayed ROI check."""
        try:
            rois = self.get_rois_data()
            if rois:
                self.on_roi_changed()
        except Exception:
            pass

    def refresh_rois(self):
        """Refresh ROI detection."""
        rois = self.get_rois_data()
        if rois:
            QMessageBox.information(self, "ROIs Found", f"Found {len(rois)} ROIs")
        else:
            QMessageBox.warning(self, "No ROIs", "No ROIs detected. Draw ROIs using napari tools.")

    def set_current_image(self, image_path):
        """Set the current image path."""
        self.current_image_path = image_path
        if image_path:
            if not self.roi_manager_initialized:
                self.status_label.setText("Status: Image loaded - Ready to initialize ROI Manager")
                self.init_roi_btn.setEnabled(True)
            else:
                self.load_image_rois(image_path)
            self.update_roi_scope_label()
            # Update ROI labels if they are currently being displayed
            if self.show_labels_check.isChecked():
                QTimer.singleShot(100, self.update_roi_labels)
    
    def set_loaded_images(self, image_list):
        """Set the list of all loaded images."""
        self.loaded_images = image_list if image_list else []
        self.update_roi_scope_label()
    
    def update_roi_scope_label(self):
        """Update the ROI scope label."""
        num_images = len(self.loaded_images)
        if num_images > 1:
            num_with_rois = sum(1 for img in self.loaded_images 
                              if img in self.per_image_rois and self.per_image_rois[img])
            self.roi_scope_label.setText(f"ROI Scope: {num_with_rois}/{num_images} images have ROIs")
            self.roi_scope_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif num_images == 1:
            has_rois = (self.current_image_path in self.per_image_rois and 
                       self.per_image_rois[self.current_image_path])
            if has_rois:
                num_rois = len(self.per_image_rois[self.current_image_path])
                self.roi_scope_label.setText(f"ROI Scope: Current image has {num_rois} ROI(s)")
            else:
                self.roi_scope_label.setText("ROI Scope: Current image has no ROIs")
            self.roi_scope_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        else:
            self.roi_scope_label.setText("ROI Scope: No images loaded")
            self.roi_scope_label.setStyleSheet("color: #FF9800; font-weight: bold;")
    
    def load_image_rois(self, image_path):
        """Load and display ROIs for a specific image."""
        if not image_path:
            return
        
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        shapes_data = self._load_rois_from_file(image_path)
        
        roi_layer, roi_layer_index = self._find_roi_layer_with_index()
        
        if roi_layer is None and self.roi_manager_initialized:
            self._create_new_roi_layer(shapes_data, image_name)
        elif roi_layer is not None:
            self._update_existing_roi_layer(roi_layer, roi_layer_index, shapes_data, image_name)
        elif not self.roi_manager_initialized:
            self.status_label.setText("Status: Click 'Initialize ROI Manager' to begin")
            self.init_roi_btn.setEnabled(True)
        
        self.update_roi_scope_label()
        
        # Update ROI labels if they are currently being displayed
        if self.show_labels_check.isChecked():
            QTimer.singleShot(100, self.update_roi_labels)

    def save_rois(self, auto=False):
        """Save ROIs to a file."""
        if not self.roi_manager_initialized:
            if not auto:
                QMessageBox.warning(self, "ROI Manager Not Ready", "Please initialize ROI Manager first.")
            return
            
        if not self.current_image_path:
            if not auto:
                QMessageBox.warning(self, "No Image", "No image loaded.")
            return

        try:
            rois = self.get_rois_data()
            if not rois:
                if not auto:
                    QMessageBox.warning(self, "No ROIs", "No ROIs to save.")
                return

            roi_file = self._get_roi_file_path(self.current_image_path)
            os.makedirs(os.path.dirname(roi_file), exist_ok=True)

            data = {
                'image_path': self.current_image_path,
                'rois': rois,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(roi_file, 'w') as f:
                json.dump(data, f, indent=2)

            if not auto:
                QMessageBox.information(self, "Success", f"ROIs saved successfully")
            
            roi_arrays = [np.array(roi['vertices']) for roi in rois if 'vertices' in roi]
            
            if self.current_image_path and roi_arrays:
                self.per_image_rois[self.current_image_path] = roi_arrays
                self.update_roi_scope_label()
            
            if roi_arrays:
                self.roi_saved.emit(roi_arrays)
                
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs: {str(e)}")
    
    def save_roi_snapshot(self):
        """Save a PNG snapshot of the current image frame with ROIs and labels."""
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        
        if not self.roi_manager_initialized:
            QMessageBox.warning(self, "ROI Manager Not Ready", "Please initialize ROI Manager first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.figure import Figure
            
            # Get the image layer
            image_layer = self._get_image_layer()
            if image_layer is None:
                QMessageBox.warning(self, "No Image", "No image found in viewer.")
                return
            
            # Get image data (first frame if it's a stack)
            image_data = image_layer.data
            
            # Handle different dimensionalities - extract first slice from each dimension until we get 2D
            frame_data = image_data
            while frame_data.ndim > 2:
                frame_data = frame_data[0]
            
            if frame_data.ndim != 2:
                QMessageBox.warning(self, "Invalid Image", 
                                  f"Could not extract 2D frame from image with shape {image_data.shape}")
                return
            
            # Get ROI layer
            roi_layer = self._find_roi_layer()
            if roi_layer is None or len(roi_layer.data) == 0:
                reply = QMessageBox.question(
                    self, "No ROIs", 
                    "No ROIs found. Save image without ROIs?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply == QMessageBox.No:
                    return
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Display image
            ax.imshow(frame_data, cmap='gray')
            ax.axis('off')
            
            # Add ROIs if they exist
            if roi_layer is not None and len(roi_layer.data) > 0:
                image_name = ""
                if self.current_image_path:
                    image_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_"
                
                for i, roi_shape in enumerate(roi_layer.data):
                    # Extract rectangle coordinates
                    try:
                        vertices = np.asarray(roi_shape)
                        
                        # Get min/max coordinates
                        y_coords = vertices[:, 0]
                        x_coords = vertices[:, 1]
                        
                        y_min = np.min(y_coords)
                        y_max = np.max(y_coords)
                        x_min = np.min(x_coords)
                        x_max = np.max(x_coords)
                        
                        width = x_max - x_min
                        height = y_max - y_min
                        
                        # Create rectangle patch
                        rect = patches.Rectangle(
                            (x_min, y_min), width, height,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                        
                        # Add label above the ROI box
                        label = f"{image_name}ROI{i+1}"
                        ax.text(
                            x_min, y_min - 5, label,
                            color='white', fontsize=12, fontweight='bold',
                            ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
                        )
                    except Exception as e:
                        print(f"Warning: Could not draw ROI {i+1}: {e}")
                        continue
            
            # Save to file
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            os.makedirs(roi_dir, exist_ok=True)
            
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            snapshot_file = os.path.join(roi_dir, f"{image_name}_ROI_snapshot.png")
            
            fig.savefig(snapshot_file, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)
            
            QMessageBox.information(
                self, "Success", 
                f"ROI snapshot saved to:\n{snapshot_file}"
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error saving ROI snapshot:\n{error_details}")
            QMessageBox.critical(self, "Error", f"Failed to save snapshot:\n{str(e)}")
    
    def save_roi_crops(self):
        """Save cropped TIF files for each ROI with all dimensions preserved."""
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "No image loaded.")
            return
        
        if not self.roi_manager_initialized:
            QMessageBox.warning(self, "ROI Manager Not Ready", "Please initialize ROI Manager first.")
            return
        
        try:
            from tifffile import imwrite
            
            # Get the image layer
            image_layer = self._get_image_layer()
            if image_layer is None:
                QMessageBox.warning(self, "No Image", "No image found in viewer.")
                return
            
            # Get full image data (all dimensions)
            image_data = image_layer.data
            
            # Get ROI layer
            roi_layer = self._find_roi_layer()
            if roi_layer is None or len(roi_layer.data) == 0:
                QMessageBox.warning(self, "No ROIs", "No ROIs found to crop.")
                return
            
            # Create cropped folder
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management', 'cropped')
            os.makedirs(roi_dir, exist_ok=True)
            
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            saved_count = 0
            
            # Process each ROI
            for i, roi_shape in enumerate(roi_layer.data):
                try:
                    vertices = np.asarray(roi_shape)
                    
                    # Get y and x coordinates (last 2 dimensions)
                    y_coords = vertices[:, 0]
                    x_coords = vertices[:, 1]
                    
                    y_min = int(np.floor(np.min(y_coords)))
                    y_max = int(np.ceil(np.max(y_coords)))
                    x_min = int(np.floor(np.min(x_coords)))
                    x_max = int(np.ceil(np.max(x_coords)))
                    
                    # Crop all dimensions, keeping time/channel/z intact
                    if image_data.ndim == 5:  # e.g., (t, z, c, y, x)
                        cropped = image_data[:, :, :, y_min:y_max, x_min:x_max]
                        num_frames, num_slices, num_channels = cropped.shape[:3]
                    elif image_data.ndim == 4:  # e.g., (t, c, y, x) or (t, z, y, x)
                        cropped = image_data[:, :, y_min:y_max, x_min:x_max]
                        # Assume (t, c, y, x) format
                        num_frames, num_channels = cropped.shape[:2]
                        num_slices = 1
                    elif image_data.ndim == 3:  # e.g., (t, y, x) or (c, y, x)
                        cropped = image_data[:, y_min:y_max, x_min:x_max]
                        # Assume (t, y, x) format
                        num_frames = cropped.shape[0]
                        num_channels = 1
                        num_slices = 1
                    elif image_data.ndim == 2:  # (y, x)
                        cropped = image_data[y_min:y_max, x_min:x_max]
                        num_frames = 1
                        num_channels = 1
                        num_slices = 1
                    else:
                        print(f"Warning: Unsupported dimensions for ROI {i+1}")
                        continue
                    
                    # Create ImageJ metadata
                    metadata = {
                        'axes': 'TZCYX' if image_data.ndim == 5 else 'TCYX' if image_data.ndim == 4 else 'TYX' if image_data.ndim == 3 else 'YX',
                        'frames': num_frames,
                        'slices': num_slices,
                        'channels': num_channels
                    }
                    
                    # Save as TIF with ImageJ metadata
                    roi_filename = f"{image_name}_ROI{i+1}_crop.tif"
                    roi_filepath = os.path.join(roi_dir, roi_filename)
                    
                    imwrite(roi_filepath, cropped, imagej=True, metadata=metadata)
                    saved_count += 1
                    
                except Exception as e:
                    print(f"Warning: Could not save crop for ROI {i+1}: {e}")
                    continue
            
            if saved_count > 0:
                QMessageBox.information(
                    self, "Success", 
                    f"Saved {saved_count} ROI crop(s) to:\n{roi_dir}"
                )
            else:
                QMessageBox.warning(self, "No Crops Saved", "Could not save any ROI crops.")
                
        except ImportError:
            QMessageBox.critical(
                self, "Missing Library", 
                "tifffile library is required to save TIF files.\n"
                "Install it with: pip install tifffile"
            )
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error saving ROI crops:\n{error_details}")
            QMessageBox.critical(self, "Error", f"Failed to save crops:\n{str(e)}")

    def get_rois_data(self):
        """Get ROI data from the ROI manager."""
        if not self.roi_manager_initialized:
            return []
            
        try:
            roi_data = []
            
            for layer in self.viewer.layers:
                if type(layer).__name__ in ['RoiManagerLayer', 'Shapes']:
                    if hasattr(layer, 'data') and len(layer.data) > 0:
                        for i, shape in enumerate(layer.data):
                            try:
                                roi_data.append({
                                    'id': i + 1,  # Use 1-based indexing
                                    'type': 'rectangle',
                                    'vertices': shape.tolist() if hasattr(shape, 'tolist') else list(shape)
                                })
                            except Exception:
                                pass
                        
                        if roi_data:
                            self.create_roi_image_layers(roi_data, layer)
                            return roi_data
            
            return []
            
        except Exception:
            return []

    def create_roi_image_layers(self, roi_data, shapes_layer):
        """Create individual image layers for each ROI."""
        if not self.current_image_path:
            return
        
        try:
            image_layer = self._get_image_layer()
            if image_layer is None:
                return
            
            image_data = image_layer.data
            self._remove_existing_roi_image_layers()
            
            for roi_info in roi_data:
                try:
                    roi_id = roi_info['id']
                    vertices = np.array(roi_info['vertices'])
                    
                    if len(vertices.shape) == 2 and vertices.shape[0] >= 4:
                        roi_image = self._extract_roi_region(image_data, vertices)
                        
                        if roi_image is not None and roi_image.size > 0:
                            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                            roi_layer_name = f"{image_name}_ROI{roi_id}"
                            self.viewer.add_image(roi_image, name=roi_layer_name, visible=False)
                        
                except Exception:
                    pass
            
        except Exception:
            pass

    def on_roi_changed(self, *args, **kwargs):
        """Handle ROI changes."""
        try:
            image_layer = self._get_image_layer()
            if image_layer is not None:
                rois_data = self.get_rois_data()
                if rois_data:
                    roi_image_layers = [layer for layer in self.viewer.layers 
                                       if hasattr(layer, 'name') and layer.name.startswith('ROI_')]
                    
                    for roi_layer in roi_image_layers:
                        try:
                            roi_image_data = roi_layer.data
                            if roi_image_data.ndim > 2:
                                while roi_image_data.ndim > 2:
                                    roi_image_data = roi_image_data[0]
                            self.roi_updated.emit(roi_image_data, image_layer)
                        except Exception:
                            pass

            if hasattr(self, 'auto_save_check') and self.auto_save_check.isChecked():
                self.save_rois(auto=True)
                
        except Exception:
            pass

    def toggle_roi_labels(self):
        """Toggle the visibility of ROI labels."""
        try:
            roi_layer = self._find_roi_layer()
            if not roi_layer or not hasattr(roi_layer, 'text'):
                return
            
            if self.show_labels_check.isChecked():
                roi_layer.text = self._get_text_properties()
            else:
                roi_layer.text = ''
        except Exception:
            pass
    
    def update_roi_labels(self):
        """Update ROI labels to show proper numbering with image name."""
        try:
            roi_layer = self._find_roi_layer()
            if not roi_layer or not hasattr(roi_layer, 'text'):
                return
            
            if len(roi_layer.data) > 0:
                image_name = ""
                if self.current_image_path:
                    image_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_"
                
                # Create labels with 1-based numbering
                labels = [f"{image_name}ROI{i+1}" for i in range(len(roi_layer.data))]
                
                roi_layer.text = {
                    'string': labels,
                    'size': 12,
                    'color': 'white',
                    'anchor': 'center',
                    'translation': [0, 0]
                }
                roi_layer.refresh()
        except Exception:
            pass
    
    def get_rois_for_image(self, image_path):
        """Get ROIs for a specific image."""
        return self.per_image_rois.get(image_path, None)

    # Helper methods
    def _has_image_loaded(self):
        """Check if an image is loaded."""
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                return True
        return False
    
    def _cleanup_existing_roi_layers(self):
        """Clean up existing ROI layers."""
        layers_to_remove = []
        for layer in self.viewer.layers:
            if hasattr(layer, 'name'):
                layer_name = layer.name.lower()
                if 'roi' in layer_name or type(layer).__name__ == 'RoiManagerLayer':
                    layers_to_remove.append(layer)
        
        for layer in layers_to_remove:
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
    
    def _enable_operations(self, enabled):
        """Enable or disable operation buttons."""
        self.save_rois_btn.setEnabled(enabled)
        self.refresh_rois_btn.setEnabled(enabled)
        self.auto_save_check.setEnabled(enabled)
        self.show_labels_check.setEnabled(enabled)
        self.save_snapshot_btn.setEnabled(enabled)
        self.save_crops_btn.setEnabled(enabled)
    
    def _get_text_properties(self):
        """Get text properties for ROI labels."""
        image_name = ""
        if self.current_image_path:
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0] + "_"
        
        # Create custom labels for each ROI with image name and 1-based numbering
        roi_layer = self._find_roi_layer()
        if roi_layer and hasattr(roi_layer, 'data'):
            num_rois = len(roi_layer.data)
            labels = [f"{image_name}ROI{i+1}" for i in range(num_rois)]
            return {
                'string': labels,
                'size': 12,
                'color': 'white',
                'anchor': 'center',
                'translation': [0, 0]
            }
        
        return {
            'string': f'{image_name}ROI{{index}}',
            'size': 12,
            'color': 'white',
            'anchor': 'center',
            'translation': [0, 0]
        }
    
    def _find_roi_layer(self):
        """Find the active ROI layer."""
        for layer in self.viewer.layers:
            if type(layer).__name__ in ['Shapes', 'RoiManagerLayer']:
                if hasattr(layer, 'name') and 'roi' in layer.name.lower():
                    return layer
        return None
    
    def _find_roi_layer_with_index(self):
        """Find the active ROI layer with its index."""
        for idx, layer in enumerate(self.viewer.layers):
            if type(layer).__name__ in ['Shapes', 'RoiManagerLayer']:
                if hasattr(layer, 'name') and 'roi' in layer.name.lower():
                    return layer, idx
        return None, None
    
    def _get_roi_file_path(self, image_path):
        """Get the ROI file path for an image."""
        roi_dir = os.path.join(os.path.dirname(image_path), 'ROI_management')
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(roi_dir, f"{image_name}_ROIs.json")
    
    def _load_rois_from_file(self, image_path):
        """Load ROI data from file and prompt user."""
        roi_file = self._get_roi_file_path(image_path)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        shapes_data = []
        
        if not os.path.exists(roi_file):
            if image_path in self.per_image_rois:
                del self.per_image_rois[image_path]
            return shapes_data
        
        try:
            with open(roi_file, 'r') as f:
                rois = json.load(f).get('rois', [])
            
            if not rois:
                return shapes_data
            
            should_load = False
            if image_path not in self.prompted_images:
                self.prompted_images.add(image_path)
                reply = QMessageBox.question(
                    self, "Existing ROIs Found", 
                    f"Found {len(rois)} ROI(s) for {image_name}. Load them?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                should_load = (reply == QMessageBox.Yes)
            else:
                should_load = (image_path in self.per_image_rois)
            
            if should_load:
                self.per_image_rois[image_path] = []
                for roi_info in rois:
                    if 'vertices' in roi_info:
                        vertices = np.array(roi_info['vertices'])
                        self.per_image_rois[image_path].append(vertices)
                        shapes_data.append(vertices)
                self.status_label.setText(f"Status: Loaded {len(rois)} ROIs for {image_name}")
            else:
                self.status_label.setText(f"Status: Ready to draw ROIs for {image_name}")
        except Exception:
            pass
        
        return shapes_data
    
    def _create_new_roi_layer(self, shapes_data, image_name):
        """Create a new ROI layer."""
        try:
            text_properties = self._get_text_properties() if self.show_labels_check.isChecked() else None
            
            roi_layer = self.viewer.add_shapes(
                shapes_data if shapes_data else [],
                name="ROIs",
                shape_type='rectangle',
                edge_color='red',
                face_color='red',
                opacity=0.3,
                text=text_properties
            )
            
            roi_layer.events.data.connect(self.on_shapes_layer_changed)
            self.roi_layer = roi_layer
            roi_layer.visible = True
            self.viewer.layers.selection.active = roi_layer
            
            if shapes_data:
                self.status_label.setText(f"Status: Loaded {len(shapes_data)} ROIs for {image_name}")
            else:
                self.status_label.setText(f"Status: Ready to draw ROIs for {image_name}")
        except Exception:
            pass
    
    def _update_existing_roi_layer(self, roi_layer, roi_layer_index, shapes_data, image_name):
        """Update an existing ROI layer."""
        try:
            if hasattr(roi_layer, 'selected_data'):
                roi_layer.selected_data = set()
            if hasattr(roi_layer, '_value'):
                roi_layer._value = (None, None)
            if hasattr(roi_layer, '_moving_value'):
                roi_layer._moving_value = (None, None)
            
            # Only update if there are no existing ROIs, or if the loaded shapes_data is not empty
            # This prevents replacing user-drawn ROIs when switching between images
            if len(roi_layer.data) == 0 or shapes_data:
                # If there are existing ROIs and we're loading new ones, preserve the existing ROIs
                if len(roi_layer.data) > 0 and shapes_data:
                    # Don't replace - keep existing ROIs
                    pass
                else:
                    # No existing ROIs, so load the saved ones
                    roi_layer.data = shapes_data
            
            roi_layer.visible = True
            
            if roi_layer_index is not None and roi_layer_index < len(self.viewer.layers) - 1:
                self.viewer.layers.move(roi_layer_index, len(self.viewer.layers) - 1)
            
            self.viewer.layers.selection.active = roi_layer
            
            num_rois = len(roi_layer.data)
            if num_rois > 0:
                self.status_label.setText(f"Status: {num_rois} ROI(s) present for {image_name}")
            else:
                self.status_label.setText(f"Status: Ready to draw ROIs for {image_name}")
        except Exception:
            pass
    
    def _get_image_layer(self):
        """Get the current image layer."""
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                if not layer.name.startswith('ROI_'):
                    return layer
        return None
    
    def _remove_existing_roi_image_layers(self):
        """Remove existing ROI image layers."""
        layers_to_remove = [layer for layer in self.viewer.layers 
                           if hasattr(layer, 'name') and layer.name.startswith('ROI_')]
        for layer in layers_to_remove:
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass
    
    def _extract_roi_region(self, image_data, vertices):
        """Extract ROI region from image data."""
        y_coords = vertices[:, 0]
        x_coords = vertices[:, 1]
        
        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
        
        if image_data.ndim == 4:
            roi_image = image_data[:, :, y_min:y_max, x_min:x_max]
        elif image_data.ndim == 3:
            roi_image = image_data[:, y_min:y_max, x_min:x_max]
        else:
            roi_image = image_data[y_min:y_max, x_min:x_max]
        
        return roi_image
