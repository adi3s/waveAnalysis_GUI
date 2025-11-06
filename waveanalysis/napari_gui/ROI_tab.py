import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QGroupBox,
    QLabel, QFileDialog, QMessageBox, QComboBox, QCheckBox,
    QSpinBox, QListWidget, QLineEdit, QSplitter, QProgressBar,
    QScrollArea
)
from qtpy.QtCore import Signal, Qt, QThread
from napari_roi_manager import QRoiManager

class ROITab(QWidget):
    """
    Tab for managing ROIs in Napari viewer using napari_roi_manager.
    
    This widget provides a comprehensive interface for:
    - Initializing and managing ROI tools
    - Creating, saving, and loading ROIs
    - Calculating measurements on ROI regions
    - Handling fallback functionality when napari-roi-manager is unavailable
    
    Signals:
        measurements_ready: Emitted when ROI measurements are calculated
        roi_saved: Emitted when ROIs are saved to file
        roi_updated: Emitted when ROI data is updated
    """
    measurements_ready = Signal(pd.DataFrame)
    roi_saved = Signal(list)
    roi_updated = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.viewer = parent.viewer
        self.current_image_path = None
        self.saved_rois = {}
        self.roi_manager = None
        self.roi_manager_initialized = False
        self.roi_layer = None  # For fallback ROI layer
        self.loaded_images = []  # Track all loaded images
        self.roi_applies_to_all = True  # ROIs apply to all images by default
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface with scrollable layout and organized sections."""
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create scroll area for the entire widget
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QScrollArea.NoFrame)  # Remove frame for cleaner look
        
        # Create container widget for the scroll area
        scroll_content = QWidget()
        layout = QVBoxLayout()
        
        # ROI Manager Setup section
        setup_group = QGroupBox("ROI Manager Setup")
        setup_layout = QVBoxLayout()
        
        self.setup_instructions = QLabel(
            "1. Load an image in your application\n"
            "2. Click 'Initialize ROI Manager' to set up ROI annotation\n"
            "3. Use the ROI manager tools to draw and manage ROIs\n"
            "4. ROIs will automatically apply to ALL loaded images"
        )
        self.setup_instructions.setWordWrap(True)
        setup_layout.addWidget(self.setup_instructions)
        
        # Add ROI scope information
        self.roi_scope_label = QLabel("ROI Scope: Will apply to ALL loaded images")
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
        
        # ROI Manager widget container with scroll area (initially hidden)
        self.roi_manager_container = QScrollArea()
        self.roi_manager_container.setWidgetResizable(True)
        self.roi_manager_container.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.roi_manager_container.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.roi_manager_container.setMaximumHeight(300)  # Set max height for scrolling
        self.roi_manager_container.setVisible(False)
        layout.addWidget(self.roi_manager_container)
        
        # ROI operations group
        roi_ops_group = QGroupBox("ROI Operations")
        roi_ops_layout = QHBoxLayout()
        
        self.save_rois_btn = QPushButton("Save ROIs")
        self.save_rois_btn.clicked.connect(self.save_rois)
        self.save_rois_btn.setEnabled(False)  # Initially disabled
        
        self.refresh_rois_btn = QPushButton("Refresh ROIs")
        self.refresh_rois_btn.clicked.connect(self.refresh_rois)
        self.refresh_rois_btn.setEnabled(False)  # Initially disabled
        
        # Auto-save option
        self.auto_save_check = QCheckBox("Auto-save ROIs on changes")
        self.auto_save_check.setChecked(True)
        self.auto_save_check.setEnabled(False)  # Initially disabled
        
        roi_ops_layout.addWidget(self.save_rois_btn)
        roi_ops_layout.addWidget(self.refresh_rois_btn)
        roi_ops_layout.addWidget(self.auto_save_check)
        roi_ops_group.setLayout(roi_ops_layout)
        layout.addWidget(roi_ops_group)
        
        # Fallback for when napari-roi-manager has issues
        self.fallback_container = QWidget()
        fallback_layout = QVBoxLayout()
        
        fallback_instructions = QLabel(
            "napari-roi-manager encountered an issue. Using fallback ROI tools.\n\n"
            "1. Create an ROI layer using the button below\n"
            "2. Use napari's rectangle tool to draw ROIs\n"
            "3. ROIs will be available for measurement"
        )
        fallback_instructions.setWordWrap(True)
        fallback_layout.addWidget(fallback_instructions)
        
        self.create_roi_btn = QPushButton("Create ROI Layer")
        self.create_roi_btn.clicked.connect(self.create_fallback_roi_layer)
        fallback_layout.addWidget(self.create_roi_btn)
        
        self.fallback_container.setLayout(fallback_layout)
        self.fallback_container.setVisible(False)
        layout.addWidget(self.fallback_container)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Add stretch to prevent excessive vertical expansion
        layout.addStretch()
        
        # Set up scroll area
        scroll_content.setLayout(layout)
        scroll_area.setWidget(scroll_content)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)
        
        # Set size constraints for the widget
        self.setMinimumSize(400, 300)
        self.setMaximumSize(800, 600)  # Reasonable maximum size
        self.resize(500, 450)  # Default size

    def initialize_roi_manager(self):
        """
        Initialize the napari-roi-manager with error handling and fallback support.
        
        Attempts to create QRoiManager instance, connects signals, and enables
        UI controls. Falls back to basic shapes layer if initialization fails.
        """
        if self.roi_manager_initialized:
            return
        
        # Check if an image is loaded
        has_image = False
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                has_image = True
                break
        
        if not has_image:
            QMessageBox.warning(self, "No Image", "Please load an image first before initializing ROI manager.")
            return
        
        try:
            # Clean up any existing ROI layers
            self.cleanup_existing_roi_layers()
            
            # Create the ROI Manager
            self.roi_manager = QRoiManager(self.viewer)
            
            # Create a container widget for the scroll area
            container_widget = QWidget()
            container_layout = QVBoxLayout()
            container_layout.addWidget(self.roi_manager)
            container_widget.setLayout(container_layout)
            
            # Add to scroll area
            self.roi_manager_container.setWidget(container_widget)
            self.roi_manager_container.setVisible(True)
            self.fallback_container.setVisible(False)
            
            # Check for existing ROI files and offer to load them
            self.check_and_offer_existing_rois()
            
            # Connect ROI manager signals if available
            try:
                if hasattr(self.roi_manager, 'roi_changed'):
                    self.roi_manager.roi_changed.connect(self.on_roi_changed)
                elif hasattr(self.roi_manager, 'event') and hasattr(self.roi_manager.event, 'roi_changed'):
                    self.roi_manager.event.roi_changed.connect(self.on_roi_changed)
            except Exception:
                pass  # Silently handle signal connection failures
            
            # Try to get the ROI layer created by the manager and connect to its events
            self.connect_to_roi_layer()
            
            # Enable the operation buttons
            self.save_rois_btn.setEnabled(True)
            self.refresh_rois_btn.setEnabled(True)
            self.auto_save_check.setEnabled(True)
            
            # Update status and button
            self.status_label.setText("Status: ROI Manager initialized successfully")
            self.init_roi_btn.setText("ROI Manager Initialized")
            self.init_roi_btn.setEnabled(False)
            self.roi_manager_initialized = True
            
            QMessageBox.information(self, "Success", "ROI Manager initialized successfully! You can now create and manage ROIs.")
            
        except Exception as e:
            error_msg = f"Error initializing QRoiManager: {str(e)}"
            self.status_label.setText(f"Status: {error_msg}")
            
            # Fall back to simple ROI manager
            self.fallback_container.setVisible(True)
            self.roi_manager_container.setVisible(False)
            QMessageBox.warning(self, "ROI Manager Error", 
                              f"Failed to initialize napari-roi-manager: {error_msg}\n\n"
                              "Using fallback ROI tools instead.")
        except ImportError as e:
            # Handle case where QRoiManager is not available
            error_msg = "napari-roi-manager not available"
            self.status_label.setText(f"Status: {error_msg}")
            
            # Use fallback ROI manager
            self.fallback_container.setVisible(True)
            self.roi_manager_container.setVisible(False)
            QMessageBox.information(self, "Fallback Mode", 
                                  "napari-roi-manager not available. Using fallback ROI tools.")
            self.roi_manager_initialized = True  # Mark as initialized to enable fallback functionality

    def connect_to_roi_layer(self):
        """Connect to the ROI layer created by the ROI manager."""
        try:
            # Look for RoiManagerLayer first (from napari-roi-manager)
            for layer in self.viewer.layers:
                if type(layer).__name__ == 'RoiManagerLayer':
                    # Connect to the layer's data change events
                    if hasattr(layer.events, 'data'):
                        layer.events.data.connect(self.on_shapes_layer_changed)
                    if hasattr(layer.events, 'current_properties'):
                        layer.events.current_properties.connect(self.on_shapes_layer_changed)
                    # Try other common event names for roi manager layers
                    if hasattr(layer.events, 'roi_changed'):
                        layer.events.roi_changed.connect(self.on_shapes_layer_changed)
                    if hasattr(layer.events, 'current_roi'):
                        layer.events.current_roi.connect(self.on_shapes_layer_changed)
                    return  # Connected to roi manager layer
            
            # Look for standard shapes layers that might be created by the ROI manager
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and type(layer).__name__ == 'Shapes':
                    # Connect to the layer's data change events
                    layer.events.data.connect(self.on_shapes_layer_changed)
                    # Also connect to selection events to trigger updates
                    if hasattr(layer.events, 'current_properties'):
                        layer.events.current_properties.connect(self.on_shapes_layer_changed)
                    # This is likely the ROI layer
                    break
        except Exception as e:
            pass

    def on_shapes_layer_changed(self, event=None):
        """Handle changes to shapes layer data."""
        try:
            # Small delay to ensure shape is fully added
            from qtpy.QtCore import QTimer
            QTimer.singleShot(100, self.delayed_roi_check)
        except Exception as e:
            pass

    def delayed_roi_check(self):
        """Delayed ROI check to ensure shapes are fully processed."""
        try:
            # Check for ROIs and create image layers
            rois = self.get_rois_data()
            if rois:
                # Call the original ROI changed handler
                self.on_roi_changed()
        except Exception as e:
            pass

    def create_fallback_roi_layer(self):
        """Create a fallback ROI layer when napari-roi-manager is not available"""
        # Clean up any existing ROI layers
        self.cleanup_existing_roi_layers()
        
        # Create new ROI layer
        self.roi_layer = self.viewer.add_shapes(
            name="ROIs",
            shape_type='rectangle',
            edge_color='red',
            face_color='red',
            opacity=0.3
        )
        
        # Connect to data change event
        self.roi_layer.events.data.connect(self.on_shapes_layer_changed)
        
        # Enable operation buttons
        self.save_rois_btn.setEnabled(True)
        self.refresh_rois_btn.setEnabled(True)
        self.auto_save_check.setEnabled(True)
        
        self.status_label.setText("Status: Fallback ROI layer created")
        self.init_roi_btn.setText("ROI Layer Created")
        self.init_roi_btn.setEnabled(False)
        self.roi_manager_initialized = True
        
        QMessageBox.information(self, "Success", 
                              "Fallback ROI layer created successfully!\n\n"
                              "You can now use napari's rectangle tool to draw ROIs.")

    def cleanup_existing_roi_layers(self):
        """Clean up any existing ROI layers to avoid duplicates"""
        layers_to_remove = []
        for layer in self.viewer.layers:
            if hasattr(layer, 'name'):
                layer_name = layer.name.lower()
                if ('roi' in layer_name or 
                    layer_name.startswith('roi_') or 
                    'roiManagerLayer' in type(layer).__name__.lower() or
                    (hasattr(layer, 'shape_type') and type(layer).__name__ == 'Shapes')):
                    layers_to_remove.append(layer)
        
        for layer in layers_to_remove:
            try:
                self.viewer.layers.remove(layer)
            except Exception:
                pass  # Silently handle layer removal errors

    def check_and_offer_existing_rois(self):
        """Check for existing ROI files and offer to load them automatically"""
        if not self.current_image_path:
            return
        
        try:
            # Check if ROI file exists for current image
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            roi_file = os.path.join(roi_dir, f"{image_name}_ROIs.json")
            
            if os.path.exists(roi_file):
                # Check if the ROI file has content
                try:
                    with open(roi_file, 'r') as f:
                        data = json.load(f)
                    
                    rois = data.get('rois', [])
                    if rois:
                        # Ask user if they want to load existing ROIs
                        reply = QMessageBox.question(
                            self, 
                            "Existing ROIs Found", 
                            f"Found existing ROIs for {image_name} ({len(rois)} ROIs).\n\n"
                            f"Would you like to automatically load them?",
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes
                        )
                        
                        if reply == QMessageBox.Yes:
                            # Load the existing ROIs
                            self.load_rois_data(rois)
                            
                            # Also create the visual ROI image layers
                            self.create_roi_image_layers_from_saved_data(rois)
                            
                            # Force a refresh of the ROI display
                            self.delayed_roi_check()
                            
                            # Refresh the viewer to ensure ROIs are visible
                            try:
                                self.viewer.reset_view()
                            except Exception:
                                pass
                            
                            # Update status
                            self.status_label.setText(f"Status: Loaded {len(rois)} ROIs from saved data")
                            
                            QMessageBox.information(
                                self, 
                                "ROIs Loaded", 
                                f"Successfully loaded {len(rois)} existing ROIs for {image_name}.\n\n"
                                f"The ROI shapes should now be visible on the image as red rectangles."
                            )
                        
                except (json.JSONDecodeError, Exception):
                    pass  # Silently handle file reading errors
                
        except Exception:
            pass  # Silently handle ROI file checking errors

    def refresh_rois(self):
        """Manually refresh ROI detection"""
        rois = self.get_rois_data()
        if rois:
            # Count ROI image layers created
            roi_layers = [layer for layer in self.viewer.layers 
                         if hasattr(layer, 'name') and layer.name.startswith('ROI_')]
            
            message = f"Found {len(rois)} ROIs!\n"
            message += f"Created {len(roi_layers)} ROI image layers.\n\n"
            message += "ROI Details:\n"
            for i, roi in enumerate(rois):
                roi_type = roi.get('shape_type', roi.get('type', 'unknown'))
                message += f"  ROI {i+1}: {roi_type}\n"
            
            QMessageBox.information(self, "ROIs Found", message)
        else:
            QMessageBox.warning(self, "No ROIs", 
                              "No ROIs detected.\n\n"
                              "Make sure you've drawn some ROIs using the napari tools.")
            
        # Also try to reconnect to shapes layers
        self.connect_to_roi_layer()

    def set_current_image(self, image_path):
        """Set the current image path"""
        self.current_image_path = image_path
        if image_path:
            # Update status to show image is loaded
            if not self.roi_manager_initialized:
                self.status_label.setText("Status: Image loaded - Ready to initialize ROI Manager")
                self.init_roi_btn.setEnabled(True)
            
            # Don't clear existing ROIs - they should persist across images
            # Update the ROI scope label to reflect the number of loaded images
            self.update_roi_scope_label()
    
    def set_loaded_images(self, image_list):
        """Set the list of all loaded images"""
        self.loaded_images = image_list if image_list else []
        self.update_roi_scope_label()
    
    def update_roi_scope_label(self):
        """Update the ROI scope label to show how many images ROIs will apply to"""
        num_images = len(self.loaded_images)
        if num_images > 1:
            self.roi_scope_label.setText(
                f"ROI Scope: Will apply to ALL {num_images} loaded images"
            )
            self.roi_scope_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif num_images == 1:
            self.roi_scope_label.setText("ROI Scope: Will apply to current image")
            self.roi_scope_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        else:
            self.roi_scope_label.setText("ROI Scope: No images loaded")
            self.roi_scope_label.setStyleSheet("color: #FF9800; font-weight: bold;")

    def save_rois(self, auto=False):
        """Save ROIs to a file"""
        if not self.roi_manager_initialized or not self.roi_manager:
            if not auto:
                QMessageBox.warning(self, "ROI Manager Not Ready", "Please initialize ROI Manager first.")
            return
            
        if not self.current_image_path:
            if not auto:
                QMessageBox.warning(self, "No Image", "No image loaded to associate ROIs with.")
            return

        try:
            # Get ROIs from ROI manager
            rois = self.get_rois_data()
            if not rois:
                if not auto:
                    QMessageBox.warning(self, "No ROIs", "No ROIs to save.")
                return

            # Create ROI_management folder
            roi_dir = os.path.join(os.path.dirname(self.current_image_path), 'ROI_management')
            os.makedirs(roi_dir, exist_ok=True)

            # Save ROIs
            image_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            roi_file = os.path.join(roi_dir, f"{image_name}_ROIs.json")

            data = {
                'image_path': self.current_image_path,
                'rois': rois,
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            with open(roi_file, 'w') as f:
                json.dump(data, f, indent=2)

            if not auto:
                QMessageBox.information(self, "Success", f"ROIs saved to {roi_file}")
            
            # Convert ROIs to numpy arrays for the signal (workflows expect numpy arrays)
            roi_arrays = []
            for roi in rois:
                if 'vertices' in roi:
                    # Convert vertices list back to numpy array
                    vertices = np.array(roi['vertices'])
                    roi_arrays.append(vertices)
            
            # Emit signal with numpy arrays so main_gui can use them for analysis
            if roi_arrays:
                self.roi_saved.emit(roi_arrays)
                
        except Exception as e:
            if not auto:
                QMessageBox.critical(self, "Error", f"Failed to save ROIs: {str(e)}")

    def get_rois_data(self):
        """Get ROI data from the ROI manager in a safe way"""
        if not self.roi_manager_initialized or not self.roi_manager:
            return []
            
        try:
            roi_data = []
            
            # Method 1: Look for RoiManagerLayer (from napari-roi-manager)
            for layer in self.viewer.layers:
                if type(layer).__name__ == 'RoiManagerLayer':
                    if len(layer.data) > 0:
                        for i, roi in enumerate(layer.data):
                            try:
                                # Try to get ROI data in different ways
                                if hasattr(roi, 'vertices') or hasattr(roi, 'data'):
                                    vertices = getattr(roi, 'vertices', getattr(roi, 'data', None))
                                    if vertices is not None:
                                        vertices_array = np.array(vertices)
                                        roi_info = {
                                            'id': i,
                                            'layer_name': layer.name,
                                            'type': 'roi',
                                            'vertices': vertices_array.tolist(),
                                            'shape_type': getattr(roi, 'shape_type', 'rectangle')
                                        }
                                        roi_data.append(roi_info)
                                elif hasattr(roi, '__dict__'):
                                    # Try to extract data from roi object attributes
                                    roi_dict = roi.__dict__
                                    
                                    # Look for common ROI data attributes
                                    vertices = None
                                    for attr in ['vertices', 'data', 'points', 'coords', 'geometry']:
                                        if attr in roi_dict:
                                            vertices = roi_dict[attr]
                                            break
                                    
                                    if vertices is not None:
                                        vertices_array = np.array(vertices)
                                        roi_info = {
                                            'id': i,
                                            'layer_name': layer.name,
                                            'type': 'roi',
                                            'vertices': vertices_array.tolist(),
                                            'shape_type': roi_dict.get('shape_type', 'rectangle')
                                        }
                                        roi_data.append(roi_info)
                                        
                            except Exception as e:
                                pass
                        
                        if roi_data:
                            # Create image layers for these ROIs
                            self.create_roi_image_layers(roi_data, layer)
                            return roi_data
            
            # Method 2: Look for any shapes layers in the viewer (standard napari shapes)
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and type(layer).__name__ == 'Shapes':
                    if len(layer.data) > 0:
                        for i, shape in enumerate(layer.data):
                            try:
                                shape_array = np.array(shape)
                                roi_info = {
                                    'id': i,
                                    'layer_name': layer.name,
                                    'type': 'rectangle',
                                    'vertices': shape_array.tolist() if hasattr(shape_array, 'tolist') else str(shape),
                                    'shape_type': getattr(layer, 'shape_type', ['rectangle'])[i] if hasattr(layer, 'shape_type') else 'rectangle'
                                }
                                roi_data.append(roi_info)
                            except Exception as e:
                                pass
                        
                        if roi_data:
                            # Also create image layers for these ROIs
                            self.create_roi_image_layers(roi_data, layer)
                            return roi_data
            
            # Method 3: Try to get ROIs from ROI manager if other methods failed
            if hasattr(self.roi_manager, 'get_rois'):
                rois = self.roi_manager.get_rois()
                if rois:
                    for i, roi in enumerate(rois):
                        try:
                            if hasattr(roi, 'to_dict'):
                                roi_data.append(roi.to_dict())
                            elif hasattr(roi, 'data'):
                                roi_data.append({
                                    'id': i,
                                    'type': 'rectangle',
                                    'data': roi.data.tolist() if hasattr(roi.data, 'tolist') else str(roi.data)
                                })
                            else:
                                roi_data.append({'id': i, 'type': 'unknown', 'data': str(roi)})
                        except Exception as e:
                            pass
                    
                    if roi_data:
                        return roi_data
            
            # Method 4: Try to access the ROI layer directly from ROI manager
            if hasattr(self.roi_manager, '_roi_layer') and self.roi_manager._roi_layer:
                roi_layer = self.roi_manager._roi_layer
                if hasattr(roi_layer, 'data') and len(roi_layer.data) > 0:
                    for i, shape in enumerate(roi_layer.data):
                        try:
                            roi_data.append({
                                'id': i,
                                'type': 'rectangle',
                                'vertices': shape.tolist() if hasattr(shape, 'tolist') else str(shape)
                            })
                        except Exception as e:
                            pass
                    
                    if roi_data:
                        return roi_data
            
            return []
            
        except Exception:
            return []  # Return empty list on any error

    def create_roi_image_layers(self, roi_data, shapes_layer):
        """Create individual image layers for each ROI for post-processing"""
        if not self.current_image_path:
            return
        
        try:
            # Get the current image data
            image_layer = None
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                    image_layer = layer
                    break
            
            if image_layer is None:
                return
            
            image_data = image_layer.data
            
            # Remove any existing ROI image layers to avoid duplicates
            layers_to_remove = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and layer.name.startswith('ROI_'):
                    layers_to_remove.append(layer)
            
            for layer in layers_to_remove:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass  # Silently handle layer removal errors
            
            # Create new ROI image layers
            for roi_info in roi_data:
                try:
                    roi_id = roi_info['id']
                    vertices = np.array(roi_info['vertices'])
                    
                    if len(vertices.shape) == 2 and vertices.shape[0] >= 4:
                        # Get bounding box (napari uses y, x coordinates)
                        y_coords = vertices[:, 0]
                        x_coords = vertices[:, 1]
                        
                        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
                        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
                        
                        # Handle multi-dimensional image data
                        if image_data.ndim == 4:  # (T, Z, Y, X) or (T, C, Y, X)
                            roi_image = image_data[:, :, y_min:y_max, x_min:x_max]
                        elif image_data.ndim == 3:  # (T, Y, X) or (Z, Y, X) or (C, Y, X)
                            roi_image = image_data[:, y_min:y_max, x_min:x_max]
                        else:  # 2D image (Y, X)
                            roi_image = image_data[y_min:y_max, x_min:x_max]
                        
                        # Ensure within image bounds
                        if roi_image.size > 0:
                            roi_layer_name = f"ROI_{roi_id + 1}_{os.path.splitext(os.path.basename(self.current_image_path))[0]}"
                            
                            # Add as new image layer
                            self.viewer.add_image(
                                roi_image, 
                                name=roi_layer_name,
                                visible=False  # Start hidden to not clutter the view
                            )
                        
                except Exception as e:
                    pass  # Silently handle individual ROI processing errors
            
        except Exception as e:
            pass  # Silently handle image layer creation errors

    def create_roi_image_layers_from_saved_data(self, rois_data):
        """Create individual image layers for each ROI from saved data"""
        if not self.current_image_path or not rois_data:
            return
        
        try:
            # Get the current image data
            image_layer = None
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                    image_layer = layer
                    break
            
            if image_layer is None:
                return

            image_data = image_layer.data
            
            # Remove any existing ROI image layers to avoid duplicates
            layers_to_remove = []
            for layer in self.viewer.layers:
                if hasattr(layer, 'name') and layer.name.startswith('ROI_'):
                    layers_to_remove.append(layer)
            
            for layer in layers_to_remove:
                try:
                    self.viewer.layers.remove(layer)
                except Exception:
                    pass  # Silently handle layer removal errors
            
            # Create new ROI image layers from saved data
            for i, roi_info in enumerate(rois_data):
                try:
                    roi_id = roi_info.get('id', i)
                    vertices = np.array(roi_info['vertices'])
                    
                    if len(vertices.shape) == 2 and vertices.shape[0] >= 4:
                        # Get bounding box (napari uses y, x coordinates)
                        y_coords = vertices[:, 0]
                        x_coords = vertices[:, 1]
                        
                        y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
                        x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
                        
                        # Handle multi-dimensional image data
                        if image_data.ndim == 4:  # (T, Z, Y, X) or (T, C, Y, X)
                            roi_image = image_data[:, :, y_min:y_max, x_min:x_max]
                        elif image_data.ndim == 3:  # (T, Y, X) or (Z, Y, X) or (C, Y, X)
                            roi_image = image_data[:, y_min:y_max, x_min:x_max]
                        else:  # 2D image (Y, X)
                            roi_image = image_data[y_min:y_max, x_min:x_max]
                        
                        # Ensure within image bounds
                        if roi_image.size > 0:
                            roi_layer_name = f"ROI_{roi_id + 1}_{os.path.splitext(os.path.basename(self.current_image_path))[0]}"
                            
                            # Add as new image layer
                            self.viewer.add_image(
                                roi_image, 
                                name=roi_layer_name,
                                visible=False  # Start hidden to not clutter the view
                            )
                        
                except Exception as e:
                    pass  # Silently handle individual ROI processing errors
            
        except Exception as e:
            pass  # Silently handle image layer creation errors

    def load_rois_data(self, rois_data):
        """Load ROI data into the ROI manager in a safe way"""
        if not self.roi_manager_initialized:
            return
            
        try:
            # Clear existing ROI shapes first
            self.cleanup_existing_roi_layers()
            
            # Create a new shapes layer to hold the loaded ROIs
            shapes_data = []
            for roi_data in rois_data:
                try:
                    if 'vertices' in roi_data:
                        vertices = np.array(roi_data['vertices'])
                        shapes_data.append(vertices)
                    elif 'data' in roi_data and isinstance(roi_data['data'], list):
                        vertices = np.array(roi_data['data'])
                        shapes_data.append(vertices)
                except Exception as e:
                    pass  # Silently handle individual ROI loading errors
            
            if shapes_data:
                # Add shapes layer with loaded ROIs
                roi_layer = self.viewer.add_shapes(
                    shapes_data,
                    name="ROIs",
                    shape_type='rectangle',
                    edge_color='red',
                    face_color='red',
                    opacity=0.3
                )
                
                # Connect to data change events
                roi_layer.events.data.connect(self.on_shapes_layer_changed)
                
                # Update the roi_layer reference for fallback mode
                if not self.roi_manager:
                    self.roi_layer = roi_layer
                
                # Make sure the ROI layer is visible and selected
                roi_layer.visible = True
                self.viewer.layers.selection.active = roi_layer
            
            # Also try to add ROIs to the ROI manager if it exists
            if self.roi_manager:
                try:
                    # Clear existing ROIs from manager
                    if hasattr(self.roi_manager, 'clear_rois'):
                        self.roi_manager.clear_rois()
                    
                    # Try different methods to add ROIs to manager
                    for roi_data in rois_data:
                        try:
                            if hasattr(self.roi_manager, 'add_roi'):
                                self.roi_manager.add_roi(roi_data)
                            elif hasattr(self.roi_manager, '_roi_layer') and self.roi_manager._roi_layer:
                                # Add to the underlying layer directly
                                roi_layer = self.roi_manager._roi_layer
                                if 'vertices' in roi_data:
                                    vertices = np.array(roi_data['vertices'])
                                    roi_layer.add([vertices], shape_type='rectangle')
                        except Exception as e:
                            pass  # Silently handle ROI manager loading errors
                except Exception as e:
                    pass  # Silently handle ROI manager operations
                    
        except Exception as e:
            pass  # Silently handle general ROI data loading errors

    def on_roi_changed(self, *args, **kwargs):
        """Handle ROI changes"""
        try:
            # Get the current image layer
            image_layer = None
            for layer in self.viewer.layers:
                # Check if it's an image layer (not a shapes layer or ROI layer)
                if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image' and not layer.name.startswith('ROI_'):
                    image_layer = layer
                    break

            if image_layer is not None:
                # Get ROIs and process each one individually
                rois_data = self.get_rois_data()
                if rois_data:
                    # Find the ROI image layers that were created
                    roi_image_layers = []
                    for layer in self.viewer.layers:
                        if hasattr(layer, 'name') and layer.name.startswith('ROI_'):
                            roi_image_layers.append(layer)
                    
                    # Emit signal for each ROI image layer
                    for roi_layer in roi_image_layers:
                        try:
                            # Extract the ROI data from the image layer
                            roi_image_data = roi_layer.data
                            
                            # Handle multi-dimensional data - get 2D slice if needed
                            if roi_image_data.ndim > 2:
                                # For time series or multi-channel, take first frame/channel
                                while roi_image_data.ndim > 2:
                                    roi_image_data = roi_image_data[0]
                            
                            self.roi_updated.emit(roi_image_data, image_layer)
                            
                        except Exception as e:
                            pass  # Silently handle individual ROI update errors

            # Auto-save if enabled
            if hasattr(self, 'auto_save_check') and self.auto_save_check.isChecked():
                self.save_rois(auto=True)
                
        except Exception as e:
            pass  # Silently handle ROI change processing errors

    def calculate_measurements(self):
        """Calculate measurements for all ROIs"""
        if not self.roi_manager_initialized or not self.roi_manager:
            QMessageBox.warning(self, "ROI Manager Not Ready", "Please initialize ROI Manager first.")
            return
            
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        try:
            rois_data = self.get_rois_data()
            if not rois_data:
                QMessageBox.warning(self, "No ROIs", "Please create some ROIs first.")
                return

            # Get the image data
            image_layer = None
            for layer in self.viewer.layers:
                if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                    image_layer = layer
                    break

            if image_layer is None:
                QMessageBox.warning(self, "No Image", "No image found to measure.")
                return

            # Calculate measurements for each ROI
            results = []
            
            # Try to use ROI manager's measurement capabilities first
            try:
                if hasattr(self.roi_manager, 'get_rois'):
                    rois = self.roi_manager.get_rois()
                    for i, roi in enumerate(rois):
                        if hasattr(roi, 'get_measurements'):
                            measurements = roi.get_measurements(image_layer.data)
                            result = {
                                'ROI_ID': i + 1,
                                'Frame': 0,
                                **measurements
                            }
                            results.append(result)
                        elif hasattr(roi, 'data'):
                            # Manual calculation fallback
                            result = self.calculate_roi_measurements(roi.data, image_layer.data, i + 1)
                            if result:
                                results.append(result)
            except Exception:
                pass  # Silently handle ROI manager measurement errors
            
            # Fallback: manual calculation using ROI layer
            if not results:
                if hasattr(self.roi_manager, '_roi_layer') and self.roi_manager._roi_layer:
                    roi_layer = self.roi_manager._roi_layer
                    if hasattr(roi_layer, 'data'):
                        for i, shape in enumerate(roi_layer.data):
                            result = self.calculate_roi_measurements(shape, image_layer.data, i + 1)
                            if result:
                                results.append(result)

            # Convert to DataFrame and emit
            if results:
                df = pd.DataFrame(results)
                self.measurements_ready.emit(df)
                QMessageBox.information(self, "Success", f"Calculated measurements for {len(results)} ROIs")
            else:
                QMessageBox.warning(self, "No Measurements", "No valid ROI measurements could be calculated.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate measurements: {str(e)}")

    def calculate_roi_measurements(self, roi_shape, image_data, roi_id):
        """Calculate measurements for a single ROI shape"""
        try:
            # Convert to numpy array if needed
            if not isinstance(roi_shape, np.ndarray):
                roi_shape = np.array(roi_shape)
            
            # Handle different shape formats
            if len(roi_shape.shape) == 2 and roi_shape.shape[0] == 4:  # Rectangle with 4 corners
                # Get bounding box coordinates (napari uses y, x)
                y_coords = roi_shape[:, 0]
                x_coords = roi_shape[:, 1]
                
                y_min, y_max = int(np.floor(y_coords.min())), int(np.ceil(y_coords.max()))
                x_min, x_max = int(np.floor(x_coords.min())), int(np.ceil(x_coords.max()))
                
                # Get image data (handle 2D or higher dimensions)
                if image_data.ndim > 2:
                    # For multi-dimensional data, take the first frame/channel
                    current_data = image_data
                    while current_data.ndim > 2:
                        current_data = current_data[0]
                else:
                    current_data = image_data
                
                # Ensure within image bounds
                y_min = max(0, y_min)
                x_min = max(0, x_min)
                y_max = min(current_data.shape[0], y_max)
                x_max = min(current_data.shape[1], x_max)
                
                if y_min < y_max and x_min < x_max:
                    # Extract ROI region
                    roi_region = current_data[y_min:y_max, x_min:x_max]
                    
                    # Calculate metrics
                    return {
                        'ROI_ID': roi_id,
                        'Frame': 0,
                        'Max_Density': float(np.max(roi_region)),
                        'Mean_Density': float(np.mean(roi_region)),
                        'Min_Density': float(np.min(roi_region)),
                        'Std_Density': float(np.std(roi_region)),
                        'Area': int((y_max - y_min) * (x_max - x_min))
                    }
            
            return None
            
        except Exception as e:
            return None