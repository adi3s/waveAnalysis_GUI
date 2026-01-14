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
        self.previous_roi_count = 0  # Track previous ROI count for auto-save
        self.is_drawing = False  # Track if actively drawing
        self.auto_save_timer = None  # Timer to debounce auto-save
        self.is_saving = False  # Flag to prevent save loops
        self.roi_count_monitor_timer = None  # Timer to monitor ROI count changes
        self.is_switching_images = False  # Flag to prevent auto-save during image switch
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
            
            # Get the ROI Manager's layer after a brief delay to ensure it's created
            QTimer.singleShot(100, self._get_roi_manager_layer)
            
            if self.current_image_path:
                QTimer.singleShot(150, lambda: self.load_image_rois(self.current_image_path))
            
            QMessageBox.information(self, "Success", "ROI Manager initialized successfully!")
            
        except Exception as e:
            self.status_label.setText(f"Status: Error - {str(e)}")
            self.fallback_container.setVisible(True)
            self.roi_manager_container.setVisible(False)
            QMessageBox.warning(self, "ROI Manager Error", 
                              f"Failed to initialize: {str(e)}\n\nUsing fallback mode.")

    def _get_roi_manager_layer(self):
        """Get the ROI layer created by the ROI Manager and connect to table changes."""
        try:
            if self.roi_manager and hasattr(self.roi_manager, '_layer'):
                self.roi_layer = self.roi_manager._layer
                
                # First, inspect the ROI Manager structure
                self._inspect_roi_manager()
                
                # Connect to ROI Manager table changes for auto-save
                if hasattr(self.roi_manager, '_roilist'):
                    roi_list = self.roi_manager._roilist
                    print(f"Found _roilist widget: {type(roi_list).__name__}")
                    
                    # Connect to model signals (QRoiListWidget is likely a QTableWidget or similar)
                    if hasattr(roi_list, 'model') and roi_list.model():
                        model = roi_list.model()
                        print(f"Model type: {type(model).__name__}")
                        
                        if hasattr(model, 'rowsInserted'):
                            model.rowsInserted.connect(self.on_roi_table_changed)
                            print("Connected to rowsInserted signal")
                        if hasattr(model, 'rowsRemoved'):
                            model.rowsRemoved.connect(self.on_roi_table_changed)
                            print("Connected to rowsRemoved signal")
                        if hasattr(model, 'dataChanged'):
                            model.dataChanged.connect(self.on_roi_table_changed)
                            print("Connected to dataChanged signal")
                    
                    # Also try direct widget signals
                    if hasattr(roi_list, 'itemChanged'):
                        roi_list.itemChanged.connect(self.on_roi_table_changed)
                        print("Connected to itemChanged signal")
                    
                    if hasattr(roi_list, 'cellChanged'):
                        roi_list.cellChanged.connect(self.on_roi_table_changed)
                        print("Connected to cellChanged signal")
                    
                    # Start monitoring timer as backup
                    self.roi_count_monitor_timer = QTimer()
                    self.roi_count_monitor_timer.timeout.connect(self.check_roi_count_changed)
                    self.roi_count_monitor_timer.start(500)  # Check every 500ms
                    print("Started ROI count monitor timer")
                    
                self.previous_roi_count = 0
            else:
                # Fallback: search for RoiManagerLayer in viewer
                for layer in self.viewer.layers:
                    if type(layer).__name__ == 'RoiManagerLayer':
                        self.roi_layer = layer
                        self.previous_roi_count = 0
                        break
        except Exception as e:
            print(f"Error getting ROI Manager layer: {e}")
    
    def _inspect_roi_manager(self):
        """Inspect ROI Manager structure to find available widgets and signals."""
        try:
            print("\n=== ROI Manager Inspection ===")
            print(f"ROI Manager type: {type(self.roi_manager).__name__}")
            
            # List all attributes including private ones
            print("\nAll ROI Manager attributes (including private):")
            for attr in dir(self.roi_manager):
                if not callable(getattr(self.roi_manager, attr, None)):
                    try:
                        value = getattr(self.roi_manager, attr)
                        print(f"  {attr}: {type(value).__name__}")
                    except:
                        pass
            
            # Check all child widgets of any type
            print("\nAll child QWidget descendants:")
            from qtpy.QtWidgets import QWidget
            for child in self.roi_manager.findChildren(QWidget):
                widget_type = type(child).__name__
                obj_name = child.objectName() if child.objectName() else "(no name)"
                print(f"  {widget_type}: {obj_name}")
                
                # Check if this widget has count() method (lists/tables)
                if hasattr(child, 'count'):
                    try:
                        count = child.count()
                        print(f"    -> Has count() method: {count} items")
                    except:
                        pass
                
                # Check for rowCount (tables)
                if hasattr(child, 'rowCount'):
                    try:
                        rows = child.rowCount()
                        print(f"    -> Has rowCount() method: {rows} rows")
                    except:
                        pass
            
            # Check all buttons
            print("\nAll buttons:")
            from qtpy.QtWidgets import QPushButton, QAbstractButton
            for btn in self.roi_manager.findChildren(QAbstractButton):
                btn_text = btn.text() if hasattr(btn, 'text') else ""
                print(f"  {type(btn).__name__}: '{btn_text}'")
                
            print("=== End Inspection ===\n")
            
        except Exception as e:
            print(f"Error inspecting ROI Manager: {e}")
            import traceback
            traceback.print_exc()
    
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
        
        # Initialize ROI count
        self.previous_roi_count = 0
        
        self._enable_operations(True)
        self.status_label.setText("Status: Fallback ROI layer created")
        self.init_roi_btn.setText("ROI Layer Created")
        self.init_roi_btn.setEnabled(False)
        self.roi_manager_initialized = True
        
        QMessageBox.information(self, "Success", "Fallback ROI layer created!")

    def on_roi_table_changed(self, *args, **kwargs):
        """Handle when ROI Manager table changes (items added/removed/modified)."""
        try:
            # Prevent triggering during save operation or image switching
            if self.is_saving or self.is_switching_images:
                return
            
            # Update labels
            QTimer.singleShot(50, self.update_roi_labels)
            
            # Trigger auto-save if enabled (with debouncing)
            if hasattr(self, 'auto_save_check') and self.auto_save_check.isChecked():
                # Cancel any pending auto-save
                if self.auto_save_timer is not None:
                    self.auto_save_timer.stop()
                    self.auto_save_timer = None
                
                # Create new timer with 300ms delay (debounce)
                self.auto_save_timer = QTimer()
                self.auto_save_timer.setSingleShot(True)
                self.auto_save_timer.timeout.connect(self.trigger_auto_save)
                self.auto_save_timer.start(300)
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    def check_roi_count_changed(self):
        """Periodically check if ROI count changed (backup method for detecting changes)."""
        try:
            # Only check if auto-save is enabled, not currently saving, and not switching images
            if not (hasattr(self, 'auto_save_check') and self.auto_save_check.isChecked()):
                return
            
            if self.is_saving or self.is_switching_images:
                return
            
            # Get current ROI count from _roilist
            current_count = 0
            if self.roi_manager and hasattr(self.roi_manager, '_roilist'):
                roi_list = self.roi_manager._roilist
                if hasattr(roi_list, 'rowCount'):
                    current_count = roi_list.rowCount()
            
            # If count changed, trigger table changed handler
            if current_count != self.previous_roi_count:
                print(f"ROI count changed: {self.previous_roi_count} -> {current_count}")
                self.previous_roi_count = current_count
                self.on_roi_table_changed()
        except Exception as e:
            pass


    
    def trigger_auto_save(self):
        """Trigger the actual save operation or delete file if table is empty."""
        try:
            if self.is_saving:
                return
                
            self.is_saving = True
            
            if self.current_image_path:
                # Check if ROI table is empty using _roilist
                table_count = 0
                if self.roi_manager and hasattr(self.roi_manager, '_roilist'):
                    roi_list = self.roi_manager._roilist
                    if hasattr(roi_list, 'rowCount'):
                        table_count = roi_list.rowCount()
                
                print(f"Auto-save triggered: ROI count = {table_count}")
                
                # Use the same file path method as save_rois
                roi_filepath = self._get_roi_file_path(self.current_image_path)
                
                print(f"ROI file path: {roi_filepath}")
                print(f"File exists: {os.path.exists(roi_filepath)}")
                
                if table_count == 0:
                    # Delete the ROI file if it exists and clear internal tracking
                    if os.path.exists(roi_filepath):
                        os.remove(roi_filepath)
                        print(f"Deleted ROI file: {roi_filepath}")
                    
                    # Update internal tracking
                    if self.current_image_path in self.per_image_rois:
                        self.per_image_rois[self.current_image_path] = []
                    self.update_roi_scope_label()
                else:
                    # Save ROIs
                    self.save_rois(auto=True)
                    print(f"Auto-saved {table_count} ROIs")
            
            self.is_saving = False
        except Exception as e:
            self.is_saving = False
            import traceback
            traceback.print_exc()

    def _clear_roi_manager_table(self):
        """Clear all ROIs from the ROI Manager table and layer."""
        try:
            from qtpy.QtWidgets import QApplication
            
            # Get the roi_layer
            roi_layer = self.roi_layer
            if not roi_layer:
                return
            
            # If there are existing shapes, remove them all
            if hasattr(roi_layer, 'data') and len(roi_layer.data) > 0:
                # Select all shapes
                roi_layer.selected_data = set(range(len(roi_layer.data)))
                QApplication.processEvents()
                
                # Remove selected shapes
                roi_layer.remove_selected()
                QApplication.processEvents()
                
                print(f"Cleared ROI Manager table")
        except Exception as e:
            print(f"Error clearing ROI Manager table: {e}")
            # Fallback: just clear the data
            try:
                if roi_layer and hasattr(roi_layer, 'data'):
                    roi_layer.data = []
            except:
                pass

    def _enable_auto_save_after_switch(self):
        """Re-enable auto-save after image switching is complete."""
        self.is_switching_images = False
        print("Auto-save monitoring resumed after image switch")

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
        
        # Disable auto-save during image switching (only if auto-save is enabled)
        auto_save_was_enabled = self.auto_save_check.isChecked()
        if auto_save_was_enabled:
            self.is_switching_images = True
        
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            shapes_data = self._load_rois_from_file(image_path)
            
            # Clear ROI Manager table if no ROIs loaded
            if not shapes_data and self.roi_manager and hasattr(self.roi_manager, '_roilist'):
                self._clear_roi_manager_table()
            
            roi_layer, roi_layer_index = self._find_roi_layer_with_index()
            
            if roi_layer is None and self.roi_manager_initialized:
                self._create_new_roi_layer(shapes_data, image_name)
            elif roi_layer is not None:
                self._update_existing_roi_layer(roi_layer, roi_layer_index, shapes_data, image_name)
            elif not self.roi_manager_initialized:
                self.status_label.setText("Status: Click 'Initialize ROI Manager' to begin")
                self.init_roi_btn.setEnabled(True)
            
            # Update ROI count tracking after loading
            if self.roi_layer and hasattr(self.roi_layer, 'data'):
                self.previous_roi_count = len(self.roi_layer.data)
            else:
                self.previous_roi_count = 0
            
            self.update_roi_scope_label()
            
            # Update ROI labels if they are currently being displayed
            if self.show_labels_check.isChecked():
                QTimer.singleShot(100, self.update_roi_labels)
        
        finally:
            # Re-enable auto-save after a short delay to ensure all operations complete
            if auto_save_was_enabled:
                QTimer.singleShot(500, self._enable_auto_save_after_switch)

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
            roi_file = self._get_roi_file_path(self.current_image_path)
            os.makedirs(os.path.dirname(roi_file), exist_ok=True)
            
            if not rois:
                # If no ROIs, don't save anything (let trigger_auto_save handle deletion)
                if not auto:
                    QMessageBox.warning(self, "No ROIs", "No ROIs to save.")
                return

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
            
            # If using ROI Manager, get data from the table, not the layer
            if self.roi_manager and hasattr(self.roi_manager, '_roilist'):
                roi_list = self.roi_manager._roilist
                if hasattr(roi_list, 'rowCount'):
                    row_count = roi_list.rowCount()
                    
                    # Get ROI layer to access the actual shape data
                    roi_layer = None
                    for layer in self.viewer.layers:
                        if type(layer).__name__ in ['RoiManagerLayer', 'Shapes']:
                            roi_layer = layer
                            break
                    
                    if roi_layer and hasattr(roi_layer, 'data'):
                        # Only get ROIs that are in the table
                        for row in range(row_count):
                            try:
                                # Get the shape index from the table
                                # The table stores which shapes are managed
                                if row < len(roi_layer.data):
                                    shape = roi_layer.data[row]
                                    roi_data.append({
                                        'id': row + 1,  # Use 1-based indexing
                                        'type': 'rectangle',
                                        'vertices': shape.tolist() if hasattr(shape, 'tolist') else list(shape)
                                    })
                            except Exception:
                                pass
                    
                    if roi_data and roi_layer:
                        self.create_roi_image_layers(roi_data, roi_layer)
                    return roi_data
            
            # Fallback: get from layer directly (for non-ROI Manager mode)
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
        """Load ROI data from file automatically if it exists."""
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
            
            # Automatically load ROIs without prompting
            self.per_image_rois[image_path] = []
            for roi_info in rois:
                if 'vertices' in roi_info:
                    vertices = np.array(roi_info['vertices'], dtype=np.float64)
                    self.per_image_rois[image_path].append(vertices)
                    shapes_data.append(vertices)
            self.status_label.setText(f"Status: Loaded {len(rois)} ROIs for {image_name}")
        except Exception as e:
            print(f"Error loading ROIs from file: {e}")
        
        return shapes_data
    
    def _create_new_roi_layer(self, shapes_data, image_name):
        """Create a new ROI layer or use the ROI Manager's layer."""
        try:
            # If we have a ROI Manager, use its layer
            if self.roi_manager and hasattr(self.roi_manager, '_layer'):
                self.roi_layer = self.roi_manager._layer
                # Add shapes to the existing layer
                if shapes_data:
                    self._add_shapes_to_roi_layer(self.roi_layer, shapes_data)
                    self.status_label.setText(f"Status: Loaded {len(shapes_data)} ROIs for {image_name}")
                else:
                    self.status_label.setText(f"Status: Ready to draw ROIs for {image_name}")
            else:
                # Fallback: create our own shapes layer
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
        except Exception as e:
            print(f"Error creating ROI layer: {e}")
    
    def _add_shapes_to_roi_layer(self, roi_layer, shapes_data):
        """Add shapes to the ROI layer, handling ROI Manager compatibility."""
        try:
            # If using ROI Manager, clear both visual layer and table, then add new shapes
            if self.roi_manager and hasattr(self.roi_manager, '_layer'):
                self._clear_roi_manager_table()
                self._sync_shapes_to_roi_manager(roi_layer, shapes_data)
            else:
                # Fallback: clear and add all shapes at once
                if len(roi_layer.data) > 0:
                    roi_layer.data = []
                if shapes_data:
                    roi_layer.add_rectangles(shapes_data)
        except Exception as e:
            print(f"Error adding shapes to ROI layer: {e}")
    
    def _sync_shapes_to_roi_manager(self, roi_layer, shapes_data):
        """Sync shapes to the ROI Manager table by adding them one at a time and clicking Add button."""
        try:
            from qtpy.QtWidgets import QPushButton, QApplication
            
            # Find the Add button in the ROI Manager
            add_btn = None
            for btn in self.roi_manager.findChildren(QPushButton):
                btn_text = btn.text().lower()
                if "add" in btn_text and "layer" not in btn_text:
                    add_btn = btn
                    break
            
            if not add_btn:
                print("Could not find Add button in ROI Manager - using fallback method")
                # Fallback: just add all shapes at once
                roi_layer.add_rectangles(shapes_data)
                return
            
            # Make sure the ROI layer is selected
            self.viewer.layers.selection.active = roi_layer
            QApplication.processEvents()
            
            # Add shapes one at a time and register each with the ROI Manager
            for i, shape in enumerate(shapes_data):
                # Add single rectangle
                roi_layer.add_rectangles([shape])
                QApplication.processEvents()
                
                # Select the newly added shape
                new_index = len(roi_layer.data) - 1
                roi_layer.selected_data = {new_index}
                QApplication.processEvents()
                
                # Click the Add button to register it in the ROI Manager table
                add_btn.click()
                QApplication.processEvents()
            
            # Clear selection after all shapes are added
            roi_layer.selected_data = set()
            print(f"Successfully synced {len(shapes_data)} ROIs to ROI Manager table")
            
        except Exception as e:
            print(f"Error syncing to ROI Manager: {e}")
            # Fallback: just add all shapes
            try:
                roi_layer.add_rectangles(shapes_data)
            except:
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
            
            # Load ROIs if we have shapes_data
            if shapes_data:
                self._add_shapes_to_roi_layer(roi_layer, shapes_data)
                self.status_label.setText(f"Status: Loaded {len(shapes_data)} ROIs for {image_name}")
            else:
                # Clear existing ROIs when switching to an image with no saved ROIs
                if len(roi_layer.data) > 0:
                    roi_layer.data = []
                self.status_label.setText(f"Status: Ready to draw ROIs for {image_name}")
            
            roi_layer.visible = True
            
            if roi_layer_index is not None and roi_layer_index < len(self.viewer.layers) - 1:
                self.viewer.layers.move(roi_layer_index, len(self.viewer.layers) - 1)
            
            self.viewer.layers.selection.active = roi_layer
            
        except Exception as e:
            print(f"Error updating ROI layer: {e}")
    
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
