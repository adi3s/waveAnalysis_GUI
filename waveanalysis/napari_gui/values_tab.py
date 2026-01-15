import os
import json
from pathlib import Path
from qtpy.QtWidgets import (
    QWidget, QScrollArea, QVBoxLayout, QPushButton, QLabel, QGroupBox,
    QComboBox, QSpinBox, QFormLayout, QCheckBox, QLineEdit, QFileDialog,
    QListWidget, QHBoxLayout
)
from qtpy.QtCore import Qt, Signal
from .styles import BUTTON_STYLE

class ValuesTab(QWidget):
    """
    Tab for setting analysis parameters and selecting analysis type.
    
    Supports loading multiple images/movies and managing analysis parameters
    for different types of wave analysis (Standard, Rolling, Kymograph).
    """
    image_loaded = Signal(str)  # Signal to notify when an image is loaded
    images_updated = Signal(list)  # Signal to notify when image list changes
    analysis_type_changed = Signal(str)  # Signal to notify when analysis type changes
    reset_requested = Signal()  # Signal to request global app reset

    def __init__(self, parent):
        """Initialize the ValuesTab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.image_files = []  # Store list of loaded image files
        self.current_image_path = None
        self.last_directory = ""  # Remember last used directory for file dialogs
        self.init_ui()
        self.setup_parameter_groups()
        self.update_visible_params()

    def init_ui(self):
        """Initialize the user interface with scrollable layout for multiple images"""
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

        # Parameter Groups
        self.standard_params = QGroupBox("Standard Parameters")
        self.rolling_params = QGroupBox("Rolling Parameters")
        self.kymo_params = QGroupBox("Kymograph Parameters")
        self.common_params = QGroupBox("Common Parameters")

        # Analysis Type Selection
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["Standard", "Rolling", "Kymograph"])
        self.analysis_combo.currentIndexChanged.connect(self.on_analysis_type_changed)

        # Load Image Section with multiple image support
        load_group = QGroupBox("Image/Movie Controls")
        load_layout = QVBoxLayout()
        
        # Image list with scroll area
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(200)  # Set maximum height to enable scrolling
        self.image_list.itemSelectionChanged.connect(self.on_image_selection_changed)
        load_layout.addWidget(QLabel("Loaded Images/Movies:"))
        load_layout.addWidget(self.image_list)
        
        # Buttons for image management
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        
        self.load_btn = QPushButton("Load")
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setToolTip("Load image or movie files")
        self.load_btn.setStyleSheet(BUTTON_STYLE)
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.remove_image)
        self.remove_btn.setToolTip("Remove selected image from list")
        self.remove_btn.setStyleSheet(BUTTON_STYLE)
        
        self.clear_all_btn = QPushButton("Clear All")
        self.clear_all_btn.clicked.connect(self.clear_all_images)
        self.clear_all_btn.setToolTip("Clear all loaded images")
        self.clear_all_btn.setStyleSheet(BUTTON_STYLE)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.request_reset)
        self.reset_btn.setToolTip("Reset the application to its initial state")
        self.reset_btn.setStyleSheet(BUTTON_STYLE)
        
        button_layout.addWidget(self.load_btn)
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.clear_all_btn)
        button_layout.addWidget(self.reset_btn)
        load_layout.addLayout(button_layout)
        
        # Movie info label
        self.movie_info_label = QLabel("Supported formats: TIFF, PNG, JPG, LSM (2D movies as TIFF stacks)")
        self.movie_info_label.setWordWrap(True)
        load_layout.addWidget(self.movie_info_label)
        
        load_group.setLayout(load_layout)

        layout.addWidget(QLabel("Analysis Type:"))
        layout.addWidget(self.analysis_combo)
        layout.addWidget(self.standard_params)
        layout.addWidget(self.rolling_params)
        layout.addWidget(self.kymo_params)
        layout.addWidget(self.common_params)
        layout.addWidget(load_group)
        
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

    def load_image(self):
        """Load one or more image/movie files and add them to the list"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Image/Movie(s)", self.last_directory, 
            "Image/Movie files (*.tif *.tiff *.png *.jpg *.jpeg *.lsm)"
        )
        
        if file_paths:
            # Save the directory of the first selected file
            self.last_directory = os.path.dirname(file_paths[0])
            
            # Track newly added files
            newly_added = []
            
            for file_path in file_paths:
                # Check if file is already loaded
                if file_path in self.image_files:
                    continue
                
                # Add to image list
                self.image_files.append(file_path)
                item_text = f"{Path(file_path).name}"
                
                # Try to get image info (this could be enhanced to read actual image data)
                try:
                    # Add file size info
                    file_size = os.path.getsize(file_path)
                    if file_size > 1024 * 1024:  # > 1MB
                        size_str = f" ({file_size / (1024 * 1024):.1f} MB)"
                    else:
                        size_str = f" ({file_size / 1024:.1f} KB)"
                    item_text += size_str
                except Exception:
                    pass  # If we can't get size, just use filename
                
                self.image_list.addItem(item_text)
                newly_added.append(file_path)
            
            if newly_added:
                # Set the first newly added as current image
                self.current_image_path = newly_added[0]
                
                # Emit signal only for the first newly added file to display it
                self.image_loaded.emit(newly_added[0])
                
                # Emit update signal once for all changes
                self.images_updated.emit(self.image_files.copy())
            else:
                from qtpy.QtWidgets import QMessageBox
                QMessageBox.information(self, "Already Loaded", 
                                      "All selected files are already in the list.")

    def remove_image(self):
        """Remove selected image from list"""
        current_row = self.image_list.currentRow()
        if current_row >= 0:
            # Remove from list
            removed_file = self.image_files.pop(current_row)
            self.image_list.takeItem(current_row)
            
            # Update current image path
            if self.image_files:
                self.current_image_path = self.image_files[0]
            else:
                self.current_image_path = None
            
            # Emit signal
            self.images_updated.emit(self.image_files.copy())

    def clear_all_images(self):
        """Clear all loaded images"""
        self.image_files.clear()
        self.image_list.clear()
        self.current_image_path = None
        self.images_updated.emit(self.image_files.copy())

    def request_reset(self):
        """Request a global app reset, preserving last directory."""
        self.reset_requested.emit()

    def reset_state(self):
        """Reset this tab to its initial state, preserving last_directory."""
        # Clear images (but keep last_directory)
        self.image_files.clear()
        self.image_list.clear()
        self.current_image_path = None
        
        # Reset analysis type to default (Standard)
        self.analysis_combo.setCurrentIndex(0)
        
        # Reset Standard parameters
        self.std_box_size.setValue(20)
        self.std_bin_shift.setValue(20)
        self.std_preview_check.setChecked(False)
        
        # Reset Rolling parameters
        self.roll_box_size.setValue(20)
        self.roll_bin_shift.setValue(20)
        self.roll_sub_size.setValue(50)
        self.roll_sub_shift.setValue(5)
        self.roll_preview_check.setChecked(False)
        
        # Reset Kymograph parameters
        self.kymo_line_width.setValue(5)
        self.kymo_bin_shift.setValue(5)
        self.calc_speed.setChecked(False)
        self.kymo_preview_check.setChecked(False)
        
        # Reset Common parameters
        self.group_names.clear()
        
        # Update visibility
        self.update_visible_params()
        
        # Emit update signal
        self.images_updated.emit(self.image_files.copy())

    def get_loaded_images(self):
        """Get list of all loaded image file paths"""
        return self.image_files.copy()

    def get_current_image(self):
        """Get the currently selected or most recently loaded image"""
        current_row = self.image_list.currentRow()
        if current_row >= 0 and current_row < len(self.image_files):
            return self.image_files[current_row]
        elif self.image_files:
            return self.image_files[0]  # Return first image if none selected
        return None
    
    def on_image_selection_changed(self):
        """Handle when user selects a different image from the list"""
        current_row = self.image_list.currentRow()
        if current_row >= 0 and current_row < len(self.image_files):
            selected_image = self.image_files[current_row]
            # Notify parent to display this image in viewer
            self.image_loaded.emit(selected_image)
    
    def on_analysis_type_changed(self):
        """Handle analysis type change"""
        self.update_visible_params()
        analysis_type = self.analysis_combo.currentText().lower()
        self.analysis_type_changed.emit(analysis_type)

    def setup_parameter_groups(self):
        """Setup the parameter groups for the 3 analysis types"""
        # Standard Parameters
        self.std_box_size = QSpinBox()
        self.std_box_size.setRange(1, 1000)
        self.std_box_size.setValue(20)  # Default value
        self.std_bin_shift = QSpinBox()
        self.std_bin_shift.setRange(1, 100)
        self.std_bin_shift.setValue(20)  # Default value
        
        std_layout = QFormLayout()
        std_layout.addRow("Box Size (px):", self.std_box_size)
        std_layout.addRow("Bin Shift (px):", self.std_bin_shift)
        
        # Add preview checkbox
        self.std_preview_check = QCheckBox("Show Box Grid Preview")
        self.std_preview_check.stateChanged.connect(lambda: self.preview_boxes("standard"))
        std_layout.addRow(self.std_preview_check)
        
        self.standard_params.setLayout(std_layout)

        # Rolling Parameters
        self.roll_box_size = QSpinBox()
        self.roll_box_size.setRange(1, 1000)
        self.roll_box_size.setValue(20)  # Default value
        self.roll_bin_shift = QSpinBox()
        self.roll_bin_shift.setRange(1, 100)
        self.roll_bin_shift.setValue(20)  # Default value
        self.roll_sub_size = QSpinBox()
        self.roll_sub_size.setRange(1, 1000)
        self.roll_sub_size.setValue(50)  # Default value
        self.roll_sub_shift = QSpinBox()
        self.roll_sub_shift.setRange(1, 100)
        self.roll_sub_shift.setValue(5)  # Default value
        
        roll_layout = QFormLayout()
        roll_layout.addRow("Box Size (px):", self.roll_box_size)
        roll_layout.addRow("Bin Shift (px):", self.roll_bin_shift)
        roll_layout.addRow("Subframe Size:", self.roll_sub_size)
        roll_layout.addRow("Subframe Shift:", self.roll_sub_shift)
        
        # Add preview checkbox
        self.roll_preview_check = QCheckBox("Show Box Grid Preview")
        self.roll_preview_check.stateChanged.connect(lambda: self.preview_boxes("rolling"))
        roll_layout.addRow(self.roll_preview_check)
        
        self.rolling_params.setLayout(roll_layout)

        # Kymograph Parameters
        self.kymo_line_width = QSpinBox()
        self.kymo_line_width.setRange(1, 1000)
        self.kymo_line_width.setValue(5)  # Default value
        self.kymo_bin_shift = QSpinBox()
        self.kymo_bin_shift.setRange(1, 100)
        self.kymo_bin_shift.setValue(5)  # Default value
        self.calc_speed = QCheckBox("Calculate Wave Speeds")
        
        kymo_layout = QFormLayout()
        kymo_layout.addRow("Line Width (px):", self.kymo_line_width)
        kymo_layout.addRow("Bin Shift (px):", self.kymo_bin_shift)
        kymo_layout.addRow(self.calc_speed)
        
        # Add preview checkbox
        self.kymo_preview_check = QCheckBox("Show Line Grid Preview")
        self.kymo_preview_check.stateChanged.connect(lambda: self.preview_boxes("kymograph"))
        kymo_layout.addRow(self.kymo_preview_check)
        
        self.kymo_params.setLayout(kymo_layout)

        # Common Parameters
        self.group_names = QLineEdit()
        self.group_names.setPlaceholderText("Comma-separated group names")
        
        common_layout = QFormLayout()
        common_layout.addRow("Group Names:", self.group_names)
        self.common_params.setLayout(common_layout)    

    def update_visible_params(self):
        """Update the visibility of parameter groups based on selected analysis type"""
        analysis_type = self.analysis_combo.currentText().lower()
        
        # Show/hide parameter groups
        self.standard_params.setVisible(analysis_type == "standard")
        self.rolling_params.setVisible(analysis_type == "rolling")
        self.kymo_params.setVisible(analysis_type == "kymograph")
        
        # Show group names for standard and kymograph analyses
        self.common_params.setVisible(analysis_type in ["standard", "kymograph"])
        
        # Notify pre-process tab about analysis type change
        if hasattr(self.parent, 'pre_process_tab'):
            self.parent.pre_process_tab.set_analysis_type(analysis_type)
    
    def save_parameters(self):
        """Save the parameters and update the parent widget"""
        params = self.get_params()
        self.parent.update_params(params)

    def get_params(self):
        """Get the parameters for the selected analysis type"""
        analysis_type = self.analysis_combo.currentText().lower()
        params = {
            "type": analysis_type,
            "group_names": [n.strip() for n in self.group_names.text().split(",") if n.strip()],
            "loaded_images": self.image_files.copy(),
            "current_image": self.get_current_image(),
            "total_images": len(self.image_files)
        }

        if analysis_type == "standard":
            params.update({
                "box_size": self.std_box_size.value(),
                "bin_shift": self.std_bin_shift.value()
            })
        elif analysis_type == "rolling":
            params.update({
                "box_size": self.roll_box_size.value(),
                "bin_shift": self.roll_bin_shift.value(),
                "subframe_size": self.roll_sub_size.value(),
                "subframe_shift": self.roll_sub_shift.value()
            })
        elif analysis_type == "kymograph":
            params.update({
                "line_width": self.kymo_line_width.value(),
                "bin_shift": self.kymo_bin_shift.value(),
                "calc_wave_speeds": self.calc_speed.isChecked()
            })
        return params

    def validate_inputs(self):
        """Validate the inputs for the selected analysis type"""
        analysis_type = self.analysis_combo.currentText().lower()
        errors = []

        if analysis_type == "standard":
            if not self.std_box_size.value():
                errors.append("Box size is required for standard analysis")
            if not self.std_bin_shift.value():
                errors.append("Bin shift is required for standard analysis")
        elif analysis_type == "rolling":
            if not self.roll_box_size.value():
                errors.append("Box size is required for rolling analysis")
            if not self.roll_bin_shift.value():
                errors.append("Bin shift is required for rolling analysis")
            if not self.roll_sub_size.value():
                errors.append("Subframe size is required for rolling analysis")
            if not self.roll_sub_shift.value():
                errors.append("Subframe shift is required for rolling analysis")
        elif analysis_type == "kymograph":
            if not self.kymo_line_width.value():
                errors.append("Line width is required for kymograph analysis")
            if not self.kymo_bin_shift.value():
                errors.append("Bin shift is required for kymograph analysis")

        # Only require group names if multiple images are loaded
        if analysis_type in ["standard", "kymograph"]:
            if len(self.image_files) > 1 and not self.group_names.text():
                errors.append("Group names are required when analyzing multiple images")

        return errors

    def save_settings(self, filename):
        """Save current settings to file"""
        params = self.get_params()
        settings = {
            "analysis_type": self.analysis_combo.currentText(),
            "group_names": self.group_names.text(),
            "loaded_images": self.image_files,
            "current_image": self.get_current_image(),
        }

        if params["type"] == "standard":
            settings.update({
                "std_box_size": self.std_box_size.value(),
                "std_bin_shift": self.std_bin_shift.value()
            })
        elif params["type"] == "rolling":
            settings.update({
                "roll_box_size": self.roll_box_size.value(),
                "roll_bin_shift": self.roll_bin_shift.value(),
                "roll_sub_size": self.roll_sub_size.value(),
                "roll_sub_shift": self.roll_sub_shift.value()
            })
        elif params["type"] == "kymograph":
            settings.update({
                "kymo_line_width": self.kymo_line_width.value(),
                "kymo_bin_shift": self.kymo_bin_shift.value(),
                "calc_speed": self.calc_speed.isChecked()
            })

        with open(filename, 'w') as f:
            json.dump(settings, f, indent=2)

    def load_settings(self, filename):
        """Load settings from file"""
        try:
            with open(filename, 'r') as f:
                settings = json.load(f)

            # Restore analysis type
            if "analysis_type" in settings:
                self.analysis_combo.setCurrentText(settings["analysis_type"])

            # Restore group names
            if "group_names" in settings:
                self.group_names.setText(settings["group_names"])

            # Restore loaded images
            if "loaded_images" in settings:
                self.clear_all_images()
                for image_path in settings["loaded_images"]:
                    if os.path.exists(image_path):
                        self.image_files.append(image_path)
                        item_text = f"{os.path.basename(image_path)} ({self._format_file_size(os.path.getsize(image_path))})"
                        self.image_list.addItem(item_text)
                
                # Set current image if available
                if "current_image" in settings and settings["current_image"] in self.image_files:
                    index = self.image_files.index(settings["current_image"])
                    self.image_list.setCurrentRow(index)

            # Restore analysis parameters
            if "std_box_size" in settings:
                self.std_box_size.setValue(settings["std_box_size"])
            if "std_bin_shift" in settings:
                self.std_bin_shift.setValue(settings["std_bin_shift"])
            if "roll_box_size" in settings:
                self.roll_box_size.setValue(settings["roll_box_size"])
            if "roll_bin_shift" in settings:
                self.roll_bin_shift.setValue(settings["roll_bin_shift"])
            if "roll_sub_size" in settings:
                self.roll_sub_size.setValue(settings["roll_sub_size"])
            if "roll_sub_shift" in settings:
                self.roll_sub_shift.setValue(settings["roll_sub_shift"])
            if "kymo_line_width" in settings:
                self.kymo_line_width.setValue(settings["kymo_line_width"])
            if "kymo_bin_shift" in settings:
                self.kymo_bin_shift.setValue(settings["kymo_bin_shift"])
            if "calc_speed" in settings:
                self.calc_speed.setChecked(settings["calc_speed"])

            self.images_updated.emit()

        except Exception as e:
            # Error loading settings - fail silently and continue with defaults
            pass
    
    def preview_boxes(self, analysis_type):
        """Preview the box/line grid on the current image in napari viewer"""
        from qtpy.QtWidgets import QMessageBox
        import numpy as np
        
        # Determine which checkbox triggered this
        if analysis_type == "standard":
            is_checked = self.std_preview_check.isChecked()
        elif analysis_type == "rolling":
            is_checked = self.roll_preview_check.isChecked()
        elif analysis_type == "kymograph":
            is_checked = self.kymo_preview_check.isChecked()
        else:
            return
        
        # Check if parent has viewer
        if not hasattr(self.parent, 'viewer'):
            # Uncheck the box if viewer not available
            if analysis_type == "standard":
                self.std_preview_check.setChecked(False)
            elif analysis_type == "rolling":
                self.roll_preview_check.setChecked(False)
            elif analysis_type == "kymograph":
                self.kymo_preview_check.setChecked(False)
            QMessageBox.warning(self, "No Viewer", "Napari viewer not available.")
            return
        
        viewer = self.parent.viewer
        
        # Remove existing preview layer if it exists
        for layer in list(viewer.layers):
            if layer.name == "Box Grid Preview":
                viewer.layers.remove(layer)
        
        # If unchecked, just remove the layer and return
        if not is_checked:
            return
        
        # Check if an image is loaded
        if not self.current_image_path:
            # Uncheck the box
            if analysis_type == "standard":
                self.std_preview_check.setChecked(False)
            elif analysis_type == "rolling":
                self.roll_preview_check.setChecked(False)
            elif analysis_type == "kymograph":
                self.kymo_preview_check.setChecked(False)
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
        
        # Check if image is in viewer
        if len(viewer.layers) == 0:
            # Uncheck the box
            if analysis_type == "standard":
                self.std_preview_check.setChecked(False)
            elif analysis_type == "rolling":
                self.roll_preview_check.setChecked(False)
            elif analysis_type == "kymograph":
                self.kymo_preview_check.setChecked(False)
            QMessageBox.warning(self, "No Image", "No image is displayed in the viewer.")
            return
        
        # Get the current image layer - find the actual Image layer, not shapes
        image_layer = None
        for layer in viewer.layers:
            if hasattr(layer, 'data') and layer.__class__.__name__ == 'Image':
                image_layer = layer
                break
        
        if image_layer is None:
            # Uncheck the box
            if analysis_type == "standard":
                self.std_preview_check.setChecked(False)
            elif analysis_type == "rolling":
                self.roll_preview_check.setChecked(False)
            elif analysis_type == "kymograph":
                self.kymo_preview_check.setChecked(False)
            QMessageBox.warning(self, "No Image", "No image layer found in viewer.")
            return
        
        image_shape = image_layer.data.shape
        
        # Get dimensions (handle both 2D and 3D/4D images)
        if len(image_shape) >= 2:
            height, width = image_shape[-2:]
        else:
            # Uncheck the box
            if analysis_type == "standard":
                self.std_preview_check.setChecked(False)
            elif analysis_type == "rolling":
                self.roll_preview_check.setChecked(False)
            elif analysis_type == "kymograph":
                self.kymo_preview_check.setChecked(False)
            QMessageBox.warning(self, "Invalid Image", "Image dimensions not supported.")
            return
        
        # Get parameters based on analysis type
        if analysis_type == "standard":
            box_size = self.std_box_size.value()
            step = self.std_bin_shift.value()
            grid_type = "boxes"
        elif analysis_type == "rolling":
            box_size = self.roll_box_size.value()
            step = self.roll_bin_shift.value()
            grid_type = "boxes"
        elif analysis_type == "kymograph":
            box_size = self.kymo_line_width.value()
            step = self.kymo_bin_shift.value()
            grid_type = "lines"
        else:
            return
        
        # Calculate grid positions
        ind = box_size // 2
        
        if grid_type == "boxes":
            # Create box grid (same logic as create_multi_frame_bin_array)
            y_positions = list(range(ind, height - ind, step))
            x_positions = list(range(ind, width - ind, step))
            
            # Create shapes for each box
            shapes_data = []
            for y in y_positions:
                for x in x_positions:
                    # Define box corners: [top-left, top-right, bottom-right, bottom-left]
                    box = np.array([
                        [y - ind, x - ind],
                        [y - ind, x + ind],
                        [y + ind, x + ind],
                        [y + ind, x - ind]
                    ])
                    shapes_data.append(box)
            
            # Add shapes layer
            viewer.add_shapes(
                shapes_data,
                shape_type='rectangle',
                edge_color='cyan',
                edge_width=2,
                face_color='transparent',
                name='Box Grid Preview'
            )
            
        elif grid_type == "lines":
            # Create vertical line grid for kymograph
            x_positions = list(range(0, width, step))
            
            # Create lines
            shapes_data = []
            for x in x_positions:
                # Define vertical line from top to bottom
                line = np.array([
                    [0, x],
                    [height - 1, x]
                ])
                shapes_data.append(line)
            
            # Add shapes layer
            viewer.add_shapes(
                shapes_data,
                shape_type='line',
                edge_color='cyan',
                edge_width=box_size,  # Line width
                name='Box Grid Preview'
            )
