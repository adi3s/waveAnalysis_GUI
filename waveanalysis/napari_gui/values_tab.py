from qtpy.QtWidgets import *
from qtpy.QtCore import Qt, Signal
import os

class ValuesTab(QWidget):
    """Tab for setting analysis parameters and selecting analysis type"""
    image_loaded = Signal(str)  # Signal to notify when an image is loaded

    def __init__(self, parent):
        """Initialize the ValuesTab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.init_ui()
        self.setup_parameter_groups()
        self.update_visible_params()

    def init_ui(self):
        """Initialize the user interface for the ValuesTab"""
        main_layout = QVBoxLayout()

        # Load Image Section
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        self.path_label = QLabel("No image loaded")
        load_layout = QVBoxLayout()
        load_layout.addWidget(self.load_btn)
        load_layout.addWidget(self.path_label)
        load_group = QGroupBox("Image Controls")
        load_group.setLayout(load_layout)

        # Analysis Type Selection
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["Standard", "Rolling", "Kymograph"])
        self.analysis_combo.currentIndexChanged.connect(self.update_visible_params)

        # Parameter Groups
        self.standard_params = QGroupBox("Standard Parameters")
        self.rolling_params = QGroupBox("Rolling Parameters")
        self.kymo_params = QGroupBox("Kymograph Parameters")
        self.common_params = QGroupBox("Common Parameters")

        main_layout.addWidget(QLabel("Analysis Type:"))
        main_layout.addWidget(self.analysis_combo)
        main_layout.addWidget(self.standard_params)
        main_layout.addWidget(self.rolling_params)
        main_layout.addWidget(self.kymo_params)
        main_layout.addWidget(self.common_params)
        main_layout.addWidget(load_group)

        save_params = QPushButton("Save Parameters")
        save_params.clicked.connect(self.save_parameters)
        main_layout.addWidget(save_params)
        self.setLayout(main_layout)

    def load_image(self):
        """Load an image file and emit the path"""
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.path_label.setText(f"Loaded: {os.path.basename(path)}")
            self.image_loaded.emit(path)  # Emit the signal with the image path

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
    
    def save_parameters(self):
        """Save the parameters and update the parent widget"""
        params = self.get_params()
        self.parent.update_params(params)

    def get_params(self):
        """Get the parameters for the selected analysis type"""
        analysis_type = self.analysis_combo.currentText().lower()
        params = {
            "type": analysis_type,
            "group_names": [n.strip() for n in self.group_names.text().split(",") if n.strip()]
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

        if analysis_type in ["standard", "kymograph"]:
            if not self.group_names.text():
                errors.append("At least one group name is required")

        return errors