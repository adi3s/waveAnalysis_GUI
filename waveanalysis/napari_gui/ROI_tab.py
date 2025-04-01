import os
from qtpy.QtWidgets import *
from qtpy.QtCore import Signal
import napari

class ROITab(QWidget):
    """Tab for loading images and saving ROIs in Napari viewer"""
    image_loaded = Signal(str)
    roi_saved = Signal(list)
    
    def __init__(self, parent):
        """Initialize the ROITab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.save_path = ""
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the ROITab"""
        layout = QVBoxLayout()
        
        # Image loading controls
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)

        # Add a button to create shapes layer
        add_rois_btn = QPushButton("Create ROIs")
        add_rois_btn.clicked.connect(self.create_rois)
        
        # ROI saving controls
        self.roi_name = QLineEdit()
        save_btn = QPushButton("Save ROI")
        save_btn.clicked.connect(self.save_roi)
        
        # Path display
        self.path_label = QLabel("No image loaded")
        
        # Grouping controls
        image_group = QGroupBox("Image Controls")
        image_layout = QVBoxLayout()
        image_layout.addWidget(load_btn)
        image_layout.addWidget(self.path_label)
        image_group.setLayout(image_layout)
        
        roi_group = QGroupBox("ROI Controls")
        roi_layout = QGridLayout()
        roi_layout.addWidget(add_rois_btn, 0, 0, 1, 2)
        roi_layout.addWidget(QLabel("ROI Name:"), 1, 0)
        roi_layout.addWidget(self.roi_name, 1, 1)
        roi_layout.addWidget(save_btn, 2, 0, 1, 2)
        roi_group.setLayout(roi_layout)
        
        layout.addWidget(image_group)
        layout.addWidget(roi_group)
        layout.addStretch()
        self.setLayout(layout)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.save_path = os.path.join(os.path.dirname(path), "cropped_images")
            os.makedirs(self.save_path, exist_ok=True)  # Critical for saving
            self.image_loaded.emit(path)
            self.path_label.setText(f"Loaded: {os.path.basename(path)}")

    def create_rois(self):
        """Create a shapes layer if none exists"""
        # Check if a shapes layer already exists
        shapes_layers = [layer for layer in self.parent.viewer.layers 
                        if isinstance(layer, napari.layers.Shapes)]
        
        if not shapes_layers:
            # Create a new shapes layer
            self.parent.viewer.add_shapes(name="ROI")
            QMessageBox.information(self, "Shapes Layer Created", 
                                "Draw your ROI using the shapes tools")
        else:
            QMessageBox.information(self, "Shapes Layer Exists", 
                                "Use the existing shapes layer to draw your ROI")
    
    def save_roi(self):
        """Save current ROI from shapes layer"""
        shapes = [layer for layer in self.parent.viewer.layers if isinstance(layer, napari.layers.Shapes)]
        if shapes:
            print("Type of shapes data:", type(shapes[0].data))
            print("Length of shapes data:", len(shapes[0].data) if hasattr(shapes[0].data, "__len__") else "Not a sequence")
            if hasattr(shapes[0].data, "__len__") and len(shapes[0].data) > 0:
                roi = shapes[0].data[-1]
                roi_list = roi.tolist() if hasattr(roi, "tolist") else roi
                self.roi_saved.emit(roi_list)
                QMessageBox.information(self, "ROI Saved", f"Saved ROI: {self.roi_name.text()}")
                self.roi_name.clear()
            else:
                QMessageBox.warning(self, "No ROI", "No shapes found to save")
        else:
            QMessageBox.warning(self, "No Shapes Layer", "Create a shapes layer first")