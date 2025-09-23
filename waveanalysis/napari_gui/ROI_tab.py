import os
from qtpy.QtWidgets import *
from qtpy.QtCore import Signal
from napari_roi_manager import QRoiManager

class ROITab(QWidget):
    """Tab for loading images and saving ROIs in Napari viewer"""
    image_loaded = Signal(str)
    roi_saved = Signal(list)  # Changed to pass the list of ROIs

    def __init__(self, parent):
        """Initialize the ROITab with the parent widget"""
        super().__init__(parent)
        self.parent = parent
        self.save_path = ""
        self.roi_manager = None
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the ROITab"""
        layout = QVBoxLayout()

        # ROI control: just a single button
        create_rois_btn = QPushButton("Create ROIs")
        create_rois_btn.clicked.connect(self.show_qroi_manager)
        roi_group = QGroupBox("ROI Controls")
        roi_layout = QVBoxLayout()
        roi_layout.addWidget(create_rois_btn)
        roi_group.setLayout(roi_layout)

        # QRoiManager placeholder, hidden by default
        self.roi_manager_group = QGroupBox("ROI Manager")
        self.roi_manager_group.setVisible(False)
        self.roi_manager_layout = QVBoxLayout()
        self.roi_manager_group.setLayout(self.roi_manager_layout)

        layout.addWidget(roi_group)
        layout.addWidget(self.roi_manager_group)
        layout.addStretch()
        self.setLayout(layout)

    def show_qroi_manager(self):
        """Show the QRoiManager for creating and managing ROIs"""
        if self.roi_manager is None:
            self.roi_manager = QRoiManager(self.parent.viewer)
            # Connect to the Save button's clicked signal
            save_button = self.roi_manager.findChild(QPushButton, "Save")
            if save_button:
                save_button.clicked.connect(self._on_roi_save)
            self.roi_manager_layout.addWidget(self.roi_manager)
        self.roi_manager_group.setVisible(True)

    def _on_roi_save(self):
        """Handler for when the Save button in ROI manager is clicked"""
        rois = self.get_rois()
        if rois:
            # Emit our signal with the list of ROIs
            self.roi_saved.emit(rois)

    def get_rois(self):
        """Retrieve the list of ROIs from the QRoiManager"""
        if self.roi_manager:
            if hasattr(self.roi_manager, 'layer') and self.roi_manager.layer:
                return self.roi_manager.layer.data
            elif hasattr(self.roi_manager, 'shapes_layer') and self.roi_manager.shapes_layer:
                return self.roi_manager.shapes_layer.data
            elif hasattr(self.roi_manager, 'roi_layer') and self.roi_manager.roi_layer:
                return self.roi_manager.roi_layer.data
        return []