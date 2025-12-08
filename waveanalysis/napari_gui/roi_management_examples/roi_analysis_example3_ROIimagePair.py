import os
import napari
import numpy as np
import pandas as pd
from pathlib import Path
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QListWidget, 
                            QFileDialog, QLabel, QHBoxLayout, QMessageBox)
from napari.layers import Image, Shapes

class MultiImageROIManager(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        # Registry to store association: 
        # { 'filename': {'image_layer': layer_obj, 'roi_layer': layer_obj, 'path': full_path} }
        self.image_registry = {} 

        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        # 1. Controls Area
        btn_layout = QHBoxLayout()
        self.btn_load = QPushButton("Load Images")
        self.btn_load.clicked.connect(self.load_images)
        
        self.btn_save = QPushButton("Save All ROIs")
        self.btn_save.clicked.connect(self.save_all_rois)
        
        btn_layout.addWidget(self.btn_load)
        btn_layout.addWidget(self.btn_save)
        layout.addLayout(btn_layout)

        # 2. Information Label
        self.lbl_info = QLabel("No images loaded.")
        layout.addWidget(self.lbl_info)

        # 3. The Image List
        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self.on_list_item_clicked)
        layout.addWidget(self.list_widget)

        self.setLayout(layout)

    def load_images(self):
        """Opens file dialog and loads images + associated ROIs."""
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.tif *.tiff *.png *.jpg)"
        )

        if not filenames:
            return

        for file_path in filenames:
            path_obj = Path(file_path)
            name = path_obj.stem
            
            # 1. Avoid reloading if already in list
            if name in self.image_registry:
                continue

            # 2. Load the Image Layer
            # We use stack=False to ensure we get individual layers
            loaded_layers = self.viewer.open(file_path, stack=False)
            if not loaded_layers:
                continue
            
            img_layer = loaded_layers[0]
            img_layer.name = name  # Clean name in napari
            img_layer.visible = False # Hidden by default until selected

            # 3. Check for associated ROI file (naming convention: name_roi.csv)
            roi_path = path_obj.parent / f"{name}_roi.csv"
            shapes_layer = None

            if roi_path.exists():
                try:
                    # Load existing ROI
                    print(f"Loading ROI for {name}...")
                    shapes_layer = self.viewer.open(str(roi_path), layer_type='shapes')[0]
                    shapes_layer.name = f"{name}_ROI"
                except Exception as e:
                    print(f"Failed to load ROI for {name}: {e}")

            # 4. If no ROI file exists, create a fresh Shapes layer
            if shapes_layer is None:
                shapes_layer = self.viewer.add_shapes(
                    name=f"{name}_ROI",
                    ndim=2,
                    edge_color='red',
                    face_color='transparent',
                    edge_width=3
                )

            shapes_layer.visible = False # Hidden by default

            # 5. Register the pair
            self.image_registry[name] = {
                'image_layer': img_layer,
                'roi_layer': shapes_layer,
                'path': path_obj
            }

            # 6. Add to UI List
            self.list_widget.addItem(name)

        self.lbl_info.setText(f"{self.list_widget.count()} images managed.")
        
        # Automatically select the first item if it's the first load
        if self.list_widget.count() > 0 and not self.list_widget.currentItem():
            self.list_widget.setCurrentRow(0)
            self.on_list_item_clicked(self.list_widget.item(0))

    def on_list_item_clicked(self, item):
        """
        Handles the logic when a user selects an image name.
        1. Hides all other layers.
        2. Shows the specific image and ROI layer.
        3. Selects the ROI layer so plugins work immediately.
        """
        selected_name = item.text()

        # Iterate through all registered pairs
        for name, data in self.image_registry.items():
            img = data['image_layer']
            roi = data['roi_layer']

            if name == selected_name:
                # Show this pair
                img.visible = True
                roi.visible = True
                
                # IMPORTANT: Set the active selection to the ROI layer.
                # This ensures napari-roi-manager targets this specific layer.
                self.viewer.layers.selection.active = roi
                
                # Adjust camera to fit this image
                extent = img.extent.world  # Returns (mins, maxs) arrays
                center = tuple((extent[0] + extent[1]) / 2)
                self.viewer.camera.center = center
                self.viewer.camera.zoom = 1 # Reset zoom or calculate based on extent
            else:
                # Hide others
                img.visible = False
                roi.visible = False

    def save_all_rois(self):
        """Saves the Shapes layer data to CSV for every image."""
        count = 0
        for name, data in self.image_registry.items():
            roi_layer = data['roi_layer']
            original_image_path = data['path']
            
            # Skip if no shapes drawn
            if len(roi_layer.data) == 0:
                continue

            # Define save path: image_name_roi.csv
            save_path = original_image_path.parent / f"{name}_roi.csv"
            
            # Using napari's built-in save for shapes
            roi_layer.save(str(save_path))
            count += 1

        msg = QMessageBox()
        msg.setWindowTitle("Save Complete")
        msg.setText(f"Saved ROIs for {count} images.")
        msg.exec_()

# ==========================================
# TEST HARNESS: Run this to test the widget
# ==========================================
if __name__ == "__main__":
    
    # 1. Generate Dummy Data for Testing
    # We create 3 images and save them to a temporary folder
    import tempfile
    import tifffile
    
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary test directory: {temp_dir}")

    # Create 3 test images
    for i in range(1, 4):
        img_data = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        # Add a visual marker so we know images are different
        img_data[i*50:i*50+50, i*50:i*50+50] = 255 
        tifffile.imwrite(os.path.join(temp_dir, f"sample_image_{i}.tif"), img_data)

    # 2. Launch Napari
    viewer = napari.Viewer()

    # 3. Initialize and dock the widget
    roi_manager_widget = MultiImageROIManager(viewer)
    viewer.window.add_dock_widget(roi_manager_widget, area='right', name="Multi-Image ROI")

    # 4. Instructions
    print("\n--- INSTRUCTIONS ---")
    print(f"1. In the widget, click 'Load Images'.")
    print(f"2. Navigate to: {temp_dir}")
    print("3. Select all three 'sample_image' files.")
    print("4. Draw shapes on different images.")
    print("5. Click 'Save All ROIs'.")
    print("6. Restart the script and load images again to see ROIs restored.")
    
    napari.run()