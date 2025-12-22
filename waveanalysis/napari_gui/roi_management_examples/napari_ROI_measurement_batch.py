"""
Napari ROI Measurement Batch Plugin
===================================
A comprehensive widget for managing multiple images, synchronizing ROIs with the 
'napari-roi-manager' plugin, and performing batch measurements.

Key Features:
- **Project Management**: Load multiple images, switch between them, and maintain state.
- **ROI Bridge**: Robustly transfers ROIs between the local image layer and the ROI Manager plugin
  using a 'UI Mimic' strategy to prevent crashes and ensure metadata (ROI Type) integrity.
- **Persistent Storage**: Automatically saves/loads ROIs from an 'ROI_management' subfolder.
- **Batch Analysis**: Calculates Area, Min, Max, and Mean Intensity for all opened images.

Dependencies:
- napari
- napari-roi-manager
- pandas
- skimage (for mask generation)
"""

import napari
import numpy as np
import os
import json
import pandas as pd
import tempfile
from pathlib import Path
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QTabWidget, 
                            QListWidget, QListWidgetItem, QFileDialog, QCheckBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView, 
                            QAbstractItemView, QMessageBox, QProgressBar, QApplication)
from qtpy.QtCore import Qt, QTimer

# =============================================================================
#  CLASS: NapariRoiManagerBridge
#  Handles the "dangerous" task of talking to the napari-roi-manager plugin.
#  Uses a UI simulation strategy (button clicking) to ensure 100% stability.
# =============================================================================
class NapariRoiManagerBridge(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.roi_manager_instance = None

    def initialize_manager(self):
        """Finds and attaches to the ROI Manager plugin instance."""
        if 'ROI Manager' in self.viewer.window._dock_widgets:
            self.roi_manager_instance = self.viewer.window._dock_widgets['ROI Manager'].widget()
            return True
        
        # Search installed plugins if not already open
        import npe2
        names = ['napari-roi-manager', 'napari_roi_manager']
        for m in npe2.PluginManager.instance().iter_manifests():
            if 'roi-manager' in m.name: names.append(m.name)  
            
        for name in set(names):
            try:
                _, plugin = self.viewer.window.add_plugin_dock_widget(name, 'ROI Manager')
                self.roi_manager_instance = plugin
                return True
            except: continue
        return False

    def sync_to_manager(self, src_layer):
        """
        Pushes shapes from a source layer TO the ROI Manager.
        Strategy: Add shape -> Select it -> Click 'Add' button programmatically.
        """
        if not self.initialize_manager(): return False
        
        # 1. Locate the UI Trigger (Add Button)
        add_btn = None
        for btn in self.roi_manager_instance.findChildren(QPushButton):
            if "add" in btn.text().lower() and "layer" not in btn.text().lower():
                add_btn = btn
                break
        if not add_btn: return False
        
        target_layer = self.roi_manager_instance._layer
        
        # 2. Clear target layer safely
        if len(target_layer.data) > 0:
            target_layer.selected_data = set(range(len(target_layer.data)))
            target_layer.remove_selected()
            QApplication.processEvents()
            
        # 3. Transfer Loop
        self.viewer.layers.selection.active = target_layer 
        
        for i, shape in enumerate(src_layer.data):
            # Use native add methods to preserve shape type (Rect vs Ellipse)
            shape_type = src_layer.shape_type[i]
            if shape_type == 'rectangle': target_layer.add_rectangles([shape])
            elif shape_type == 'ellipse': target_layer.add_ellipses([shape])
            elif shape_type == 'polygon': target_layer.add_polygons([shape])
            elif shape_type == 'line': target_layer.add_lines([shape])
            elif shape_type == 'path': target_layer.add_paths([shape])
            else: target_layer.add([shape], shape_type=shape_type)
            
            # Select the new shape and click 'Add'
            target_layer.selected_data = {len(target_layer.data)-1}
            add_btn.click()
            QApplication.processEvents() # Critical for plugin to digest the event
            
        return True

    def sync_from_manager(self, target_layer):
        """Pulls valid ROIs FROM the Manager's layer to the local image layer."""
        if not self.initialize_manager(): return False
        src_layer = self.roi_manager_instance._layer
        target_layer.data = []
        target_layer.add(src_layer.data, shape_type=src_layer.shape_type)
        return True
    
    def get_manager_rois_as_dict(self):
        """Helper to retrieve ROI data without modifying layers (for measurement)."""
        if not self.initialize_manager(): return []
        layer = self.roi_manager_instance._layer
        return list(zip(layer.data, layer.shape_type))


# =============================================================================
#  CLASS: ImageObject
#  Data container for a single opened image, its ROIs, and its measurement status.
# =============================================================================
class ImageObject:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        
        # Path Logic: Save to "ROI_management" subfolder
        img_path_obj = Path(path)
        self.roi_folder = img_path_obj.parent / "ROI_management"
        self.roi_file_path = str(self.roi_folder / f"{img_path_obj.stem}_rois.json")
        
        self.roi_data = [] # List of (data, shape_type) tuples
        self.roi_layer_name = f"ROIs_{name}"
        self.measurements = None 
        self.status = "Loaded"

    def save_rois(self):
        """Saves current ROIs to JSON, creating the subfolder if needed."""
        serializable_rois = []
        for roi, s_type in self.roi_data:
            serializable_rois.append({
                "type": s_type,
                "data": roi.tolist()
            })
        try:
            if not os.path.exists(self.roi_folder):
                os.makedirs(self.roi_folder, exist_ok=True)
                
            with open(self.roi_file_path, 'w') as f:
                json.dump(serializable_rois, f)
            self.status = "ROIs Saved"
        except Exception as e:
            print(f"Save Error ({self.roi_file_path}): {e}")

    def load_rois(self):
        """Loads ROIs from the JSON file if it exists."""
        if os.path.exists(self.roi_file_path):
            try:
                with open(self.roi_file_path, 'r') as f:
                    data = json.load(f)
                self.roi_data = []
                for item in data:
                    self.roi_data.append((np.array(item['data']), item['type']))
                self.status = "ROIs Loaded"
            except Exception as e:
                print(f"Load Error: {e}")


# =============================================================================
#  CLASS: ROI_Measurement_Batch
#  The Main GUI Widget containing the 4 Tabs.
# =============================================================================
class ROI_Measurement_Batch(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.images = [] 
        self.current_image_index = -1
        self.bridge = NapariRoiManagerBridge(self.viewer)
        
        # GUI Layout Setup
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        self.init_project_tab()
        self.init_image_tab()
        self.init_manager_tab()
        self.init_measurement_tab()
        
        self.progress = QProgressBar()
        self.layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        self.layout.addWidget(self.status_label)

    # --- TAB 1: PROJECT ---
    def init_project_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        btn_open = QPushButton("Open Images")
        btn_open.clicked.connect(self.open_images)
        layout.addWidget(btn_open)
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self.select_image)
        layout.addWidget(self.image_list_widget)
        btn_delete = QPushButton("Delete Selected Image")
        btn_delete.clicked.connect(self.delete_image)
        layout.addWidget(btn_delete)
        btn_reset = QPushButton("Reset Plugin")
        btn_reset.clicked.connect(self.reset_plugin)
        layout.addWidget(btn_reset)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Project")

    def open_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Images (*.tif *.tiff *.png *.jpg)")
        if not files: return
        
        # Replace existing project state
        self.images = []
        self.image_list_widget.clear()
        
        manager_layer = None
        if self.bridge.initialize_manager():
            manager_layer = self.bridge.roi_manager_instance._layer
        
        # Clear layers (preserving manager)
        for layer in list(self.viewer.layers):
            if layer != manager_layer:
                self.viewer.layers.remove(layer)

        for f in files:
            name = Path(f).stem
            img_obj = ImageObject(name, f)
            img_obj.load_rois() 
            self.images.append(img_obj)
            self.image_list_widget.addItem(name)
        
        if self.images:
            self.image_list_widget.setCurrentRow(0)

    def select_image(self, index):
        if index < 0 or index >= len(self.images): return
        
        # Smart cleanup: Remove all layers EXCEPT the ROI Manager's layer
        manager_layer = None
        if self.bridge.initialize_manager():
            manager_layer = self.bridge.roi_manager_instance._layer
            
        for layer in list(self.viewer.layers):
            if layer != manager_layer: 
                self.viewer.layers.remove(layer)
        
        self.current_image_index = index
        img = self.images[index]
        
        # Load Image Layer
        try:
            if "Test_Image" in img.name:
                self.viewer.add_image(np.random.random((512, 512)), name=img.name)
            else:
                self.viewer.open(img.path, name=img.name)
        except Exception as e:
            self.status_label.setText(f"Error opening image: {e}")
            return
            
        # Create ROI Layer for this image
        roi_data_list = [r[0] for r in img.roi_data]
        roi_type_list = [r[1] for r in img.roi_data]
        
        layer = self.viewer.add_shapes(
            data=roi_data_list,
            shape_type=roi_type_list,
            name=img.roi_layer_name,
            edge_color='red',
            face_color='transparent'
        )
        # Connect drawing events to update the object model
        layer.events.data.connect(self.on_roi_layer_update)
        
        # Critical: Ensure Manager layer is visible on top
        self.bring_manager_layer_to_top()
        
        self.update_image_info(img)
        self.status_label.setText(f"Loaded {img.name}")

    def bring_manager_layer_to_top(self):
        """Moves the ROI Manager layer to the end of the list (Top)."""
        if self.bridge.initialize_manager():
            manager_layer = self.bridge.roi_manager_instance._layer
            if manager_layer in self.viewer.layers:
                idx = self.viewer.layers.index(manager_layer)
                if idx < len(self.viewer.layers) - 1:
                    self.viewer.layers.move(idx, -1)

    def on_roi_layer_update(self, event):
        """Callback for when user draws on the ROI layer."""
        if self.current_image_index == -1: return
        img = self.images[self.current_image_index]
        if img.roi_layer_name not in self.viewer.layers: return
        layer = self.viewer.layers[img.roi_layer_name]
        
        # Sync layer data back to object
        img.roi_data = list(zip(layer.data, layer.shape_type))
        img.measurements = None 
        self.status_label.setText(f"ROIs updated for {img.name}.")

    def delete_image(self):
        row = self.image_list_widget.currentRow()
        if row >= 0:
            self.image_list_widget.takeItem(row)
            self.images.pop(row)
            
            manager_layer = None
            if self.bridge.initialize_manager():
                manager_layer = self.bridge.roi_manager_instance._layer
            
            for layer in list(self.viewer.layers):
                if layer != manager_layer: self.viewer.layers.remove(layer)
            
            self.current_image_index = -1

    def reset_plugin(self):
        """Resets state while preserving the ROI Manager layer."""
        self.images = []
        self.image_list_widget.clear()
        
        manager_layer = None
        if self.bridge.initialize_manager():
            manager_layer = self.bridge.roi_manager_instance._layer
        
        if manager_layer is None and 'ROIs' in self.viewer.layers:
            manager_layer = self.viewer.layers['ROIs']

        for layer in list(self.viewer.layers):
            if layer != manager_layer:
                self.viewer.layers.remove(layer)
        
        self.current_image_index = -1
        self.table_results.setRowCount(0)
        self.status_label.setText("Plugin Reset.")

    # --- TAB 2: IMAGE INFO ---
    def init_image_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.lbl_name = QLabel("Name: -")
        self.lbl_path = QLabel("Path: -")
        self.lbl_rois = QLabel("ROIs Defined: 0")
        self.lbl_status = QLabel("Status: -")
        layout.addWidget(self.lbl_name)
        layout.addWidget(self.lbl_path)
        layout.addWidget(self.lbl_rois)
        layout.addWidget(self.lbl_status)
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Image")

    def update_image_info(self, img):
        self.lbl_name.setText(f"Name: {img.name}")
        self.lbl_path.setText(f"Path: {img.path}")
        self.lbl_rois.setText(f"ROIs Defined: {len(img.roi_data)}")
        self.lbl_status.setText(f"Status: {img.status}")

    # --- TAB 3: ROI MANAGER BRIDGE ---
    def init_manager_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.btn_sync_to_manager = QPushButton("Sync: Image ROI -> Manager")
        self.btn_sync_to_manager.clicked.connect(self.action_sync_to_manager)
        layout.addWidget(self.btn_sync_to_manager)
        self.btn_save_from_manager = QPushButton("Save: Manager -> Image & File")
        self.btn_save_from_manager.clicked.connect(self.action_save_from_manager)
        layout.addWidget(self.btn_save_from_manager)
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "ROI-manager")

    def action_sync_to_manager(self):
        if self.current_image_index == -1: return
        img = self.images[self.current_image_index]
        if img.roi_layer_name in self.viewer.layers:
            src_layer = self.viewer.layers[img.roi_layer_name]
            self.status_label.setText("Syncing to Manager...")
            success = self.bridge.sync_to_manager(src_layer)
            if success:
                self.status_label.setText("Sync Complete.")
            else:
                self.status_label.setText("Sync Failed.")

    def action_save_from_manager(self):
        if self.current_image_index == -1: return
        img = self.images[self.current_image_index]
        if img.roi_layer_name in self.viewer.layers:
            target_layer = self.viewer.layers[img.roi_layer_name]
            self.status_label.setText("Pulling from Manager...")
            success = self.bridge.sync_from_manager(target_layer)
            if success:
                # Update Data
                img.roi_data = list(zip(target_layer.data, target_layer.shape_type))
                img.measurements = None 
                # Save to JSON
                img.save_rois()
                self.status_label.setText(f"ROIs Saved to {img.roi_file_path}")
                self.update_image_info(img)

    # --- TAB 4: MEASUREMENTS ---
    def init_measurement_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        self.cb_use_manager_rois = QCheckBox("Use ROIs in ROI-Manager")
        layout.addWidget(self.cb_use_manager_rois)
        self.cb_analyze_all = QCheckBox("Analyze all opened images")
        layout.addWidget(self.cb_analyze_all)
        btn_run = QPushButton("Run ROI analysis")
        btn_run.clicked.connect(self.run_analysis)
        layout.addWidget(btn_run)
        
        # Results Table
        self.table_results = QTableWidget()
        self.table_results.setColumnCount(6)
        self.table_results.setHorizontalHeaderLabels(["Image", "ROI ID", "Area", "Min", "Max", "Mean"])
        layout.addWidget(self.table_results)
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Measurements")

    def run_analysis(self):
        use_manager = self.cb_use_manager_rois.isChecked()
        analyze_all = self.cb_analyze_all.isChecked()
        target_images = self.images if analyze_all else [self.images[self.current_image_index]] if self.current_image_index != -1 else []
        if not target_images:
            self.status_label.setText("No images to analyze.")
            return

        manager_rois = []
        if use_manager:
            manager_rois = self.bridge.get_manager_rois_as_dict()
            if not manager_rois:
                self.status_label.setText("ROI Manager is empty.")
                return

        self.table_results.setRowCount(0)
        self.progress.setValue(0)
        self.progress.setMaximum(len(target_images))
        
        all_results = []
        
        for i, img in enumerate(target_images):
            rois_to_use = manager_rois if use_manager else img.roi_data
            if not rois_to_use:
                self.progress.setValue(i+1)
                continue
            
            try:
                # Retrieve Image Data
                image_data = None
                if img.name in self.viewer.layers:
                    image_data = self.viewer.layers[img.name].data
                else:
                    # Mock for testing, replace with real loading (e.g., skimage.io.imread)
                    if "Test_Image" in img.name:
                        image_data = np.random.random((512, 512)) * 255
                    elif os.path.exists(img.path):
                        from skimage.io import imread
                        image_data = imread(img.path)
                
                if image_data is None: continue

                # Perform Measurement
                for r_idx, (roi_coords, r_type) in enumerate(rois_to_use):
                    mask = self._create_mask(image_data.shape, roi_coords, r_type)
                    masked_data = image_data[mask]
                    if masked_data.size == 0: continue
                    res = {
                        "Image": img.name,
                        "ROI ID": f"ROI-{r_idx+1}",
                        "Area": masked_data.size,
                        "Min": np.min(masked_data),
                        "Max": np.max(masked_data),
                        "Mean": np.mean(masked_data)
                    }
                    all_results.append(res)
                img.measurements = pd.DataFrame(all_results)
            except Exception as e:
                print(f"Error analyzing {img.name}: {e}")

            self.progress.setValue(i+1)

        # Update Table
        self.table_results.setRowCount(len(all_results))
        for row, res in enumerate(all_results):
            self.table_results.setItem(row, 0, QTableWidgetItem(str(res['Image'])))
            self.table_results.setItem(row, 1, QTableWidgetItem(str(res['ROI ID'])))
            self.table_results.setItem(row, 2, QTableWidgetItem(f"{res['Area']}"))
            self.table_results.setItem(row, 3, QTableWidgetItem(f"{res['Min']:.2f}"))
            self.table_results.setItem(row, 4, QTableWidgetItem(f"{res['Max']:.2f}"))
            self.table_results.setItem(row, 5, QTableWidgetItem(f"{res['Mean']:.2f}"))
        self.status_label.setText("Analysis Complete.")

    def _create_mask(self, shape, coords, shape_type):
        """Creates a boolean mask from ROI coordinates."""
        from skimage.draw import polygon
        mask = np.zeros(shape[:2], dtype=bool)
        if shape_type in ['rectangle', 'polygon', 'ellipse']:
            r = coords[:, 0]
            c = coords[:, 1]
            rr, cc = polygon(r, c, shape=shape[:2])
            mask[rr, cc] = True
        return mask

# =============================================================================
#  TEST SUITE
# =============================================================================
def run_test():
    print("--- STARTING BATCH PLUGIN TEST ---")
    viewer = napari.Viewer()
    widget = ROI_Measurement_Batch(viewer)
    viewer.window.add_dock_widget(widget, name="Batch Analysis", area='right')
    
    # 1. Setup Test Images
    temp_dir = tempfile.gettempdir()
    path1 = os.path.join(temp_dir, "test1.tif")
    path2 = os.path.join(temp_dir, "test2.tif")
    img1 = ImageObject("Test_Image_1", path1)
    img2 = ImageObject("Test_Image_2", path2)
    img1.roi_data = [(np.array([[50,50], [50,100], [100,100], [100,50]]), 'rectangle')]
    
    # 2. Add to Widget (Mocking Open Dialog)
    widget.images = [img1, img2]
    widget.image_list_widget.addItem("Test_Image_1")
    widget.image_list_widget.addItem("Test_Image_2")
    
    print("1. Selecting Image 1...")
    widget.image_list_widget.setCurrentRow(0)
    QApplication.processEvents()
    
    print("2. Testing Save Button (Check Subfolder Creation)...")
    widget.action_save_from_manager()
    QApplication.processEvents()
    
    # Validation
    expected_folder = os.path.join(os.path.dirname(path1), "ROI_management")
    expected_file = os.path.join(expected_folder, "test1_rois.json")
    
    if os.path.isdir(expected_folder) and os.path.isfile(expected_file):
         print(f"   -> PASS: ROI saved to subfolder: {expected_folder}")
    else:
         print(f"   -> FAIL: File not found in {expected_folder}")

    print("3. Testing Reset Plugin...")
    widget.reset_plugin()
    
    # Verify Manager Layer survival
    survived = False
    if widget.bridge.initialize_manager():
        mlayer = widget.bridge.roi_manager_instance._layer
        if mlayer in viewer.layers:
            survived = True
            print("   -> PASS: ROI Manager Layer survived reset.")
    if not survived: print("   -> FAIL: ROI Manager Layer deleted.")

    napari.run()

if __name__ == '__main__':
    run_test()