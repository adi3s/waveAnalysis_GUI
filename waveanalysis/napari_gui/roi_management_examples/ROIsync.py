"""
Research ROI Sync Tool - Final Implementation
---------------------------------------------
Strategy: UI Mimicry (Button Click Simulation)

Key Operations:
1. Capture Source: Lock onto the user's data layer.
2. Find Trigger: Locate the plugin's "Add" QPushButton.
3. Mimic Loop: Add Shape -> Select Shape -> Click Button -> Process Events.
"""

import napari
import numpy as np
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                            QApplication, QTableWidget)
from qtpy.QtCore import Qt
import npe2 
import time

class NapariRoiManagerBridge(QWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.roi_manager_instance = None
        
        # UI Layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(QLabel("<b>Research ROI Sync Tool</b>"))
        
        self.btn_init = QPushButton("1. Open ROI Manager")
        self.btn_init.clicked.connect(self._initialize_manager)
        self.layout.addWidget(self.btn_init)
        
        self.btn_to_manager = QPushButton("2. Image ROI ➔ ROI Manager")
        self.btn_to_manager.clicked.connect(self.sync_to_manager)
        self.layout.addWidget(self.btn_to_manager)
        
        self.btn_from_manager = QPushButton("3. ROI Manager ➔ Image ROI")
        self.btn_from_manager.clicked.connect(self.sync_from_manager)
        self.layout.addWidget(self.btn_from_manager)

    def _initialize_manager(self):
        """Robustly launches or retrieves the ROI Manager plugin."""
        if 'ROI Manager' in self.viewer.window._dock_widgets:
            self.roi_manager_instance = self.viewer.window._dock_widgets['ROI Manager'].widget()
            return

        pm = npe2.PluginManager.instance()
        names = ['napari-roi-manager', 'napari_roi_manager']
        for m in pm.iter_manifests():
            if 'roi-manager' in m.name: names.append(m.name)
            
        for name in set(names):
            try:
                _, plugin = self.viewer.window.add_plugin_dock_widget(name, 'ROI Manager')
                self.roi_manager_instance = plugin
                print(f"DEBUG: Initialized '{name}'.")
                return
            except Exception:
                continue
        print("ERROR: ROI Manager plugin not found.")

    def _get_active_manager_layer(self):
        """Finds the layer currently attached to the plugin."""
        if self.roi_manager_instance:
            for attr in ['_layer', 'layer', 'roi_layer', '_shapes']:
                if hasattr(self.roi_manager_instance, attr):
                    obj = getattr(self.roi_manager_instance, attr, None)
                    if isinstance(obj, napari.layers.Shapes): return obj
        if 'ROIs' in self.viewer.layers: return self.viewer.layers['ROIs']
        return None

    def sync_to_manager(self):
        """
        Transfers shapes by simulating user selection and button clicks.
        This ensures 100% compatibility with the plugin's internal logic.
        """
        if self.roi_manager_instance is None: self._initialize_manager()

        # --- STEP 1: CAPTURE SOURCE ---
        src_layer = self.viewer.layers.selection.active
        if not isinstance(src_layer, napari.layers.Shapes):
            print("ABORT: Please select your 'Source' Shapes layer first.")
            return
        if len(src_layer.data) == 0:
            print(f"ABORT: Source layer '{src_layer.name}' is empty!")
            return

        print(f"   -> Source: '{src_layer.name}' ({len(src_layer.data)} shapes).")

        # --- STEP 2: SETUP TARGET LAYER ---
        target_layer = self._get_active_manager_layer()
        if target_layer is None:
            print("   -> Initializing 'ROIs' layer...")
            target_layer = self.viewer.add_shapes(name="ROIs", edge_color='cyan')
            if hasattr(self.roi_manager_instance, '_layer'):
                self.roi_manager_instance._layer = target_layer
        
        if src_layer == target_layer:
            print("ABORT: Source and Target are same. Select Data Layer.")
            return

        # --- STEP 3: FIND THE UI TRIGGER ---
        # We explicitly look for the 'Add' QPushButton.
        # This avoids TypeError issues with internal methods like .add(data)
        add_btn = None
        for btn in self.roi_manager_instance.findChildren(QPushButton):
            txt = btn.text().lower()
            if "add" in txt and "layer" not in txt:
                add_btn = btn
                break
        
        # Fallback search
        if not add_btn:
            for btn in self.roi_manager_instance.findChildren(QPushButton):
                if "add" in btn.text().lower(): 
                    add_btn = btn
                    break

        if not add_btn:
            print("CRITICAL ERROR: Could not find 'Add' button in ROI Manager UI.")
            return
        else:
            print(f"DEBUG: Found Trigger Button: '{add_btn.text()}'")

        # --- STEP 4: CLEAN SLATE ---
        # Clear target to ensure indices start at 0
        if len(target_layer.data) > 0:
            target_layer.selected_data = set(range(len(target_layer.data)))
            target_layer.remove_selected()
            QApplication.processEvents()

        # --- STEP 5: THE MIMIC LOOP (CRITICAL) ---
        print(f"   -> Transferring {len(src_layer.data)} shapes...")
        
        # Make Target Active so Plugin sees the selection we are about to make
        self.viewer.layers.selection.active = target_layer

        for i, shape in enumerate(src_layer.data):
            # A. Add Shape (Native Methods)
            # We use add_rectangles/ellipses so napari knows the correct shape type
            shape_type = src_layer.shape_type[i]
            if shape_type == 'rectangle':
                target_layer.add_rectangles([shape])
            elif shape_type == 'ellipse':
                target_layer.add_ellipses([shape])
            elif shape_type == 'polygon':
                target_layer.add_polygons([shape])
            elif shape_type == 'line':
                target_layer.add_lines([shape])
            elif shape_type == 'path':
                target_layer.add_paths([shape])
            else: 
                target_layer.add([shape], shape_type=shape_type)
            
            # B. Select the New Shape
            # The plugin listens for 'current selection', so we highlight the one we just added
            new_index = len(target_layer.data) - 1
            target_layer.selected_data = {new_index}
            
            # C. Click the Button
            # This triggers the plugin to register the shape, create the table row, and detect type
            add_btn.click()
            
            # D. Digest
            # Wait for the UI update to finish before moving to the next shape
            QApplication.processEvents()

        print(f"   -> Sync Complete. {len(target_layer.data)} shapes processed.")

    def sync_from_manager(self):
        """Pulls ROIs back from the Manager to the active layer."""
        src_layer = self._get_active_manager_layer()
        if src_layer is None: return
        target_layer = self.viewer.layers.selection.active
        if not isinstance(target_layer, napari.layers.Shapes) or target_layer == src_layer: return
        
        # Simple bulk transfer is fine for this direction
        target_layer.data = []
        target_layer.add(src_layer.data, shape_type=src_layer.shape_type)
        print("   -> Pulled ROIs from Manager.")

# --- TEST RUNNER ---
def run_test():
    print("--- STARTING FINAL VALIDATION ---")
    viewer = napari.Viewer()
    viewer.add_image(np.random.random((512, 512)), name="Image")
    widget = NapariRoiManagerBridge(viewer)
    viewer.window.add_dock_widget(widget, name="Sync Tool", area='right')
    
    # Create Source with Mixed Types
    src = viewer.add_shapes(name="Source")
    src.add_rectangles([[50,50],[100,100]])     # 1. Rectangle
    src.add_ellipses([[300,300],[400,400]])     # 2. Ellipse
    
    # Open Plugin
    widget._initialize_manager()
    QApplication.processEvents()
    
    print("\n--- Pushing Data ---")
    
    # Important: User must have Source selected
    viewer.layers.selection.active = src
    
    widget.sync_to_manager()
    QApplication.processEvents()
    
    print("DONE. Success Criteria:")
    print("1. ROI Manager List has 2 items.")
    print("2. Item 1 is 'Rectangle', Item 2 is 'Ellipse'.")
    print("3. Clicking items highlights shapes without crashing.")
    napari.run()

if __name__ == '__main__':
    run_test()