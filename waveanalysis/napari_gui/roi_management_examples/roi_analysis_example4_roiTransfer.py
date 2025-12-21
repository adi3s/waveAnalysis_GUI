import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QApplication

def transfer_to_roi_manager(viewer, source_layer_name):
    if source_layer_name not in viewer.layers:
        print(f"Error: Layer '{source_layer_name}' not found.")
        return
    
    src_layer = viewer.layers[source_layer_name]
    
    # 1. FORCE THE LAYER TO BE ACTIVE
    viewer.layers.selection.clear()
    viewer.layers.selection.add(src_layer)
    viewer.layers.selection.active = src_layer

    # 2. Access the ROI Manager Widget
    if 'ROI Manager' in viewer.window._dock_widgets:
        roi_manager_instance = viewer.window._dock_widgets['ROI Manager'].widget()
    else:
        _, roi_manager_instance = viewer.window.add_plugin_dock_widget(
            plugin_name='napari-roi-manager', 
            widget_name='ROI Manager'
        )

    # 3. Find the Add Button
    add_button = None
    for btn in roi_manager_instance.findChildren(QPushButton):
        if 'Add' in btn.text():
            add_button = btn
            break

    if add_button:
        # 4. Transfer with GUI Refresh
        for i in range(len(src_layer.data)):
            # Select the shape
            src_layer.selected_data = {i}
            
            # CRITICAL: Let Qt process the selection event so the plugin sees it
            QApplication.processEvents()
            
            # Programmatically click
            add_button.click()
            
            # Let the plugin finish adding before the next loop
            QApplication.processEvents()
            
        print(f"Transfer attempt complete for {len(src_layer.data)} ROIs.")
    else:
        print("Add button not found.")

# --- Simulation Test ---
viewer = napari.Viewer()
img = np.random.random((512, 512))
viewer.add_image(img, name='Research_Image')

# Create two distinct rectangles
shapes = [
    np.array([[50, 50], [50, 150], [150, 150], [150, 50]]),
    np.array([[200, 200], [200, 300], [300, 300], [300, 200]])
]
viewer.add_shapes(shapes, shape_type='rectangle', name='Source_ROIs')

# Run the transfer
transfer_to_roi_manager(viewer, 'Source_ROIs')

napari.run()