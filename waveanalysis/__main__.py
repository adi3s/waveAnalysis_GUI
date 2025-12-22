"""
Wave Analysis GUI Application

This module launches the napari-based wave analysis GUI.
All analysis workflows are now controlled through the GUI interface.
"""

from waveanalysis.napari_gui.main_gui import WaveAnalysisWidget
import napari


def main():
    """
    Launch the wave analysis GUI in a napari viewer.
    
    The GUI provides a tabbed interface for:
    - Loading and managing images
    - Creating and managing ROIs
    - Configuring analysis parameters
    - Running analysis workflows
    - Viewing and exporting results
    """
    viewer = napari.Viewer()
    gui = WaveAnalysisWidget(viewer)
    viewer.window.add_dock_widget(gui, area="right", name="Wave Analysis")
    napari.run()


if __name__ == '__main__':
    main()