import numpy as np
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt
import pandas as pd

class PostProcessingTab(QWidget):
    """Tab for displaying analysis results and visualizations"""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface for the PostProcessingTab"""
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Results Table
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["ROI", "Parameter", "Mean", "Std", "Min", "Max"])
        self.table.itemSelectionChanged.connect(self.update_plots)
        
        # Plot Canvas
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        
        self.tabs.addTab(self.table, "Table View")
        self.tabs.addTab(self.canvas, "Graphs")
        
        layout.addWidget(QLabel("Analysis Results:"))
        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def show_results(self, results, params):
        self.results = results
        self.table.setRowCount(0)
        
        for idx, result in enumerate(results):
            row_pos = self.table.rowCount()
            self.table.insertRow(row_pos)
            
            metrics = {
                "Period (s)": np.nanmean(result['period']),
                "Peak Amplitude": np.nanmean([p['heights'] for p in result['peak_props'][-1].values()]),
                "Wave Speed (μm/s)": np.nanmean(result.get('wave_speed', [np.nan]))
            }
            
            # Populate each metric in separate rows
            for metric_idx, (name, value) in enumerate(metrics.items()):
                self.table.setItem(row_pos + metric_idx, 0, QTableWidgetItem(f"ROI {idx+1}"))
                self.table.setItem(row_pos + metric_idx, 1, QTableWidgetItem(name))
                self.table.setItem(row_pos + metric_idx, 2, QTableWidgetItem(f"{value:.2f}"))
                # Add Std, Min, Max if available

    def update_plots(self):
        """Update plots based on selected metrics from the table"""
        selected_items = self.table.selectedItems()
        if not selected_items:
            return
        
        selected_metrics = set()
        for item in selected_items:
            if item.column() > 0 and item.column() < 4:  # Only consider Mean, Std, Min, Max columns
                selected_metrics.add(self.table.horizontalHeaderItem(item.column()).text())
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot parameter curves based on selected metrics
        for result in self.results:
            for metric in selected_metrics:
                if metric == "Period (s)":
                    data = np.nanmean(result['period'])
                elif metric == "Peak Amplitude":
                    data = np.nanmean([p['heights'] for p in result['peak_props'][-1].values()])
                elif metric == "Wave Speed (μm/s)":
                    data = np.nanmean(result.get('wave_speed', [np.nan]))
                else:
                    continue
                ax.plot(data, label=f"ROI {self.results.index(result)+1} - {metric}")
        
        ax.set_title("Selected Metrics Curves")
        ax.legend()
        self.canvas.draw()