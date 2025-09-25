import os
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGroupBox,
    QComboBox, QPushButton, QCheckBox, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem
)
from qtpy.QtCore import Qt, Signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.image import imread
import pandas as pd

class PostProcessingTab(QWidget):
    """Tab for displaying and exporting analysis results"""
    # Add signal for status updates
    status_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.results_dir = None
        self.results = None
        self.params = None
        self.canvases = []
        self.has_roi_results = False
        
        # Main layout
        layout = QVBoxLayout()
        
        # Add status bar at the top
        self.status_label = QLabel("No results loaded")
        layout.addWidget(self.status_label)
        
        # Results view area with a scrollable widget
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create a widget to hold all results
        self.results_widget = QWidget()
        self.results_layout = QHBoxLayout()
        self.results_layout.setSpacing(10)  # Add spacing between plots
        self.results_layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        self.results_layout.setAlignment(Qt.AlignLeft)  # Align plots to the left
        
        self.results_widget.setLayout(self.results_layout)
        self.scroll_area.setWidget(self.results_widget)
        layout.addWidget(self.scroll_area)
        
        # Controls area
        controls_group = QGroupBox("Display & Export Controls")
        controls_layout = QHBoxLayout()
        
        # Display options
        display_box = QVBoxLayout()
        display_label = QLabel("Display Type:")
        display_label.setStyleSheet("font-weight: bold;")
        display_box.addWidget(display_label)

        # Export controls
        export_box = QVBoxLayout()
        export_label = QLabel("Export Options:")
        export_label.setStyleSheet("font-weight: bold;")
        export_box.addWidget(export_label)
        
        self.export_combo = QComboBox()
        self.export_combo.addItems([
            "All Plots",
            "Summary Plots Only",
            "Individual Plots Only"
        ])
        export_box.addWidget(self.export_combo)
        
        export_btn = QPushButton("Export")
        export_btn.clicked.connect(self.export_plots)
        export_box.addWidget(export_btn)
        
        self.display_combo = QComboBox()
        self.display_combo.addItems([
            "ROIs", 
            "Summary",
            "ACF Plots",
            "CCF Plots", 
            "Peak Properties"
        ])
        self.display_combo.currentIndexChanged.connect(self.on_display_type_changed)
        display_box.addWidget(self.display_combo)
        
        # Add individual plot checkbox
        self.show_individual = QCheckBox("Show Individual Plots")
        self.show_individual.stateChanged.connect(self.on_show_individual_changed)
        self.show_individual.setVisible(False)  # Only show for plot types that support it
        display_box.addWidget(self.show_individual)
        
        # Add ROI selector
        self.roi_selector = QComboBox()
        self.roi_selector.addItem("All ROIs")
        self.roi_selector.setVisible(False)  # Hidden by default
        self.roi_selector.currentIndexChanged.connect(self.on_roi_selection_changed)
        display_box.addWidget(self.roi_selector)
        
        controls_layout.addLayout(display_box)
        controls_layout.addLayout(export_box)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        self.setLayout(layout)

    def set_results_directory(self, results_dir):
        """Set the directory where the results are stored."""
        if isinstance(results_dir, str) and os.path.isdir(results_dir):
            self.results_dir = results_dir
            print(f"Results directory set to: {self.results_dir}")
        else:
            print(f"Invalid results directory: {results_dir}")
            self.results_dir = None

    def show_results(self, results=None, params=None):
        """Display processed images or data from the results directory."""
        if isinstance(results, pd.DataFrame):
            self.results = results
            # Update ROI information
            self.has_roi_results = True
            if self.parent and hasattr(self.parent, 'crops') and self.parent.crops:
                # Update ROI selector with actual ROI information
                self.roi_selector.clear()
                self.roi_selector.addItem("All ROIs")
                for i in range(len(self.parent.crops)):
                    self.roi_selector.addItem(f"ROI_{i+1}")
                self.roi_selector.setVisible(True)
                print(f"Added {len(self.parent.crops)} ROIs to selector")
            else:
                self.has_roi_results = False
                self.roi_selector.setVisible(False)
        if isinstance(params, dict):
            self.params = params

        if not isinstance(self.results_dir, str) or not os.path.exists(self.results_dir):
            QMessageBox.warning(self, "Error", "Invalid or missing results directory.")
            print(f"Current results directory: {self.results_dir}")
            return
            
        print(f"Looking for results in: {self.results_dir}")

        # Clear existing widgets from the layout while preserving the original files
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)  # Detach from layout without deleting
                widget.deleteLater()  # Schedule for deletion after display is updated
        
        # Clear the stored canvas references (doesn't affect original files)
        self.canvases.clear()

        # Locate processed files based on the selected display type
        display_type = self.display_combo.currentText()
        print(f"\nShowing results for display type: {display_type}")
        if hasattr(self, 'show_individual'):
            print(f"Show Individual Plots checkbox state: {self.show_individual.isChecked()}")
        
        file_paths = self.locate_processed_files(display_type)
        
        print(f"Found {len(file_paths)} files:")
        for path in file_paths:
            print(f"  - {os.path.basename(path)}")

        if not file_paths:
            self.status_label.setText(f"No processed files found for {display_type}.")
            return

        self.status_label.setText(f"Displaying {len(file_paths)} {display_type} result(s).")
        
        # Create a frame for each image to maintain consistent sizing
        for file_path in file_paths:
            if file_path.endswith(".png"):
                # Create a container widget to hold the canvas
                container = QWidget()
                container_layout = QVBoxLayout()
                container_layout.setContentsMargins(0, 0, 0, 0)
                container.setLayout(container_layout)
                
                # Create and add the canvas
                fig = Figure(figsize=(6, 5), dpi=100)  # Slightly larger figures
                canvas = FigureCanvas(fig)
                canvas.setMinimumWidth(400)  # Set minimum width to prevent too small plots
                canvas.setMinimumHeight(350)  # Set minimum height
                self.canvases.append(canvas)
                
                # Add the canvas to the container
                container_layout.addWidget(canvas)
                
                # Add the container to the main layout
                self.results_layout.addWidget(container)
                
                # Display the image
                self.display_image(canvas, file_path)
            elif file_path.endswith(".csv") and display_type == "Summary":
                # Create a widget to hold the table with its own scrollbars
                scroll_widget = QWidget()
                scroll_layout = QVBoxLayout()
                scroll_layout.setContentsMargins(0, 0, 0, 0)
                scroll_widget.setLayout(scroll_layout)
                
                # Add the table to this widget
                self.display_csv(file_path)
                
                # Add the scroll widget to main layout
                self.results_layout.addWidget(scroll_widget)
        
        # Add a stretch at the end to keep plots left-aligned
        self.results_layout.addStretch()
        
        # Ensure the results widget is wide enough to accommodate all plots
        total_width = len(file_paths) * 420  # 400px width + 20px spacing
        self.results_widget.setMinimumWidth(total_width)

    def locate_processed_files(self, display_type):
        """Locate processed files based on the display type."""
        if not self.results_dir:
            return []

        keywords = {
            "ROIs": "ROI",  # For ROI-specific files
            "Summary": "summary",  # Summary data and mean plots
            "ACF Plots": {"mean": "Mean ACF", "indv": ["Individual_ACF_plots", "ACF"]},
            "CCF Plots": {"mean": "Mean CCF", "indv": ["Individual_CCF_plots", "CCF"]},
            "Peak Properties": {"mean": "Peak Props", "indv": ["Individual_peak_plots", "Peak"]}
        }

        file_paths = []
        
        # Handle different display types
        if display_type == "Summary":
            # Look for summary CSV files in main directory
            for root, dirs, files in os.walk(self.results_dir):
                for f in files:
                    if f.endswith(".csv") and keywords[display_type] in f:
                        file_paths.append(os.path.join(root, f))
        elif display_type == "ROIs":
            # Look for ROI-related files
            for root, dirs, files in os.walk(self.results_dir):
                for f in files:
                    if f.endswith(".png") and keywords[display_type] in f:
                        file_paths.append(os.path.join(root, f))
        else:
            # Handle plot types that can show either summary or individual
            selected_type = display_type  # The type the user selected to view
            keyword_info = keywords.get(selected_type)  # Get keywords for selected type
            if keyword_info:
                print(f"Looking for {selected_type} plots...")
                # Check if individual plots are requested
                if hasattr(self, 'show_individual') and self.show_individual.isChecked():
                    # Show only individual plots
                    print(f"Looking for individual plots for {selected_type}...")
                    indv_dir, indv_keyword = keyword_info["indv"]
                    for root, dirs, files in os.walk(self.results_dir):
                        if os.path.basename(root) == indv_dir:
                            for f in files:
                                if f.endswith(".png") and indv_keyword in f:
                                    print(f"Found individual plot: {f}")
                                    file_paths.append(os.path.join(root, f))
                else:
                    # Show only summary plots
                    print(f"Looking for summary plots for {selected_type}...")
                    for root, dirs, files in os.walk(self.results_dir):
                        for f in files:
                            if f.endswith(".png") and keyword_info["mean"] in f:
                                print(f"Found summary plot: {f}")
                                file_paths.append(os.path.join(root, f))
                                    
        return sorted(file_paths)

    def display_image(self, canvas, image_path):
        """Load and display an image on the given canvas."""
        if not os.path.exists(image_path):
            # Handle missing file case
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Image not found", fontsize=12, ha="center", va="center")
            fig.tight_layout()
            canvas.draw()
            return
            
        try:
            # Load image without modifying the original file
            img = imread(image_path)  # This creates a copy of the image data
            
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(os.path.basename(image_path), fontsize=10)
            fig.tight_layout()
            canvas.draw()
        except Exception as e:
            print(f"Error displaying image {image_path}: {e}")
            # Handle error without affecting original file
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Error loading image", fontsize=12, ha="center", va="center")
            fig.tight_layout()
            canvas.draw()

    def display_csv(self, csv_path):
        """Display the contents of a CSV file in a table widget."""
        if not os.path.exists(csv_path):
            QMessageBox.warning(self, "Error", f"CSV file not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
            table_widget = QTableWidget()
            table_widget.setRowCount(len(df))
            table_widget.setColumnCount(len(df.columns))
            table_widget.setHorizontalHeaderLabels(df.columns)

            # Calculate the width needed for the table
            table_width = table_widget.verticalHeader().width()
            for i in range(table_widget.columnCount()):
                table_width += table_widget.columnWidth(i)

            # Calculate the height needed for the table
            table_height = table_widget.horizontalHeader().height()
            for i in range(table_widget.rowCount()):
                table_height += table_widget.rowHeight(i)

            # Set fixed size based on content (with some padding)
            table_widget.setMinimumWidth(min(table_width + 50, 800))  # Cap at 800px width
            table_widget.setMinimumHeight(min(table_height + 50, 600))  # Cap at 600px height

            # Fill the table
            for i, row in df.iterrows():
                for j, value in enumerate(row):
                    table_widget.setItem(i, j, QTableWidgetItem(str(value)))

            # Create a container widget with vertical layout
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.addWidget(table_widget)
            container.setLayout(container_layout)

            # Add to results layout
            self.results_layout.addWidget(container)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV file: {e}")

    def display_roi_results(self, roi_selection="All ROIs"):
        """Display results specific to the selected ROI."""
        if not self.results is None and hasattr(self.parent, 'crops') and self.parent.crops:
            # Clear existing display while preserving original data
            while self.results_layout.count() > 0:
                item = self.results_layout.takeAt(0)
                if item.widget():
                    widget = item.widget()
                    widget.setParent(None)  # Detach from layout without deleting
                    widget.deleteLater()  # Schedule for deletion after display is updated

            # Create a table widget to display results
            table_widget = QTableWidget()
            
            if roi_selection == "All ROIs":
                # Show results for all ROIs
                display_data = pd.DataFrame()
                for i, roi in enumerate(self.parent.crops):
                    roi_data = pd.DataFrame({"ROI": [f"ROI_{i+1}"],
                                          "X_min": [roi[:, 0].min()],
                                          "X_max": [roi[:, 0].max()],
                                          "Y_min": [roi[:, 1].min()],
                                          "Y_max": [roi[:, 1].max()],
                                          "Area": [len(roi)]})
                    display_data = pd.concat([display_data, roi_data], ignore_index=True)
            else:
                # Show data for specific ROI
                roi_idx = int(roi_selection.split("_")[1]) - 1
                roi = self.parent.crops[roi_idx]
                display_data = pd.DataFrame({"Parameter": ["X_min", "X_max", "Y_min", "Y_max", "Area"],
                                           "Value": [roi[:, 0].min(), roi[:, 0].max(),
                                                   roi[:, 1].min(), roi[:, 1].max(),
                                                   len(roi)]})

            # Set up table dimensions
            rows = len(display_data)
            cols = len(display_data.columns)
            headers = display_data.columns

            table_widget.setRowCount(rows)
            table_widget.setColumnCount(cols)
            table_widget.setHorizontalHeaderLabels(headers)

            # Fill the table with data
            for i in range(rows):
                for j, col in enumerate(headers):
                    value = display_data.iloc[i][col]
                    table_widget.setItem(i, j, QTableWidgetItem(str(value)))

            # Add the table to the layout
            self.results_layout.addWidget(table_widget)
            
            # Add stretch to keep everything aligned
            self.results_layout.addStretch()

    def on_roi_selection_changed(self):
        """Handle changes in ROI selection."""
        if self.display_combo.currentText() == "ROIs" and self.has_roi_results:
            self.display_roi_results(self.roi_selector.currentText())

    def set_acf_visibility(self, visible):
        """Set visibility of the Individual ACF button."""
        if hasattr(self, "individual_acf_btn"):
            self.individual_acf_btn.setVisible(visible)

    def set_ccf_visibility(self, visible):
        """Set visibility of the Individual CCF button."""
        if hasattr(self, "individual_ccf_btn"):
            self.individual_ccf_btn.setVisible(visible)

    def set_peaks_visibility(self, visible):
        """Set visibility of the Individual Peaks button."""
        if hasattr(self, "individual_peaks_btn"):
            self.individual_peaks_btn.setVisible(visible)

    def on_display_type_changed(self):
        """Handle changes in the display type dropdown."""
        display_type = self.display_combo.currentText()
        
        # Show/hide individual plot checkbox based on display type
        self.show_individual.setVisible(display_type in ["ACF Plots", "CCF Plots", "Peak Properties"])
        
        if display_type == "ROIs":
            # Show ROI selector and results for ROIs view
            if self.parent and hasattr(self.parent, 'crops') and self.parent.crops:
                self.has_roi_results = True
                self.roi_selector.clear()
                self.roi_selector.addItem("All ROIs")
                for i in range(len(self.parent.crops)):
                    self.roi_selector.addItem(f"ROI_{i+1}")
                self.roi_selector.setVisible(True)
                self.display_roi_results(self.roi_selector.currentText())
            else:
                self.has_roi_results = False
                self.roi_selector.setVisible(False)
                if self.results_dir and os.path.exists(self.results_dir):
                    self.show_results()
        else:
            # Hide ROI selector for other views
            self.roi_selector.setVisible(False)
            if self.results_dir and os.path.exists(self.results_dir):
                self.show_results()
                print(f"Showing results for type: {display_type}")
            else:
                print("No results directory set or directory does not exist")
                
    def on_show_individual_changed(self, state):
        """Handle changes in the individual plots checkbox."""
        print(f"\nIndividual plots checkbox changed to: {'checked' if state else 'unchecked'}")
        print(f"Current display type: {self.display_combo.currentText()}")
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()  # This will update the display based on current selection

    def export_plots(self):
        """Export plots based on selected options."""
        if not self.results_dir or not os.path.exists(self.results_dir):
            QMessageBox.warning(self, "Error", "No results directory available.")
            return

        # Get target directory from user
        target_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory")
        if not target_dir:
            return  # User cancelled

        # Create export directory
        export_dir = os.path.join(target_dir, "wave_analysis_plots")
        os.makedirs(export_dir, exist_ok=True)

        try:
            # Determine which plots to export based on selection
            export_type = self.export_combo.currentText()

            if export_type == "Summary Plots Only" or export_type == "All Plots":
                # Export summary plots and data
                for plot_type in ["Summary", "ACF Plots", "CCF Plots", "Peak Properties"]:
                    files = self.locate_processed_files(plot_type)
                    for src in files:
                        if os.path.exists(src):
                            dest = os.path.join(export_dir, os.path.basename(src))
                            with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
                                fdst.write(fsrc.read())

            if export_type == "Individual Plots Only" or export_type == "All Plots":
                # Export individual plots
                for plot_type in ["Individual ACF Plots", "Individual CCF Plots", "Individual Peak Plots"]:
                    files = self.locate_processed_files(plot_type)
                    if files:  # Only create subdir if files exist
                        subdir = plot_type.replace(" ", "_")
                        os.makedirs(os.path.join(export_dir, subdir), exist_ok=True)
                        for src in files:
                            if os.path.exists(src):
                                dest = os.path.join(export_dir, subdir, os.path.basename(src))
                                with open(src, 'rb') as fsrc, open(dest, 'wb') as fdst:
                                    fdst.write(fsrc.read())

            QMessageBox.information(self, "Export Complete", f"Plots have been exported to:\n{export_dir}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export plots:\n{str(e)}")
            return