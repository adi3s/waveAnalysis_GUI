import os
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QGroupBox,
    QComboBox, QPushButton, QCheckBox, QMessageBox, QFileDialog,
    QTableWidget, QTableWidgetItem, QGridLayout
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
        self.loaded_images = []  # Store list of loaded images
        self.canvases = []
        
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
        display_label = QLabel("Display Options:")
        display_label.setStyleSheet("font-weight: bold;")
        display_box.addWidget(display_label)

        # Data source selection with help text (hidden for Summary view)
        self.data_source_widget = QWidget()
        data_source_widget_layout = QVBoxLayout()
        data_source_widget_layout.setContentsMargins(0, 0, 0, 0)
        
        data_source_help = QLabel("Select data to display:")
        data_source_help.setStyleSheet("font-size: 10px; color: gray;")
        data_source_widget_layout.addWidget(data_source_help)
        
        data_source_layout = QHBoxLayout()
        self.whole_image_checkbox = QCheckBox("Whole Image")
        self.whole_image_checkbox.setChecked(True)
        self.whole_image_checkbox.setToolTip("Show analysis results for the entire image")
        self.whole_image_checkbox.stateChanged.connect(self.on_data_source_changed)
        
        self.roi_data_checkbox = QCheckBox("ROI Data")
        self.roi_data_checkbox.setChecked(False)
        self.roi_data_checkbox.setToolTip("Show analysis results for individual ROIs")
        self.roi_data_checkbox.stateChanged.connect(self.on_data_source_changed)
        
        data_source_layout.addWidget(self.whole_image_checkbox)
        data_source_layout.addWidget(self.roi_data_checkbox)
        data_source_widget_layout.addLayout(data_source_layout)
        
        self.data_source_widget.setLayout(data_source_widget_layout)
        self.data_source_widget.setVisible(False)  # Hidden by default for Summary
        display_box.addWidget(self.data_source_widget)

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

        # Display type dropdown
        display_type_label = QLabel("Plot Type:")
        display_box.addWidget(display_type_label)
        
        # ROI selector removed as ROIs are now processed automatically
        
        controls_layout.addLayout(display_box)
        controls_layout.addLayout(export_box)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        self.setLayout(layout)

    def set_results_directory(self, results_dir):
        """Set the directory where the results are stored."""
        if isinstance(results_dir, str) and os.path.isdir(results_dir):
            self.results_dir = results_dir
        else:
            self.results_dir = None

    def set_loaded_images(self, image_list):
        """Set the list of loaded images for multi-image analysis display"""
        self.loaded_images = image_list if image_list else []

    def set_roi_results(self, results_df):
        """Handle ROI measurement results (deprecated)"""
        pass  # ROI results are now handled automatically through the analysis process

    def show_results(self, results=None, params=None):
        """Display processed images or data from the results directory."""
        if isinstance(results, pd.DataFrame):
            self.results = results
        if isinstance(params, dict):
            self.params = params

        if not isinstance(self.results_dir, str) or not os.path.exists(self.results_dir):
            QMessageBox.warning(self, "Error", "Invalid or missing results directory.")
            return

        # Clear existing widgets from the layout while preserving the original files
        while self.results_layout.count() > 0:
            item = self.results_layout.takeAt(0)
            if item.widget():
                widget = item.widget()
                widget.setParent(None)
                widget.deleteLater()
        
        # Clear the stored canvas references
        self.canvases.clear()
        
        # Create a grid layout for organized display of ROI plots
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)
        self.results_layout.addWidget(grid_widget)

        # Get display type and files
        display_type = self.display_combo.currentText()
        file_paths = self.locate_processed_files(display_type)
        
        if not file_paths:
            # Check if user selected ROI data but no ROI files exist
            if self.roi_data_checkbox.isChecked() and not self.whole_image_checkbox.isChecked():
                self.status_label.setText(f"No ROI processed files found for {display_type}. Try running analysis with ROIs first.")
            else:
                self.status_label.setText(f"No processed files found for {display_type}.")
            
            # Display a helpful message
            container = QWidget()
            container_layout = QVBoxLayout()
            
            if self.roi_data_checkbox.isChecked() and not self.whole_image_checkbox.isChecked():
                message = QLabel(
                    "No ROI analysis results found.\n\n"
                    "To see ROI-specific results:\n"
                    "1. Go to the ROI tab and create ROIs\n"
                    "2. Run analysis from the Pre-Processing tab\n"
                    "3. ROI-specific results will appear here"
                )
            else:
                message = QLabel(f"No processed files found for {display_type}.")
            
            message.setAlignment(Qt.AlignCenter)
            message.setStyleSheet("color: gray; font-size: 12px; padding: 20px;")
            container_layout.addWidget(message)
            container.setLayout(container_layout)
            self.results_layout.addWidget(container)
            return
            
        # Group files by ROI or Group (enhanced logic for multi-image analysis)
        grouped_files = {}
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            filepath_parts = file_path.split(os.sep)
            
            # Look for ROI or Group identifiers in path (prioritize directory structure)
            group_identifier = None
            
            # First check for ROI directories in the path (highest priority)
            for part in filepath_parts:
                if part.startswith("ROI_"):
                    group_identifier = part
                    break
            
            # Then check for group directories or image names in the path
            if not group_identifier:
                for part in filepath_parts:
                    if part.startswith("1_Group") or part.startswith("Image_"):
                        group_identifier = part
                        break
            
            # Then check for ROI in filename
            if not group_identifier:
                if "ROI_" in filename:
                    group_identifier = "ROI_" + filename.split("ROI_")[1].split("_")[0]
                elif "roi_" in filename:
                    group_identifier = "ROI_" + filename.split("roi_")[1].split("_")[0]
            
            # If no specific identifier found, use the parent directory name or display type
            if not group_identifier:
                if len(filepath_parts) > 1:
                    parent_dir = os.path.basename(os.path.dirname(file_path))
                    if parent_dir and parent_dir != os.path.basename(self.results_dir):
                        group_identifier = parent_dir
                    else:
                        group_identifier = display_type
                else:
                    group_identifier = display_type
            
            if group_identifier not in grouped_files:
                grouped_files[group_identifier] = []
            grouped_files[group_identifier].append(file_path)
        
        # Display files in a grid, one row per group
        row = 0
        for group_name, files in grouped_files.items():
            # Add group label
            label = QLabel(group_name)
            label.setStyleSheet("font-weight: bold; color: #ffcc00;")
            grid_layout.addWidget(label, row, 0)
            col = 1
            
            # Add plots for this group
            for file_path in files:
                if file_path.endswith(".png"):
                    # Create container and canvas
                    container = QWidget()
                    container_layout = QVBoxLayout()
                    container_layout.setContentsMargins(5, 5, 5, 5)
                    container.setLayout(container_layout)
                    
                    fig = Figure(figsize=(5, 4), dpi=100)
                    canvas = FigureCanvas(fig)
                    canvas.setMinimumWidth(300)
                    canvas.setMinimumHeight(250)
                    self.canvases.append(canvas)
                    
                    container_layout.addWidget(canvas)
                    grid_layout.addWidget(container, row, col)
                    
                    self.display_image(canvas, file_path)
                    col += 1
                elif file_path.endswith(".csv") and display_type == "Summary":
                    scroll_widget = QWidget()
                    scroll_layout = QVBoxLayout()
                    scroll_layout.setContentsMargins(5, 5, 5, 5)
                    scroll_widget.setLayout(scroll_layout)
                    
                    self.display_csv(file_path)
                    grid_layout.addWidget(scroll_widget, row, col)
                    col += 1
            row += 1
        
        # Update status message based on data source selection
        if display_type == "Summary":
            # For Summary, always show all data
            data_source_text = "All Data"
        else:
            data_source = []
            if self.whole_image_checkbox.isChecked():
                data_source.append("Whole Image")
            if self.roi_data_checkbox.isChecked():
                data_source.append("ROI Data")
            
            data_source_text = " & ".join(data_source) if data_source else "No Data"
        
        self.status_label.setText(f"Displaying {len(grouped_files)} result groups for {display_type} ({data_source_text}).")
        
        # Add a stretch at the end to keep plots left-aligned
        self.results_layout.addStretch()
        
        # Ensure the results widget is wide enough to accommodate all plots
        total_width = len(file_paths) * 420  # 400px width + 20px spacing
        self.results_widget.setMinimumWidth(total_width)

    def locate_processed_files(self, display_type):
        """Locate processed files based on the display type and data source selection."""
        if not self.results_dir:
            return []

        keywords = {
            "Summary": "summary",  # Summary data and mean plots
            "ACF Plots": {"mean": "Mean ACF", "indv": ["Individual_ACF_plots", "ACF"]},
            "CCF Plots": {"mean": "Mean CCF", "indv": ["Individual_CCF_plots", "CCF"]},
            "Peak Properties": {"mean": "Peak Props", "indv": ["Individual_peak_plots", "Peak"]}
        }

        file_paths = []
        
        # Determine data source filter
        # For Summary, always show both whole image and ROI data
        if display_type == "Summary":
            show_whole_image = True
            show_roi_data = True
        else:
            show_whole_image = self.whole_image_checkbox.isChecked()
            show_roi_data = self.roi_data_checkbox.isChecked()
        
        # Handle different display types - search in original structure
        if display_type == "Summary":
            # Look for summary CSV files in main directory and subdirectories
            for root, dirs, files in os.walk(self.results_dir):
                for f in files:
                    if f.endswith(".csv") and keywords[display_type] in f:
                        file_path = os.path.join(root, f)
                        # Filter based on data source selection (always both for Summary)
                        if self._should_include_file(f, file_path, show_whole_image, show_roi_data):
                            file_paths.append(file_path)
        else:
            # Handle plot types that can show either summary or individual
            selected_type = display_type  # The type the user selected to view
            keyword_info = keywords.get(selected_type)  # Get keywords for selected type
            if keyword_info:
                # Check if individual plots are requested
                if hasattr(self, 'show_individual') and self.show_individual.isChecked():
                    # Show only individual plots
                    indv_dir, indv_keyword = keyword_info["indv"]
                    for root, dirs, files in os.walk(self.results_dir):
                        if os.path.basename(root) == indv_dir:
                            for f in files:
                                if f.endswith(".png") and indv_keyword in f:
                                    file_path = os.path.join(root, f)
                                    if self._should_include_file(f, file_path, show_whole_image, show_roi_data):
                                        file_paths.append(file_path)
                else:
                    # Show only summary plots
                    for root, dirs, files in os.walk(self.results_dir):
                        for f in files:
                            if f.endswith(".png") and keyword_info["mean"] in f:
                                file_path = os.path.join(root, f)
                                if self._should_include_file(f, file_path, show_whole_image, show_roi_data):
                                    file_paths.append(file_path)
                                    
        return sorted(file_paths)

    def _should_include_file(self, filename, file_path, show_whole_image, show_roi_data):
        """Determine if a file should be included based on data source selection."""
        # Normalize path separators for consistent checking
        normalized_path = file_path.replace("\\", "/")
        path_parts = normalized_path.split("/")
        
        # Check if file is in an ROI subdirectory or has ROI in filename
        is_roi_file = False
        for part in path_parts:
            if part.startswith("ROI_"):
                is_roi_file = True
                break
        
        # Also check filename directly
        if "ROI_" in filename or "roi_" in filename.lower():
            is_roi_file = True
        
        # Include file based on selection
        if is_roi_file:
            return show_roi_data
        else:
            return show_whole_image

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

    # ROI-specific methods removed as ROIs are now processed automatically with analysis

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
        
        # Show/hide data source selection based on display type
        # For Summary, show everything; for plots, allow filtering
        if display_type == "Summary":
            self.data_source_widget.setVisible(False)
        else:
            self.data_source_widget.setVisible(True)
        
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()
                
    def on_show_individual_changed(self, state):
        """Handle changes in the individual plots checkbox."""
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()  # This will update the display based on current selection
    
    def on_data_source_changed(self, state):
        """Handle changes in data source selection (whole image vs ROI)."""
        # Ensure at least one option is selected
        if not self.whole_image_checkbox.isChecked() and not self.roi_data_checkbox.isChecked():
            # If user unchecked the last option, recheck it
            sender = self.sender()
            sender.setChecked(True)
            return
        
        # Update display if results are available
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()

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