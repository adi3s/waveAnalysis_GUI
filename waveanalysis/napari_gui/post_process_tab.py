import os
import numpy as np
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
        self.loaded_image_names = []  # Store list of loaded image names
        self.canvases = []
        # Store which summary plots were requested
        self.plot_preferences = {
            "plot_summary_acfs": True,
            "plot_summary_ccfs": True,
            "plot_summary_peaks": True
        }
        # Store what type of analysis was performed
        self.analysis_scope = {
            "analyze_whole_image": False,
            "analyze_roi_data": False
        }
        
        # Main layout
        layout = QVBoxLayout()
        
        # Add status bar at the top
        self.status_label = QLabel("No results loaded")
        layout.addWidget(self.status_label)
        
        # Add image selector dropdown for filtering (initially hidden)
        self.image_filter_widget = QWidget()
        filter_layout = QHBoxLayout()
        filter_layout.setContentsMargins(5, 5, 5, 5)
        filter_label = QLabel("Filter by Image:")
        filter_label.setStyleSheet("font-weight: bold; padding: 5px;")
        filter_layout.addWidget(filter_label)
        
        self.image_filter_combo = QComboBox()
        self.image_filter_combo.addItem("Show All Images")
        self.image_filter_combo.currentIndexChanged.connect(self.on_image_filter_changed)
        filter_layout.addWidget(self.image_filter_combo)
        filter_layout.addStretch()
        
        self.image_filter_widget.setLayout(filter_layout)
        self.image_filter_widget.setVisible(False)
        layout.addWidget(self.image_filter_widget)
        
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
            "Peak Properties",
            "ROI Statistics"
        ])
        self.display_combo.currentIndexChanged.connect(self.on_display_type_changed)
        display_box.addWidget(self.display_combo)
        
        # Add individual plot checkbox
        self.show_individual = QCheckBox("Show Individual Plots")
        self.show_individual.stateChanged.connect(self.on_show_individual_changed)
        self.show_individual.setVisible(False)  # Only show for plot types that support it
        display_box.addWidget(self.show_individual)
        
        # Add statistic selector for ROI Statistics view
        stat_selector_label = QLabel("Statistic:")
        display_box.addWidget(stat_selector_label)
        self.stat_selector = QComboBox()
        self.stat_selector.addItems(["Mean", "Median", "StdDev", "SEM"])
        self.stat_selector.currentIndexChanged.connect(self.on_stat_selector_changed)
        self.stat_selector.setVisible(False)  # Only show for ROI Statistics view
        display_box.addWidget(self.stat_selector)
        stat_selector_label.setVisible(False)
        self.stat_selector_label = stat_selector_label

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
            self.update_display_options()
        else:
            self.results_dir = None
    
    def is_rolling_analysis(self):
        """Detect if the current results are from rolling analysis by checking for summary_plots directory."""
        if not self.results_dir or not os.path.exists(self.results_dir):
            return False
        
        # Check for summary_plots subdirectory which is specific to rolling analysis
        for root, dirs, files in os.walk(self.results_dir):
            if "summary_plots" in dirs:
                return True
        return False
    
    def update_display_options(self):
        """Update display combo box options based on analysis type."""
        current_selection = self.display_combo.currentText()
        self.display_combo.blockSignals(True)  # Prevent triggering change event
        self.display_combo.clear()
        
        # Always add Summary
        self.display_combo.addItem("Summary")
        
        if self.is_rolling_analysis():
            # Rolling analysis uses different terminology
            # Only add if they were generated
            if self.plot_preferences.get("plot_summary_acfs", True):
                self.display_combo.addItem("Period Plots")
            if self.plot_preferences.get("plot_summary_ccfs", True):
                self.display_combo.addItem("Shift Plots")
            if self.plot_preferences.get("plot_summary_peaks", True):
                self.display_combo.addItem("Peak Properties")
            
            # Try to maintain selection with mapping
            mapping = {
                "ACF Plots": "Period Plots",
                "CCF Plots": "Shift Plots",
                "Peak Properties": "Peak Properties",
                "Summary": "Summary",
                "ROI Statistics": "ROI Statistics"
            }
            new_selection = mapping.get(current_selection, "Summary")
        else:
            # Standard/Kymograph analysis uses traditional names
            # Only add if they were generated
            if self.plot_preferences.get("plot_summary_acfs", True):
                self.display_combo.addItem("ACF Plots")
            if self.plot_preferences.get("plot_summary_ccfs", True):
                self.display_combo.addItem("CCF Plots")
            if self.plot_preferences.get("plot_summary_peaks", True):
                self.display_combo.addItem("Peak Properties")
            
            # Try to maintain selection with reverse mapping
            mapping = {
                "Period Plots": "ACF Plots",
                "Shift Plots": "CCF Plots",
                "Peak Properties": "Peak Properties",
                "Summary": "Summary",
                "ROI Statistics": "ROI Statistics"
            }
            new_selection = mapping.get(current_selection, "Summary")
        
        # Always add ROI Statistics option
        self.display_combo.addItem("ROI Statistics")
        
        # Restore selection if possible
        index = self.display_combo.findText(new_selection)
        if index >= 0:
            self.display_combo.setCurrentIndex(index)
        
        self.display_combo.blockSignals(False)

    def set_loaded_images(self, image_list):
        """Set the list of loaded images for multi-image analysis display"""
        self.loaded_images = image_list if image_list else []

    def set_loaded_image_names(self, image_names):
        """Set the list of loaded image names (without extension) for labeling"""
        self.loaded_image_names = image_names if image_names else []
        
        # Update the filter dropdown
        self.update_image_filter()
    
    def update_image_filter(self):
        """Update the image filter dropdown"""
        # Clear and repopulate the filter combo
        self.image_filter_combo.blockSignals(True)
        self.image_filter_combo.clear()
        self.image_filter_combo.addItem("Show All Images")
        
        if len(self.loaded_image_names) > 1:
            for name in self.loaded_image_names:
                self.image_filter_combo.addItem(name)
            self.image_filter_widget.setVisible(True)
        else:
            self.image_filter_widget.setVisible(False)
        
        self.image_filter_combo.blockSignals(False)
    
    def on_image_filter_changed(self, index):
        """Handle image filter selection change"""
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()

    def set_plot_preferences(self, pre_params):
        """Set which summary plots were requested during analysis"""
        if pre_params:
            self.plot_preferences = {
                "plot_summary_acfs": pre_params.get("plot_summary_acfs", True),
                "plot_summary_ccfs": pre_params.get("plot_summary_ccfs", True),
                "plot_summary_peaks": pre_params.get("plot_summary_peaks", True)
            }
            # Also capture what type of analysis was performed
            self.analysis_scope = {
                "analyze_whole_image": pre_params.get("analyze_whole_image", False),
                "analyze_roi_data": pre_params.get("analyze_roi_data", False)
            }
        self.update_display_options_for_preferences()

    def update_display_options_for_preferences(self):
        """Update display combo box to only show plot types that were generated"""
        if not hasattr(self, 'display_combo'):
            return
            
        current_selection = self.display_combo.currentText()
        self.display_combo.blockSignals(True)
        
        # Get current items to preserve rolling vs standard naming
        current_items = [self.display_combo.itemText(i) for i in range(self.display_combo.count())]
        is_rolling = "Period Plots" in current_items or "Shift Plots" in current_items
        
        # Clear and rebuild
        self.display_combo.clear()
        
        # Always add Summary
        self.display_combo.addItem("Summary")
        
        # Add ACF/Period plots if they were generated
        if self.plot_preferences.get("plot_summary_acfs", True):
            if is_rolling:
                self.display_combo.addItem("Period Plots")
            else:
                self.display_combo.addItem("ACF Plots")
        
        # Add CCF/Shift plots if they were generated
        if self.plot_preferences.get("plot_summary_ccfs", True):
            if is_rolling:
                self.display_combo.addItem("Shift Plots")
            else:
                self.display_combo.addItem("CCF Plots")
        
        # Add Peak Properties if they were generated
        if self.plot_preferences.get("plot_summary_peaks", True):
            self.display_combo.addItem("Peak Properties")
        
        # Always add ROI Statistics ( doesn't depend on plot preferences)
        self.display_combo.addItem("ROI Statistics")
        
        # Try to restore previous selection if still available
        index = self.display_combo.findText(current_selection)
        if index >= 0:
            self.display_combo.setCurrentIndex(index)
        else:
            self.display_combo.setCurrentIndex(0)  # Default to Summary
        
        self.display_combo.blockSignals(False)

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
        
        # Get display type
        display_type = self.display_combo.currentText()
        
        # Handle ROI Statistics display separately
        if display_type == "ROI Statistics":
            self.show_roi_statistics()
            return
        
        # Create a grid layout for organized display of ROI plots
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)
        self.results_layout.addWidget(grid_widget)

        # Get display type and files
        display_type = self.display_combo.currentText()
        file_paths = self.locate_processed_files(display_type)
        
        if not file_paths:
            self.status_label.setText(f"No processed files found for {display_type}.")
            
            # Display a helpful message
            container = QWidget()
            container_layout = QVBoxLayout()
            
            message = QLabel(f"No processed files found for {display_type}.")
            
            message.setAlignment(Qt.AlignCenter)
            message.setStyleSheet("color: gray; font-size: 12px; padding: 20px;")
            container_layout.addWidget(message)
            container.setLayout(container_layout)
            self.results_layout.addWidget(container)
            return
            
        # Group files by Image and ROI (enhanced logic for multi-image analysis)
        grouped_files = {}
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            filepath_parts = file_path.split(os.sep)
            
            # Extract image name and ROI information for better labeling
            image_name = None
            roi_id = None
            
            # Look for ROI directory in path (for combined/standard workflow)
            for part in filepath_parts:
                if part.startswith("ROI_"):
                    roi_id = part  # e.g., "ROI_1"
                    break
            
            # For rolling workflow, ROI info is in the filename, not the path
            # Check if this is a rolling analysis file (in summary_plots folder)
            if "summary_plots" in filepath_parts and not roi_id:
                # Check for ROI pattern in filename (e.g., "imagename_ROI_1_Period.png")
                if "_ROI_" in filename:
                    try:
                        roi_num = filename.split("_ROI_")[1].split("_")[0]
                        roi_id = f"ROI_{roi_num}"
                    except:
                        pass
            
            # Try to match with loaded image names first (highest priority)
            for loaded_name in self.loaded_image_names:
                if loaded_name in filename or loaded_name in os.sep.join(filepath_parts):
                    image_name = loaded_name
                    break
            
            # If not found in loaded names, extract from filename or path
            if not image_name:
                # Check if filename contains ROI pattern (e.g., "imagename_ROI_1_...")
                if "_ROI_" in filename:
                    # Extract image name before _ROI_
                    image_name = filename.split("_ROI_")[0]
                
                # If no image name found yet, try to extract from parent directory
                if not image_name:
                    # Check for common image name patterns in parent directories
                    for part in filepath_parts:
                        if not part.startswith("ROI_") and not part.startswith("0_signalProcessing"):
                            # This might be the image name directory
                            if part not in ["Individual_ACF_plots", "Individual_CCF_plots", 
                                          "Individual_peak_plots", "mean_parameter_measurements",
                                          "group_comparison_graphs", "summary_plots"]:
                                image_name = part
                    
                    # Fallback: extract from filename
                    if not image_name:
                        # Remove common suffixes to get image name
                        image_name = filename.replace("_Mean ACF.png", "").replace("_Mean CCF.png", "")
                        image_name = image_name.replace("_Peak Props.png", "").replace("_summary.csv", "")
                        image_name = image_name.split("_Individual_")[0]
                        # For rolling analysis, remove the metric suffixes
                        for suffix in ["_Period", "_Shift", "_Width", "_Max", "_Min", "_Amp"]:
                            if image_name.endswith(suffix):
                                image_name = image_name[:-len(suffix)]
                                break
            
            # Create group identifier - prioritize ROI organization
            # Format: ROI_1, ROI_2, etc. OR "Whole Image" for non-ROI results
            if roi_id:
                # ROI-based analysis - use just ROI number for grouping
                group_identifier = roi_id
            else:
                # Whole image analysis
                group_identifier = "Whole Image"
            
            if group_identifier not in grouped_files:
                grouped_files[group_identifier] = []
            grouped_files[group_identifier].append(file_path)
        
        # Display files in a grid, one row per ROI
        # Sort groups: Whole Image first, then ROI_1, ROI_2, etc.
        def sort_key(item):
            group_name = item[0]
            if group_name == "Whole Image":
                return (0, "")  # Whole image comes first
            elif group_name.startswith("ROI_"):
                # Extract ROI number for proper numeric sorting
                try:
                    roi_num = int(group_name.replace("ROI_", ""))
                    return (1, roi_num)  # ROIs come after, sorted by number
                except:
                    return (1, group_name)
            else:
                return (2, group_name)  # Everything else comes last
        
        sorted_groups = sorted(grouped_files.items(), key=sort_key)
        
        row = 0
        for group_name, files in sorted_groups:
            # Skip Whole_Image group if only ROI analysis was performed
            if group_name == "Whole Image":
                if not self.analysis_scope.get("analyze_whole_image", False):
                    # User didn't request whole image analysis, skip this group
                    continue
            
            # Skip ROI groups if only whole image analysis was performed
            if group_name.startswith("ROI_"):
                if not self.analysis_scope.get("analyze_roi_data", False):
                    # User didn't request ROI analysis, skip this group
                    continue
            
            # Get the selected filter
            selected_filter = self.image_filter_combo.currentText()
            
            # Filter files based on selected image
            filtered_files = []
            group_images = set()
            
            for file_path in files:
                # Determine which image this file belongs to
                file_image = None
                for loaded_name in self.loaded_image_names:
                    if loaded_name in file_path or loaded_name in os.sep.join(file_path.split(os.sep)):
                        file_image = loaded_name
                        group_images.add(loaded_name)
                        break
                
                # Apply filter
                if selected_filter == "Show All Images":
                    filtered_files.append(file_path)
                elif file_image == selected_filter:
                    filtered_files.append(file_path)
            
            # Skip this group if no files match the filter
            if not filtered_files:
                continue
            
            # Create clear, simple label text
            label_text = group_name
            if len(group_images) == 1:
                label_text += f" ({list(group_images)[0]})"
            elif len(group_images) > 1:
                label_text += f" (multiple images)"
            
            label = QLabel(label_text)
            label.setStyleSheet(
                "font-weight: bold; font-size: 13px; "
                "padding: 6px; background-color: rgba(100, 100, 100, 0.2); "
                "border-left: 3px solid #888; border-radius: 2px;"
            )
            label.setWordWrap(True)
            grid_layout.addWidget(label, row, 0)
            col = 1
            
            # Add plots for this group (using filtered files)
            for file_path in filtered_files:
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
                    # Display CSV table for this group
                    table_widget = self.create_csv_table(file_path)
                    if table_widget:
                        grid_layout.addWidget(table_widget, row, col)
                        col += 1
            row += 1
        
        # Update status message
        self.status_label.setText(f"Displaying {len(grouped_files)} result groups for {display_type}.")
        
        # Add a stretch at the end to keep plots left-aligned
        self.results_layout.addStretch()
        
        # Ensure the results widget is wide enough to accommodate all plots
        total_width = len(file_paths) * 420  # 400px width + 20px spacing
        self.results_widget.setMinimumWidth(total_width)

    def locate_processed_files(self, display_type):
        """Locate processed files based on the display type."""
        if not self.results_dir:
            return []

        # Map rolling-specific display types to standard ones for keyword lookup
        display_type_mapping = {
            "Period Plots": "ACF Plots",
            "Shift Plots": "CCF Plots"
        }
        
        # Use mapped type for keyword lookup
        lookup_type = display_type_mapping.get(display_type, display_type)

        keywords = {
            "Summary": "summary",  # Summary data and mean plots
            "ACF Plots": {"mean": "Mean ACF", "indv": ["Individual_ACF_plots", "ACF"]},
            "CCF Plots": {"mean": "Mean CCF", "indv": ["Individual_CCF_plots", "CCF"]},
            "Peak Properties": {"mean": "Peak Props", "indv": ["Individual_peak_plots", "Peak"]}
        }

        # Rolling analysis specific keywords for summary plots
        rolling_keywords = {
            "ACF Plots": ["Period"],  # Rolling analysis shows period plots
            "CCF Plots": ["Shift"],   # Rolling analysis shows shift plots
            "Peak Properties": ["Width", "Max", "Min", "Amp"]  # Rolling analysis peak property plots
        }

        file_paths = []
        
        # Handle different display types - search in original structure
        if lookup_type == "Summary":
            # Look for summary CSV files in main directory and subdirectories
            # Include both files with "summary" in name and general CSV files in results directories
            for root, dirs, files in os.walk(self.results_dir):
                for f in files:
                    if f.endswith(".csv"):
                        # Include if it has "summary" in the name
                        if keywords[lookup_type] in f.lower():
                            file_path = os.path.join(root, f)
                            file_paths.append(file_path)
                        # Also include CSV files in ROI subdirectories or image subdirectories
                        elif any(part.startswith("ROI_") for part in root.split(os.sep)):
                            file_path = os.path.join(root, f)
                            file_paths.append(file_path)
                        # Include CSV files in "Whole Image" subdirectories
                        elif "Whole" in root or "whole" in root.lower():
                            file_path = os.path.join(root, f)
                            file_paths.append(file_path)
        else:
            # Handle plot types that can show either summary or individual
            keyword_info = keywords.get(lookup_type)  # Get keywords for selected type
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
                                    file_paths.append(file_path)
                else:
                    # Show only summary plots
                    for root, dirs, files in os.walk(self.results_dir):
                        for f in files:
                            # Check for standard/kymograph analysis plots
                            if f.endswith(".png") and keyword_info["mean"] in f:
                                file_path = os.path.join(root, f)
                                file_paths.append(file_path)
                            # Check for rolling analysis plots (in summary_plots subdirectory)
                            elif f.endswith(".png") and os.path.basename(root) == "summary_plots":
                                # Check if file matches rolling analysis keywords
                                if lookup_type in rolling_keywords:
                                    for rolling_kw in rolling_keywords[lookup_type]:
                                        if rolling_kw in f:
                                            file_path = os.path.join(root, f)
                                            file_paths.append(file_path)
                                            break
                                    
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
            
            # Check if image has valid data
            if img is None or (hasattr(img, 'size') and img.size == 0):
                ax.text(0.5, 0.5, "Empty image", fontsize=12, ha="center", va="center")
            elif np.all(np.isnan(img)):
                ax.text(0.5, 0.5, "Image contains only NaN values", fontsize=12, ha="center", va="center")
            else:
                # Filter out NaN values for display if there are some valid values
                if np.any(np.isnan(img)):
                    # Create a masked array to handle NaN values
                    img_display = np.ma.masked_invalid(img)
                    ax.imshow(img_display)
                else:
                    ax.imshow(img)
            
            ax.axis("off")
            
            # Simple, clear title with image name
            filename = os.path.basename(image_path)
            
            # Try to extract image name from path or filename
            image_identifier = None
            filepath_parts = image_path.split(os.sep)
            
            # Look for loaded image names in the path
            for loaded_name in self.loaded_image_names:
                if loaded_name in filename or loaded_name in os.sep.join(filepath_parts):
                    image_identifier = loaded_name
                    break
            
            # Create simple, clear title
            if image_identifier and len(self.loaded_image_names) > 1:
                # Only show image identifier when multiple images are loaded
                title = f"{image_identifier}\n{filename}"
                ax.set_title(title, fontsize=9, weight='bold')
            else:
                ax.set_title(filename, fontsize=10)
            
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

    def create_csv_table(self, csv_path):
        """Create a table widget from a CSV file and return it."""
        if not os.path.exists(csv_path):
            return None

        try:
            df = pd.read_csv(csv_path)
            table_widget = QTableWidget()
            table_widget.setRowCount(len(df))
            table_widget.setColumnCount(len(df.columns))
            table_widget.setHorizontalHeaderLabels(df.columns)

            # Fill the table
            for i, row in df.iterrows():
                for j, value in enumerate(row):
                    table_widget.setItem(i, j, QTableWidgetItem(str(value)))

            # Set reasonable size constraints
            table_widget.setMinimumWidth(400)
            table_widget.setMaximumWidth(800)
            table_widget.setMinimumHeight(200)
            table_widget.setMaximumHeight(600)
            
            # Enable sorting
            table_widget.setSortingEnabled(True)
            
            return table_widget

        except Exception as e:
            return None

    def display_csv(self, csv_path):
        """Display the contents of a CSV file in a table widget (legacy method)."""
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
        # Works for both standard (ACF/CCF/Peak) and rolling (Period/Shift/Peak) names
        self.show_individual.setVisible(display_type in [
            "ACF Plots", "CCF Plots", "Peak Properties",  # Standard/Kymograph
            "Period Plots", "Shift Plots"  # Rolling
        ])
        
        # Show/hide statistic selector for ROI Statistics view
        is_roi_stats = display_type == "ROI Statistics"
        self.stat_selector.setVisible(is_roi_stats)
        self.stat_selector_label.setVisible(is_roi_stats)
        
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()
                
    def on_show_individual_changed(self, state):
        """Handle changes in the individual plots checkbox."""
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()  # This will update the display based on current selection
    
    def on_stat_selector_changed(self, index):
        """Handle changes in the statistic selector."""
        if self.results_dir and os.path.exists(self.results_dir):
            self.show_results()
    
    def extract_roi_statistics_from_csvs(self):
        """Extract statistics for each ROI from summary CSV files.
        
        Returns:
            pd.DataFrame: DataFrame with columns ['ROI_ID', 'Image_Name', 'Parameter', 'Statistic', 'Value']
        """
        if not self.results_dir or not os.path.exists(self.results_dir):
            return None
        
        all_roi_data = []
        
        # Find all summary CSV files
        for root, dirs, files in os.walk(self.results_dir):
            for f in files:
                if f.endswith('.csv') and ('summary' in f.lower() or 'ROI_' in root):
                    csv_path = os.path.join(root, f)
                    
                    # Determine ROI ID and Image name from path
                    path_parts = root.split(os.sep)
                    roi_id = None
                    image_name = None
                    
                    # Look for ROI directory
                    for part in path_parts:
                        if part.startswith("ROI_"):
                            roi_id = part
                        elif part not in ['0_signalProcessing', 'Whole_Image'] and not part.startswith('ROI_'):
                            # Try to match with loaded image names
                            for loaded_name in self.loaded_image_names:
                                if loaded_name in part:
                                    image_name = loaded_name
                                    break
                    
                    # If no ROI found, check if it's whole image
                    if not roi_id and 'Whole_Image' in root:
                        roi_id = 'Whole_Image'
                    
                    if not roi_id:
                        continue  # Skip if we can't determine ROI
                    
                    # Read CSV and extract statistics
                    try:
                        df = pd.read_csv(csv_path)
                        
                        # Check if this is a standard summary format
                        if 'Parameter' in df.columns:
                            # Standard format has columns: Parameter, Mean, Median, StdDev, SEM, Bin 0, Bin 1, ...
                            for idx, row in df.iterrows():
                                parameter = row['Parameter']
                                for stat in ['Mean', 'Median', 'StdDev', 'SEM']:
                                    if stat in df.columns:
                                        all_roi_data.append({
                                            'ROI_ID': roi_id,
                                            'Image_Name': image_name,
                                            'Parameter': parameter,
                                            'Statistic': stat,
                                            'Value': row[stat]
                                        })
                        elif 'Submovie' in df.columns:
                            # Rolling analysis format
                            # Extract all columns that contain statistics
                            for idx, row in df.iterrows():
                                for col in df.columns:
                                    if col == 'Submovie':
                                        continue
                                    # Parse column name to extract parameter and statistic
                                    # Format: "Ch X Mean Period", "Ch1-Ch2 Mean Shift", etc.
                                    if any(stat in col for stat in ['Mean', 'Median', 'StdDev', 'Pcnt']):
                                        for stat in ['Mean', 'Median', 'StdDev']:
                                            if stat in col:
                                                parameter = col.replace(f' {stat}', '').strip()
                                                all_roi_data.append({
                                                    'ROI_ID': roi_id,
                                                    'Image_Name': image_name,
                                                    'Parameter': parameter,
                                                    'Statistic': stat,
                                                    'Value': row[col]
                                                })
                                                break
                    except Exception as e:
                        print(f"Error reading CSV {csv_path}: {e}")
                        continue
        
        if not all_roi_data:
            return None
        
        return pd.DataFrame(all_roi_data)
    
    def generate_roi_comparison_plots(self, selected_statistic='Mean'):
        """Generate comparison plots for ROIs across different parameters.
        
        Args:
            selected_statistic (str): The statistic to plot (Mean, Median, StdDev, or SEM)
        
        Returns:
            dict: Dictionary of matplotlib figures, keyed by parameter name
        """
        roi_stats_df = self.extract_roi_statistics_from_csvs()
        
        if roi_stats_df is None or roi_stats_df.empty:
            return {}
        
        # Filter by selected statistic
        filtered_df = roi_stats_df[roi_stats_df['Statistic'] == selected_statistic].copy()
        
        if filtered_df.empty:
            return {}
        
        # Get unique parameters
        parameters = filtered_df['Parameter'].unique()
        
        figures = {}
        
        # Generate a plot for each parameter
        for param in parameters:
            param_df = filtered_df[filtered_df['Parameter'] == param].copy()
            
            # Skip if we don't have enough data
            if len(param_df) < 2:
                continue
            
            try:
                # Import plotting libraries
                import seaborn as sns
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create boxplot
                sns.boxplot(x='ROI_ID', 
                           y='Value', 
                           data=param_df,
                           showfliers=False,
                           ax=ax)
                
                # Create swarmplot on top
                sns.swarmplot(x='ROI_ID', 
                             y='Value', 
                             data=param_df,
                             color=".25",
                             ax=ax)
                
                ax.set_xlabel('ROI', fontsize=12, weight='bold')
                ax.set_ylabel(f'{selected_statistic} Value', fontsize=12, weight='bold')
                ax.set_title(f'{param} - {selected_statistic} Comparison', fontsize=14, weight='bold')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Adjust layout
                plt.tight_layout()
                
                figures[param] = fig
                
            except Exception as e:
                print(f"Error generating plot for {param}: {e}")
                continue
        
        return figures
    
    def show_roi_statistics(self):
        """Display ROI statistics comparison plots."""
        selected_statistic = self.stat_selector.currentText()
        
        # Generate comparison plots
        comparison_plots = self.generate_roi_comparison_plots(selected_statistic)
        
        if not comparison_plots:
            self.status_label.setText("No ROI statistics available for comparison.")
            
            # Display a helpful message
            container = QWidget()
            container_layout = QVBoxLayout()
            
            message = QLabel("No ROI statistics found.\n\nMake sure you have analyzed multiple ROIs.")
            message.setAlignment(Qt.AlignCenter)
            message.setStyleSheet("color: gray; font-size: 12px; padding: 20px;")
            container_layout.addWidget(message)
            container.setLayout(container_layout)
            self.results_layout.addWidget(container)
            return
        
        # Create a grid layout for organized display
        grid_widget = QWidget()
        grid_layout = QGridLayout()
        grid_widget.setLayout(grid_layout)
        self.results_layout.addWidget(grid_widget)
        
        # Display plots in a grid (2 columns)
        row = 0
        col = 0
        max_cols = 2
        
        for param_name, fig in comparison_plots.items():
            # Create container and canvas
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(5, 5, 5, 5)
            container.setLayout(container_layout)
            
            # Create canvas from existing figure
            canvas = FigureCanvas(fig)
            canvas.setMinimumWidth(400)
            canvas.setMinimumHeight(350)
            self.canvases.append(canvas)
            
            container_layout.addWidget(canvas)
            grid_layout.addWidget(container, row, col)
            
            canvas.draw()
            
            # Update grid position
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Update status message
        self.status_label.setText(f"Displaying {len(comparison_plots)} ROI comparison plots for {selected_statistic}.")
        
        # Add a stretch at the end
        self.results_layout.addStretch()
    
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