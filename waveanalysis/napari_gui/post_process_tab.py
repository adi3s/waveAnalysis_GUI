import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt, Signal

class PostProcessingTab(QWidget):
    """Tab for displaying and exporting analysis results"""
    # Add signal for status updates
    status_updated = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.results = None
        self.params = None
        self.canvases = []  # Keep track of all canvases
        
        # Main layout
        layout = QVBoxLayout()
        
        # Add status bar at the top
        self.status_label = QLabel("No results loaded")
        layout.addWidget(self.status_label)
        
        # Results view area with tabs
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Controls area in a group box for better visibility
        controls_group = QGroupBox("Display Controls")
        controls_layout = QHBoxLayout()
        
        # Display options
        display_box = QVBoxLayout()
        display_label = QLabel("Display Type:")
        display_label.setStyleSheet("font-weight: bold;")
        display_box.addWidget(display_label)
        
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Summary", "ACF", "CCF", "Period", "Peak Properties"])
        self.display_combo.currentIndexChanged.connect(self.update_display)
        display_box.addWidget(self.display_combo)
        controls_layout.addLayout(display_box)
        
        # Figure options
        figure_box = QVBoxLayout()
        figure_label = QLabel("Figure Options:")
        figure_label.setStyleSheet("font-weight: bold;")
        figure_box.addWidget(figure_label)
        
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.update_display)
        figure_box.addWidget(self.grid_checkbox)
        
        controls_layout.addLayout(figure_box)
        
        # Export options
        export_box = QVBoxLayout()
        export_label = QLabel("Export:")
        export_label.setStyleSheet("font-weight: bold;")
        export_box.addWidget(export_label)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_results)
        export_box.addWidget(self.export_btn)
        
        controls_layout.addLayout(export_box)
        
        # Stretch to push controls to the left
        controls_layout.addStretch(1)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.setLayout(layout)
        
    def show_results(self, results, params):
        """Display analysis results in the tab"""
        if not results or len(results) == 0:
            self.status_label.setText("No valid results to display")
            return
            
        self.results = results
        self.params = params
        self.canvases = []  # Clear canvas references
        
        # Clear existing tabs
        while self.results_tabs.count() > 0:
            self.results_tabs.removeTab(0)
        
        # Update status
        self.status_label.setText(f"Displaying {len(results)} result set(s)")
        
        # Process each result
        for i, result in enumerate(results):
            # Create a summary tab
            try:
                # Extract all height values and flatten them
                all_heights = []
                if 'peak_props' in result and len(result['peak_props']) > 0:
                    for p in result['peak_props'][-1].values():
                        if 'heights' in p and p['heights'] is not None:
                            # Check if heights is a list/array and extend
                            if isinstance(p['heights'], (list, np.ndarray)):
                                all_heights.extend(p['heights'])
                            else:
                                all_heights.append(p['heights'])
                
                # Calculate mean if we have heights
                peak_amplitude = np.nanmean(all_heights) if all_heights else np.nan
            except Exception as e:
                print(f"Error calculating peak amplitude: {str(e)}")
                peak_amplitude = np.nan
            
            # Build summary dictionary with more robust error handling
            summary = {
                "Analysis Type": params.get("type", "Unknown"),
                "Period": np.nanmean(result.get("period", [np.nan])) if isinstance(result.get("period", []), (list, np.ndarray)) else np.nan,
                "Peak Amplitude": peak_amplitude,
            }
            
            # Add wave speed if available
            if "wave_speed" in result:
                summary["Wave Speed"] = np.nanmean(result["wave_speed"]) if isinstance(result["wave_speed"], (list, np.ndarray)) else result["wave_speed"]
                
            # Create summary tab with improved styling
            summary_tab = QWidget()
            summary_layout = QVBoxLayout()
            
            # Add a title for the summary
            title_label = QLabel(f"Analysis Result {i+1}")
            title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
            summary_layout.addWidget(title_label)
            
            # Create summary table with better visibility
            table = QTableWidget(len(summary), 2)
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            table.setAlternatingRowColors(True)
            
            for row, (key, value) in enumerate(summary.items()):
                metric_item = QTableWidgetItem(key)
                metric_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                table.setItem(row, 0, metric_item)
                
                # Format numerical values to 3 decimal places for better readability
                if isinstance(value, (int, float)) and not np.isnan(value):
                    formatted_value = f"{value:.3f}"
                elif np.isnan(value):
                    formatted_value = "N/A"
                else:
                    formatted_value = str(value)
                    
                value_item = QTableWidgetItem(formatted_value)
                value_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(row, 1, value_item)
            
            # Make sure the table is visible and sized properly
            table.setMinimumHeight(len(summary) * 30 + 50)  # Allow space for headers and rows
            summary_layout.addWidget(table)
            
            # Add notes about data quality if needed
            if np.isnan(summary["Period"]):
                note_label = QLabel("Note: Some data values could not be calculated")
                note_label.setStyleSheet("color: red;")
                summary_layout.addWidget(note_label)
                
            summary_layout.addStretch(1)  # Add stretch to keep everything at the top
            summary_tab.setLayout(summary_layout)
            
            # Create visualization tabs with improved visibility
            figure_tab = QWidget()
            figure_layout = QVBoxLayout()
            
            # Create matplotlib figure for plots with better sizing
            fig = Figure(figsize=(10, 8), dpi=100)
            canvas = FigureCanvas(fig)
            figure_layout.addWidget(canvas)
            self.canvases.append(canvas)  # Keep reference to canvas
            
            # Plot ACF as default with better visibility
            if "acf" in result and result["acf"] is not None:
                ax = fig.add_subplot(111)
                
                # Check data shape and handle appropriately
                mean_acf = np.array(result["acf"])
                if mean_acf.ndim > 2:  # If we have channels, bins, etc.
                    mean_acf = np.nanmean(mean_acf, axis=1)
                
                # Ensure we have valid data to plot
                if mean_acf.size > 0:
                    # Plot each channel if multiple exist
                    for ch_idx in range(mean_acf.shape[0] if mean_acf.ndim > 1 else 1):
                        if mean_acf.ndim > 1:
                            y_data = mean_acf[ch_idx]
                        else:
                            y_data = mean_acf
                            
                        # Check for NaN or infinite values
                        if np.any(np.isfinite(y_data)):
                            ax.plot(y_data, linewidth=2, label=f"Channel {ch_idx}")
                
                ax.set_title("Average Auto-Correlation Function", fontsize=14)
                ax.set_xlabel("Lag", fontsize=12)
                ax.set_ylabel("Correlation", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                ax.grid(True)
                ax.legend(fontsize=10)
                
                fig.tight_layout()
                canvas.draw()
                
            figure_tab.setLayout(figure_layout)
            
            # Add the tabs with clear titles
            self.results_tabs.addTab(summary_tab, f"Summary {i+1}")
            self.results_tabs.addTab(figure_tab, f"Plots {i+1}")
        
        # Enable export button if we have results
        self.export_btn.setEnabled(len(results) > 0)
        
    def update_display(self):
        """Change the displayed visualization based on selection with improved visibility"""
        if not self.results:
            return
            
        display_type = self.display_combo.currentText()
        show_grid = self.grid_checkbox.isChecked()
        
        # Find visualization tabs and update them
        for i in range(self.results_tabs.count()):
            tab_text = self.results_tabs.tabText(i)
            if "Plots" in tab_text:
                idx = int(tab_text.split()[-1]) - 1
                if idx < len(self.results):
                    result = self.results[idx]
                    
                    # Get the tab widget
                    tab = self.results_tabs.widget(i)
                    canvas = None
                    
                    # Find the canvas in the tab
                    for child in tab.children():
                        if isinstance(child, FigureCanvas):
                            canvas = child
                            break
                    
                    if canvas:
                        # Clear the figure
                        fig = canvas.figure
                        fig.clear()
                        ax = fig.add_subplot(111)
                        
                        # Plot based on display type with improved visibility
                        if display_type == "ACF" and "acf" in result and result["acf"] is not None:
                            # Handle different array shapes
                            acf_data = np.array(result["acf"])
                            if acf_data.ndim > 2:
                                mean_acf = np.nanmean(acf_data, axis=1)
                            else:
                                mean_acf = acf_data
                                
                            # Plot each channel
                            if mean_acf.size > 0:
                                for ch_idx in range(mean_acf.shape[0] if mean_acf.ndim > 1 else 1):
                                    if mean_acf.ndim > 1:
                                        y_data = mean_acf[ch_idx]
                                    else:
                                        y_data = mean_acf
                                        
                                    # Check for valid data
                                    if np.any(np.isfinite(y_data)):
                                        ax.plot(y_data, linewidth=2, label=f"Channel {ch_idx}")
                                        
                            ax.set_title("Average Auto-Correlation Function", fontsize=14)
                            ax.set_xlabel("Lag", fontsize=12)
                            ax.set_ylabel("Correlation", fontsize=12)
                            
                        elif display_type == "CCF" and "ccf" in result and result["ccf"] is not None:
                            ccf_data = np.array(result["ccf"])
                            if ccf_data.ndim > 2:
                                mean_ccf = np.nanmean(ccf_data, axis=1)
                            else:
                                mean_ccf = ccf_data
                                
                            if mean_ccf.size > 0:
                                for ch_idx in range(mean_ccf.shape[0] if mean_ccf.ndim > 1 else 1):
                                    if mean_ccf.ndim > 1:
                                        data = mean_ccf[ch_idx]
                                    else:
                                        data = mean_ccf
                                        
                                    if np.any(np.isfinite(data)):
                                        ch1, ch2 = ch_idx//2, ch_idx%2
                                        ax.plot(data, linewidth=2, label=f"Channels {ch1} & {ch2}")
                                        
                            ax.set_title("Average Cross-Correlation Function", fontsize=14)
                            ax.set_xlabel("Lag", fontsize=12)
                            ax.set_ylabel("Correlation", fontsize=12)
                            
                        elif display_type == "Period" and "period" in result and result["period"] is not None:
                            # Create heatmap of periods with better visibility
                            period_data = np.array(result["period"])
                            if np.any(np.isfinite(period_data)):
                                im = ax.imshow(period_data, aspect='auto', cmap='viridis')
                                cbar = fig.colorbar(im, ax=ax, label="Period")
                                cbar.ax.tick_params(labelsize=10)
                                
                                # Add text values on the heatmap for better visibility
                                if period_data.size < 50:  # Only add text for smaller arrays
                                    for i in range(period_data.shape[0]):
                                        for j in range(period_data.shape[1]):
                                            if np.isfinite(period_data[i, j]):
                                                ax.text(j, i, f"{period_data[i, j]:.1f}", 
                                                       ha="center", va="center", 
                                                       color="white" if period_data[i, j] > np.nanmean(period_data) else "black")
                                
                            ax.set_title("Period by Bin and Channel", fontsize=14)
                            ax.set_xlabel("Bin", fontsize=12)
                            ax.set_ylabel("Channel", fontsize=12)
                            
                        elif display_type == "Peak Properties" and "peak_widths" in result and result["peak_widths"] is not None:
                            # Create heatmap of peak widths with better visibility
                            width_data = np.array(result["peak_widths"])
                            if np.any(np.isfinite(width_data)):
                                im = ax.imshow(width_data, aspect='auto', cmap='plasma')
                                cbar = fig.colorbar(im, ax=ax, label="Width")
                                cbar.ax.tick_params(labelsize=10)
                                
                                # Add text values for better visibility
                                if width_data.size < 50:  # Only add text for smaller arrays
                                    for i in range(width_data.shape[0]):
                                        for j in range(width_data.shape[1]):
                                            if np.isfinite(width_data[i, j]):
                                                ax.text(j, i, f"{width_data[i, j]:.1f}", 
                                                       ha="center", va="center", 
                                                       color="white" if width_data[i, j] > np.nanmean(width_data) else "black")
                                
                            ax.set_title("Peak Widths by Bin and Channel", fontsize=14)
                            ax.set_xlabel("Bin", fontsize=12)
                            ax.set_ylabel("Channel", fontsize=12)
                        
                        elif display_type == "Summary":
                            # Create a comprehensive summary plot
                            # Split into 2x2 grid
                            fig.clear()
                            ax1 = fig.add_subplot(221)
                            ax2 = fig.add_subplot(222)
                            ax3 = fig.add_subplot(223)
                            ax4 = fig.add_subplot(224)
                            
                            # Plot ACF in first panel
                            if "acf" in result and result["acf"] is not None:
                                acf_data = np.array(result["acf"])
                                if acf_data.ndim > 2:
                                    mean_acf = np.nanmean(acf_data, axis=1)
                                else:
                                    mean_acf = acf_data
                                    
                                if mean_acf.size > 0 and np.any(np.isfinite(mean_acf)):
                                    if mean_acf.ndim > 1:
                                        ax1.plot(mean_acf[0], linewidth=2)
                                    else:
                                        ax1.plot(mean_acf, linewidth=2)
                                        
                                ax1.set_title("ACF", fontsize=10)
                                ax1.grid(show_grid)
                            
                            # Plot CCF in second panel
                            if "ccf" in result and result["ccf"] is not None:
                                ccf_data = np.array(result["ccf"])
                                if ccf_data.ndim > 2:
                                    mean_ccf = np.nanmean(ccf_data, axis=1)
                                else:
                                    mean_ccf = ccf_data
                                    
                                if mean_ccf.size > 0 and np.any(np.isfinite(mean_ccf)):
                                    if mean_ccf.ndim > 1:
                                        ax2.plot(mean_ccf[0], linewidth=2)
                                    else:
                                        ax2.plot(mean_ccf, linewidth=2)
                                        
                                ax2.set_title("CCF", fontsize=10)
                                ax2.grid(show_grid)
                            
                            # Plot Period in third panel
                            if "period" in result and result["period"] is not None:
                                period_data = np.array(result["period"])
                                if np.any(np.isfinite(period_data)):
                                    im = ax3.imshow(period_data, aspect='auto', cmap='viridis')
                                    fig.colorbar(im, ax=ax3, label="Period", fraction=0.046, pad=0.04)
                                ax3.set_title("Period", fontsize=10)
                                
                            # Plot Wave Speed or Peak info in fourth panel
                            if "wave_speed" in result and result["wave_speed"] is not None:
                                ws_data = np.array(result["wave_speed"])
                                if np.any(np.isfinite(ws_data)):
                                    im = ax4.imshow(ws_data, aspect='auto', cmap='coolwarm')
                                    fig.colorbar(im, ax=ax4, label="Speed", fraction=0.046, pad=0.04)
                                ax4.set_title("Wave Speed", fontsize=10)
                            elif "peak_widths" in result and result["peak_widths"] is not None:
                                width_data = np.array(result["peak_widths"])
                                if np.any(np.isfinite(width_data)):
                                    im = ax4.imshow(width_data, aspect='auto', cmap='plasma')
                                    fig.colorbar(im, ax=ax4, label="Width", fraction=0.046, pad=0.04)
                                ax4.set_title("Peak Widths", fontsize=10)
                            
                        # Common settings
                        if display_type != "Summary":  # Not needed for summary which has multiple axes
                            ax.tick_params(axis='both', which='major', labelsize=10)
                            ax.grid(show_grid)
                            if ax.has_data():
                                ax.legend(fontsize=10, loc='best')
                        
                        fig.tight_layout()
                        canvas.draw()
    
    def export_results(self):
        """Export analysis results to files with improved error handling"""
        if not self.results or not self.params:
            QMessageBox.warning(self, "Export Warning", "No results available to export")
            return
        
        # Ask user for directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", 
            self.parent.log_params.get("Base Directory", os.path.expanduser("~"))
        )
        
        if not export_dir:
            return
        
        try:
            # Create progress dialog
            progress = QProgressDialog("Exporting results...", "Cancel", 0, len(self.results) * 5)
            progress.setWindowTitle("Export Progress")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            progress_value = 0
            
            # Export each result
            for i, result in enumerate(self.results):
                # Create a more descriptive base filename
                analysis_type = self.params.get("type", "unknown")
                timestamp = result.get("timestamp", "")
                if timestamp:
                    base_name = f"wave_analysis_{analysis_type}_{timestamp}_{i+1}"
                else:
                    import datetime
                    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"wave_analysis_{analysis_type}_{now}_{i+1}"
                
                # Save summary as CSV
                with open(os.path.join(export_dir, f"{base_name}_summary.csv"), 'w') as f:
                    # Add header row
                    f.write("Metric,Value\n")
                    
                    # Add metadata
                    f.write(f"Analysis Type,{self.params.get('type', 'Unknown')}\n")
                    f.write(f"Export Date,{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    # Add period with robust error handling
                    try:
                        period_data = result.get("period", None)
                        if period_data is not None and np.any(np.isfinite(period_data)):
                            period_mean = np.nanmean(period_data)
                            f.write(f"Period,{period_mean:.4f}\n")
                        else:
                            f.write("Period,N/A\n")
                    except Exception as e:
                        print(f"Error exporting period: {str(e)}")
                        f.write("Period,N/A\n")
                    
                    # Add peak amplitude
                    try:
                        all_heights = []
                        if 'peak_props' in result and len(result['peak_props']) > 0:
                            for p in result['peak_props'][-1].values():
                                if 'heights' in p and p['heights'] is not None:
                                    if isinstance(p['heights'], (list, np.ndarray)):
                                        all_heights.extend(p['heights'])
                                    else:
                                        all_heights.append(p['heights'])
                        
                        if all_heights and np.any(np.isfinite(all_heights)):
                            peak_amplitude = np.nanmean(all_heights)
                            f.write(f"Peak Amplitude,{peak_amplitude:.4f}\n")
                        else:
                            f.write("Peak Amplitude,N/A\n")
                    except Exception as e:
                        print(f"Error exporting peak amplitude: {str(e)}")
                        f.write("Peak Amplitude,N/A\n")
                    
                    # Add wave speed if available
                    try:
                        if "wave_speed" in result and result["wave_speed"] is not None:
                            ws_data = np.array(result["wave_speed"])
                            if np.any(np.isfinite(ws_data)):
                                wave_speed_mean = np.nanmean(ws_data)
                                f.write(f"Wave Speed,{wave_speed_mean:.4f}\n")
                            else:
                                f.write("Wave Speed,N/A\n")
                    except Exception as e:
                        print(f"Error exporting wave speed: {str(e)}")
                        f.write("Wave Speed,N/A\n")
                
                progress_value += 1
                progress.setValue(progress_value)
                if progress.wasCanceled():
                    break
                
                # Save numerical data as NPZ with error handling
                try:
                    # Filter out None values and non-serializable objects
                    save_dict = {}
                    for key, value in result.items():
                        if value is not None and isinstance(value, (np.ndarray, list, dict, int, float, str)):
                            save_dict[key] = value
                    
                    # Add parameters to save dict
                    save_dict['params'] = {k: v for k, v in self.params.items() 
                                         if v is not None and isinstance(v, (np.ndarray, list, dict, int, float, str))}
                    
                    np.savez(os.path.join(export_dir, f"{base_name}_data.npz"), **save_dict)
                except Exception as e:
                    print(f"Error saving NPZ: {str(e)}")
                    QMessageBox.warning(self, "Export Warning", 
                                     f"Could not save numerical data for result {i+1}: {str(e)}")
                
                progress_value += 1
                progress.setValue(progress_value)
                if progress.wasCanceled():
                    break
                
                # Create and save figures with error handling
                self._export_figures(result, os.path.join(export_dir, base_name), progress)
                
                if progress.wasCanceled():
                    break
            
            progress.close()
            QMessageBox.information(self, "Export Complete", 
                                   f"Results exported to {export_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")