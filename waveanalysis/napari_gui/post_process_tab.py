import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from qtpy.QtWidgets import *
from qtpy.QtCore import Qt

class PostProcessingTab(QWidget):
    """Tab for displaying and exporting analysis results"""
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.results = None
        self.params = None
        
        # Main layout
        layout = QVBoxLayout()
        
        # Results view area with tabs
        self.results_tabs = QTabWidget()
        layout.addWidget(self.results_tabs)
        
        # Controls area
        controls_layout = QHBoxLayout()
        
        # Display options
        display_box = QVBoxLayout()
        display_box.addWidget(QLabel("Display:"))
        self.display_combo = QComboBox()
        self.display_combo.addItems(["Summary", "ACF", "CCF", "Period", "Peak Properties"])
        self.display_combo.currentIndexChanged.connect(self.update_display)
        display_box.addWidget(self.display_combo)
        controls_layout.addLayout(display_box)
        
        # Export button
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self.export_results)
        controls_layout.addWidget(self.export_btn)
        
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
    def show_results(self, results, params):
        """Display analysis results in the tab"""
        self.results = results
        self.params = params
        
        # Clear existing tabs
        while self.results_tabs.count() > 0:
            self.results_tabs.removeTab(0)
        
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
            
            summary = {
                "Analysis Type": params["type"],
                "Period": np.nanmean(result["period"]),
                "Peak Amplitude": peak_amplitude,
            }
            
            # Add wave speed if available
            if "wave_speed" in result:
                summary["Wave Speed"] = np.nanmean(result["wave_speed"])
                
            # Create summary tab
            summary_tab = QWidget()
            summary_layout = QVBoxLayout()
            
            # Create summary table
            table = QTableWidget(len(summary), 2)
            table.setHorizontalHeaderLabels(["Metric", "Value"])
            
            for row, (key, value) in enumerate(summary.items()):
                table.setItem(row, 0, QTableWidgetItem(key))
                # Format numerical values to 2 decimal places
                if isinstance(value, (int, float)) and not np.isnan(value):
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = str(value)
                table.setItem(row, 1, QTableWidgetItem(formatted_value))
            
            table.resizeColumnsToContents()
            summary_layout.addWidget(table)
            summary_tab.setLayout(summary_layout)
            
            # Create visualization tabs
            figure_tab = QWidget()
            figure_layout = QVBoxLayout()
            
            # Create matplotlib figure for plots
            fig = Figure(figsize=(8, 6))
            canvas = FigureCanvas(fig)
            figure_layout.addWidget(canvas)
            
            # Plot ACF as default
            if "acf" in result:
                ax = fig.add_subplot(111)
                
                # Plot average ACF across all bins
                mean_acf = np.nanmean(result["acf"], axis=1)
                for ch_idx in range(mean_acf.shape[0]):
                    ax.plot(mean_acf[ch_idx], label=f"Channel {ch_idx}")
                
                ax.set_title("Average Auto-Correlation Function")
                ax.set_xlabel("Lag")
                ax.set_ylabel("Correlation")
                ax.grid(True)
                ax.legend()
                
                fig.tight_layout()
                canvas.draw()
                
            figure_tab.setLayout(figure_layout)
            
            # Add the tabs
            self.results_tabs.addTab(summary_tab, f"Summary {i+1}")
            self.results_tabs.addTab(figure_tab, f"Visualization {i+1}")
        
        # Check if we have results and enable export button
        self.export_btn.setEnabled(len(results) > 0)
        
    def update_display(self):
        """Change the displayed visualization based on selection"""
        if not self.results:
            return
            
        display_type = self.display_combo.currentText()
        
        # Find visualization tabs and update them
        for i in range(self.results_tabs.count()):
            tab_text = self.results_tabs.tabText(i)
            if "Visualization" in tab_text:
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
                        
                        # Plot based on display type
                        if display_type == "ACF" and "acf" in result:
                            mean_acf = np.nanmean(result["acf"], axis=1)
                            for ch_idx in range(mean_acf.shape[0]):
                                ax.plot(mean_acf[ch_idx], label=f"Channel {ch_idx}")
                            ax.set_title("Average Auto-Correlation Function")
                            ax.set_xlabel("Lag")
                            ax.set_ylabel("Correlation")
                            
                        elif display_type == "CCF" and "ccf" in result:
                            mean_ccf = np.nanmean(result["ccf"], axis=1)
                            for ch_idx in range(mean_ccf.shape[0]):
                                ax.plot(mean_ccf[ch_idx], label=f"Channels {ch_idx//2} & {ch_idx%2}")
                            ax.set_title("Average Cross-Correlation Function")
                            ax.set_xlabel("Lag")
                            ax.set_ylabel("Correlation")
                            
                        elif display_type == "Period" and "period" in result:
                            # Create heatmap of periods
                            im = ax.imshow(result["period"], aspect='auto', cmap='viridis')
                            ax.set_title("Period by Bin and Channel")
                            ax.set_xlabel("Bin")
                            ax.set_ylabel("Channel")
                            fig.colorbar(im, ax=ax, label="Period")
                            
                        elif display_type == "Peak Properties" and "peak_widths" in result:
                            # Create heatmap of peak widths
                            im = ax.imshow(result["peak_widths"], aspect='auto', cmap='viridis')
                            ax.set_title("Peak Widths by Bin and Channel")
                            ax.set_xlabel("Bin")
                            ax.set_ylabel("Channel")
                            fig.colorbar(im, ax=ax, label="Width")
                        
                        # Common settings
                        ax.grid(True)
                        ax.legend()
                        fig.tight_layout()
                        canvas.draw()
    
    def export_results(self):
        """Export analysis results to files"""
        if not self.results or not self.params:
            return
        
        # Ask user for directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", 
            self.parent.log_params.get("Base Directory", "")
        )
        
        if not export_dir:
            return
        
        try:
            # Export each result
            for i, result in enumerate(self.results):
                # Create a base filename
                base_name = f"wave_analysis_result_{i+1}"
                
                # Save summary as CSV
                with open(os.path.join(export_dir, f"{base_name}_summary.csv"), 'w') as f:
                    # Add header row
                    f.write("Metric,Value\n")
                    
                    # Add analysis type
                    f.write(f"Analysis Type,{self.params['type']}\n")
                    
                    # Add period
                    period_mean = np.nanmean(result["period"])
                    f.write(f"Period,{period_mean:.4f}\n")
                    
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
                        peak_amplitude = np.nanmean(all_heights) if all_heights else np.nan
                        f.write(f"Peak Amplitude,{peak_amplitude:.4f}\n")
                    except Exception as e:
                        print(f"Error exporting peak amplitude: {str(e)}")
                        f.write(f"Peak Amplitude,N/A\n")
                    
                    # Add wave speed if available
                    if "wave_speed" in result:
                        wave_speed_mean = np.nanmean(result["wave_speed"])
                        f.write(f"Wave Speed,{wave_speed_mean:.4f}\n")
                
                # Save numerical data as NPZ
                np.savez(
                    os.path.join(export_dir, f"{base_name}_data.npz"),
                    acf=result.get("acf", None),
                    ccf=result.get("ccf", None),
                    period=result.get("period", None),
                    peak_widths=result.get("peak_widths", None),
                    peak_maxs=result.get("peak_maxs", None),
                    peak_mins=result.get("peak_mins", None),
                    peak_offsets=result.get("peak_offsets", None),
                    wave_speed=result.get("wave_speed", None),
                    params=self.params
                )
                
                # Create and save figures
                self._export_figures(result, os.path.join(export_dir, base_name))
            
            QMessageBox.information(self, "Export Complete", 
                                   f"Results exported to {export_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting results: {str(e)}")
    
    def _export_figures(self, result, base_path):
        """Create and save visualization figures"""
        # Figure 1: ACF
        if "acf" in result:
            fig, ax = plt.subplots(figsize=(10, 6))
            mean_acf = np.nanmean(result["acf"], axis=1)
            for ch_idx in range(mean_acf.shape[0]):
                ax.plot(mean_acf[ch_idx], label=f"Channel {ch_idx}")
            ax.set_title("Average Auto-Correlation Function")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{base_path}_acf.png", dpi=300)
            plt.close(fig)
        
        # Figure 2: CCF (if available)
        if "ccf" in result:
            fig, ax = plt.subplots(figsize=(10, 6))
            mean_ccf = np.nanmean(result["ccf"], axis=1)
            for ch_idx in range(mean_ccf.shape[0]):
                ax.plot(mean_ccf[ch_idx], label=f"Channels {ch_idx//2} & {ch_idx%2}")
            ax.set_title("Average Cross-Correlation Function")
            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()
            fig.savefig(f"{base_path}_ccf.png", dpi=300)
            plt.close(fig)
        
        # Figure 3: Period heatmap
        if "period" in result:
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(result["period"], aspect='auto', cmap='viridis')
            ax.set_title("Period by Bin and Channel")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Channel")
            fig.colorbar(im, ax=ax, label="Period")
            fig.tight_layout()
            fig.savefig(f"{base_path}_period.png", dpi=300)
            plt.close(fig)
        
        # Figure 4: Peak widths heatmap
        if "peak_widths" in result:
            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(result["peak_widths"], aspect='auto', cmap='viridis')
            ax.set_title("Peak Widths by Bin and Channel")
            ax.set_xlabel("Bin")
            ax.set_ylabel("Channel")
            fig.colorbar(im, ax=ax, label="Width")
            fig.tight_layout()
            fig.savefig(f"{base_path}_peak_widths.png", dpi=300)
            plt.close(fig)