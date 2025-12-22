import warnings
import numpy as np
import scipy.signal as sig

warnings.filterwarnings("ignore") # Ignore warnings

def calc_indv_peak_props_workflow(
    bin_values:np.ndarray,
    img_props:dict
) -> tuple:
    '''
    Calculate individual peak properties for each channel and bin.

    Args:
        bin_values (np.ndarray): The input array of bin values.
        img_props (dict): A dictionary containing image properties.

    Returns:
        tuple: A tuple containing the calculated individual peak properties, including:
            - indv_peak_widths (np.ndarray): Array of mean peak widths for each channel and bin.
            - indv_peak_maxs (np.ndarray): Array of mean peak maximums for each channel and bin.
            - indv_peak_mins (np.ndarray): Array of mean peak minimums for each channel and bin.
            - indv_peak_offsets (np.ndarray): Array of mean peak offsets for each channel and bin.
            - indv_peak_props (dict): A dictionary containing individual peak properties for each channel and bin.
    '''
    # Extract image properties from the dictionary
    num_channels = img_props['num_channels']
    num_bins = img_props['num_bins']
    analysis_type = img_props['analysis_type']

    # Initialize arrays to store the individual peak properties
    indv_peak_widths = np.zeros(shape=(num_channels, num_bins))
    indv_peak_maxs = np.zeros(shape=(num_channels, num_bins))
    indv_peak_mins = np.zeros(shape=(num_channels, num_bins))
    indv_peak_offsets = np.zeros(shape=(num_channels, num_bins))
    indv_peak_props = {}

    # Loop through each channel and bin
    for channel in range(num_channels):
        for bin in range(num_bins):
            # Extract the bin values for the current channel and bin
            signal = bin_values[:, channel, bin] if analysis_type == 'standard' else bin_values[channel, bin]
            
            # Diagnostic: Check signal properties (only for first few bins to avoid spam)
            if bin < 3:
                print(f"\n*** Bin Diagnostics: Channel {channel}, Bin {bin} ***")
                print(f"Signal length: {len(signal)}")
                print(f"Signal range: [{np.min(signal):.2f}, {np.max(signal):.2f}]")
                print(f"Signal variance: {np.var(signal):.4f}")
                print(f"Non-zero values: {np.count_nonzero(signal)}/{len(signal)}")
                print(f"First 10 values: {signal[:10]}")
            
            # Try to apply Savitzky-Golay filter, but handle SVD errors gracefully
            try:
                signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)
            except np.linalg.LinAlgError as e:
                # If SVD doesn't converge (e.g., ROI too small or insufficient variance), skip filtering
                if bin < 3:  # Only print detailed warning for first few bins
                    print(f"Warning: Could not apply Savitzky-Golay filter for channel {channel}, bin {bin}: {str(e)}")
                    print(f"Processing will continue without filtering for this bin.")
                    print(f"*** End Bin Diagnostics ***\n")
                
            peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

            # If peaks detected, calculate properties, otherwise return NaNs
            if len(peaks) > 0:
                # Calculate the peak properties
                widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
                proms, _, _ = sig.peak_prominences(signal, peaks)

                # Calculate the mean of the peak widths, maximums, and minimums
                mean_width = np.mean(widths, axis=0)
                mean_max = np.mean(signal[peaks], axis = 0)
                mean_min = np.mean(signal[peaks]-proms, axis = 0)

                # calculate the left and right bases of the peaks, then midpoints and peak offsets
                _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=.99)
                midpoints = (leftIndex + rightIndex) / 2
                peak_offsets = peaks - midpoints

                # Check if one peak entirely encompasses another
                for i in range(len(peaks)):
                    for j in range(len(peaks)):
                        if i != j:  # Avoid self-comparison
                            if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                                # Peak j is entirely encompassed by peak i
                                left_bases[i] = np.nan
                                right_bases[i] = np.nan
                                peak_offsets[i] = np.nan
                                midpoints[i] = np.nan
                
                # Drop NaN values because it will mess up the mean calculation
                valid_indices = ~np.isnan(peak_offsets)
                valid_offsets = peak_offsets[valid_indices]

                # Calculate the mean of valid peak offsets
                mean_offset = np.nanmean(valid_offsets)
            else:
                # If no peaks detected, return NaNs
                mean_width = np.nan
                mean_max = np.nan
                mean_min = np.nan
                mean_offset = np.nan
                peaks = np.nan
                proms = np.nan 
                heights = np.nan
                leftIndex = np.nan
                rightIndex = np.nan
                midpoints = np.nan
                peak_offsets = np.nan
                left_bases = np.nan
                right_bases = np.nan

            # Store the mean peak properties in the arrays
            indv_peak_widths[channel, bin] = mean_width
            indv_peak_maxs[channel, bin] = mean_max
            indv_peak_mins[channel, bin] = mean_min
            indv_peak_offsets[channel, bin] = mean_offset

            # Store the individual peak properties in the dictionary
            indv_peak_props[f'Ch {channel} Bin {bin}'] = {'smoothed': signal, 
                                                                'peaks': peaks,
                                                                'proms': proms, 
                                                                'heights': heights, 
                                                                'leftIndex': leftIndex, 
                                                                'rightIndex': rightIndex,
                                                                'midpoints': midpoints,
                                                                'peak_offsets': peak_offsets,
                                                                'left_base': left_bases,
                                                                'right_base': right_bases}
                        
                        # TODO: rename the keys to be more descriptive
    
    return indv_peak_widths, indv_peak_maxs, indv_peak_mins, indv_peak_offsets, indv_peak_props

def calc_indv_peak_props_rolling(signal: np.ndarray) -> tuple:
    '''
    Calculate the individual peak properties of a signal using rolling window.

    Parameters:
        signal (np.ndarray): The input signal.

    Returns:
        tuple: A tuple containing the mean width, mean maximum, mean minimum, and mean offset of the peaks. If no peaks are detected, NaN values are returned.
    '''
    # Calculate the peak properties
    try:
        signal = sig.savgol_filter(signal, window_length = 11, polyorder = 2)
    except np.linalg.LinAlgError as e:
        # If SVD doesn't converge, skip filtering
        print(f"Warning: Could not apply Savitzky-Golay filter in calc_indv_peak_props: {str(e)}")
    peaks, _ = sig.find_peaks(signal, prominence=(np.max(signal)-np.min(signal))*0.1)

    # If peaks detected, calculate properties, otherwise return NaNs
    if len(peaks) > 0:
        # Calculate the peak properties
        widths, heights, leftIndex, rightIndex = sig.peak_widths(signal, peaks, rel_height=0.5)
        proms, _, _ = sig.peak_prominences(signal, peaks)
        # Calculate the mean of the peak widths, maximums, and minimums
        mean_width = np.mean(widths, axis=0)
        mean_max = np.mean(signal[peaks], axis = 0)
        mean_min = np.mean(signal[peaks]-proms, axis = 0)

        # calculate the left and right bases of the peaks, then midpoints and peak offsets
        _, _, left_bases, right_bases = sig.peak_widths(signal, peaks, rel_height=.99)
        midpoints = (leftIndex + rightIndex) / 2
        peak_offsets = peaks - midpoints
        # Check if one peak entirely encompasses another
        for i in range(len(peaks)):
            for j in range(len(peaks)):
                if i != j:  # Avoid self-comparison
                    if left_bases[j] >= left_bases[i] and right_bases[j] <= right_bases[i]:
                        # Peak j is entirely encompassed by peak i
                        left_bases[i] = np.nan
                        right_bases[i] = np.nan
                        peak_offsets[i] = np.nan
                        midpoints[i] = np.nan
        
        # Drop NaN values because it will mess up the mean calculation
        valid_indices = ~np.isnan(peak_offsets)
        valid_offsets = peak_offsets[valid_indices]
        # Calculate the mean of valid peak offsets
        mean_offset = np.nanmean(valid_offsets)
    else:
        # If no peaks detected, return NaNs
        mean_width = np.nan
        mean_max = np.nan
        mean_min = np.nan
        mean_offset = np.nan

    return mean_width, mean_max, mean_min, mean_offset
