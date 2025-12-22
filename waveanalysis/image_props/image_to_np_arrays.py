import tifffile
import numpy as np

def tiff_to_np_array_single_frame(file_path: str, roi=None) -> np.ndarray:
    """
    Convert a TIFF file to a NumPy array representing a single frame.

    Args:
        file_path (str): The path to the TIFF file.
        roi (np.ndarray, optional): ROI vertices defining the region to crop. Shape should be (N,2) for N vertices.
            If None, entire image is returned.

    Returns:
        np.ndarray: A NumPy array representing the image data of a single frame.
    """
    image = tifffile.imread(file_path)

    with tifffile.TiffFile(file_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)

    image = image.reshape(num_channels, 
                            image.shape[-2],  # cols
                            image.shape[-1])  # rows
    
    if roi is not None:
        print(f"\n*** ROI Diagnostics (Single Frame) ***")
        print(f"ROI shape: {roi.shape}")
        print(f"ROI coordinates (first 5 points): {roi[:5]}")
        print(f"ROI bounds: Y=[{roi[:, 0].min():.1f}, {roi[:, 0].max():.1f}], X=[{roi[:, 1].min():.1f}, {roi[:, 1].max():.1f}]")
        print(f"Image shape: {image.shape}")
        
        # Create a mask from ROI vertices
        # Note: napari uses (y, x) coordinates, but matplotlib.path.Path expects (x, y)
        # So we need to swap the coordinates
        from matplotlib.path import Path
        mask = np.zeros(image.shape[-2:], dtype=bool)
        
        # Swap from (y, x) to (x, y) for matplotlib Path
        roi_xy = roi[:, [1, 0]] if roi.shape[1] == 2 else roi
        roi_path = Path(roi_xy)
        
        y, x = np.mgrid[:image.shape[-2], :image.shape[-1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = roi_path.contains_points(points).reshape(image.shape[-2:])
        
        print(f"Mask shape: {mask.shape}")
        print(f"Pixels in ROI: {np.sum(mask)} out of {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
        
        # Apply mask to all channels
        masked_image = np.zeros_like(image)
        for c in range(num_channels):
            masked_image[c] = np.where(mask, image[c], 0)
            non_zero = np.count_nonzero(masked_image[c])
            print(f"Channel {c}: {non_zero} non-zero pixels after masking")
        image = masked_image
        print(f"*** End ROI Diagnostics ***\n")
    
    return image

def tiff_to_np_array_multi_frame(file_path: str, roi=None) -> np.ndarray:
    """
    Convert a multi-frame TIFF file to a numpy array.

    Args:
        file_path (str): The path to the TIFF file.
        roi (np.ndarray, optional): ROI vertices defining the region to crop. Shape should be (N,2) for N vertices.
            If None, entire image is returned.

    Returns:
        np.ndarray: The numpy array representing the TIFF file, cropped to ROI if provided.
    """
    # Load the TIFF file into a numpy array
    image = tifffile.imread(file_path)

    with tifffile.TiffFile(file_path) as tif_file:
        metadata = tif_file.imagej_metadata
    num_channels = metadata.get('channels', 1)
    num_frames = metadata.get('frames', 1)
    num_slices = metadata.get('slices', 1)

    # Max project if multiple slices
    if num_slices > 1:
        print('Max projecting image stack')
        image = np.max(image, axis=1)
        num_slices = 1
        
    image = image.reshape(num_frames, 
                        num_slices, 
                        num_channels, 
                        *image.shape[-2:])
    
    if roi is not None:
        print(f"\n*** ROI Diagnostics (Multi Frame) ***")
        print(f"ROI shape: {roi.shape}")
        print(f"ROI coordinates (first 5 points): {roi[:5]}")
        print(f"ROI bounds: Y=[{roi[:, 0].min():.1f}, {roi[:, 0].max():.1f}], X=[{roi[:, 1].min():.1f}, {roi[:, 1].max():.1f}]")
        print(f"Image shape before ROI: {image.shape}")
        
        # Create a mask from ROI vertices
        # Note: napari uses (y, x) coordinates, but matplotlib.path.Path expects (x, y)
        # So we need to swap the coordinates
        from matplotlib.path import Path
        mask = np.zeros(image.shape[-2:], dtype=bool)
        
        # Swap from (y, x) to (x, y) for matplotlib Path
        roi_xy = roi[:, [1, 0]] if roi.shape[1] == 2 else roi
        roi_path = Path(roi_xy)
        
        y, x = np.mgrid[:image.shape[-2], :image.shape[-1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = roi_path.contains_points(points).reshape(image.shape[-2:])
        
        print(f"Mask shape: {mask.shape}")
        print(f"Pixels in ROI: {np.sum(mask)} out of {mask.size} ({100*np.sum(mask)/mask.size:.1f}%)")
        
        # Apply mask to all frames and channels
        masked_image = np.zeros_like(image)
        for f in range(num_frames):
            for c in range(num_channels):
                masked_image[f, 0, c] = np.where(mask, image[f, 0, c], 0)
        
        # Sample diagnostic for first frame
        sample_non_zero = np.count_nonzero(masked_image[0, 0, 0])
        print(f"Frame 0, Channel 0: {sample_non_zero} non-zero pixels after masking")
        image = masked_image
        print(f"*** End ROI Diagnostics ***\n")

    return image