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
        # Create a mask from ROI vertices
        from matplotlib.path import Path
        mask = np.zeros(image.shape[-2:], dtype=bool)
        roi_path = Path(roi)
        y, x = np.mgrid[:image.shape[-2], :image.shape[-1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = roi_path.contains_points(points).reshape(image.shape[-2:])
        
        # Apply mask to all channels
        masked_image = np.zeros_like(image)
        for c in range(num_channels):
            masked_image[c] = np.where(mask, image[c], 0)
        image = masked_image
    
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
        # Create a mask from ROI vertices
        from matplotlib.path import Path
        mask = np.zeros(image.shape[-2:], dtype=bool)
        roi_path = Path(roi)
        y, x = np.mgrid[:image.shape[-2], :image.shape[-1]]
        points = np.vstack((x.ravel(), y.ravel())).T
        mask = roi_path.contains_points(points).reshape(image.shape[-2:])
        
        # Apply mask to all frames and channels
        masked_image = np.zeros_like(image)
        for f in range(num_frames):
            for c in range(num_channels):
                masked_image[f, 0, c] = np.where(mask, image[f, 0, c], 0)
        image = masked_image

    return image