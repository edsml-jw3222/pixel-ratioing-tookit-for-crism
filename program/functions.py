import numpy as np

from spectral.io import envi
from scipy.signal import medfilt

# all bands of TRDR data
num_bands = 438

def crism_to_mat(hdr_path):
    """
    Convert a CRISM image file into a dictionary that resembles a MATLAB .mat file.

    Parameters:
    hdr_path (str): Path to the hdr file of CRISM image file to be converted.

    Returns:
    mdict (dict): A dictionary with keys 'IF', 'x', and 'y'. 'IF' corresponds to the 
    image data reshaped into a 2D array, 'x' and 'y' correspond to the pixel coordinates 
    of the image data.
    """
   
    # Open the image file with ENVI
    img = envi.open(hdr_path)

    # Load the image data into a numpy array
    arr = img.load()

    # Initialize the dictionary with the image data under the key 'IF'
    mdict = {'IF': arr}
   
    # Create a meshgrid of the x and y coordinates
    xx_, yy_ = np.meshgrid(np.arange(arr.shape[1]),
                           np.arange(arr.shape[0]))

    # Update the dictionary with the x and y coordinates, incremented by 1 to align with MATLAB's 1-indexing
    mdict.update({'x': xx_.ravel() + 1, 'y': yy_.ravel() + 1})

    # Reshape the image data and store it under the key 'IF' in the dictionary
    mdict['IF'] = mdict['IF'].reshape((-1, num_bands))

    return mdict


def filter_bad_pixels(mdict_spec):
    """
    Filters out 'bad' pixels from an image array.

    Parameters:
    mdict_spec (np.array): The original spectra array.

    Returns:
    mdict_spec (np.array): The spectra array after bad pixels have been filtered and replaced.

    'Bad' pixels are defined as those that have values greater than 1e3 or are non-finite 
    (i.e., they are either not a number (NaN), positive infinity, or negative infinity). 
    These pixels are replaced with the mean value of the remaining 'good' pixels.
    """

    # Create a copy of the input array to prevent modifying the original data
    mdict_spec = mdict_spec.copy()

    # Identify 'bad' pixels
    bad = (mdict_spec < -1) | (mdict_spec > 2) | ~np.isfinite(mdict_spec)

    # If there are any bad pixels, replace them with the mean of the 'good' pixels
    if np.any(bad):
        mdict_spec[bad] = np.mean(mdict_spec[~bad])

    return mdict_spec


def remove_spikes_column(mdict, image, window_size):
    """
    Removes 'spikes' in the data using a column-wise median filter.

    Parameters:
    mdict (dict): The dictionary containing the image data in 'mat' format.
    image (np.array): The original image array to be processed.
    window_size (int): The size of the window to be used in the median filter.

    Returns:
    result (np.array): The image array after spikes have been removed.

    'Spikes' are defined as values that deviate from the median by more than 5 standard deviations. 
    These spikes are replaced with the median value within a window of specified size.
    """

    # Get the dimensions of the image
    height = np.max(mdict['y'])
    width = np.max(mdict['x'])

    # Reshape the image to the original dimensions
    image = image.reshape(height, width, num_bands)
    
    # Initialize an empty array of the same shape as the input image to store the processed data
    result = np.empty((height, width, num_bands))

    # Operate on each column and each channel
    for col in range(width):
        for ch in range(num_bands):
            # Extract the current channel of the current column
            column_channel = image[:, col, ch]
            
            # Apply the median filter to the current channel of the current column
            filtered_column_channel = medfilt(column_channel, kernel_size=window_size)

            # Compute the difference between the median-filtered data and the original data
            diff = np.abs(column_channel - filtered_column_channel)
            
            # Compute the standard deviation of the difference
            std_diff = np.std(diff)
            
            # Identify spikes as values where the difference is more than 5 standard deviations
            spikes = diff > 5 * std_diff
            
            # Replace the spikes with the median-filtered data
            column_channel[spikes] = filtered_column_channel[spikes]

            # Put the processed column into the result array
            result[:, col, ch] = column_channel
            
    return result.reshape(-1, num_bands)


def line_pixel(mat, if_, pix_coor, model, scaler, window_size, num_bland):
    
    # unratioed
    spectra_index = np.where((mat['x'] == pix_coor[0]) & (mat['y'] == pix_coor[1]))[0]
    unratioed = if_[spectra_index][0]
    
    # ratioed_median
    line_pixels = if_[np.where(mat['x'] == pix_coor[0])[0]]
    median_spectra = np.median(line_pixels, axis=0)
    ratioed_median = np.divide(unratioed, median_spectra, where=median_spectra!=0, out=np.full_like(unratioed, np.nan))
    
    # ratioed_bland
    x_indices = np.where(mat['x'] == pix_coor[0])[0]
    y_values = mat['y'][x_indices]
    central_index = np.where(y_values == pix_coor[1])[0][0]
    # determine the range of window
    start_index = max(0, central_index - window_size)
    end_index = min(len(y_values) - 1, central_index + window_size)
    # get the indices of the pixels inside the window
    indices_above = x_indices[start_index:central_index]
    indices_below = x_indices[central_index+1:end_index+1]
    indices = np.concatenate((indices_above, indices_below))
    window = if_[indices]
    # scale
    window = scaler.transform(window)
    # predict
    window = model.predict_proba(window)
    # select bland pixels based on the probablity
    highest_proba_indices = np.argsort(window[:, 1])[-num_bland:]
    bland_indices = indices[highest_proba_indices]
    ave_bland = np.mean(if_[bland_indices], axis=0)
    ratioed_bland = np.divide(unratioed, ave_bland, where=ave_bland!=0, out=np.full_like(unratioed, np.nan))
    
    return unratioed, ratioed_bland, ratioed_median


def line_ROI(mat, if_, pix_coor, model, scaler, ROI, window_size, num_bland):

    # unratioed
    x_min = max(0, pix_coor[0] - ROI // 2)
    x_max = min(mat['x'].max(), pix_coor[0] + ROI // 2)
    y_min = max(0, pix_coor[1] - ROI // 2)
    y_max = min(mat['y'].max(), pix_coor[1] + ROI // 2)
    region_indices = np.where((mat['x'] >= x_min) & (mat['x'] <= x_max) & (mat['y'] >= y_min) & (mat['y'] <= y_max))[0]
    region_spectra = if_[region_indices]
    unratioed = region_spectra.mean(axis=0)

   # ratioed_median
    ratioed_spectra_median = np.empty((0, num_bands))
    for i in range(x_min, x_max+1):
        line_indices = np.where((mat['x'] == i))[0]
        line_pixels = if_[line_indices]
        median_spectrum = np.median(line_pixels, axis=0)
        region_indices = np.where((mat['x'] == i) & (mat['y'] >= y_min) & (mat['y'] <= y_max))[0]
        region_spectra = if_[region_indices].sum(axis=0)
        ratioed_region_spectra = \
        np.divide(region_spectra, median_spectrum, where=median_spectrum!=0, out=np.full_like(region_spectra, np.nan))
        ratioed_spectra_median = np.vstack((ratioed_spectra_median, ratioed_region_spectra))
    ratioed_median = np.mean(ratioed_spectra_median, axis=0)
    
    # ratioed_bland
    ratioed_spectra_bland = np.empty((0, num_bands))
    for i in range(x_min, x_max+1):
        for j in range(y_min, y_max+1):
            window_y_min = max(0, j - window_size)
            window_y_max = min(mat['y'].max(), j + window_size)
            window_indices = np.where((mat['x']==i) & (mat['y']>=window_y_min) & (mat['y']<=window_y_max) & ((mat['y']<y_min) | (mat['y']>y_max)))[0]
            window = if_[window_indices]
            window = scaler.transform(window)
            window = model.predict_proba(window)
            highest_proba_indices = np.argsort(window[:, 1])[-num_bland:]
            bland_indices = window_indices[highest_proba_indices]
            ave_bland = np.mean(if_[bland_indices], axis=0)
            pixel_index = np.where((mat['x'] == i) & (mat['y'] == j))[0]
            pixel_spectra = if_[pixel_index][0]
            ratioed_pixel_spectra = \
            np.divide(pixel_spectra, ave_bland, where=ave_bland!=0, out=np.full_like(pixel_spectra, np.nan))
            ratioed_spectra_bland = np.vstack((ratioed_spectra_bland, ratioed_pixel_spectra))
    ratioed_bland = np.mean(ratioed_spectra_bland, axis=0)

    return unratioed, ratioed_bland, ratioed_median

def get_spectrum(hdr_path, pix_coor, model, scaler, ROI=None, window_size=50, num_bland=3):
    """
    This function extracts and processes the spectrum of a specified pixel or a region of interest (ROI)
    from a given hyperspectral image. It performs pixel ratioing based on bland spectra or median spectra. 

    Parameters:
    -----------
    hdr_path : str
        Path to the hdr file of hyperspectral image.
    pix_coor : tuple
        The x, y coordinate of the pixel for which the spectrum is to be extracted.
    ROI : int, optional
        Size of the region of interest around the pixel. If specified, the function computes the mean spectrum
        within this region. The region is a square with the specified size and the target pixel at its center.
    window_size : int, optional
        Size of the window used for bland pixel detection, based on machine learning model prediction. 
        Default is 50.
    num_bland : int, optional
        Number of bland pixels to consider when computing the mean bland spectra for ratioing. Default is 3.

    Returns:
    --------
    unratioed : numpy.ndarray
        The unratioed mean spectra of the specified pixel or ROI.
    ratioed_bland : numpy.ndarray
        The mean spectra of the specified pixel or ROI, ratioed to the mean bland spectra.
    ratioed_median : numpy.ndarray
        The mean spectra of the specified pixel or ROI, ratioed to the column median spectra.

    Notes:
    ------
    This function requires the presence of a trained machine learning model and 
    a fitted scaler saved under the folder 'models'. These are used for the bland pixel detection.
    """
    
    mat = crism_to_mat(hdr_path)
    if_ = filter_bad_pixels(mat['IF'])
    # remove spikes
    if_ = remove_spikes_column(mat, if_, 3)

    if ROI == None:
        unratioed, ratioed_bland, ratioed_median = line_pixel(mat, if_, pix_coor, model, scaler, window_size, num_bland)

    if ROI != None:
        unratioed, ratioed_bland, ratioed_median = line_ROI(mat, if_, pix_coor, model, scaler, ROI, window_size, num_bland)

    return unratioed, ratioed_bland, ratioed_median