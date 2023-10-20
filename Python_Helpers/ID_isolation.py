import numpy as np
from google.cloud import vision

import numpy as np
import matplotlib.pyplot as plt
import scipy

def IsolateIdentity(Processed_Image_Dictionary, padx=80, pady=20, delta_width=15, verbose=False):
    # Reduce the dictionary to contain only ID characters.
    Processed_Image_Dictionary = RetainID(Processed_Image_Dictionary, verbose=verbose)
    
    # Check character heights (ensure no other characters have been mixed with the title).
    Processed_Image_Dictionary = CheckHeights(Processed_Image_Dictionary, verbose=verbose)

    # Find the ID bounding box.
    bounding_box = CombineBoxes(Processed_Image_Dictionary['character_bounds'], verbose=verbose)

    # Reduce the image to contain only the ID (with some added padding).
    Processed_Image_Dictionary = ReduceImage(Processed_Image_Dictionary, bounding_box, padx=padx, pady=pady, verbose=verbose)

    # Mask the image, removing white background and black title/date text.
    Processed_Image_Dictionary = MaskAndRemove(Processed_Image_Dictionary, verbose=verbose)

    # Clear the border and remove any thin strands from the image.
    Processed_Image_Dictionary = RemoveDeltas(Processed_Image_Dictionary, delta_width=delta_width, padx=padx, pady=pady, verbose=verbose)

    # Run some tests to see if the ID image satisfies certain conditions.
    shape = np.shape(Processed_Image_Dictionary['frame'])
    counts = np.count_nonzero(Processed_Image_Dictionary['frame'])
    validity_test = counts / (shape[0] * shape[1])

    if validity_test < 0.065 or shape[0] < 200 or shape[0] > 450 or shape[1] > 1100:
        if verbose: print(validity_test, shape)
        raise ValueError('ID image has failed final condition checks.', 'Vadlidity (<0/065):', validity_test, 'Shape (shape[0]>200, shape[1]<1100):', shape)

    return Processed_Image_Dictionary


def RetainID(Processed_Image_Dictionary, verbose=False):
    if verbose: print("\nIdentifying and retaining only characters in the ID.")

    word_params = [x[-2] for x in Processed_Image_Dictionary['character_params']]

    ID_min_index = min(i for i, x in enumerate(word_params) if x == 1)
    ID_max_index = max(i for i, x in enumerate(word_params) if x == 1) + 1

    Processed_Image_Dictionary['characters'] = Processed_Image_Dictionary['characters'][ID_min_index:ID_max_index]
    Processed_Image_Dictionary['character_params'] = Processed_Image_Dictionary['character_params'][ID_min_index:ID_max_index]
    Processed_Image_Dictionary['character_bounds'] = Processed_Image_Dictionary['character_bounds'][ID_min_index:ID_max_index]
    Processed_Image_Dictionary['character_confidences'] = Processed_Image_Dictionary['character_confidences'][ID_min_index:ID_max_index]

    if verbose: print("Remaining characters are", Processed_Image_Dictionary['characters'])

    return Processed_Image_Dictionary

def CheckHeights(Processed_Image_Dictionary, verbose=False):
    if verbose: print("\nChecking the heights of ID characters.")
    # Each character bound will have 2 lower and 2 upper corners. Seperate the lower and upper
    # corners and take the mean.
    lower_heights = [x[1][:2] for x in Processed_Image_Dictionary['character_bounds']]
    upper_heights = [x[1][2:] for x in Processed_Image_Dictionary['character_bounds']]

    mean_lower_heights = []
    mean_upper_heights = []

    for i, heights in enumerate(lower_heights):
        mean_height = np.mean(heights)
        mean_lower_heights.append(mean_height)

    for i, heights in enumerate(upper_heights):
        mean_height = np.mean(heights)
        mean_upper_heights.append(mean_height)

    # Use the central index (most likely to actually be in the ID) as a reference.
    reference_lower = mean_lower_heights[len(mean_lower_heights) // 2]
    reference_upper = mean_upper_heights[len(mean_upper_heights) // 2]

    include = []
    # Cycle through and find all characters with similar heights.
    for i in range(len(mean_lower_heights)):
        lower_difference = abs((reference_lower - mean_lower_heights[i]))
        upper_difference = abs((reference_upper - mean_upper_heights[i]))

        if max(lower_difference, upper_difference) < 75:
            include.append(i)
        else:
            if verbose: print('\n', Processed_Image_Dictionary['characters'][i], 'does not fall in the height bounds. Removing from the ID.')

    Processed_Image_Dictionary['characters'] = [Processed_Image_Dictionary['characters'][i] for i in include]
    Processed_Image_Dictionary['character_params'] = [Processed_Image_Dictionary['character_params'][i] for i in include]
    Processed_Image_Dictionary['character_bounds'] = [Processed_Image_Dictionary['character_bounds'][i] for i in include]
    Processed_Image_Dictionary['character_confidences'] = [Processed_Image_Dictionary['character_confidences'][i] for i in include]

    return Processed_Image_Dictionary

def CombineBoxes(boxes, verbose=False):
    # Given a list of boxes, combines them into a single box that spans from (x_min, y_min) to (x_max, y_max).
    X_points = []
    Y_points = []
    for box in boxes:
        X_points.append(box[0])
        Y_points.append(box[1])

    X_points = np.sort(np.array(X_points))
    Y_points = np.sort(np.array(Y_points))


    Y_min = Y_points.min() - 20
    Y_max = Y_points.max() + 20
    X_min = X_points.min() - 20
    X_max = X_points.max() + 20

    para_box = np.zeros((4,2), dtype=np.int32)

    para_box[0] = [X_min, Y_min]
    para_box[1] = [X_max, Y_min]
    para_box[2] = [X_max, Y_max]
    para_box[3] = [X_min, Y_max]
    
    return para_box


def ReduceImage(Processed_Image_Dictionary, boudning_box, padx=80, pady=20, verbose=False):
    """Takes an image and a bounding box and reduces the image to include only the image inside the (padded!) bounding box."""
    image = Processed_Image_Dictionary['frame']

    if len(Processed_Image_Dictionary['characters']) <= 3:
        if verbose: print("\nLess than three characters in the ID. Doubling the horizontal padding.")
        padx = 2 * padx

    adapted_box = boudning_box.copy()

    x = np.argsort(boudning_box[:,0])
    y = np.argsort(boudning_box[:,1])


    adapted_box[x[:2],0] = np.max( np.stack(( boudning_box[x[:2],0] - padx , [0,0] )), axis=0)

    adapted_box[x[2:],0] = np.min( np.stack(( boudning_box[x[2:],0] + 1.2*padx , [image.shape[1] - 1,image.shape[1] - 1] )), axis=0)

    adapted_box[y[:2],1] = np.max( np.stack(( boudning_box[y[:2],1] - pady, [0,0] )), axis=0) 
    adapted_box[y[2:],1] = np.min( np.stack(( boudning_box[y[2:],1] + pady , [image.shape[0] - 1,image.shape[0] - 1])), axis=0)


    miny, maxy, minx, maxx = np.min(adapted_box[:,0]), np.max(adapted_box[:,0]), np.min(adapted_box[:,1]), np.max(adapted_box[:,1])

    Processed_Image_Dictionary['frame'] = image[minx:maxx, miny:maxy]

    return Processed_Image_Dictionary


def MaskAndRemove(Processed_Image_Dictionary, verbose=False):
    """Applies an upper and lower threshold to the image, removing the white background and black text respectively. Returns an image
    with only 0's and 255's depending on whether the pixel is within the thresholds. """
    frame = Processed_Image_Dictionary['frame']
    
    if verbose: fig,ax = plt.subplots(1,3,dpi=250)
    if verbose: ax[0].imshow(frame)

    # Average out the RGB axis
    averaged_frame = np.mean(frame,axis=2)
    if verbose: ax[1].imshow(averaged_frame)


    # White background masking
    hist = np.histogram(averaged_frame, bins=100)
    centers = 0.5*(hist[1][1:]+ hist[1][:-1])

    thresh_val = np.argmax(hist[0] > np.max(hist[0]) * 0.25)
    upper_thresh = centers[thresh_val] * 0.95

    # Dark writing masking. Mask based on central row of frame
    frame_shape = np.shape(averaged_frame)
    row1 = -averaged_frame[2 * frame_shape[0] // 5][200:-200]
    row2 = -averaged_frame[3 * frame_shape[0] // 5][200:-200]
    rows = np.concatenate((row1, row2))
    row_mean = np.mean(rows)
    row_max = np.max(rows)
    row_min = np.min(rows)
    peaks,_ = scipy.signal.find_peaks(rows, height = row_mean + ((row_max - row_min) / 5), distance=4)

    values = []
    for peak in peaks:
        values.append(-rows[peak])

    sorted_values = np.sort(values)

    lower_thresh = sorted_values[0] * 1

    averaged_frame[averaged_frame>upper_thresh] = 0
    averaged_frame[averaged_frame<lower_thresh] = 0
    averaged_frame[averaged_frame!=0] = 255

    if verbose: ax[2].imshow(averaged_frame)

    Processed_Image_Dictionary['frame'] = averaged_frame

    return Processed_Image_Dictionary

def RemoveDeltas(Processed_Image_Dictionary, delta_width, padx=80, pady=20, verbose=False):
    """Removes thin strands from the image and clears the border. The border clearing is limited to within the padding zone. If anything
    is removed from within the bounding box, this is undone. This stops whole characters being removed due to boundary clearing."""
    frame = Processed_Image_Dictionary['frame']

    # if len(Processed_Image_Dictionary['characters']) == 3:
    #     padx = 2 * padx
    #     pady = 2 * pady 
    
    if verbose: fig,ax = plt.subplots(1,3,dpi=250)

    frame_shape = np.shape(frame)

    for i, row in enumerate(frame):
        for j, column in enumerate(row):
            if i == 0:
                pass
            else:
                if frame[i, j] != 0:
                    i_lower = max(i - delta_width, 0)
                    j_lower = max(j - delta_width, 0)
                    i_higher = min(i + delta_width, frame_shape[0])
                    j_higher = min(j + delta_width, frame_shape[1])

                    surroundings = frame[i_lower:i_higher, j_lower:j_higher]
                    condition = surroundings == 255.
                    count = np.count_nonzero(condition)

                    surroundings_shape = np.shape(surroundings)
                    surroundings_elements_number = surroundings_shape[0] * surroundings_shape[1]

                    if count < surroundings_elements_number // 5:
                        frame[i, j] = 0

    if verbose: ax[0].imshow(frame)

    from skimage.segmentation import clear_border
    cleared_frame = clear_border(frame)

    # Undo any clearing of content within the unpadded bounding box (to ensure whole characters arent removed).
    if padx == 0:
        cleared_frame[pady:-pady, :] = frame[pady:-pady, :]
    else:
        cleared_frame[pady:-pady, padx:-padx] = frame[pady:-pady, padx:-padx]

    if verbose: ax[1].imshow(cleared_frame)

    # Remove deltas again after clearing the frame.

    frame = cleared_frame

    frame_shape = np.shape(frame)

    for i, row in enumerate(frame):
        for j, column in enumerate(row):
            if i == 0:
                pass
            else:
                if frame[i, j] != 0:
                    i_lower = max(i - delta_width, 0)
                    j_lower = max(j - delta_width, 0)
                    i_higher = min(i + delta_width, frame_shape[0])
                    j_higher = min(j + delta_width, frame_shape[1])

                    surroundings = frame[i_lower:i_higher, j_lower:j_higher]
                    condition = surroundings == 255.
                    count = np.count_nonzero(condition)

                    surroundings_shape = np.shape(surroundings)
                    surroundings_elements_number = surroundings_shape[0] * surroundings_shape[1]

                    if count < surroundings_elements_number // 5:
                        frame[i, j] = 0

    if verbose: ax[2].imshow(frame)

    Processed_Image_Dictionary['frame'] = frame  

    return Processed_Image_Dictionary
