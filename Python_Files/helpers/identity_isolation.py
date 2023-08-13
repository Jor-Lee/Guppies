from google.cloud import vision
import os

import matplotlib.pyplot as plt
import scipy
import numpy as np

from skimage.transform import rotate
from skimage.draw import polygon
import tensorflow as tf
import cv2

from helpers.corrections import *
from helpers.image_loading import * 
from helpers.handling import *

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"../guppies-test-4c48569421d8.json"


def extract_vertices(bounding_box):
    # Groups values that are returned from the bounding_box function into a vertex coordinate.
    vertices = []
    for vertex in bounding_box.vertices:
        vertices.append((vertex.x, vertex.y))
    return vertices

def combine_boxes(boxes):
    # Given a list of boxes, combines them into a single box that spans from (x_min, y_min) to (x_max, y_max).
    X_points = []
    Y_points = []
    for box in boxes:
        for vertex in box:
            X_points.append(vertex[0])
            Y_points.append(vertex[1])

    X_points = np.sort(np.array(X_points))
    Y_points = np.sort(np.array(Y_points))

    number_of_points = len(Y_points)

    X_Low = X_points[:number_of_points//2]
    X_High = X_points[number_of_points//2:]
    Y_Low = Y_points[:number_of_points//2]
    Y_High = Y_points[number_of_points//2:]

    Y_min = np.mean(Y_Low)
    Y_max = np.mean(Y_High)

    para_box = np.zeros((4,2), dtype=np.int32)

    para_box[0] = [X_points.min(), Y_min]
    para_box[1] = [X_points.max(), Y_min]
    para_box[2] = [X_points.max(), Y_max]
    para_box[3] = [X_points.min(), Y_max]

    return para_box

def reduce_image(image, box, padx=60, pady=30):
    """Takes an image and a bounding box and reduces the image to include only the image inside the (padded!) bounding box."""
    adapted_box = box.copy()

    x = np.argsort(box[:,0])
    y = np.argsort(box[:,1])


    adapted_box[x[:2],0] = np.max( np.stack(( box[x[:2],0] - padx , [0,0] )), axis=0)

    adapted_box[x[2:],0] = np.min( np.stack(( box[x[2:],0] + 1.2*padx , [image.shape[1] - 1,image.shape[1] - 1] )), axis=0)

    adapted_box[y[:2],1] = np.max( np.stack(( box[y[:2],1] - pady, [0,0] )), axis=0) 
    adapted_box[y[2:],1] = np.min( np.stack(( box[y[2:],1] + pady , [image.shape[0] - 1,image.shape[0] - 1])), axis=0)


    miny, maxy, minx, maxx = np.min(adapted_box[:,0]), np.max(adapted_box[:,0]), np.min(adapted_box[:,1]), np.max(adapted_box[:,1])

    frame = image[minx:maxx, miny:maxy]

    return frame

def rotate_image_and_transform_coordinates(image, coordinates, angle):
    angle = np.rad2deg(angle)

    # Calculate the transformation matrix.
    transform_matrix = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)

    # Apply the transformation matrix to the coordinates.
    image = cv2.warpAffine(image, transform_matrix, (image.shape[1], image.shape[0]))

    transformed_coordinates = [(transform_matrix[:,:2] @ coordinates[i].T).T + transform_matrix[:,2] for i in range(len(coordinates))]

    return image, transformed_coordinates

def zero_printed_text(image, threshold=40, verbose=False):

    mask = np.zeros(image.shape, dtype=np.uint8)

    mask[(image>0) & (image<threshold)] = 1

    # nx, ny = image.shape

    # mask[nx//6:4*nx//6, ny//6:5*ny//6] = 0 #havent worked this bit out yet

    dilated = cv2.dilate(mask, np.ones((9,9), np.uint8), iterations=1)

    return image * (1-dilated)

def mask_and_remove(frame, verbose=False):
    """Applies an upper and lower threshold to the image, removing the white background and black text respectively. Returns an image
    with only 0's and 255's depending on whether the pixel is within the thresholds. """
    if verbose: fig,ax = plt.subplots(1,3,dpi=250)
    if verbose: ax[0].imshow(frame)

    # Average out the RGB axis
    averaged_frame = np.mean(frame,axis=2)
    if verbose: ax[1].imshow(averaged_frame)


    # White background masking
    hist = np.histogram(averaged_frame, bins=100)
    centers = 0.5*(hist[1][1:]+ hist[1][:-1])

    thresh_val = np.argmax(hist[0] > np.max(hist[0]) * 0.2)
    upper_thresh = centers[thresh_val] * 0.95

    # Dark writing masking. Mask based on central row of frame
    frame_shape = np.shape(averaged_frame)
    row1 = -averaged_frame[2 * frame_shape[0] // 5][100:-100]
    row2 = -averaged_frame[3 * frame_shape[0] // 5][100:-100]
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

    if verbose: print('(upper thresh, lower_thresh) = (%.2f, %.2f)' %(upper_thresh, lower_thresh))

    averaged_frame[averaged_frame>upper_thresh] = 0
    averaged_frame[averaged_frame<lower_thresh] = 0
    averaged_frame[averaged_frame!=0] = 255

    if verbose: ax[2].imshow(averaged_frame)

    return averaged_frame

def remove_deltas(frame, width, padx=60, pady=30, verbose=False):
    """Removes thin strands from the image and clears the border. The border clearing is limited to within the padding zone. If anything
    is removed from within the bounding box, this is undone. This stops whole characters being removed due to boundary clearing."""
    if verbose: fig,ax = plt.subplots(1,2,dpi=250)

    frame_shape = np.shape(frame)

    for i, row in enumerate(frame):
        for j, column in enumerate(row):
            if i == 0:
                pass
            else:
                if frame[i, j] != 0:
                    i_lower = max(i - width, 0)
                    j_lower = max(j - width, 0)
                    i_higher = min(i+width, frame_shape[0])
                    j_higher = min(j+width, frame_shape[1])

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
    cleared_frame[pady:-pady, padx:-padx] = frame[pady:-pady, padx:-padx]

    if verbose: ax[1].imshow(cleared_frame)

    return cleared_frame


def GetImageAndParaBox(image_in_bytes, client, verbose=False):
    # Takes the whole image and returns the initially read output string, an image which contains 
    # the label (either cropped or whole), and the boxes around each of the charaters in the identity.
    client = vision.ImageAnnotatorClient()
    whole_image = image_in_bytes

    img_byte_array = CroppedImage(whole_image, verbose=False)

    decoded = np.frombuffer(img_byte_array.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(decoded, cv2.IMREAD_COLOR) 

    content = img_byte_array.getvalue()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["en"]})

    characters = []
    character_params = []
    character_heights = [] 

    output_string = ''
    word_confidences = []

    for a, page in enumerate(response.full_text_annotation.pages):
        for b, block in enumerate(page.blocks):
            for c, paragraph in enumerate(block.paragraphs):
                for d, word in enumerate(paragraph.words):
                    word_text = "".join([symbol.text for symbol in word.symbols])

                    if '-' in word_text:
                        """Sometimes dates are written with'-', not '/'. This line 
                        should short this out and replace the '-' before they 
                        get passed to the output string"""
                        if verbose: print('Replaced "-" in text with "/".')
                        word_text = word_text.replace('-', '/')
                    
                    output_string += '%s-' %word_text
                    word_confidences.append(word.confidence)
                    
                    for e, symbol in enumerate(word.symbols):

                        # Save character details
                        characters.append(symbol.text)
                        character_params.append([a, b, c, d, e])
                        if verbose: print(characters[-1], 'pageno:'+str(a), 'blockno:'+str(b), 'paragraphno:'+str(c), 'wordno:'+str(d), 'symbolno:'+str(e))

                        # Save the height of each character.
                        vertices = extract_vertices(symbol.bounding_box)
                        X = []
                        Y = []

                        for vertex in vertices:
                            X.append(vertex[0])
                            Y.append(vertex[1])
                            
                        X = np.array(X)
                        Y = np.array(Y)

                        character_heights.append(min(Y))
                        if verbose: print('character:', symbol.text, 'upper height:', max(Y), 'lower height:', min(Y))

    output_string = output_string.upper()
    if verbose: print('\nInitial label:', output_string[:-1].upper(), "\nConfidence:", np.prod(word_confidences))

    character_params = np.array(character_params)

    # If we have not read enough characters, the label may be moved outside the cropping box. Resort to reading the whole image and repeat above.
    if len(characters) < 7:
        if verbose: print('\nNot enough characters in ID, running with the whole image.')

        client = vision.ImageAnnotatorClient()
        content = whole_image.getvalue()
        image = vision.Image(content=content)
        response = client.document_text_detection(image=image, image_context={"language_hints": ["en"]})

        decoded = np.frombuffer(whole_image.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(decoded, cv2.IMREAD_COLOR) 

        characters = []
        character_params = []
        character_heights = [] 

        output_string = ''
        word_confidences = []

        for a, page in enumerate(response.full_text_annotation.pages):
            for b, block in enumerate(page.blocks):
                for c, paragraph in enumerate(block.paragraphs):
                    for d, word in enumerate(paragraph.words):

                        word_text = "".join([symbol.text for symbol in word.symbols])

                        if '-' in word_text:
                            """Sometimes dates are written with'-', not '/'. This line 
                            should short this out and replace the '-' before they 
                            get passed to the output string"""
                            if verbose: print('Replaced "-" in text with "/".')
                            word_text = word_text.replace('-', '/')
                        
                        output_string += '%s-' %word_text
                        word_confidences.append(word.confidence)

                        for e, symbol in enumerate(word.symbols):

                            # Save character details
                            characters.append(symbol.text)
                            character_params.append([a, b, c, d, e])
                            if verbose: print(characters[-1], 'pageno:'+str(a), 'blockno:'+str(b), 'paragraphno:'+str(c), 'wordno:'+str(d), 'symbolno:'+str(e))

                            # Save the height of each character.
                            vertices = extract_vertices(symbol.bounding_box)
                            X = []
                            Y = []

                            for vertex in vertices:
                                X.append(vertex[0])
                                Y.append(vertex[1])

                            X = np.array(X)
                            Y = np.array(Y)

                            character_heights.append(min(Y))
                            if verbose: print('character:', symbol.text, 'upper height:', max(Y), 'lower height:', min(Y))

        character_params = np.array(character_params)

    # If M or F are in the list of characters, remove everything before M or F.
    if 'M' in characters or 'F' in characters:
        lead_character = characters[0]
        while lead_character != 'F' and lead_character != 'M':

            if verbose: print('\nRemoving everything before F and M')
            if verbose: print('removing character', characters[0])

            characters = characters[1:]
            character_params = character_params[1:]
            character_heights = character_heights[1:]
            lead_character = characters[0]

            if verbose: print('Reduced characters are now:', characters)
            if verbose: print('Lead character is now:', lead_character)

            # If we drop below 7 characters then end then break
            if len(characters) < 7:
                if verbose: print("Not enough characters to continue. Breaking the cycle.")
                return 1

    # Remove any trailing *'s or x's
    if characters[0] == '*' or characters[0] == 'x' or characters[0] == 'X':
        characters = characters[1:]
        character_params = character_params[1:]
        character_heights = character_heights[1:]
        if verbose: print('Removing special character after title.')

    # Removing the three title characters:
    if verbose: print('Removing the title characters')
    title = characters[:3]
    characters = characters[3:]
    character_params = character_params[3:]
    character_heights = character_heights[3:]
    if verbose: print('Remaining characters are:', characters)

    # Removing potential *'s from the title:
    if characters[0] == '*' or characters[0] == 'x':
        characters = characters[1:]
        character_params = character_params[1:]
        character_heights = character_heights[1:]

    if len(characters) < 4:
                if verbose: print("Not enough characters to continue. Breaking the cycle.")
                return 1

    # Remove the date and everything after
    if verbose: print('Removing date and characters after date')
    date_index = np.argmax(['/' in i for i in characters])
    if verbose: print('date index is', date_index)
    
    if date_index == 0:
        if verbose: print("No '/' characters, searching for '-'")
        date_index = np.argmax(['-' in i for i in characters])
        if date_index == 0:
            if verbose: print("No '-' characters either. Cannot find date. Skipping step.")
            pass
        
    
    characters = characters[:date_index - 1]
    character_params = character_params[:date_index - 1]
    character_heights = character_heights[:date_index - 1]
    if verbose: 
        print('Remaining characters are:', characters)
        print('Ramaining character heights:', character_heights)

    if len(characters) < 3:
        if verbose: print("Not enough characters to continue. Breaking the cycle.")
        return 1, 1, 1

    # Character in the 3rd position of height.
    ordered_height = np.argsort(character_heights)
    if verbose: print('Character heights are:', character_heights)
    if verbose: print('Character height order is:', ordered_height)
    # focused_index = ordered_height[5]
    focused_index = 2
    if verbose: print('Character and height in position 3 is:', characters[focused_index], character_heights[focused_index])

    # 3rd character bounding box is:
    focused_height = character_heights[focused_index]
    if verbose: print('character height is:', focused_height)

    # Cycle through and find include all characters with similar height
    ID_indices = []
    for i, character_height in enumerate(character_heights):
        if focused_height >= 0:
            if (focused_height - (focused_height * 0.09) ) <= character_height <= (focused_height + (focused_height * 0.09) ):
                ID_indices.append(i)
                if verbose: print('Character', characters[i], 'falls within height bounds')
        else:
            if (focused_height + (focused_height * 0.09) ) <= character_height <= (focused_height - (focused_height * 0.09) ):
                ID_indices.append(i)
                if verbose: print('Character', characters[i], 'falls within height bounds')

    # Remove any that are too far away from the central indicies (removes and stray date characters included)
    indices_diff = np.diff(ID_indices)
    for i, diff in enumerate(indices_diff):
        if diff > 3:
            if verbose: print('Removed %s as too far away from central indices' %characters[ID_indices[i+1]])
            ID_indices = ID_indices[:i + 1]

    # If less than 4 characters in the ID, then manually add more.
    while len(ID_indices) < 4:
        if np.min(np.array(ID_indices)) == 0:
            also_include_index = np.max(np.array(ID_indices)) + 1
        else:
            also_include_index = np.min(np.array(ID_indices)) - 1
        ID_indices.insert(0, np.array(also_include_index))
        if verbose: print('Less than 4 chracters in ID. Adding', characters[also_include_index])

    ID_characters = []
    ID_character_params = []
    ID_vertices = []
    
    for index in ID_indices:
        try:
            character = characters[index]
            character_param = character_params[index]
            page_no, block_no, paragraph_no, word_no, symbol_no = character_param
            character_bounds = extract_vertices(page.blocks[block_no].paragraphs[paragraph_no].words[word_no].symbols[symbol_no].bounding_box)

            ID_characters.append(character)
            ID_character_params.append(character_param)
            ID_vertices.append(extract_vertices(page.blocks[block_no].paragraphs[paragraph_no].words[word_no].symbols[symbol_no].bounding_box))
        except: pass

    if verbose: print('Included characters and their indicies are:', ID_characters, ID_indices)

    if verbose: print('Number of bounding boxes', len(ID_vertices))

    character_boxes = np.array(ID_vertices)

    if verbose: plt.imshow(frame)

    return output_string[:-1], frame, character_boxes


def extract_ID_handwriting(frame, paragraph_vertices, verbose=False):
        
    padx = 60
    pady = 30

    #firstly find the box around the identity, and also calculate the approx angle of rotation of that box. 
    adapted_para_box = paragraph_vertices.copy()

    x = np.argsort(paragraph_vertices[:,0])
    y = np.argsort(paragraph_vertices[:,1])


    adapted_para_box[x[:2],0] = np.max( np.stack(( paragraph_vertices[x[:2],0] - padx , [0,0] )), axis=0)

    adapted_para_box[x[2:],0] = np.min( np.stack(( paragraph_vertices[x[2:],0] + 1.2*padx , [frame.shape[1] - 1,frame.shape[1] - 1] )), axis=0)

    adapted_para_box[y[:2],1] = np.max( np.stack(( paragraph_vertices[y[:2],1] - pady, [0,0] )), axis=0) 
    adapted_para_box[y[2:],1] = np.min( np.stack(( paragraph_vertices[y[2:],1] + pady , [frame.shape[0] - 1,frame.shape[0] - 1])), axis=0)


    rel_area = polygon(adapted_para_box[:,1], adapted_para_box[:,0]) 
    mask1 = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask1[rel_area] = 1

    miny, maxy, minx, maxx = np.min(adapted_para_box[:,0]), np.max(adapted_para_box[:,0]), np.min(adapted_para_box[:,1]), np.max(adapted_para_box[:,1])


    idx = paragraph_vertices[:,1].argsort()[:2]
    dx = paragraph_vertices[idx[1],0] - paragraph_vertices[idx[0],0]
    dy = paragraph_vertices[idx[1],1] - paragraph_vertices[idx[0],1]

    angle = np.arctan2(dy,dx)


    if verbose:
        fig,ax = plt.subplots(1,6,dpi=250)
        ax[0].imshow(frame)
    frame = frame*mask1[...,None] #mask the image

    if verbose: ax[1].imshow(frame)

    frame = np.mean(frame, axis=2)
    frame = frame[minx:maxx, miny:maxy]

    if verbose: ax[2].imshow(frame)

    frame = rotate(frame, -angle, resize=False, ) #rotate the image
    # frame = (1-frame) #invert and threshold.
    # frame[frame < 0.5] = 0
    # frame[frame !=0] = 1

    if verbose: ax[3].imshow(frame)


    frame = zero_printed_text(frame, threshold = 30) #threshold needs to be found algorithmically.


    hist = np.histogram(frame, bins=100)
    peaks,_ = scipy.signal.find_peaks(hist[0], height=1000, distance=7)

    if len(peaks) == 2:
        peak_handwriting = peaks[0]
        peak_background = peaks[1]

        thresh = 1*hist[1][peak_background]/4  + 3*hist[1][peak_handwriting]/4
    else:
        thresh = frame.max()*0.85
        


    frame[frame>thresh] = 0 
    frame[frame!=0] = 1

    if verbose: ax[4].imshow(frame)


    from skimage.segmentation import clear_border
    # kernel = np.ones((5, 5), np.uint8)
    # frame = cv2.erode(frame, kernel, iterations=1)
    
    # frame = clear_border(frame)

    if verbose: ax[5].imshow(frame)

    return frame