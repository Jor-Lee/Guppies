import os
import numpy as np
import torch 
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt


def UseYOLO(model, ID_image, mapping, probability_threshold=0.5, verbose=False):
    processed_ID_image = prepare_image(ID_image)

    file = './Temp_YOLO_File.jpg'

    file, processed_ID_image.save(file)

    results = model.predict(file)[0]

    os.remove(file)

    boxes, pred_vec, probs = analyse_results(results.boxes.data, probability_threshold)

    characters = []
    for vec in pred_vec:
        characters.append(mapping[vec])

    return characters, boxes, probs, processed_ID_image


def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    # image = tf.transpose(image, perm=[1, 0, 2])
    # image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image, img_size):
    image = tf.convert_to_tensor(image)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def prepare_image(ID_image):
    ID_image_width, ID_image_height = 256, 64
    processed_ID_image = preprocess_image(ID_image[...,None],(ID_image_width,ID_image_height)).numpy()[None,...,0][0]
    processed_ID_image = (((processed_ID_image - processed_ID_image.min()) / (processed_ID_image.max() - processed_ID_image.min())) * 255.9)
    processed_ID_image = Image.fromarray(np.uint8(processed_ID_image))
    processed_ID_image = processed_ID_image.convert("L")

    return processed_ID_image

def analyse_results(data, prob_threshold):

    #first bring together the letters and numbers.

    data = data.detach().clone()
    
    idx_data = torch.arange(data.shape[0], dtype=torch.int) #create an index tensor

    #discard characters with low probability
    keep = data[:,-2] > prob_threshold
    data = data[keep] 
    idx_data = idx_data[keep]

    #sort by x coordinate to get the right order
    orde = torch.argsort(data[:,0],dim=0)
    data = data[orde] 
    idx_data = idx_data[orde]

    
    if data.shape[0] > 0:
        # discard any characters that are too short?
        height_threshold = (data[:,3] - data[:,1]).max() * 0.5
        
        keep = data[:,3] - data[:,1] > height_threshold
        data = data[keep]
        idx_data = idx_data[keep] 

        data, idx_data = discard_overlapping_boxes(data, idx_data) #discard overlapping boxes
        
    probs = data[:,-2].numpy()
    classes = data[:,-1].to(torch.int).numpy()

    boxes = data[:,:-2].numpy().astype(int)
    
    return boxes, classes, probs
    

def discard_overlapping_boxes(data, idx_data):
    #if the model predicted two characters that overlap alot, then remove the smaller one?

    intersections = np.array([len(list(range(max(int(data[i,0]), int(data[i+1,0])), min(int(data[i,2]), int(data[i+1,2]))+1))) for i in range(data.shape[0]-1)])#get the x intersection between each box
    area = (data[:,2]-data[:,0])*(data[:,3]-data[:,1])

    x_ranges = data[:,2]-data[:,0]

    x_ranges = np.array([torch.min(x_ranges[i],x_ranges[i+1]) for i in range(x_ranges.shape[0]-1)])

    intersections = intersections/x_ranges

    problemos = np.where(intersections>0.3)[0] #if 0.3 of a letter 

    for n in range(len(problemos)):
        i = problemos[n]
        if area[i] < area[i+1]:
            data = np.delete(data, i, axis=0)
            idx_data = np.delete(idx_data, i, axis=0)
        else:
            data = np.delete(data, i+1, axis=0)
            idx_data = np.delete(idx_data, i+1, axis=0)
        problemos = problemos - 1

    return data, idx_data