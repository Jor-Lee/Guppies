from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os
import tensorflow as tf
import torch.nn as nn
import torch

import tensorflow.keras as keras

def ReadImage(img_byte_array, verbose=False):
    """Reads an image as an array of bytes and returns the text in the output format required."""
    from google.cloud import vision
    from PIL import Image
    import numpy as np

    client = vision.ImageAnnotatorClient()
    content = img_byte_array.getvalue()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["en"]})

    output_string = ''

    word_confidences = []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:

            for paragraph in block.paragraphs:
                
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])

                    output_string += '%s-' %word_text
                    word_confidences.append(word.confidence)

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    if verbose: print('\nOutput:', output_string[:-1].upper(), "\nConfidence:", np.prod(word_confidences))
    return output_string[:-1].upper(), word_confidences



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


def string_to_vector(labelstring, char_dict, max_len=8):
    #this file takes a label written in characters (a string) and converts it to a vector of integers based on the dictionary.
    space_token = len(char_dict)

    label = [list(char_dict.values()).index(char) for char in labelstring]
    if len(labelstring) < max_len:
        label = label + [list(char_dict.values()).index('')]*(max_len-len(label))
    return np.array(label, dtype=np.int32)

def vector_to_string(labelvector, char_dict):
    return [char_dict[label] for label in labelvector]


# class process_image_labels:
#     def __init__(self, h5file):
#         self.h5file = h5file

#     def __call__(self, idx):
#         image = self.h5file['images'][idx]
#         label = self.h5file['labels'][idx]
#         return {"image":image[...,None], "label": label}


# def process_images_labels(image, label):
#     # image = preprocess_image(image_path)
#     # label = vectorize_label(label)
#     return {"image": image, "label": label}


# def prepare_dataset(h5file, listids):
#     AUTOTUNE = tf.data.AUTOTUNE
#     dataset = tf.data.Dataset.from_tensor_slices((listids)).map(
#         process_image_labels, num_parallel_calls=AUTOTUNE
#     )
#     return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, h5_file, idxs, batch_size=32, shuffle=True, ctc=False):
        'Initialization'
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.list_IDs = idxs
        self.shuffle = shuffle
        self.ctc = ctc
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.sort(self.indexes[index*self.batch_size:(index+1)*self.batch_size])

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        # X, y = self.__data_generation(list_IDs_temp)
        X= self.__data_generation(list_IDs_temp)

        return X#,y


    def on_epoch_end(self):

        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)



    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = self.h5_file['images'][list_IDs_temp]

        if self.ctc:
            Y = self.h5_file['labels'][list_IDs_temp]
            return {"image":X[...,None], "label": Y}

        else:
            Y = labels_to_logits(self.h5_file['labels'][list_IDs_temp])

            return X[...,None], Y 


def labels_to_logits(labels):
    logits = np.zeros((len(labels), 8,23))

    for n in range(len(labels)):
        for i, label in enumerate(labels[n]):
            if label != 99:
                logits[n, i, label] = 1
            else:
                logits[n, i, -1] = 1
    return logits


class CNN(nn.Module):
    def __init__(self, no_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, no_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization

