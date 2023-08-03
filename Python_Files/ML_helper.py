from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os
import tensorflow as tf

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


def vectorize_label(label, char_to_num, max_len, padding_token):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


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
    def __init__(self, h5_file, idxs, batch_size=32, shuffle=True):
        'Initialization'
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.list_IDs = idxs
        self.shuffle = shuffle
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
        Y = self.h5_file['labels'][list_IDs_temp]


        return {"image":X[...,None], "label": Y}    