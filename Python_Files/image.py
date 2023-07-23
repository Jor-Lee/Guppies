from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os
import csv

from corrections import *

def RetreiveImage(file, verbose=False): 
    """Retreives an image from the google cloud bucket and returns it as an array of bytes."""
    bucket_name = "guppy_images"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    file_list = storage_client.list_blobs(bucket_name)
    file_list = [file.name for file in file_list]

    blob = bucket.blob(file)
    
    img_bytes = BytesIO(blob.download_as_bytes())

    if verbose: print("\nImage has been read from google bucket.")

    return img_bytes


def LoadImage(file, verbose=False):
    """Reads a local image and returns an array of bytes. Similar to RetreiveImage function but for local data."""
    image = Image.open(file)
    byte_arr = BytesIO()
    image.save(byte_arr, format='jpeg')
    img_bytes = BytesIO(byte_arr.getvalue())

    if verbose: print("\nImage has been read from local file.")

    return img_bytes

def CroppedImage(img_bytes, verbose=False):
    """Takes an image as an array of bytes and returns a cropped image as an array of bytes."""
    img = Image.open(img_bytes)
    width, height = img.size

    left = 1 * width / 4
    right = 3 * width / 4
    top = 0
    bottom = height / 3
    cropped_image = img.crop((left, top, right, bottom))

    cropped_byte_arr = BytesIO()
    cropped_image.save(cropped_byte_arr, format='jpeg')
    cropped_byte_arr = BytesIO(cropped_byte_arr.getvalue())

    if verbose: print("\nImage has been cropped.")

    return cropped_byte_arr
    