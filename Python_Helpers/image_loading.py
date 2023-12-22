from google.cloud import storage
from google.cloud import vision

from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dropbox
import json


def ListGoogleFiles(bucket_name, prefix='', verbose=False):
    """Lists all avaliable files in the bucket. Useful for cycling through all files in a loop."""
    storage_client = storage.Client()
    file_list = storage_client.list_blobs(bucket_name, prefix=prefix)
    file_list = [file.name for file in file_list]
    if verbose: print("\nFiles have been read from Google.")
    return file_list


def LoadDropboxDetails(path, verbose=False):
    """Reads DropBox token details."""
    with open(path) as f: 
        data = f.read()
        js = json.loads(data) 
    
    return js


def ListDropBoxFiles(dbx, bucket_name, prefix='', suffix='JPG', verbose=False):
    file_list = []
    has_more_files = True # because we haven't queried yet
    cursor = None # because we haven't queried yet

    while has_more_files:
        if cursor is None: # if it is our first time querying
            results = dbx.files_list_folder('/%s/%s' %(bucket_name, prefix), recursive=True)
        else:
            results = dbx.files_list_folder_continue(cursor)
        
        for result in results.entries:
            if 'size' in dir(result):
                file = result.path_display
                if file[-len(suffix):] == suffix or suffix == '':
                    file_list.append(file)
                else: continue
            else: continue

        cursor = results.cursor
        has_more_files = results.has_more

    if verbose: print("\nFiles have been read from Dropbox.")
    return file_list

def RetreiveImageGoogle(bucket_name, file, verbose=False): 
    """Retreives an image from the google cloud bucket and returns it as an array of bytes."""
    # bucket_name = "guppy_images"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(file)

    img_bytes = BytesIO(blob.download_as_bytes())

    if verbose: print("\nImage has been read from google bucket.")
    return img_bytes


def RetreiveImageDropBox(dbx, file_name, verbose=False): 
    """Retreives an image from a dropbox folder and returns it as an array of bytes."""
    img_bytes = BytesIO(dbx.files_download(file_name)[1].content)

    if verbose: print("\nImage has been read from google DropBox.")
    return img_bytes


def RetreiveImageLocal(file, verbose=False):
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

    left = 0 * width / 5
    right = 5 * width / 5
    top = 0
    bottom = height * (3 / 5)
    cropped_image = img.crop((left, top, right, bottom))

    cropped_byte_arr = BytesIO()
    cropped_image.save(cropped_byte_arr, format='jpeg')
    cropped_byte_arr = BytesIO(cropped_byte_arr.getvalue())

    if verbose: print("\nImage has been cropped.")

    return cropped_byte_arr


def GoogleRead(img_bytes, verbose=False):
    """Reads an image as an array of bytes and returns the google cloud vision output.
    Initially tries to read a cropped version of the image for increased accuracy. If not enough
    characters are read from the cropped image, the process is retried with the whole image.
    This output contains information on full words and individual characters, inclluding: 
    bounding boxes, character predictions and confidences."""
    
    whole_image = img_bytes
    cropped_image = CroppedImage(img_bytes, verbose=False)

    # First try with cropped image
    decoded = np.frombuffer(cropped_image.getvalue(), dtype=np.uint8)
    frame = cv2.imdecode(decoded, cv2.IMREAD_COLOR) 

    client = vision.ImageAnnotatorClient()
    content = cropped_image.getvalue()
    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["en"]})

    characters = []
    character_params = []
    character_bounds = [] 
    character_confidences = []

    words = []
    word_confidences = []

    for a, page in enumerate(response.full_text_annotation.pages):
        for b, block in enumerate(page.blocks):
            for c, paragraph in enumerate(block.paragraphs):
                for d, word in enumerate(paragraph.words):
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    words.append(word_text)
                    word_confidences.append(word.confidence)

                    for e, symbol in enumerate(word.symbols):
                        characters.append(symbol.text)
                        character_params.append([a, b, c, d, e])
                        character_confidences.append(symbol.confidence)

                        if verbose: print(symbol.text, 'pageno:'+str(a), 'blockno:'+str(b), 'paragraphno:'+str(c), 'wordno:'+str(d), 'symbolno:'+str(e))

                        # Save the height of each character.
                        character_box = symbol.bounding_box.vertices
                        X = []
                        Y = []

                        for vertex in character_box:
                            X.append(vertex.x)
                            Y.append(vertex.y)

                        X = np.array(X)
                        Y = np.array(Y)

                        vertices = [X, Y]

                        character_bounds.append(vertices)


    # If we have not read enough characters, the label may be moved outside the cropping box. Resort to reading the whole image and repeat above.
    if len(characters) < 11:
        if verbose: print('\nNot enough characters in label, running with the whole image.')

        # First try with cropped image
        decoded = np.frombuffer(whole_image.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(decoded, cv2.IMREAD_COLOR) 

        client = vision.ImageAnnotatorClient()
        content = whole_image.getvalue()
        image = vision.Image(content=content)

        response = client.document_text_detection(image=image, image_context={"language_hints": ["en"]})

        characters = []
        character_params = []
        character_bounds = [] 
        character_confidences = []

        words = []
        word_confidences = []

        for a, page in enumerate(response.full_text_annotation.pages):
            for b, block in enumerate(page.blocks):
                for c, paragraph in enumerate(block.paragraphs):
                    for d, word in enumerate(paragraph.words):
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        words.append(word_text)
                        word_confidences.append(word.confidence)

                        for e, symbol in enumerate(word.symbols):
                            characters.append(symbol.text)
                            character_params.append([a, b, c, d, e])
                            character_confidences.append(symbol.confidence)

                            if verbose: print(symbol.text, 'pageno:'+str(a), 'blockno:'+str(b), 'paragraphno:'+str(c), 'wordno:'+str(d), 'symbolno:'+str(e))

                            # Save the height of each character.
                            character_box = symbol.bounding_box.vertices
                            X = []
                            Y = []

                            for vertex in character_box:
                                X.append(vertex.x)
                                Y.append(vertex.y)

                            X = np.array(X)
                            Y = np.array(Y)

                            vertices = [X, Y]

                            character_bounds.append(vertices)

    if verbose: plt.imshow(frame)

    Image_Output = {'words': words,
                    'word_confidences': word_confidences,
                    'characters': characters,
                    'character_params': character_params,
                    'character_bounds': character_bounds,
                    'character_confidences': character_confidences,
                    'frame': frame}

    return Image_Output