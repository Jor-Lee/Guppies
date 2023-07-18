from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"/Users/jordan/Desktop/Guppies/guppies-test-4c48569421d8.json"


def ListAvaliableFiles(bucket_name, verbose=False):
    """Lists all avaliable files in the bucket. Useful for cycling through all files in a loop."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    file_list = storage_client.list_blobs(bucket_name)
    file_list = [file.name for file in file_list]
    if verbose: print("\nFiles have been read.")
    return file_list


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


ReplaceSpecialCharacter = [['(', '1'], ['\\', '1'], ['+', 'G'], ['~~', 'W']]
def RemoveSpecialCharacters(output_string, verbose=False):
    """Function removes all special characters that are read by the OCR."""
    if verbose: print("\nRemoving special characters from the output string (e.g. '.', '|').")

    for character in output_string:
        for element in ReplaceSpecialCharacter:
            if character in element:
                output_string = output_string.replace(element[0], element[1])
                if verbose: print("characters %s have been replaced with %s" %(element[0], element[1]))
        
    for character in output_string:
        if 'A' <= character <= 'Z' or '0' <= character <= '9' or character == '/' or character == '-':
            if verbose:  print("Character %s is fine." %character)
        else:
            if verbose: print("Character %s has been removed." %character)
            output_string = output_string.replace(character, "")

    return output_string


def RemoveDeadElements(output_split, verbose=False):
    """Function removes paragraphs that are not the right size. These paragraphs are often formed when the 
    OCR reads spetial characters from the image."""
    if verbose: print("\nRemoving paragraphs without three (3), four (4), five (5) or eight (8) elements.")
    empty = []

    for element in output_split:
        if len(element) == 3 or len(element) == 4 or len(element) == 5 or len(element) == 8 or len(element) == 9:
            empty.append(element)
        else:
            if element == "":
                if verbose: print("Removed empty paragraph")
            else:
                if verbose: print("Paragraph removed:", element)
                if verbose: print("Length of removed paragraph:", len(element))
    
    output_split = empty

    if verbose: print("Label after removing dead paragraphs:", output_split)

    return output_split


def TitleErrors(title, verbose=False):
    """Finds errors in the title. The title should be three (3) characters long and contain only english capital letters."""
    if verbose: print("\nLooking for errors in the title (%s)." %title)

    if len(title) != 3:
        if verbose: print("Incorrect title length. Title has %i elements, three (3) are required." %len(title))
        return 1

    for character in title:
        if not 'A' <= character <= 'Z':
            if verbose: print("Incorrect character (%s) in the title" %character)
            return 1

    if verbose: print("Final title:", title)

    return title


def IdentityErrors(identity, verbose=False):
    """Finds errors in the identity. The identy should be four (4) or five (5) characters long and should follow the
    pattern [number, letter, number, letter number*] with the final number being optional. Any other format is an error."""
    if verbose: print("\nLooking for errors in the identity (%s)." %identity)

    acceptable_numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    acceptable_letters = ['B', 'F', 'G', 'K', 'N', 'O', 'P', 'R', 'S', 'V', 'W', 'Y']

    if len(identity) < 4:
        if verbose: print("Incorrect identity length. Identity has %i elements, four (4) or five (5) are required." %len(identity))
        return 1

    for i, character in enumerate(identity):
        if i%2 == 0:
            if character not in acceptable_numbers:
                if verbose: print("Replacing erroneous letter (%s) at index %s with a matched alternative." %(identity[i], i))
                identity = ReplaceLetter(identity, i)
                new_number = identity[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

        if i%2 == 1:
            if character not in acceptable_letters:
                if verbose: print("Replacing erroneous number (%s) with a matched alternative." %identity[i])
                identity = ReplaceNumber(identity, i)
                new_number = identity[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

    if verbose: print("Final identity:", identity)
    
    return identity


def DateErrors(whole_date, verbose=False):
    """Finds errors in the date. The format of the date is mm/dd/yy."""
    if verbose: print("\nLooking for errors in the date (%s)." %whole_date)

    if len(whole_date) != 8 and len(whole_date) != 9:
        if verbose: print("length of the date is incorrect. Current length is %i, required length is eight (8)." %len(whole_date))
        return 1
    
    if len(whole_date) == 8:
        for i, character in enumerate(whole_date):
            if i in [0, 1, 3, 4, 6, 7]:
                if not '0' <= character <= '9':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
            if i in [2, 5]:
                if character != '/':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))

    if len(whole_date) == 9:
            for i, character in enumerate(whole_date):
                if i in [0, 1, 2, 4, 5, 7, 8]:
                    if not '0' <= character <= '9':
                        if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
                if i in [4, 6]:
                    if character != '/':
                        if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
    

    if verbose: print("Final date:", whole_date)
    return whole_date


insert_letter = [['0', 'O'], ['1', 'I'],['2', 'S'], ['3', 'B'], ['4', 'Y'], ['5', 'S'], ['6', 'G'], ['7', 'Y'], ['U', 'V'], ['8', 'F'], ['E', 'F'], ['X', 'K']]
insert_number = [['G', '6'], ['B', '3'], ['S', '5'], ['Y', '7'], ['T', '1'], ['A', '7'], ['Z', '2'], ['E', '8'], ['I', '1'], ['U', '4']]

def ReplaceNumber(identity, i, verbose=False):
    """Replaces the erroneous number at index i with a matched alternative from numbers_to_letters. (e.g.
    '0' goes to 'O', '2' goes to 'S' etc.)"""
    character_error = identity[i]

    for element in insert_letter:
        if character_error in element:
            new_character = element[1]
            split_identity = list(identity)
            split_identity[i] = new_character
            identity = ''.join(split_identity)

    return identity


def ReplaceLetter(identity, i, verbose=False):
    """Similar to above but replaces letters with matched numbers."""
    character_error = identity[i]

    for element in insert_number:
        if character_error in element:
            new_character = element[1]
            split_identity = list(identity)
            split_identity[i] = new_character
            identity = ''.join(split_identity)

    return identity


def preprocess_string(output_string, verbose=False):
    '''
    We noticed that sometimes the identity gets split across two sections.
    '''

    output_split = output_string.split('-')

    # Remove any empty elements.
    output_split[:] = [x for x in output_split if len(x) > 0]

    # First drop anything before the title. 
    if output_split[0][0] != 'F' and output_split[0][0] != 'M':
        output_split = output_split[1:]

    # Removes any characters added to the end of the title. Often A's or X's from people puting stars on the label. e.g. FCAX -> FCA
    if len(output_split[0]) == 4:
        output_split[0] = output_split[0][0:3]

    # Removes lone A's or X's that have been read as a sperate line. Again, these are typically read because 
    # people have starred the label. e.g. FCA-X-3B7Y -> FCA-3B7Y
    for i, output in enumerate(output_split):
        if output == 'X' or output == 'A':
            output_split.pop(i)

    # drop anything after the date. 
    date_element = np.argmax(['/' in i for i in output_split])
    output_split = output_split[:date_element+1]

    # now append everything in the middle together.
    identity = ''
    for i in output_split[1:-1]:
        identity += i

    new_split = [output_split[0], identity, output_split[-1]]

    if verbose: print("String after preprocessing: %s" %new_split)

    return new_split





def FindErrors(output_string, verbose=False):
    """Function to find all the errors associated with the inital output with subfunctions which act to rectify these errors and 
    present the output in a suitable format."""

    if output_string == "":
        if verbose: print("Output string is empty.")
        return 1

    output_string = RemoveSpecialCharacters(output_string, verbose=verbose)

    output_split = preprocess_string(output_string, verbose=verbose)

    # output_split = RemoveDeadElements(output_split, verbose=verbose)

    if len(output_split) != 3:
        if verbose: print("Incorrect number of paragraphs. There are %i, there should be three (3)" %len(output_split))
        return 1
    else:
        if verbose: print("\nCorrect number of paragraphs.")

    title, identity, whole_date = output_split

    title = TitleErrors(title,  verbose=verbose)
    identity = IdentityErrors(identity, verbose=verbose)
    whole_date = DateErrors(whole_date, verbose=verbose)

    if title == 1:
        if verbose: print("Incorrectly labelled the title.")
        return 1
    
    elif identity == 1:
        if verbose: print("Incorrectly labelled the title.")
        return 1
    
    elif whole_date == 1:
        if verbose: print("Incorrectly labelled the date.")
        return 1

    else:
        label = "%s-%s-%s" %(str(title), str(identity), str(whole_date))
        if verbose: print("\nFinal label is:", label)
    
        return label