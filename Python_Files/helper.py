from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np

def CroppedImage(file):
    """Crops the image to only include the label."""
    img = Image.open(file)
    width, height = img.size

    left = 1 * width / 4
    right = 3 * width / 4
    top = 0
    bottom = height / 3
    cropped_image = img.crop((left, top, right, bottom))

    cropstring = file.split('raw')[0] + 'cropped' + file.split('raw')[1]

    cropped_image.save(cropstring)


def ReadImage(cropped_path, verbose = False):


    client = vision.ImageAnnotatorClient()

    with open(cropped_path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image, image_context={"language_hints": ["ko"]})

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

    if verbose:  print('Output:', output_string[:-1].upper(), "\nConfidence:", np.prod(word_confidences))

    return output_string[:-1].upper(), word_confidences





def RemoveSpecialCharacters(output_string,verbose=False):
    """Function removes all special characters that are read by the OCR."""
    if verbose: print("\nRemoving special characters from the output string (e.g. '.', '|').")
    for character in output_string:
        if 'A' <= character <= 'Z' or '0' <= character <= '9' or character == '/' or character == '-':
            if verbose:  print("Character %s is fine." %character)
        else:
            if verbose: print("Character %s has been removed." %character)
            output_string = output_string.replace(character, "")

    return output_string

def RemoveDeadElements(output_split, verbose=False):
    """Function removes paragraphs that are not the right size. These paragraphs are often formed when the 
    OCR reads spetial characters frin the image."""
    if verbose: print("\nRemoving paragraphs without three (3), four (4), five (5) or eight (8) elements.")
    empty = []

    for element in output_split:
        if len(element) == 3 or len(element) == 4 or len(element) == 5 or len(element) == 8:
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
    """Finds errors in the title. The title should be three (3) characters long and contain only english capitcal letters."""
    if verbose: print("\nLooking for errors in the title (%s)." %title)

    if len(title) != 3:
        if verbose: print("Incorrect title length. Title has %i elements, three (3) are required." %len(title))

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

    if len(identity) != 4 and len(identity) != 5:
        if verbose: print("Incorrect identity length. Identity has %i elements, four (4) or five (5) are required." %len(identity))

    for i, character in enumerate(identity):
        if i == 0 or i == 2 or i == 4:
            if character not in acceptable_numbers:
                identity = ReplaceLetter(identity, i)
                new_number = identity[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

        if i == 1 or i == 3:
            if character not in acceptable_letters:
                identity = ReplaceNumber(identity, i)
                new_number = identity[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

    if verbose: print("Final identity:", identity)
    
    return identity

def DateErrors(whole_date, verbose=False):
    """Finds errors in the date. The format of the date is mm/dd/yy."""
    if verbose: print("\nLooking for errors in the date (%s)." %whole_date)

    if len(whole_date) != 8:
        if verbose: print("length of the date is incorrect. Current length is %i, required length is eight (8)." %len(whole_date))
        return 1
    
    for i, character in enumerate(whole_date):
        if i in [0, 1, 3, 4, 6, 7]:
            if not '0' <= character <= '9':
                if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
        if i in [2, 5]:
            if character != '/':
                if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))

    if verbose: print("Final date:", whole_date)
    return whole_date


numbers_to_letters = [['0', 'O'], ['1', 'I'],['2', 'S'], ['3', 'B'], ['4', 'Y'], ['5', 'S'], ['6', 'G'], ['7', 'Y']]
letters_to_numbers = [['G', '6'], ['B', '3'], ['S', '5'], ['Y', '7'], ['T', '1'], ['A', '7'], ['Z', '2']]

def ReplaceNumber(identity, i, verbose=False):
    """Replaces the erroneous number at index i with a matched alternative from numbers_to_letters. (e.g.
    '0' goes to 'O', '2' goes to 'S' etc.)"""
    if verbose: print("Replacing erroneous number (%s) with a matched alternative." %identity[i])

    number_error = identity[i]

    for element in numbers_to_letters:
        if number_error in element:
            new_character = element[1]
            if verbose: print("Replaced %s with %s at index %s" %(number_error, new_character, i))
            split_identity = list(identity)
            split_identity[i] = new_character
            identity = ''.join(split_identity)

    return identity

def ReplaceLetter(identity, i,verbose=False):
    """Similar to above but replaces letters with matched numbers."""
    if verbose: print("Replacing erroneous letter (%s) at index %s with a matched alternative." %(identity[i], i))

    letter_error = identity[i]

    for element in letters_to_numbers:
        if letter_error in element:
            new_character = element[1]
            if verbose:  print("Replacing '%s' with '%s' at index %s" %(letter_error, new_character, i))
            split_identity = list(identity)
            split_identity[i] = new_character
            identity = ''.join(split_identity)

    return identity



def FindErrors(output_string, verbose=False):
    """Function to find all the errors associated with the inital output with subfunctions which act to rectify these errors and 
    present the output in a suitable format."""
    output_string = RemoveSpecialCharacters(output_string, verbose=verbose)

    output_split = output_string.split('-')
    output_split = RemoveDeadElements(output_split, verbose=verbose)

    if len(output_split) != 3:
        if verbose: print("Incorrect number of paragraphs. There are %i, there should be three (3)" %len(output_split))
        return 1
    else:
        if verbose: print("\nCorrect number of paragraphs.")

    title, identity, whole_date = output_split

    title = TitleErrors(title,  verbose=verbose)
    identity = IdentityErrors(identity, verbose=verbose)
    whole_date = DateErrors(whole_date, verbose=verbose)

    label = "%s-%s-%s" %(str(title), str(identity), str(whole_date))
    if verbose: print("\nFinal label is:", label)
    
    return label