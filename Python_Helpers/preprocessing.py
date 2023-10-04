def PreprocessLabel(Image_Dictionary, verbose=False):
    if verbose: print('\nThe initial character list is', Image_Dictionary['characters'])

    # Remove special characters and change '-' to '/' (sometimes dates are written 
    # 04-05-12 instead of 04/05/12)
    Image_Dictionary = ReplaceSpecialCharacter(Image_Dictionary, verbose=verbose)

    # Remove characters before the title (title starting with 'F' or 'M').
    Image_Dictionary = RemoveBeginning(Image_Dictionary, verbose=verbose)

    # Reduce title to three (3) characters.
    Image_Dictionary = ReduceTitle(Image_Dictionary, verbose=verbose)

    # Remove any stars that typically trail the title.
    Image_Dictionary = RemoveStars(Image_Dictionary, verbose=verbose)

    # Remove everything after the date.
    Image_Dictionary = RemoveEnd(Image_Dictionary, verbose=verbose)

    # Modify the character parameters to ensure the title, ID and date are contained
    # in single, seperate words.
    # Title characters have parameters [0, 0, 0, 0, i]
    # ID characters have parameters [0, 0, 0, 1, i]
    # Date characters have parameters [0, 0, 0, 2, i]
    Image_Dictionary = ModifyParameters(Image_Dictionary, verbose=verbose)

    title, ID, date = IndentifyWords(Image_Dictionary, verbose=verbose)

    return Image_Dictionary, title, ID, date


def ReplaceSpecialCharacter(Image_Dictionary, verbose=False):
    character_replacements = [['(', '1'], ['\\', '1'], ['√', 'V'], ['-', '/']]

    if verbose: print("\nReplacing/removing special characters.")

    for i, character in enumerate(Image_Dictionary['characters']):
        if character in [x[0] for x in character_replacements]:
            replacement_index = [x[0] for x in character_replacements].index(character)
            replacement_character = character_replacements[replacement_index][1]
            Image_Dictionary['characters'][i] = replacement_character
            if verbose: print("Replaced '" + character + "' in index", i, "with '" + replacement_character + "'")


    keep_indices = []
    for i, character in enumerate(Image_Dictionary['characters']):
        if 'A' <= character <= 'Z' or '0' <= character <= '9' or character == '/' or character == '-':
            if verbose:  print("Character %s is fine." %character)
            keep_indices.append(i)
        elif 'a' <= character <= 'z':
            if verbose: print("Converting character", character, "at index", i, "to upper case.")
            Image_Dictionary['characters'][i] = Image_Dictionary['characters'][i].upper()
            keep_indices.append(i)
        else:
            if verbose: print("Character %s at index %i has been removed." %(character, i))

    Image_Dictionary['characters'] = [Image_Dictionary['characters'][i] for i in keep_indices]
    Image_Dictionary['character_params'] = [Image_Dictionary['character_params'][i] for i in keep_indices]
    Image_Dictionary['character_bounds'] = [Image_Dictionary['character_bounds'][i] for i in keep_indices]
    Image_Dictionary['character_confidences'] = [Image_Dictionary['character_confidences'][i] for i in keep_indices]
            
    return Image_Dictionary
    

def RemoveBeginning(Image_Dictionary, verbose=False):
    """Removes everything before the start of the label ('F' or 'M')."""

    if verbose: print("\nRemoving all characters before 'F' and 'M'.")

    # If not enough characters have initially been read then return error
    if len(Image_Dictionary['characters']) < 7:
        if verbose: print("Not enough characters are in the label.")
        raise ValueError("Less than 7 characters in label.", 'Label:', Image_Dictionary['characters'])

    if 'M' in Image_Dictionary['characters'] or 'F' in Image_Dictionary['characters']:
        lead_character = Image_Dictionary['characters'][0]
        while lead_character != 'F' and lead_character != 'M':
            if verbose: print('\nRemoving everything before F and M')
            if verbose: print('removing character', Image_Dictionary['characters'])

            Image_Dictionary['characters'] = Image_Dictionary['characters'][1:]
            Image_Dictionary['character_params'] = Image_Dictionary['character_params'][1:]
            Image_Dictionary['character_bounds'] = Image_Dictionary['character_bounds'][1:]
            Image_Dictionary['character_confidences'] = Image_Dictionary['character_confidences'][1:]
            lead_character = Image_Dictionary['characters'][0]

            if verbose: print('Reduced characters are now:', Image_Dictionary['characters'])
            if verbose: print('Lead character is now:', lead_character)

            # If we drop below 7 characters then end.
            if len(Image_Dictionary['characters']) < 7:
                if verbose: print("Not enough characters to continue. Breaking the cycle.")
                raise ValueError("Less than 7 characters in label.", 'Label:', Image_Dictionary['characters'])
    
    # If there is no 'M' or 'F' in the character list then can't identify the title. 
    # Return error.
    else:
        if verbose: print("No 'M' or 'F' in the character list. Can't identify the title.")
        raise ValueError("No 'M' or 'F' in the character list.", 'List:', Image_Dictionary['characters'])

    if verbose: print('The remaining character list', Image_Dictionary['characters'])
            
    return Image_Dictionary


def ReduceTitle(Image_Dictionary, verbose=False):
    # Check first character is 'M' or 'F'.
    if verbose: print("\nReducing the title.")

    if Image_Dictionary['characters'][0] != 'M' and Image_Dictionary['characters'][0] != 'F':
        if verbose: print("Starting character is not 'M' or 'F', make sure to remove \
                          all elements before the title (RemoveBeginning funciton).")
        raise ValueError("No 'M' or 'F' at the start of the character list.")
    
    # Identify the title word.
    title_word_parameters = Image_Dictionary['character_params'][0][:-1]

    # Add all characters in the same word to the title.
    title = []
    for i, params in enumerate(Image_Dictionary['character_params']):
        if params[:-1] == title_word_parameters:
            title.append(Image_Dictionary['characters'][i])

    if len(title) < 3:
        if verbose: 
            print('The title is', ''.join(title))
            print('Too few characters in the title, there should be three (3).')
            raise ValueError("Title is too short.", 'Title:', title)


    if len(title) > 3:
        if verbose: 
            print('The title is', ''.join(title))
            print('Too many characters in the title word, retaining only the first three (3).')
            
        Image_Dictionary['characters'] = Image_Dictionary['characters'][:3] + Image_Dictionary['characters'][len(title):]
        Image_Dictionary['character_params'] = Image_Dictionary['character_params'][:3] + Image_Dictionary['character_params'][len(title):]
        Image_Dictionary['character_bounds'] = Image_Dictionary['character_bounds'][:3] + Image_Dictionary['character_bounds'][len(title):]
        Image_Dictionary['character_confidences'] = Image_Dictionary['character_confidences'][:3] + Image_Dictionary['character_confidences'][len(title):]
        title = title[:3]

    if verbose: 
        print('The final title is', ''.join(title))
        print('The remaining characters are', Image_Dictionary['characters'])

    return Image_Dictionary


def RemoveStars(Image_Dictionary, verbose=False):
    stars = ['X', 'A', '*', '•', 'x', '☆']

    if Image_Dictionary['characters'][3] in stars:
        if verbose: 
            print("\nRemoving any stars trailing the title.")
            print('Removing star trailing the title.')
        Image_Dictionary['characters'] = Image_Dictionary['characters'][:3] + Image_Dictionary['characters'][4:]
        Image_Dictionary['character_params'] = Image_Dictionary['character_params'][:3] + Image_Dictionary['character_params'][4:]
        Image_Dictionary['character_bounds'] = Image_Dictionary['character_bounds'][:3] + Image_Dictionary['character_bounds'][4:]
        Image_Dictionary['character_confidences'] = Image_Dictionary['character_confidences'][:3] + Image_Dictionary['character_confidences'][4:]

        if verbose: print('Remaining characters are now', Image_Dictionary['characters'])

    return Image_Dictionary


def RemoveEnd(Image_Dictionary, verbose=False):
    if verbose: print('\nRemoving everything after the date.')

    if '/' not in Image_Dictionary['characters']:
        if verbose: 
            print("No '/' or '-' characters have been read from the label.")
            print("Can't identify the date.")
            raise ValueError("Can't identify the date, no'/'.", 'Characters:', Image_Dictionary['characters'])

    joined_characters = ''.join(Image_Dictionary['characters'])

    final_date_index = joined_characters.rfind('/') + 3

    Image_Dictionary['characters'] = Image_Dictionary['characters'][:final_date_index]
    Image_Dictionary['character_params'] = Image_Dictionary['character_params'][:final_date_index]
    Image_Dictionary['character_bounds'] = Image_Dictionary['character_bounds'][:final_date_index]
    Image_Dictionary['character_confidences'] = Image_Dictionary['character_confidences'][:final_date_index]

    if verbose: print('Remaining characters are now', Image_Dictionary['characters'])

    return Image_Dictionary


def ModifyParameters(Image_Dictionary, verbose=False):

    # Identify the title word.
    title_word_parameters = Image_Dictionary['character_params'][0][:-1]

    # Identify the date word.
    date_word_parameters = Image_Dictionary['character_params'][-1][:-1]

    for i, params in enumerate(Image_Dictionary['character_params']):
        if params[:-1] == title_word_parameters:
            Image_Dictionary['character_params'][i] = [0, 0, 0, 0, i]

        elif params[:-1] == date_word_parameters:
            Image_Dictionary['character_params'][i] = [0, 0, 0, 2, i]

        else:
            Image_Dictionary['character_params'][i] = [0, 0, 0, 1, i]

    return Image_Dictionary


def IndentifyWords(Image_Dictionary, verbose=False):

    title = ''
    ID = ''
    date = ''

    for i, param in enumerate(Image_Dictionary['character_params']):
        if param[-2] == 0:
            title += Image_Dictionary['characters'][i]
        
        if param[-2] == 1:
            ID += Image_Dictionary['characters'][i]

        if param[-2] == 2:
            date += Image_Dictionary['characters'][i]

    if len(title) != 3:
        if verbose: print('\nTitle:', title)
        raise ValueError("Incorrect title length.", 'Title:', title)
    
    if len(date) < 5 or len(date) > 9:
        if verbose: print('\nDate:', date)
        raise ValueError("Date is the incorrect length.", 'Date', date)

    return title, ID, date