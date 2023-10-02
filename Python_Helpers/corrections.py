def CorrectOutput(title, ID, date, verbose=False):
    ID = CorrectID(ID, verbose=verbose)

    date = CorrectDate(date, verbose=verbose)

    return title, ID, date

def ReplaceNumber(ID, i, verbose=False):
    """Replaces the erroneous number at index i with a matched alternative from numbers_to_letters. (e.g.
    '0' goes to 'O', '2' goes to 'S' etc.)"""

    insert_letter = [['0', 'O'], ['1', 'K'],['2', 'S'], ['3', 'B'], ['4', 'Y'], ['5', 'S'], ['6', 'G'], ['7', 'Y'], ['8', 'F'], \
                 ['U', 'V'], ['E', 'F'], ['X', 'K'], ['I', 'K'], ['C', 'G']]

    character_error = ID[i]

    for element in insert_letter:
        if character_error in element:
            new_character = element[1]
            split_ID = list(ID)
            split_ID[i] = new_character
            ID = ''.join(split_ID)

    return ID


def ReplaceLetter(ID, i, verbose=False):
    """Similar to above but replaces letters with matched numbers."""

    insert_number = [['G', '6'], ['B', '3'], ['S', '5'], ['Y', '7'], ['T', '1'], ['A', '7'], ['Z', '2'], ['E', '8'], ['I', '1'], \
                 ['U', '4'], ['Q', '2'], ['J', '1'], ['H', '4'], ['9', '4'], ['/', '1'], ['L', '6']]

    character_error = ID[i]

    for element in insert_number:
        if character_error in element:
            new_character = element[1]
            split_ID = list(ID)
            split_ID[i] = new_character
            ID = ''.join(split_ID)

    return ID


def CorrectID(ID, verbose=False):
    """Finds errors in the ID. The identy should be four (4) or five (5) characters long and should follow the
    pattern [number, letter, number, letter number*] with the final number being optional. Any other format is an error."""
    if verbose: print("\nLooking for errors in the ID (%s)." %ID)

    acceptable_numbers = ['1', '2', '3', '4', '5', '6', '7', '8']
    acceptable_letters = ['B', 'F', 'G', 'K', 'N', 'O', 'P', 'R', 'S', 'V', 'W', 'Y']

    if len(ID) < 4:
        if verbose: print("Incorrect ID length. ID has %i elements, must be more than four (4)." %len(ID))
        return 1

    for i, character in enumerate(ID):
        if i%2 == 0:
            if character not in acceptable_numbers:
                if verbose: print("Replacing erroneous letter (%s) at index %s with a matched alternative." %(ID[i], i))
                ID = ReplaceLetter(ID, i)
                new_number = ID[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

        if i%2 == 1:
            if character not in acceptable_letters:
                if verbose: print("Replacing erroneous number (%s) with a matched alternative." %ID[i])
                ID = ReplaceNumber(ID, i)
                
                new_number = ID[i]
                if verbose: print("Character %s at index %s has been replace with %s" %(character, i, new_number))

    if verbose: print("Final ID:", ID)
    
    return ID


def CorrectDate(date, verbose=False):
    """Finds errors in the date."""
    if verbose: print("\nLooking for errors in the date (%s)." %date)

    if len(date) != 6 and len(date) != 7 and len(date) != 8 and len(date) != 9:
        if verbose: print("length of the date is incorrect. Current length is %i, required length is six (6), seven (7), eight (8) or nine (9)." %len(date))
        return 1
    
    if len(date) == 6:
        for i, character in enumerate(date):
            if i in [0, 2, 4, 5]:
                if not '0' <= character <= '9':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
            if i in [1, 3]:
                if character != '/':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))

    if len(date) == 7:
        for i, character in enumerate(date):
            if i in [0, 2, 3, 5, 6]:
                if not '0' <= character <= '9':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
            if i in [1, 4]:
                if character != '/':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
    
    if len(date) == 8:
        for i, character in enumerate(date):
            if i in [0, 1, 3, 4, 6, 7]:
                if not '0' <= character <= '9':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
            if i in [2, 5]:
                if character != '/':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))

    if len(date) == 9:
        for i, character in enumerate(date):
            if i in [0, 1, 2, 4, 5, 7, 8]:
                if not '0' <= character <= '9':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))
            if i in [4, 6]:
                if character != '/':
                    if verbose: print("Invalid date. Date contains %s at index %i" %(character, i))

    split_date = date.split('/')
    for i, element in enumerate(split_date):
        if len(element) == 1:
            split_date[i] = '0' + element
        if len(element) == 3:
            split_date[i] = element[1:]

    date = '/'.join(split_date)

    if verbose: print("Final date:", date)
    return date