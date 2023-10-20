def DecideID(YOLO_ID, google_ID, verbose=False):
    if verbose: print('\nChoosing between google and YOLO IDs.')
    Allowed_Characters = ['B', 'F', 'G', 'K', 'N', 'O', 'P', 'R', 'S', 'V', 'W', 'Y']
    Allowed_Numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    YOLO_numbers = [x for i, x in enumerate(YOLO_ID) if i%2 == 0]
    YOLO_letters = [x for i, x in enumerate(YOLO_ID) if i%2 == 1] 

    Google_numbers = [x for i, x in enumerate(google_ID) if i%2 == 0]
    Google_letters = [x for i, x in enumerate(google_ID) if i%2 == 1] 

    # Decide which predictions are valid. If both are valid then use YOLO.
    if len(YOLO_ID) >= 4 and (len(YOLO_ID) % 2) == 0 and all(number in Allowed_Numbers for number in YOLO_numbers) and all(letter in Allowed_Characters for letter in YOLO_letters):
        final_ID = YOLO_ID
        if verbose: print('Valid YOLO ID. Final ID will use this:', final_ID)
    elif len(google_ID) >= 4 and (len(google_ID) % 2) == 0 and all(number in Allowed_Numbers for number in Google_numbers) and all(letter in Allowed_Characters for letter in Google_letters):
        final_ID = google_ID
        if verbose: print('Invalid YOLO ID. Final ID will use the google prediction:', final_ID)
    else:
        if verbose: print('Invalid YOLO and google IDs. No final ID.')
        final_ID = ''

    return final_ID