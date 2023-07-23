from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os
import csv

from corrections import *
from image import * 

def ListAvaliableFiles(bucket_name, verbose=False):
    """Lists all avaliable files in the bucket. Useful for cycling through all files in a loop."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    file_list = storage_client.list_blobs(bucket_name)
    file_list = [file.name for file in file_list]
    if verbose: print("\nFiles have been read.")
    return file_list


def CorrectedLabel(file, storage, verbose=False):
    if storage == 'local':
        iamge = LoadImage(file, verbose=verbose)
    
    elif storage == 'remote':
        image = RetreiveImage(file, verbose=verbose)

    else:
        print("Invalid storage type. Storage type should either be 'local' if stored \
              on a local pc or 'remote' if stored on the google cloud.")
        
    cropped_image = CroppedImage(image, verbose=verbose)
    output_string, word_confidences = ReadImage(cropped_image, verbose=verbose)
    label = FindErrors(output_string, verbose=verbose)
    
    print("\n   Initial label:", output_string,
        "\nCorrected label:", label)
    
    return image, output_string, label, word_confidences


def ReanalysePredictions(prediction_file, verbose=False):
    with open(f'../Data/{prediction_file}_new.csv', 'w') as f_new:
        with open(f'../Data/{prediction_file}.csv', 'r') as f:
            filereader = csv.reader(f)
            writer = csv.writer(f_new)
            for n, row in enumerate(filereader):
                if row != []:
                    new_row = row[:2]

                    new_prediction = FindErrors(row[1])

                    new_row.append(new_prediction)
                    new_row.append(row[3])

                    writer.writerow(new_row)

    if verbose: print("Reanalysed predictions")


def AccuracyCheck(truth_file, verbose=False):
    correct_files = []
    incorrect_files = []
    invalid_files = []

    character_confusions = []

    with open(truth_file, 'r') as f:
        filereader = csv.reader(f)
        for n, row in enumerate(filereader):
            if row != []:
                correct = int(row[3] == row[2])
                if correct:
                    correct_files.append(row[0])

                if not correct:
                    if row[2] == '1':
                        invalid_files.append(row[0])
                        if verbose: print(n + 1, "Invalid")

                    else: 
                        true = row[3].split('-')
                        pred = row[2].split('-')

                        incorrect_files.append(row[0])

                        if len(true) != 3 or len(pred) != 3:
                            if verbose: print(n, 'error')
                            continue

                        for i in range(3):
                            if true[i] != pred[i]:
                                if verbose: print(n + 1, true[i], pred[i])

                                if (i == 1) and (len(true[i]) == len(pred[i])): #mistake in identity.
                                    for j in range(len(true[i])):
                                        if true[i][j] != pred[i][j]:
                                            character_confusions.append((true[i][j], pred[i][j]))




    print("\nNumber Correct:", len(correct_files), 
        "\nNumber Incorrect:", len(incorrect_files),
        "\nNumber Invalid:", len(invalid_files))

    character_confusions = np.array(character_confusions)

    print("Attempted % Correct:", len(correct_files) / (len(correct_files) + len(incorrect_files)))
    print("Total % Correct:", len(correct_files) / (len(correct_files) + len(incorrect_files) + len(invalid_files)))

    return correct_files, incorrect_files, invalid_files, character_confusions