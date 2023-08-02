from google.cloud import vision
from google.cloud import storage
from PIL import Image
import numpy as np
from io import BytesIO
import os
import csv
from tempfile import NamedTemporaryFile
import shutil

from corrections import *
from image import * 

def ListAvaliableFiles(bucket_name, prefix, verbose=False):
    """Lists all avaliable files in the bucket. Useful for cycling through all files in a loop."""
    storage_client = storage.Client()
    file_list = storage_client.list_blobs(bucket_name, prefix=prefix)
    file_list = [file.name for file in file_list]
    if verbose: print("\nFiles have been read.")
    return file_list


def CorrectedLabel(file, storage_type='remote', verbose=False):
    """Reads the image (either locally or remotely) and returns the initially read and the corrected label."""
    if storage_type == 'local':
        image = LoadImage(file, verbose=verbose)
    
    elif storage_type == 'remote':
        image = RetreiveImage(file, verbose=verbose)

    else:
        print("Invalid storage type. Storage type should either be 'local' if stored \
              on a local pc or 'remote' if stored on the google cloud.")
        
    cropped_image = CroppedImage(image, verbose=verbose)
    output_string, word_confidences = ReadImage(cropped_image, verbose=verbose)
    label = FindErrors(output_string, verbose=verbose)
    
    print("\nInitial label:", output_string,
        "\nCorrected label:", label)
    
    return image, output_string, label, word_confidences


def ReanalysePredictions(prediction_file, verbose=False):
    """Updates the prediction file with any edits we have made to the correction code."""
    filename = prediction_file
    tempfile = NamedTemporaryFile('w+t', newline='', delete=False)
    with open(filename, 'r', newline='') as csvFile, tempfile:
        reader = csv.reader(csvFile, delimiter=',', quotechar='"')
        writer = csv.writer(tempfile, delimiter=',', quotechar='"')
        for n, row in enumerate(reader):
            if row != []:
                new_row = row[:2]

                new_prediction = FindErrors(row[1])

                new_row.append(new_prediction)
                new_row.append(row[3])

                writer.writerow(new_row)

    shutil.move(tempfile.name, filename)

    if verbose: print("Reanalysed predictions")


def AccuracyCheck(truth_file, verbose=False):
    """Compares the corrected predictions to the truth and determines if the prediction is correct, incorrect (attempted and wrong)
    or invalide (not attempted). Characters that were confused are added to the character_confusions list."""
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


def ConfusionMatrix(character_confusions):
    true_chars = [X[0] for X in character_confusions]
    true_chars_unique = np.unique(true_chars)
    pred_chars = [X[1] for X in character_confusions]
    pred_chars_unique = np.unique(pred_chars)

    confusion_matrix =  np.zeros((len(true_chars_unique), len(pred_chars_unique)))

    for i in range(len(true_chars_unique)):
        for j in range(len(pred_chars_unique)):
            confusion_matrix[i,j] = np.sum((character_confusions[:,0] == true_chars_unique[i]) & (character_confusions[:,1] == pred_chars_unique[j]))

    return true_chars_unique, pred_chars_unique, confusion_matrix


def TruthFromFileName(filename):
    """Return truth from a manually labelled filename."""
    filename = filename.replace(".JPG", "")
    filename = filename.replace("(1)", "")
    filename = filename.split("/")[-1]

    title = filename[0:3]
    identity = filename[3:-4]
    date = filename[-4:]

    label = title + "-" + identity + "-" + date[0:2] + "/" + date[-2:]

    return label