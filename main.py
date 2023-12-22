# Fully parallelised code to run through a Dropbox directory and produce a csv file of guppy label data.
# executed using: `python main.py -d <Dropbox Directory> -f <csv file> -n <Processors> --clean_up'
# e.g. python .\main.py -d Guppy_Images -f ./results.csv -n 12 --clean_up

import time
start_time = time.perf_counter()

import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-d', '--Dropbox_Directory', type=str, default='',
                    help="Path to Dropbox directory containing images to process")

parser.add_argument('-f', '--csv_file_path', type=str, default='./Results.csv',
                    help="Path to the csv file. csv file will either be create or, if already existing, appended to.")

parser.add_argument("-n", "--Processors", type=int, default=os.cpu_count(),
                    help="Max number of processors to utilise.")

parser.add_argument('--clean_up', action='store_true')

args = parser.parse_args()


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import csv
import ultralytics
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
ultralytics.checks()

# Our scripts
from Python_Helpers.image_loading import * 
from Python_Helpers.preprocessing import * 
from Python_Helpers.corrections import *
from Python_Helpers.ID_isolation import *
from Python_Helpers.YOLO import *
from Python_Helpers.analysis import *

# YOLO stuff.
character_mapping = {0:'0',1:'1',12:'2',14:'3',15:'4',16:'5',17:'6',18:'7',19:'8',20:'B',
                    2:'F',3:'G',4:'K',5:'N',6:'O',7:'P',8:'R',9:'S',10:'V',11:'W',13:'Y', 21:''}
model = YOLO(r'YOLO_data/best.pt')

# Dropbox authentication
import dropbox
DropboxDict = LoadDropboxDetails('Dropbox_token.json')
dbx = dropbox.Dropbox(
            app_key = DropboxDict['app_key'],
            app_secret = DropboxDict['app_secret'],
            oauth2_refresh_token = DropboxDict['oauth2_refresh_token']
)
dropbox_file_list = ListDropBoxFiles(dbx, args.Dropbox_Directory)

print('Processing %i files.' %len(dropbox_file_list))

# Create results csv.
if os.path.exists(args.csv_file_path):
    print('csv file path exists. Appending results to file.')
else:
    print('csv file does not exist, creating file and adding results.')
    open(args.csv_file_path, 'a').close()


def MainFunction(image_in_bytes, padx=80, pady=20, delta_width=10, probability_threshold=0.5, verbose=False):
    # Initial google read and preprocessing of image.
    decoded = np.frombuffer(image_in_bytes.getvalue(), dtype=np.uint8)
    initial_image = cv2.imdecode(decoded, cv2.IMREAD_COLOR)

    if verbose: plt.imshow(initial_image)

    Initial_Results = GoogleRead(image_in_bytes, verbose=verbose)
    Processed_Results, title, ID, date = PreprocessLabel(Initial_Results, verbose=verbose)

    # Corrected google prediction.
    title, google_ID, date = CorrectOutput(title, ID, date, verbose=verbose)

    # Identity isolation and YOLO ID results
    ID_Dictionary = IsolateIdentity(Processed_Results, padx=padx, pady=pady, delta_width=delta_width, activate_validity_test=False, verbose=verbose)
    characters, boxes, probs, processed_ID_image = UseYOLO(model, ID_Dictionary['frame'], character_mapping, probability_threshold=probability_threshold, verbose=verbose)
    YOLO_ID = ''.join(characters)

    final_ID = DecideID(YOLO_ID, google_ID, verbose=verbose)

    if verbose:
        print('\nGoogle prediction:', '-'.join([title, google_ID, date]))
        print('YOLO prediction:', '-'.join([title, YOLO_ID, date]))
        print('Final prediction:', '-'.join([title, final_ID, date]))

        plt.imshow(ID_Dictionary['frame'])

    return '-'.join([title, google_ID, date]), '-'.join([title, YOLO_ID, date]), '-'.join([title, final_ID, date]), initial_image, ID_Dictionary['frame'], processed_ID_image

def WriteResult(csv_file, Dropbox_file_name, Google, YOLO, Final):
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        writer.writerow([Dropbox_file_name, Google, YOLO, Final])


file_names = dropbox_file_list
retries = 3

def Parallelised_Function(file_name):
    for n in range(retries):
        try:
            image_in_bytes = RetreiveImageDropBox(dbx, file_name)
            Google_Prediction, YOLO_Prediction, Final_Prediction, initial_image, ID_image, processed_ID_image = MainFunction(image_in_bytes)
            WriteResult(args.csv_file_path, file_name, Google_Prediction, YOLO_Prediction, Final_Prediction)
            break
        except ValueError as value_error:
            print('ValueError. Image unreadable.')
            WriteResult(args.csv_file_path, file_name, '-', '-', '-')
            break
        except Exception as other_error:
            print('Unknown error, retrying.')
            continue
    return

with ThreadPoolExecutor(max_workers=args.Processors) as executor:
    executor.map(Parallelised_Function, file_names)

if args.clean_up:
    print('Cleaning up directory (removing odd saved images).')
    import glob
    for file in glob.glob('./*.jpg'):
        os.remove(file)

end_time = time.perf_counter()
print('\n%i images were processed using' %len(file_names), args.Processors, 'processors and took %.2f' %(end_time - start_time), 'seconds')