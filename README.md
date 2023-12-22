# Guppy OCR
Optical character recognition code for guppy labels. 

## Requirements:
1) Python (>= 3.9). 
2) Google cloud account and project for billing.
3) Guppy Images (Local/DropBox/Google Cloud Storage).

## Setup
1) Python:

Actiavte your python environment and pip install the required packages. This can be simply done using:

`pip install -r requirements.txt`

2) Google Cloud:

To set up a google cloud account, project and SDK, follow the below commands.

- Create a google cloud account and a new project.
- Enable billing and the cloud vision API on this project.
- Download the google cloud SDK shell and for the setup steps [here](https://cloud.google.com/sdk/docs/install-sdk).
- `gcloud auth application-default login` can be used to login via web browser.

3) DropBox:

To set up a DropBox app with a refresh token follow the below commands.

- Create a DropBox account.
- Go to the [DropBox App Console](https://www.dropbox.com/developers/apps).
- Using details from this app, follow the first answer [here](https://stackoverflow.com/questions/70641660/how-do-you-get-and-use-a-refresh-token-for-the-dropbox-api-python-3-x) to generate a refresh token.
- Input these details to the `Dropbox_token.json` file.

## Running

Once python has been setup with the required packages and Google has been authenticated, the code can be used. Images of labelled guppy images can be read locally and remotely (via google-cloud-storage or in Dropbox). For DropBox compatibility, section 3 in the setup above must be done. To individually process images the jupyter notebook `main.ipynb` can be used and is helpful for diagnosing any issues. To process images in parallel, the `main.py` script can be used. The python script can be executed using the following command:

`python main.py -d <Dropbox Directory> -f <csv file> -n <Processors> --clean_up`

By default, the script will read all files stored in the Dropbox, write results to a `Results.csv` file and run on the maximum number of avaliable processers. If `--clean_up` is not used, some rogue jpeg files will be generated in the working directory. If `--clean_up` is used, all jpg's in the working directy will be deleted when the script ends, so use with caution. 