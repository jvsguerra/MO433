import os
import gdown
from zipfile import ZipFile


def download_data():
    # Download MSRA-B dataset
    url = "https://drive.google.com/u/1/uc?id=1XodzMAtVKkNnCkYwdzaCmW2xDPx3HKlr"
    path = "./input/dataset/"
    filename = path + "MSRA-B.zip"

    if not os.path.isfile(filename):
        gdown.download(url, filename, quiet=False)
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path)
    
    # Download unsupervised labels
    url = "https://drive.google.com/u/1/uc?id=1iQygYQuLTvKd7ZDWQpoPRCG1mKzTKjsK"
    path = "./input/"
    filename = path + "unsup_labels.zip"

    if not os.path.isfile(filename):
        gdown.download(url, filename, quiet=False)
        with ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(path)


def unzip_results():
    # Unzip results 
    path = "./input/"
    filename = path + "results.zip"

    with ZipFile(filename, 'r') as zip_ref:
        print("Results extracted: ")
        zip_ref.printdir()
        zip_ref.extractall(path)


def download_models():
    # Download 20 epoch models
    url = "https://drive.google.com/u/1/uc?id=1U9lDd2lcR53oHOQaLuI9Zi2m7ej9v1ET"
    path = "./input/"
    filename = path + "models.zip"

    if not os.path.isfile(filename):
        gdown.download(url, filename, quiet=False)
        with ZipFile(filename, 'r') as zip_ref:
            print("Models extracted: ")
            zip_ref.printdir()
            zip_ref.extractall(path)

if __name__ == '__main__':
    download_data()
