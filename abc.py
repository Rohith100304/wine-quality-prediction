import zipfile
import os

# Only extract if not already extracted
if not os.path.exists("wine.pkl"):
    with zipfile.ZipFile("wine.zip", "r") as zip_ref:
        zip_ref.extractall()  # Extracts all files into the current directory
