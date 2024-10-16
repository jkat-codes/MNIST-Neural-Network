import zipfile
with zipfile.ZipFile("D:\\Projects\\Python\\AI\\MNIST\\data\\digit-recognizer.zip", 'r') as zip_ref:
    zip_ref.extractall("D:\\Projects\\Python\\AI\\MNIST\\data")
