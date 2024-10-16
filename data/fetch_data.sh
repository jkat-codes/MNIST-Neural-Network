kaggle competitions download -c digit-recognizer -p "D:\Projects\Python\AI\MNIST\data" 
echo "Dataset downloaded"
python data/unzip.py
echo "File unzipped"
rm -rf data/digit-recognizer.zip
echo "Zip file removed"