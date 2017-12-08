wget https://www.dropbox.com/s/wnivyt2n43q3liv/model.zip?dl=1
unzip model.zip?dl=1
python3 src/train.py --test_path $1 --output $2 --ensemble
