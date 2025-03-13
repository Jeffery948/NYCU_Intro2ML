# Requirement
You should first enter the conda virtual environment(my python version is 3.9) and install the required packages in requirements.txt one by one. Or you can just run the following:
```
pip install -r requirements.txt
```
And for the GPU utility in pytorch, I use the following command to build:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
You can modify this to your version

# For training
If you want to run the training code:
```
python3 train.py
```

# Reproduce my kaggle submission file
First download my four model weights for bagging, and place model directory(which contains four weights) at the same directory with inference.py. Run the following:
```
python3 inference.py
```
And you can get my kaggle best public submission file