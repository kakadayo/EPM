## Setting
torch=1.12 && python=3.7
## Training and evaluation
#First training model on the source data.

python train_src.py

#Then adapting source model to target domain, with only the unlabeled target data.

python train_tar.py

