<h2 align="center">Image Classification</h2>

**Data**

1. Download [101_ObjectCategories.tar.gz](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)

2. Unzip 101_ObjectCategories.tar.gz to data/101_ObjectCategories

**Usage**

1. Create directory: `mkdir checkpoints`

2. Train on GPU0: `python3 ./train.py --data data/101_ObjectCategories --model resnet18 --classes 102 --checkpoints checkpoints --batch 32 --workers 8 --gpu 0`

3. Train on GPUs: `python3 ./train.py --data data/101_ObjectCategories --model resnet18 --classes 102 --checkpoints checkpoints --batch 64 --workers 8 --gpus 0,1,2,3`
