
# Proto2Proto [[arxiv](https://arxiv.org/abs/2204.11830)|poster]

### To appear in CVPR 2021

## Creating conda Environment

```
conda env create -f environment.yml -n myenv python=3.6
conda activate myenv
```

## Preparing Dataset

- Refer https://github.com/M-Nauta/ProtoTree to download and preprocess cars dataset
- For augmentation, run lib/protopnet/cars_augment.py (Change the dataset paths if required)
- Create a symbolic link to the dataset folder as datasets
- We need the dataset paths as follows

```
  trainDir: datasets/cars/train_augmented # Path-to-dataset
  projectDir: datasets/cars/train # Path-to-dataset
  testDir: datasets/cars/test # Path-to-dataset
```

## Training

```
sh train_teacher.sh  # For teacher training
sh train_baseline.sh # For baseline student training
sh train_kd.sh       # For proto2proto student training
```
**_NOTE:_** For proto2proto student training, set the teacher path in Experiments/Resnet50_18_cars/kd_Resnet50_18/args.yaml, backbone.loadPath. Use the teacher model trained previously. For eg. 
```
loadPath: Experiments/Resnet50_18_cars/teacher_Resnet50/org/models/protopnet_xyz.pth
```

## Evaluation

Set model paths in Experiments/Resnet50_18_cars/eval_setting/args.yaml: Teacherbackbone.loadPath, StudentBaselinebackbone.loadPath), StudentKDbackbone.loadPath. And Run

```
sh eval_setting.sh
```

## Things to remember

- Dataset path should be set appropriately
- Model path should be set in KD (1 place) and eval setting (3 places)
- Set CUDA_VISIBLE_DEVICES depending on the GPUs, change batchSize if required