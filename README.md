
# Proto2Proto [[arxiv](https://arxiv.org/abs/2204.11830)]

### To appear in CVPR 2021

## Create conda Environment

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

## Teacher training

```
sh train_teacher.sh
```
## Baseline student training

Run
```
sh train_baseline.sh
```

## KD student training

- Set teacher path in Experiments/Resnet50_18_cars/kd_Resnet50_18/args.yaml (backbone -> loadPath). Use the best teacher model trained previously

Run
```
sh train_kd.sh
```

## Evaluation

- Set teacher model path in Experiments/Resnet50_18_cars/eval_setting/args.yaml (Teacherbackbone -> loadPath)
- Set baseline model path in Experiments/Resnet50_18_cars/eval_setting/args.yaml (StudentBaselinebackbone -> loadPath)
- Set kd model path in Experiments/Resnet50_18_cars/eval_setting/args.yaml (StudentKDbackbone -> loadPath)

Run
```
sh eval_setting.sh
```

## Things to remember

- Dataset path should be set appropriately
- Model path should be set in KD (1 place) and eval setting (3 places)
- Set CUDA_VISIBLE_DEVICES depending on the GPUs, change batchSize if required