# CLORE
The public repository for model CLORE in our ACL Findings 2023 paper ["Zero-Shot Classification by Logical Reasoning on Natural Language Explanations"](https://arxiv.org/abs/2211.03252).


## Requirements

- torch >= 1.9.0
- transformers >= 4.9.0
- scikit-learn >= 0.24.1
- tqdm >= 4.61.1
- numpy >= 1.20.1
- datasets >= 1.18.1
- nltk >= 3.5
- scipy >= 1.6.3

If you want to run on CLIP backbone, you also need to install [clip](https://github.com/openai/CLIP) package.


## Datasets and Resources

In this work we experimented on two benchmarks:

- [CLUES](https://clues-benchmark.github.io), which tests zero-shot classifier learning using natural language explanations. CLUES consists of 36 real-world (CLUES-Real) and 144 synthetic (CLUES-Synthetic) classification tasks, each task coming with ~10 task label explanations.

- CUB-Explanations, where we collected [explanations for each bird species](guidelines/CUB/definitions.txt) to augment the widely studied [CUB-200-2011](https://paperswithcode.com/dataset/cub-200-2011) dataset.



## Running Experiments


### CLUES Dataset


First, follow the [dataset download](https://cs.unc.edu/~rrmenon/data/clues.tar.gz) link described in the [benchmarkd webpage](https://clues-benchmark.github.io) to your local disk, say `./clues_data/`. You will refer to the folder as `CLUES_DATA` variable for the following commands.


Then, command for running CLORE model on CLUES dataset:


```
CLUES_DATA=./clues_data/
LOG_DIR=./logs/
TRIAL_ID=clore_clues
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/clues.py \
    --output_dir ${LOG_DIR}/${TRIAL_ID} --data_dir $CLUES_DATA/data/ --subtask clues_real \
    --model_name_or_path bert-base-uncased --num_train_epochs 15 --learning_rate 1e-5 --seed 1 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_accumulation_steps 1 \
    --guided_model main --zero_shot --individual_scores --normalize_datasets \
    --guideline_feature_layer_num_head 1 --encoder_feature_layer_num_head 1 --num_feature_layer 1 --batch_per_epoch 100
```

To re-run our experiment which pre-trains on synthetic tasks and fine-tunes on real tasks:

```
CLUES_DATA=./clues_data/
LOG_DIR=./logs/
TRIAL_ID=clore_clues_pretrained
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/clues.py \
    --output_dir ${LOG_DIR}/${TRIAL_ID} --data_dir $CLUES_DATA/data/ --subtask clues_syn \
    --model_name_or_path bert-base-uncased --num_train_epochs 1 --learning_rate 1e-5 --seed 1 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_accumulation_steps 1 \
    --guided_model main --normalize_datasets \
    --guideline_feature_layer_num_head 1 --encoder_feature_layer_num_head 1 --num_feature_layer 1 --batch_per_epoch 10
PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 python experiments/clues.py \
    --output_dir $LOG_DIR/${TRIAL_ID} --data_dir $CLUES_DATA/data/ --subtask clues_real \
    --model_name_or_path bert-base-uncased --num_train_epochs 16 --learning_rate 1e-5 --seed 1 \
    --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --gradient_accumulation_steps 1 --eval_accumulation_steps 1 \
    --guided_model main --zero_shot --individual_scores --normalize_datasets \
    --guideline_feature_layer_num_head 1 --encoder_feature_layer_num_head 1 --num_feature_layer 1 --batch_per_epoch 100
```



## CUB-Definitions

First, download the dataset from [CUB_200_2011 website](http://www.vision.caltech.edu/datasets/cub_200_2011/) to your local disk.
Again you can use a variable `CUB_DATA` to refer to it.

Then in this work we follow the data split used in [TF-VAEGAN](https://github.com/akshitac8/tfvaegan), where a `att_splits.mat` file speficies the trainint, validation and test set.

To run CLORE model on CUB-Definitions:

```
CUB_DATA=./cub_data/
LOG_DIR=./logs/
TRIAL_ID=clore_cub
SPLIT_FILE=guidelines/CUB/att_splits.mat
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/cub_definitions.py \
    --output_dir $LOG_DIR/${TRIAL_ID} --data_dir $CUB_DATA/images \
    --split_file $SPLIT_FILE \
    --definition_file guidelines/CUB/definitions.txt \
    --class_names_file guidelines/CUB/CUB_classes.txt \
    --model_name_or_path bert-base-uncased --vision_model IMAGENET1K_V2 --num_train_epochs 10 --learning_rate 1e-4 --seed 1 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
    --phase fix_bert --do_eval --do_test \
    --guideline_feature_layer_num_head 4 --encoder_feature_layer_num_head 4
```


To use the CLIP backbone, you need to first install the [CLIP package](https://github.com/openai/CLIP). Then run the following two-stage training command:

```
CUB_DATA=./cub_data/
LOG_DIR=./logs/
TRIAL_ID=clore_cub_clip
SPLIT_FILE=guidelines/CUB/att_splits.mat
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/cub_definitions.py \
    --output_dir $LOG_DIR/${TRIAL_ID} --data_dir $CUB_DATA/images \
    --split_file $SPLIT_FILE \
    --definition_file guidelines/CUB/definitions.txt \
    --class_names_file guidelines/CUB/CUB_classes.txt \
    --model_name_or_path clip --vision_model clip --num_train_epochs 25 --learning_rate 1e-3 --seed 1 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
    --phase fix_both --program_pooler max --do_eval --do_test \
    --guideline_feature_layer None --encoder_feature_layer None

PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python experiments/cub_definitions.py \
    --output_dir $LOG_DIR/${TRIAL_ID} --data_dir $CUB_DATA/images \
    --split_file $SPLIT_FILE \
    --definition_file guidelines/CUB/definitions.txt \
    --class_names_file guidelines/CUB/CUB_classes.txt \
    --model_name_or_path clip --vision_model clip --num_train_epochs 35 --learning_rate 1e-6 --seed 1 \
    --per_device_train_batch_size 32 --per_device_eval_batch_size 64 \
    --phase fix_visual --program_pooler max --do_eval --do_test --do_not_load_optimizer \
    --guideline_feature_layer None --encoder_feature_layer None
```

## Citation

To cite our paper, please use the following BibTeX:

```
@article{han2022zero,
  title={Zero-Shot Classification by Logical Reasoning on Natural Language Explanations},
  author={Han, Chi and Pei, Hengzhi and Du, Xinya and Ji, Heng},
  journal={arXiv preprint arXiv:2211.03252},
  year={2022}
}
```
