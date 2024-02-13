Anonymous codes for KDD-2024 submission:  
**TEXT CAN BE FAIR: Mitigating Popularity Bias with PLMs by Learning Relative Preference**

### STEP 1: Preparing the dataset.
Please keep the following folder structure, download the dataset from the official URL in the paper, and download the pre-trained model from the Huggingface:
```python
-staple.py
-run_staple.py
...
-dataset
  -Metadata
    -meta_Office_Products.json.gz
  -Ratings
    -Office_Products.csv
-plm
  -opt-125m
```

### STEP 2: Preprocess the dataset.
* python preprocess_amazon.py --dataset=Office

### STEP 3: Pre-training the ranker model
* For multi GPUs:
  * nohup python run_staple.py --dataset=Office --num_gpus=6 --batch_size=16 --distributed --multiGPU --valid_ratio=0.1 --train_stage=1 > Office_stage1.log 2>&1 &  
* For single GPU:
  * nohup python run_staple.py --dataset=Office --batch_size=16 --valid_ratio=0.1 --train_stage=1 > Office_stage1.log 2>&1 &

### STEP 4: Fine-tuning the ranker model
* For multi GPUs:
  * nohup python run_staple.py --dataset=Office --num_gpus=6 --batch_size=16 --distributed --multiGPU --valid_ratio=0.1 --train_stage=2 > Office_stage2.log 2>&1 &  
* For single GPU:
  * nohup python run_staple.py --dataset=Office --batch_size=16 --valid_ratio=0.1 --train_stage=2 > Office_stage2.log 2>&1 &  

### STEP 5: Self-distillation
* For multi GPUs:
  * nohup python run_staple.py --dataset=Office --num_gpus=6 --batch_size=16 --distributed --multiGPU --valid_ratio=0.1 --train_stage=3 > Office_stage3.log 2>&1 &  
* For single GPU:
  * nohup python run_staple.py --dataset=Office --batch_size=16 --valid_ratio=0.1 --train_stage=3 > Office_stage3.log 2>&1 &  

### Notes:
* In our experiments, we use 6 gpus and the batch size on each gpu is 16. Thus the total batch size is 96.
* The batch size on each gpu is important in stage 3 because the value of sampled cross-entropy loss is related to the negatives, and we sample (batch_size * 10) negative items for each batch. Changing the batch size may not get the ideal output.
* We have tested the codes for a quick start and reproducing the results reported in the paper. If you have any questions, we will answer them during the Rebattle period.
