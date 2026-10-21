## Preparing
- Download the codes from this [Google Drive](https://drive.google.com/drive/folders/1-8XzG8mVo8zBJk-1ALeFLr72lj3R_-Pw?usp=sharing).
- Then, the folder structure should be similar to the one below, where the `data` folder is same as the one provided from TAs.

```
|- data
    |- train.jsonl
    |- public.jsonl
|- train.py
```

## Train the mt5-small Model
- Pleace run the following command:
```
bash train.sh mt5
```
- After running the above bash script, the trained model will be stored in a folder called `checkpoints_mt5`, which is shown below. 
- Also, the json file of the evaluation rouge scores (both precision, recall and f1-score) are included in the checkpoint folder, which is shown below.
```
|- ...
|- data
|- train.sh
|- checkpoints_mt5
    |- config.json
    |- generation_config.json
    |- model.safetensors
    |- ...
    |- eval_precisions.json
    |- eval_recalls.json
    |- eval_f1.json
```

## Train the GPT-2 Model
- Pleace run the following command:
```
bash train.sh gpt2
```
- After running the above bash script, the trained model will be stored in a folder called `checkpoints_gpt`, which is similar to the one as the mt5-small model (shown below).

```
|- ...
|- dataset
|- train.sh
|- checkpoints_mt5
    |- config.json
    |- generation_config.json
    |- model.safetensors
    |- ...
```