## Preparing
- Download the codes from this [Google Drive](https://drive.google.com/drive/folders/1-8XzG8mVo8zBJk-1ALeFLr72lj3R_-Pw?usp=sharing).
- Then, the folder structure should be similar to the one below, where the `data` folder is same as the one provided from TAs.

```
|- data
    |- train.jsonl
    |- public.jsonl
|- train.py
```

## Training
### mt5-small
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

### GPT-2
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

## Inference
- Run the following command to generate summaries with the trained mT5 model:
```
bash run.sh <test_file> <output_file>
```
- `<test_file>` should be a JSONL file containing an `id` and a `maintext` field for each example.
- `<output_file>` is the output JSONL file containing the generated `id` and `title` fields.
- For example:
```
bash run.sh ./data/public.jsonl ./predictions_mt5.jsonl
```