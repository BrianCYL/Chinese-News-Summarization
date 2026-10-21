import torch
import pandas as pd
import json
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader
import argparse
import nltk

class Summarizer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path, trust_remote_code=True)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.args.model_path, trust_remote_code=True).to(self.DEVICE)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True, return_tensors="pt")

    def preprocess(self, examples):
        inputs = [f"summarize: {ex.lstrip()}" for ex in examples['maintext']]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_input_length, truncation=True)
        return model_inputs
    
    def postprocess(self, preds):
        preds = [pred.strip() for pred in preds]
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        return preds
    
    def load_data(self):
        dataset = load_dataset('json', data_files=self.args.test_file_dir)
        dataset = dataset.map(self.preprocess, batched=True, remove_columns=dataset["train"].column_names)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.test_dataloader = DataLoader(
            dataset["train"],
            shuffle=False,
            collate_fn=self.data_collator,
            batch_size=self.args.test_batch_size,
        )

    def generate_summary(self, batch):
        batch['input_ids'] = batch['input_ids'].to(self.DEVICE)
        batch['attention_mask'] = batch['attention_mask'].to(self.DEVICE)
        if self.args.type == 'greedy':
            generated_tokens = self.model.generate(input_ids=batch['input_ids'], 
                                                   attention_mask=batch['attention_mask'], 
                                                   max_length=self.args.max_out_length,
                                                   do_sample=False)
        elif self.args.type == 'beam':
            generated_tokens = self.model.generate(input_ids=batch['input_ids'], 
                                                   attention_mask=batch['attention_mask'], 
                                                   max_length=self.args.max_out_length, 
                                                   num_beams=self.args.num_beams, 
                                                   temperature=self.args.temp, 
                                                   early_stopping=True)
        elif self.args.type == 'topk':
            generated_tokens = self.model.generate(input_ids=batch['input_ids'], 
                                                   attention_mask=batch['attention_mask'], 
                                                   max_length=self.args.max_out_length, 
                                                   top_k=self.args.top_k, 
                                                   temperature=self.args.temp, 
                                                   do_sample=True)
        elif self.args.type == 'topp':
            generated_tokens = self.model.generate(input_ids=batch['input_ids'], 
                                                   attention_mask=batch['attention_mask'], 
                                                   max_length=self.args.max_out_length, 
                                                   top_p=self.args.top_p, 
                                                   temperature=self.args.temp, 
                                                   do_sample=True)
        else:
            raise ValueError("Invalid type, the type should be one of 'greedy', 'beam', 'topk', 'topp'.")
        return generated_tokens
    
    def inference(self):
        self.load_data()
        self.model.eval()
        preds = []
        progress_bar = tqdm(range(len(self.test_dataloader)))
        for step, batch in enumerate(self.test_dataloader):
            generated_tokens = self.generate_summary(batch)
            pred = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            pred = self.postprocess(pred)
            progress_bar.update(1)
            preds.extend(pred)
        return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='google/mt5-small')
    parser.add_argument('--test_file_dir', type=str, default='data.json')
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_out_length', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--type', type=str, default='greedy')
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--output_file_name', type=str, default='preds.jsonl')
    args = parser.parse_args()
    summarizer = Summarizer(args)
    preds = summarizer.inference()
    ids = pd.read_json(args.test_file_dir, lines=True)['id'].astype(str).tolist()
    df = pd.DataFrame({'id': ids, 'title': preds})
    df.to_json(args.output_file_name, orient='records', lines=True)

if __name__ == '__main__':
    main()



