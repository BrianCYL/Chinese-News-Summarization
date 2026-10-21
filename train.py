import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, AutoTokenizer, get_scheduler
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW

from tqdm import tqdm
import numpy as np
from tw_rouge import get_rouge
import argparse
import math
import os
from accelerate import Accelerator
import json

class SummaryTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained('google/mt5-small', trust_remote_code=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small", trust_remote_code=True)
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model, padding=True, return_tensors="pt")
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)

    def preprocess(self, examples):
        inputs = [f"summarize: {ex}" for ex in examples['maintext']]
        model_inputs = self.tokenizer(inputs, max_length=self.args.max_input_length, truncation=True)
        labels = self.tokenizer(examples["title"], max_length=self.args.max_out_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def load_data(self):
        dataset = load_dataset('json', data_files=self.args.file_path)
        dataset = dataset["train"].train_test_split(test_size=self.args.test_size, seed=self.args.seed)
        dataset = dataset.map(self.preprocess, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataset = dataset.remove_columns(["maintext", "title"])
        self.train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            collate_fn=self.data_collator,
            batch_size=self.args.train_batch_size,
        )
        self.eval_dataloader = DataLoader(
            dataset["test"], collate_fn=self.data_collator, batch_size=self.args.eval_batch_size
        )

    def compute_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decode_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decode_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        cleaned_output = [(preds, labels) for preds, labels in zip(decode_preds, decode_labels) if preds]
        decode_preds, decode_labels = map(list, zip(*cleaned_output))

        rouge = get_rouge(decode_preds, decode_labels)
        f1_score = {k: v['f'] for k, v in rouge.items()}
        precision = {k: v['p'] for k, v in rouge.items()}
        recall = {k: v['r'] for k, v in rouge.items()}
        return f1_score, precision, recall


    def train(self):
        accelerator_log_kwargs = {}
        accelerator = Accelerator(gradient_accumulation_steps=self.args.gradient_accumulation_steps, **accelerator_log_kwargs)

        self.model.to(accelerator.device)

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        max_train_steps = self.args.n_epochs * num_update_steps_per_epoch
        self.args.n_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_ratio * max_train_steps,
            num_training_steps=max_train_steps,
        )

        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader, self.lr_scheduler
        )

        progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
        best_rouge1, best_rouge2, best_rougel = 0, 0, 0
        eval_f1, eval_precisions, eval_recalls = [], [], []
        for epoch in range(self.args.n_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
 
            model.eval()
            f1_scores, precisions, recalls = [], [], []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_length=self.args.max_out_length,
                        top_p=0.9, 
                        temperature=0.7, 
                        do_sample=True
                    )
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=self.tokenizer.pad_token_id
                    )

                    # If we did not pad to max length, we need to pad the labels too
                    label = accelerator.pad_across_processes(
                        batch["labels"], dim=1, pad_index=self.tokenizer.pad_token_id
                    )

                    generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                    label = accelerator.gather(label).cpu().numpy()
                f1_score, precision, recall = self.compute_metrics(generated_tokens, label)
                f1_scores.append(f1_score)
                precisions.append(precision)
                recalls.append(recall)
            eval_metrics = {k: np.mean([x[k] for x in f1_scores]) for k in f1_scores[0]}
            eval_f1.append(eval_metrics)
            eval_precisions.append({k: np.mean([x[k] for x in precisions]) for k in precisions[0]})
            eval_recalls.append({k: np.mean([x[k] for x in recalls]) for k in recalls[0]}) 
            if eval_metrics['rouge-1'] >= best_rouge1 or eval_metrics['rouge-2'] >= best_rouge2 or eval_metrics['rouge-l'] >= best_rougel:
                best_rouge1 = eval_metrics['rouge-1']
                best_rouge2 = eval_metrics['rouge-2']
                best_rougel = eval_metrics['rouge-l']
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(self.args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.args.output_dir)                
            accelerator.print(f"epoch {epoch}: {eval_metrics}")
        accelerator.print(f"best rouge-1: {best_rouge1}, best rouge-2: {best_rouge2}, best rouge-l: {best_rougel}")
        json.dump(eval_f1, open(self.args.output_dir+'eval_f1.json', 'w'))
        json.dump(eval_precisions, open(self.args.output_dir+'eval_precisions.json', 'w'))
        json.dump(eval_recalls, open(self.args.output_dir+'eval_recalls.json', 'w'))
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='data.json')
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_input_length', type=int, default=256)
    parser.add_argument('--max_out_length', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=int, default=0.1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--n_epochs', type=int, default=3)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    trainer = SummaryTrainer(args)
    trainer.load_data()
    trainer.train()

if __name__ == '__main__':
    main()