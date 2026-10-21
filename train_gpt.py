import torch
from transformers import BertTokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling, get_scheduler, GPT2Tokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from tqdm import tqdm
import numpy as np
import evaluate
import argparse
import math
import os
from accelerate import Accelerator
import json

class SummaryTrainer():
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizerFast.from_pretrained('ckiplab/gpt2-base-chinese', padding_side='left')
        self.summary_token = "摘要："
        self.tokenizer.add_special_tokens({"bos_token": "<|startoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|pad|>"})
        self.tokenizer.add_tokens([self.summary_token])
        self.model = AutoModelForCausalLM.from_pretrained("ckiplab/gpt2-base-chinese")
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, return_tensors="pt", mlm=False)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.lr)
        self.eval_fn = evaluate.load("rouge")

    def preprocess_for_train(self, examples):
        # tokenize the maintext and title
        maintexts = [f"<|startoftext|> {para}" for para in examples['maintext']]
        main_tokens = self.tokenizer(maintexts, padding="max_length", truncation=True, return_tensors="pt", max_length=959)['input_ids']
        titles = [f"{title} <|endoftext|>" for title in examples['title']]
        title_tokens = self.tokenizer(titles, padding="max_length", truncation=True, return_tensors="pt", max_length=64)['input_ids']

        # remove padding tokens
        maintext_mask = main_tokens != self.tokenizer.pad_token_id
        title_mask = title_tokens != self.tokenizer.pad_token_id
        main_tokens = [article[mask].tolist() for article, mask in zip(main_tokens, maintext_mask)]
        title_tokens = [title[mask].tolist() for title, mask in zip(title_tokens, title_mask)]

        # concatenate the maintext and title and padding to max length
        special_token_id = self.tokenizer.convert_tokens_to_ids(self.summary_token)
        inputs = [article_token + [special_token_id] + title_token for article_token, title_token in zip(main_tokens, title_tokens)]
        encodings = [{'input_ids': input_ids} for input_ids in inputs]
        model_input = self.tokenizer.pad(encodings, padding='max_length', max_length=self.args.max_input_length, return_tensors='pt')

        # create labels
        labels = model_input["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        assert model_input["input_ids"].shape == labels.shape == model_input["attention_mask"].shape
        return {"input_ids": model_input['input_ids'], "attention_mask": model_input["attention_mask"], "labels": labels}
    
    def preprocess_for_eval(self, examples):
        maintexts = [f"<|startoftext|> {para}" for para in examples['maintext']]
        main_tokens = self.tokenizer(maintexts, padding="max_length", truncation=True, return_tensors="pt", max_length=959)['input_ids']
        titles = [f"{title} <|endoftext|>" for title in examples['title']]
        title_tokens = self.tokenizer(titles, padding="max_length", truncation=True, return_tensors="pt", max_length=64)['input_ids']

        maintext_mask = main_tokens != self.tokenizer.pad_token_id
        title_mask = title_tokens != self.tokenizer.pad_token_id
        main_tokens = [article[mask].tolist() for article, mask in zip(main_tokens, maintext_mask)]   
        title_tokens = [title[mask].tolist() for title, mask in zip(title_tokens, title_mask)]    

        special_token_id = self.tokenizer.convert_tokens_to_ids(self.summary_token)
        model_inputs = [article_token + [special_token_id] for article_token in main_tokens]
        encodings = [{'input_ids': input_ids} for input_ids in model_inputs]
        model_inputs = self.tokenizer.pad(encodings, padding=True, return_tensors='pt')

        targets = [article_token + [special_token_id] + title_token for article_token, title_token in zip(main_tokens, title_tokens)]
        encodings = [{'input_ids': input_ids} for input_ids in targets]
        targets = self.tokenizer.pad(encodings, padding='max_length', max_length=self.args.max_input_length, return_tensors='pt')["input_ids"]
        targets[targets == self.tokenizer.pad_token_id] = -100
        return {"input_ids": model_inputs["input_ids"], "attention_mask": model_inputs["attention_mask"], "labels": targets}

    def postprocess(self, texts):
        texts = [str(text.split(self.summary_token)[1]).replace(" ", "").strip() for text in texts]
        return texts
    
    def load_data(self):
        dataset = load_dataset('json', data_files=self.args.file_path)
        dataset = dataset["train"].train_test_split(test_size=self.args.test_size, seed=self.args.seed)
        dataset['train'] = dataset['train'].map(self.preprocess_for_train, batched=True)
        dataset['test'] = dataset['test'].map(self.preprocess_for_eval, batched=True)
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

    def loss_fn(self, outputs, doc_len):
        logits = outputs.logits
        labels = outputs["labels"]
        summary_logits = logits[:, doc_len-1:-1, :]
        summary_labels = labels[:, doc_len:]
        summary_logits = summary_logits.view(-1, summary_logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
        summary_labels = summary_labels.view(-1)
        loss = self.criterion(summary_logits, summary_labels)
        return loss 

    def compute_metrics(self, preds, labels):
        if isinstance(preds, tuple):
            preds = preds[0]
        decode_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decode_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decode_preds = self.postprocess(decode_preds)
        decode_labels = self.postprocess(decode_labels)
        cleaned_output = [(preds, labels) for preds, labels in zip(decode_preds, decode_labels) if preds]
        decode_preds, decode_labels = map(list, zip(*cleaned_output))
        self.eval_fn.add_batch(predictions=decode_preds, references=decode_labels)

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
        eval_f1, eval_precisions, eval_recalls = [], [], []
        epoch_cnt = 0
        best_rouge1, best_rouge2, best_rougel = 0, 0, 0
        best_loss = float('inf')
        loos_track = []
        for epoch in range(self.args.n_epochs):
            total_loss = 0.0
            model.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
 
            model.eval()
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    generated_tokens = model.generate(
                        batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=self.args.max_out_length,
                        pad_token_id=self.tokenizer.pad_token_id,
                        no_repeat_ngram_size=3,
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
                self.compute_metrics(generated_tokens, label)
            eval_metrics = self.eval_fn.compute()
            eval_f1.append(eval_metrics)
            # if eval_metrics["rouge1"] > best_rouge1 or eval_metrics["rouge2"] > best_rouge2 or eval_metrics["rougeL"] > best_rougel:
            if best_loss > total_loss/len(train_dataloader):
                # best_rouge1 = eval_metrics["rouge1"]
                # best_rouge2 = eval_metrics["rouge2"]
                # best_rougel = eval_metrics["rougeL"]
                best_loss = total_loss/len(train_dataloader)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(self.args.output_dir, save_function=accelerator.save)
                if accelerator.is_main_process:
                    self.tokenizer.save_pretrained(self.args.output_dir)  
                # epoch_cnt = 0
            accelerator.print(f"epoch {epoch}: {eval_metrics}, training loss: {total_loss/len(train_dataloader)}")
            loos_track.append(total_loss/len(train_dataloader))
            epoch_cnt += 1
            # if epoch_cnt >= 10:
            #     break
        # accelerator.print(f"best rouge-1: {best_rouge1}, best rouge-2: {best_rouge2}, best rouge-l: {best_rougel}")
        json.dump(eval_f1, open(self.args.output_dir+'eval_f1.json', 'w'))
        json.dump(eval_precisions, open(self.args.output_dir+'eval_precisions.json', 'w'))
        json.dump(eval_recalls, open(self.args.output_dir+'eval_recalls.json', 'w'))
        json.dump(loos_track, open(self.args.output_dir+'loss.json', 'w'))
    
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