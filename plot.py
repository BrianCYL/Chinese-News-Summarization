import matplotlib.pyplot as plt
import json
import pandas as pd
import os
import argparse

def load_metrics(file_path):
    with open(file_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def plot_metrics(eval_f1, eval_precisions, eval_recalls, args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    data_list = [eval_f1, eval_precisions, eval_recalls]
    titles = ['F1 Score', 'Precision', 'Recall']
    for data, title in zip(data_list, titles):
        try:
            df = pd.DataFrame(data) * 100
            plt.figure(figsize=(10, 5))
            df.plot()
            plt.title(f'{title} of Rouge vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Rouge Score')
            save_dir = os.path.join(args.output_dir, f'{title.lower()}_{args.type}.pdf')
            plt.savefig(save_dir)
        except:
            pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='result_img')
    parser.add_argument('--type', type=str, default='mt5')
    args = parser.parse_args()
    eval_f1 = load_metrics(os.path.join(args.folder_dir, 'eval_f1.json'))
    eval_precisions = load_metrics(os.path.join(args.folder_dir, 'eval_precisions.json'))
    eval_recalls = load_metrics(os.path.join(args.folder_dir, 'eval_recalls.json'))
    plot_metrics(eval_f1, eval_precisions, eval_recalls, args)

if __name__ == '__main__':
    main()