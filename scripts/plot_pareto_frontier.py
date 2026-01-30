import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Use 'Agg' backend for headless environments
import matplotlib
matplotlib.use('Agg')

def plot_pareto(csv_path, output_dir='./test_results/plots'):
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Performance vs VRAM (Train)
    plt.figure(figsize=(10, 6))
    
    # Baseline vs VG-iT
    models = df['model'].unique()
    colors = plt.cm.get_cmap('tab10', len(models))

    for i, model in enumerate(models):
        mask = df['model'] == model
        subset = df[mask]
        
        plt.scatter(subset['train_vram_GB'], subset['mse'], 
                    label=model, alpha=0.7, s=100, color=colors(i))
        
        # Annotate with num_groups and pooling
        for idx, row in subset.iterrows():
            info = f"G{row['num_groups']}_{row['pooling']}" if row['model'] == 'VG_iTransformer' else ""
            plt.annotate(info, (row['train_vram_GB'], row['mse']), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.xlabel('Training Peak VRAM (GB)')
    plt.ylabel('Performance (MSE)')
    plt.title('Pareto Frontier: Performance vs VRAM')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'pareto_vram_mse.png'))
    plt.close()

    # 2. Performance vs Inference Latency
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        mask = df['model'] == model
        subset = df[mask]
        plt.scatter(subset['infer_latency_s'], subset['mse'], 
                    label=model, alpha=0.7, s=100, color=colors(i))

    plt.xlabel('Inference Latency (s/iter)')
    plt.ylabel('Performance (MSE)')
    plt.title('Pareto Frontier: Performance vs Latency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'pareto_latency_mse.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    plot_pareto('./test_results/summary.csv')
