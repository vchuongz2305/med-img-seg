
import matplotlib.pyplot as plt
import numpy as np
import os

def create_multi_metric_chart(output_path):
    metrics = ['Dice', 'Accuracy', 'Precision', 'Recall']
    # Dữ liệu trung bình từ kết quả tính toán
    vanilla_scores = [0.8786, 0.9867, 0.8923, 0.8656]
    improved_scores = [0.9075, 0.9896, 0.9036, 0.9122]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, vanilla_dice, width, label='Vanilla WNet', color='#34495e', alpha=0.8)
    rects2 = ax.bar(x + width/2, improved_dice, width, label='Improved WNet', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Scores (0.0 - 1.0)', fontsize=12)
    ax.set_title('Overall Performance Comparison (Average across Validation Cases)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(loc='lower right')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.ylim(0.8, 1.05) # Zoom vào dải cao để thấy sự khác biệt
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Multi-metric chart saved to: {output_path}")

if __name__ == "__main__":
    # Fix variable names from previous script attempt
    metrics = ['Dice', 'Accuracy', 'Precision', 'Recall']
    vanilla_scores = [0.8786, 0.9867, 0.8923, 0.8656]
    improved_scores = [0.9075, 0.9896, 0.9036, 0.9122]
    
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width/2, vanilla_scores, width, label='Vanilla WNet', color='#34495e', alpha=0.8)
    rects2 = ax.bar(x + width/2, improved_scores, width, label='Improved WNet', color='#2ecc71', alpha=0.8)
    ax.set_ylabel('Scores', fontsize=12); ax.set_title('Performance Comparison', fontsize=14)
    ax.set_xticks(x); ax.set_xticklabels(metrics); ax.legend()
    plt.ylim(0.8, 1.02); plt.tight_layout()
    plt.savefig("f:/Workspace/med-img-seg/nnUNet_data/visualizations/multi_metric_comparison.png", dpi=300)
    print("Success")
