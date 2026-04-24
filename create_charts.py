
import matplotlib.pyplot as plt
import numpy as np
import os

def create_metrics_chart(output_path):
    cases = ['Case 017', 'Case 019', 'Case 033']
    vanilla_dice = [0.8851, 0.8185, 0.8699]
    improved_dice = [0.9131, 0.8484, 0.8998]

    x = np.arange(len(cases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, vanilla_dice, width, label='Vanilla WNet', color='#3498db', alpha=0.8)
    rects2 = ax.bar(x + width/2, improved_dice, width, label='Improved WNet (Attention+Focal)', color='#e74c3c', alpha=0.8)

    ax.set_ylabel('Mean Dice Score', fontsize=12)
    ax.set_title('Comparison of Segmentation Accuracy on Validation Set', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cases)
    ax.legend()

    # Thêm giá trị trên đầu cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.ylim(0.7, 1.0) # Zoom vào dải quan trọng
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    out_dir = "f:/Workspace/med-img-seg/nnUNet_data/visualizations"
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    create_metrics_chart(f"{out_dir}/dice_comparison_chart.png")
