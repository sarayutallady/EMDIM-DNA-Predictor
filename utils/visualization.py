import matplotlib.pyplot as plt
import numpy as np

def plot_mutation_map(sequence, mutation_scores, title="Mutation Map", save_path=None):
    plt.figure(figsize=(16, 2))
    x = np.arange(len(sequence))
    y = mutation_scores[:len(sequence)]

    plt.bar(x, y, color='tomato', alpha=0.8)
    plt.xticks(x[::10], list(sequence)[::10], fontsize=6, rotation=90)
    plt.xlabel("Base Position")
    plt.ylabel("Mutation Score")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
