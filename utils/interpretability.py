import torch
import numpy as np

def interpret_top_kmers(sequence, mutation_map, k=6, top_n=3):
    """
    Generate top N most influential k-mers based on mutation scores.

    Args:
        sequence (str): DNA sequence.
        mutation_map (Tensor or list): Output mutation logits from the model.
        k (int): K-mer size.
        top_n (int): Number of top k-mers to return.

    Returns:
        top_kmers (list): List of (k-mer, position, importance_score).
        summary (str): Natural language explanation.
    """
    # ✅ Ensure mutation_map is a tensor
    if isinstance(mutation_map, list):
        mutation_map = torch.tensor(mutation_map)

    # ✅ Apply sigmoid and convert to numpy
    mutation_scores = torch.sigmoid(mutation_map).squeeze().cpu().numpy()

    # ✅ Generate all k-mers and their mean scores
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        window = mutation_scores[i:i + k]
        score = float(np.mean(window)) if len(window) == k else 0.0
        kmers.append((kmer, i, score))

    # ✅ Sort and take top-N
    seen = set()
    top_kmers = []
    for kmer, pos, score in sorted(kmers, key=lambda x: x[2], reverse=True):
        if (kmer, pos) not in seen:
            top_kmers.append((kmer, pos, round(score, 4)))
            seen.add((kmer, pos))
        if len(top_kmers) == top_n:
            break

    # ✅ Generate summary
    positions = [str(pos) for _, pos, _ in top_kmers]
    kmers_text = [f"'{kmer}'" for kmer, _, _ in top_kmers]
    summary = f"The model focused on {top_n} influential k-mers at positions {', '.join(positions)}\n"
    summary += f"(e.g., {', '.join(kmers_text)}) which contributed most to predicting the given class."

    return top_kmers, summary
