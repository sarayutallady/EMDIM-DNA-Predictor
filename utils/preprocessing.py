def kmer_encode(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def build_vocab(seqs, k=6):
    from collections import Counter
    all_kmers = []
    for s in seqs:
        all_kmers.extend(kmer_encode(s, k))
    vocab = {kmer: idx+1 for idx, (kmer, _) in enumerate(Counter(all_kmers).items())}
    vocab["<PAD>"] = 0
    return vocab

def encode_sequence(seq, vocab, k=6, max_len=256):
    kmers = kmer_encode(seq, k)
    idxs = [vocab.get(kmer, 0) for kmer in kmers]
    if len(idxs) < max_len:
        idxs += [0] * (max_len - len(idxs))
    else:
        idxs = idxs[:max_len]
    return idxs
