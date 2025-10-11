import math
from collections import Counter
import gzip
import bz2

# Practical 
def compressed_size_gzip(text):
    data = text.encode('utf-8')
    compressed = gzip.compress(data)
    return len(compressed) * 8  # bits

def compressed_size_bz2(text):
    data = text.encode('utf-8')
    compressed = bz2.compress(data)
    return len(compressed) * 8  # bits


# Theoretical
def entropy(text):
    if not text:
        return 0
    freq = Counter(text)
    print(freq)
    total = len(text)
    probs = [count / total for count in freq.values()]
    H = -sum(p * math.log2(p) for p in probs)
    return H

def compare(text):
    H = entropy(text)
    n = len(text)
    bits_theoretical = H * n
    gzip_bits = compressed_size_gzip(text)
    bz2_bits = compressed_size_bz2(text)

    print(f"Text length: {n} chars")
    print(f"Entropy: {H:.3f} bits/char")
    print(f"Theoretical minimum size: {bits_theoretical/8:.2f} bytes")
    print(f"Gzip compressed size: {gzip_bits/8:.2f} bytes")
    print(f"Bz2 compressed size: {bz2_bits/8:.2f} bytes")
    print("-" * 40)


# print(entropy("This is Text entropy calculator"))
samples = [
    "AAAAAA",           # highly predictable (low entropy)
    "ABABABAB",         # repetitive pattern
    "ABCDEFGH",         # all unique (max entropy)
    "Hello world!",     # real-world text
]

# for text in samples:
    # print(f"Text: {text} -> Entropy: {entropy(text):.3f} bits/char")

texts = {
    "Low entropy": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
    "Medium entropy": "ABABABABABABABABABABABABABABABAB",
    "High entropy": "qwertyuiopasdfghjklzxcvbnm1234567890!@#$",
    "English text": "Entropy measures uncertainty or surprise in data.",
}

for label, txt in texts.items():
    print(f"\n--- {label} ---")
    compare(txt)

# Observation:
    # Low-entropy text → compresses extremely well.
    # High-entropy/random text → barely compresses (entropy ≈ compressed ratio).

import matplotlib.pyplot as plt

def plot_frequencies(text):
    freq = Counter(text)
    symbols, counts = zip(*sorted(freq.items(), key=lambda x: -x[1]))
    probs = [c / len(text) for c in counts]
    
    plt.bar(symbols, probs)
    plt.title("Character Probability Distribution")
    plt.ylabel("Probability")
    plt.show()

plot_frequencies("Entropy measures uncertainty or surprise in data.")