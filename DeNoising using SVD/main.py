#  De-noising a Simple Line
import numpy as np
import matplotlib.pyplot as plt

# ---Part 1: Create the Data (Signal and Noise)

# 1. Define the number of data points we want to work with.
N = 50
# 2. Create the 'x' values (like time or position)
x_values = np.linspace(0, 10, N)

# 3. Create the TRUE SIGNAL. This is what we want to find.
# We make a simple straight line (a Rank 1 matrix)
# We repeat this line 5 times to form a small matrix (5 columns/features)
# The signal is just 'x' values in a 50x1 shape multiplied by 5 columns of ones.
TRUE_SIGNAL = x_values.reshape(-1, 1) @ np.ones((1, 5))

# 4. Create RANDOM NOISE.
# (mean=0, standard_deviation=0.5, shape)
np.random.seed(42)
NOISE = np.random.normal(0, 0.5, TRUE_SIGNAL.shape)

# 5. Create the NOISY DATA
NOISY_DATA = TRUE_SIGNAL + NOISE

# --- Part 2: Perform Singular Value Decomposition (SVD) ---

# SVD breaks the NOISY_DATA matrix (A) into three parts: A = U * S * Vh
# U, Vh are rotation matrices, and S contains the 'strength' of the components.
U, S, Vh = np.linalg.svd(NOISY_DATA)

# --- Part 3: De-noising by Low-Rank Approximation ---

# 1. Decide on the RANK (k) for reconstruction.
# Since the TRUE_SIGNAL was a perfect line (Rank 1), we choose k=1.
k = 1

# 2. Extract only the 'k' most important parts from each component:
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vh_k = Vh[:k, :]

# 3. Reconstruct the data using ONLY these 'k' parts.
CLEAN_DATA = U_k @ S_k @ Vh_k

# --- Part 4: Visualization (See the Magic!) ---

# We only need to plot one of the 5 columns to see the result clearly.
column_to_plot = 0
x_indices = np.arange(N) # indexing for simple plotting

plt.figure(figsize=(10, 6))

# Plot 1: The TRUE SIGNAL
plt.plot(x_indices, TRUE_SIGNAL[:, column_to_plot], 'g-', linewidth=3, label='1. True Signal (The Goal)')

# Plot 2: The NOISY DATA
plt.scatter(x_indices, NOISY_DATA[:, column_to_plot], c='r', s=20, alpha=0.6, label='2. Noisy Data (What We See)')

# Plot 3: The SVD De-noised Data
plt.plot(x_indices, CLEAN_DATA[:, column_to_plot], 'b--', linewidth=2, label=f'3. SVD De-noised Data (k={k})')

plt.title('SVD De-noising: Separating Signal from Noise')
plt.xlabel('Data Point Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() #