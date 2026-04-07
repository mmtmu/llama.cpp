#!/usr/bin/env python3
"""
Demonstrates that K-shift is incorrect when online rotation (attn_rot_k) is active
and the K-shift view uses n_rot dimensions instead of n_embd_head_k.

The bug: build_graph_shift creates a view of only the first n_rot=64 dimensions,
but the online rotation mixes all n_embd_head_k=128 dimensions. The rotation
undo/redo in build_rope_shift operates on the wrong subset of dimensions.

Expected output: the "buggy" path produces different (incorrect) results
compared to the "correct" path, with measurable error.
"""

import numpy as np

np.random.seed(42)

# Model dimensions (MiniMax M2)
n_embd_head_k = 128  # full head dimension
n_rot = 64           # RoPE only applies to first 64 dims


def apply_rope(x, pos, n_rot):
    """Apply RoPE positional encoding to the first n_rot dimensions."""
    result = x.copy()
    for i in range(0, n_rot, 2):
        freq = 1.0 / (5000000.0 ** (i / n_rot))  # freq_base=5000000 like MiniMax
        cos_val = np.cos(pos * freq)
        sin_val = np.sin(pos * freq)
        x0 = result[i]
        x1 = result[i + 1]
        result[i]     = x0 * cos_val - x1 * sin_val
        result[i + 1] = x0 * sin_val + x1 * cos_val
    return result


def rope_shift(x, old_pos, new_pos, n_rot):
    """Apply RoPE position shift: undo old_pos, apply new_pos (equivalent to delta rotation)."""
    result = x.copy()
    delta = new_pos - old_pos
    for i in range(0, n_rot, 2):
        freq = 1.0 / (5000000.0 ** (i / n_rot))
        cos_val = np.cos(delta * freq)
        sin_val = np.sin(delta * freq)
        x0 = result[i]
        x1 = result[i + 1]
        result[i]     = x0 * cos_val - x1 * sin_val
        result[i + 1] = x0 * sin_val + x1 * cos_val
    return result


def make_hadamard_rotation(n):
    """Create a random orthogonal rotation matrix (simulating the online rotation).
    Uses QR decomposition of a random matrix to get a proper orthogonal matrix.
    The actual llama.cpp rotation is a Hadamard-like matrix where rot^2 = I."""
    # For simplicity, create a matrix where R^2 = I (involution)
    # A Householder reflection is an involution: H = I - 2*v*v^T
    # Stack multiple to get a richer rotation that's still involutory
    # Actually, just use a random orthogonal matrix and note that R^{-1} = R^T
    # But llama.cpp assumes R^2 = I, so let's make a proper involution.
    # The simplest: a permutation matrix that swaps pairs, combined with sign flips
    R = np.zeros((n, n))
    indices = np.random.permutation(n)
    for i in range(0, n, 2):
        j, k = indices[i], indices[i + 1]
        # Random 2x2 rotation that is its own inverse (reflection)
        angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(angle), np.sin(angle)
        # Reflection: [[c, s], [s, -c]] has R^2 = I
        R[j, j] = c
        R[j, k] = s
        R[k, j] = s
        R[k, k] = -c
    return R


# Create the rotation matrix (128x128, involution: R^2 = I)
R_full = make_hadamard_rotation(n_embd_head_k)

# Verify R^2 = I
assert np.allclose(R_full @ R_full, np.eye(n_embd_head_k), atol=1e-10), "R^2 != I"

# Create a random K vector (before any encoding)
K_raw = np.random.randn(n_embd_head_k)

old_pos = 100
new_pos = 50  # shifting back by 50 positions (e.g., system prompt got shorter)

# === Forward pass: what gets stored in the KV cache ===
# Step 1: Apply RoPE at old position
K_roped = apply_rope(K_raw, old_pos, n_rot)
# Step 2: Apply online rotation (full 128-dim)
K_stored = R_full @ K_roped

print("=" * 70)
print("K-SHIFT WITH ONLINE ROTATION: BUG DEMONSTRATION")
print("=" * 70)
print(f"n_embd_head_k = {n_embd_head_k}, n_rot = {n_rot}")
print(f"Shifting from pos {old_pos} to pos {new_pos}")
print()

# === CORRECT K-shift (what should happen) ===
# 1. Undo rotation on ALL 128 dims
K_unrotated = R_full @ K_stored  # R^2 = I, so this undoes the rotation
# 2. Apply RoPE shift on first 64 dims
K_shifted = rope_shift(K_unrotated, old_pos, new_pos, n_rot)
# 3. Redo rotation on ALL 128 dims
K_correct = R_full @ K_shifted

# Verify: this should equal Rot(RoPE(K_raw, new_pos))
K_expected = R_full @ apply_rope(K_raw, new_pos, n_rot)
correct_error = np.max(np.abs(K_correct - K_expected))
print(f"CORRECT path (full 128-dim rotation undo/redo):")
print(f"  Max error vs expected: {correct_error:.2e}")
assert correct_error < 1e-10, "Correct path should be exact"
print(f"  PASS - matches expected result exactly")
print()

# === BUGGY K-shift (what the code actually does) ===
# build_graph_shift creates a VIEW of only the first n_rot=64 dimensions
# Then build_rope_shift tries to undo/redo the 128×128 rotation on this 64-dim view

# The buggy path: only touch the first 64 dimensions
K_buggy = K_stored.copy()

# Sub-rotation: what happens when you apply a 128×128 matrix to only 64 dims
# In practice, ggml_mul_mat_aux on a 64-dim view would use the top-left 64×64 block
# or produce garbage. Let's simulate the most charitable interpretation:
# applying R's effect on just the first 64 dims (treating the rest as zero)
R_sub = R_full[:n_rot, :n_rot]  # top-left 64x64 block

# 1. "Undo" rotation on first 64 dims only
K_buggy[:n_rot] = R_sub @ K_buggy[:n_rot]
# 2. Apply RoPE shift on first 64 dims
K_buggy[:n_rot] = rope_shift(K_buggy[:n_rot], old_pos, new_pos, n_rot)
# Dims 64-127 are UNTOUCHED by the shift
# 3. "Redo" rotation on first 64 dims only
K_buggy[:n_rot] = R_sub @ K_buggy[:n_rot]

buggy_error = np.max(np.abs(K_buggy - K_expected))
print(f"BUGGY path (64-dim partial rotation undo/redo):")
print(f"  Max error vs expected: {buggy_error:.2e}")
print()

# === Compare ===
diff = np.max(np.abs(K_buggy - K_correct))
rel_diff = diff / (np.max(np.abs(K_correct)) + 1e-15)

print(f"DIFFERENCE between correct and buggy paths:")
print(f"  Max absolute difference: {diff:.6f}")
print(f"  Relative difference:     {rel_diff:.4%}")
print()

# Show per-dimension errors
errors = np.abs(K_buggy - K_correct)
print(f"  Dims 0-63 (RoPE dims) max error:  {np.max(errors[:n_rot]):.6f}")
print(f"  Dims 64-127 (nope dims) max error: {np.max(errors[n_rot:]):.6f}")
print()

if diff > 1e-6:
    print("BUG CONFIRMED: The 64-dim partial rotation produces different results")
    print("than the correct 128-dim full rotation. K-shift corrupts the KV cache")
    print("when online rotation (attn_rot_k) is active.")
    print()
    print("Root cause: build_graph_shift() creates a view of n_rot=64 dimensions,")
    print("but the online rotation mixes all n_embd_head_k=128 dimensions.")
    print("The rotation undo/redo in build_rope_shift operates on wrong dimensions.")
    print()
    print("Fix: expand the K-shift view to n_embd_head_k when attn_rot_k is active.")
else:
    print("No bug detected (unexpected)")

# Also demonstrate cumulative error with multiple shifts
print()
print("=" * 70)
print("CUMULATIVE ERROR WITH MULTIPLE SHIFTS")
print("=" * 70)
K_multi = K_stored.copy()
K_multi_correct = K_stored.copy()

shifts = [(100, 50), (50, 120), (120, 30), (30, 200), (200, 75)]
for old_p, new_p in shifts:
    # Buggy path
    K_multi[:n_rot] = R_sub @ K_multi[:n_rot]
    K_multi[:n_rot] = rope_shift(K_multi[:n_rot], old_p, new_p, n_rot)
    K_multi[:n_rot] = R_sub @ K_multi[:n_rot]

    # Correct path
    K_multi_correct = R_full @ K_multi_correct
    K_multi_correct = rope_shift(K_multi_correct, old_p, new_p, n_rot)
    K_multi_correct = R_full @ K_multi_correct

final_error = np.max(np.abs(K_multi - K_multi_correct))
final_rel = final_error / (np.max(np.abs(K_multi_correct)) + 1e-15)
print(f"After {len(shifts)} shifts:")
print(f"  Max absolute error: {final_error:.6f}")
print(f"  Relative error:     {final_rel:.4%}")
print(f"  Error grows with each shift, corrupting cache-reuse quality further")
