# src/utils/physics.py
import numpy as np
import torch

# Physics Utility Functions for S-Parameter Conversions and Passivity Checks
# This module provides functions to convert single-ended S-parameters to mixed-mode and to check the passivity of S-parameter matrices using eigenvalue analysis.
def convert_to_mixed_mode(s_se):
    """
    Converts a 4x4 Single-Ended S-parameter matrix (TX+, TX-, RX+, RX-) 
    to a 4x4 Mixed-Mode matrix (Sdd, Sdc, Scd, Scc).
    """
    M = np.array([
        [ 1, -1,  0,  0],  # D1 (Differential 1)
        [ 0,  0,  1, -1],  # D2 (Differential 2)
        [ 1,  1,  0,  0],  # C1 (Common 1)
        [ 0,  0,  1,  1]   # C2 (Common 2)
    ]) / np.sqrt(2)
    
    M_inv = np.linalg.inv(M)
    
    s_mm = np.zeros_like(s_se, dtype=complex)
    for i in range(s_se.shape[0]):
        s_mm[i] = M @ s_se[i] @ M_inv
        
    return s_mm

# Passivity Check Function
def check_passivity(s_matrix, threshold=-1e-6):
    """
    Eigenvalue Passivity Check. 
    Returns (is_passive_boolean, min_eigenvalue)
    """
    I = np.eye(s_matrix.shape[1])
    min_eigenvalue = np.inf
    
    for i in range(s_matrix.shape[0]):
        S = s_matrix[i]
        Q = I - np.conj(S).T @ S
        eigenvalues = np.linalg.eigvalsh(Q)
        min_eigenvalue = min(min_eigenvalue, eigenvalues.min())
        
    is_passive = min_eigenvalue >= threshold
    return is_passive, min_eigenvalue

#unscaling function for physical interpretation of model outputs
def unscale_tensor(scaled_val, mean, std):
    """
    Inverse Z-score transformation.
    Physical = (Scaled * Std) + Mean
    """
    return (scaled_val * std) + mean

#S-parameter to Decibel conversion with a noise floor to prevent learning simulator artifacts
def s_to_db(s_matrix, epsilon=1e-4):
    """
    Safe conversion of complex S-parameters to Decibels.
    Epsilon acts as a 'Noise Floor' to keep the AI from learning simulator artifacts.
    """
    if isinstance(s_matrix, torch.Tensor):
        return 20 * torch.log10(torch.abs(s_matrix) + epsilon)
    return 20 * np.log10(np.abs(s_matrix) + epsilon)

# Reciprocity Enforcement Function
def enforce_reciprocity(s_matrix):
    """
    Enforces Sij = Sji to ensure physical symmetry.
    Reduces the AI search space by 50%.
    """
    if isinstance(s_matrix, np.ndarray):
        # Handle NumPy arrays (Batch, Ports, Ports) or (Freqs, Ports, Ports)
        return 0.5 * (s_matrix + np.transpose(s_matrix, axes=(0, 2, 1)))
    # Handle PyTorch Tensors
    return 0.5 * (s_matrix + s_matrix.transpose(-1, -2))