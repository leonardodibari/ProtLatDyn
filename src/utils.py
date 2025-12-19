import torch
from typing import Dict, List, Any
from sklearn.decomposition import PCA  # Assuming PCA object is of this type
import gc
import time
from typing import Tuple



def generate_dms_and_project_batch_timed(
    msa_onehot: torch.Tensor,
    pca: PCA,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    """
    Generates Deep Mutational Scan (DMS) variants, projects them using PCA in a 
    single batch, and measures the time taken for Generation, Projection, and Unstacking.

    Args:
        msa_onehot (torch.Tensor): One-hot encoded MSA with shape (N_seq, L, q).
        pca (PCA): Pre-fitted scikit-learn PCA object.

    Returns:
        Tuple[List[torch.Tensor], Dict[str, float]]: 
            - List of PC projections for all sequences.
            - Dictionary of time elapsed for each phase (Generation, Projection, Unstacking).
    """
    
    N_seq, L, q = msa_onehot.shape
    total_variants = N_seq * L * q
    
    # Timing dictionary initialization
    timing = {}

    # --- 1. GENERATION PHASE ---
    start_time_gen = time.time()
    
    # 1a. Generate the Template for Mutations (L*q, L*q)
    mutant_templates = torch.zeros(L * q, L, q, dtype=msa_onehot.dtype, device=msa_onehot.device)
    positions = torch.arange(L).repeat_interleave(q)
    residues = torch.arange(q).repeat(L)
    mutant_templates[torch.arange(L * q), positions, residues] = 1
    
    # 1b. Prepare batch indexing and repeated tensors
    mutant_templates_batch = mutant_templates.repeat(N_seq, 1, 1)
    repeated_wt_oh = msa_onehot.repeat_interleave(L * q, dim=0).float()
    positions_batch = positions.repeat(N_seq)
    indices_batch = torch.arange(total_variants)

    # 1c. Create the final one-hot batch: (N_seq * L*q, L, q)
    repeated_wt_oh[indices_batch, positions_batch, :] = 0 # Remove original residue
    all_dms_oh_batch = repeated_wt_oh + mutant_templates_batch # Add mutant residue
    
    # 1d. Prepare final input tensor for PCA
    all_dms_flat_batch = all_dms_oh_batch.view(total_variants, -1)
    wt_flat_batch = msa_onehot.view(N_seq, -1).float()
    all_pcs_input = torch.cat([wt_flat_batch, all_dms_flat_batch], dim=0) # (N_seq + total_variants, L*q)

    # Convert to NumPy for scikit-learn PCA (This is often a bottleneck itself)
    all_pcs_input_np = all_pcs_input.cpu().numpy().astype('float32')
    
    timing['generation_s'] = time.time() - start_time_gen
    
    # --- 2. PCA PROJECTION PHASE ---
    start_time_pca = time.time()
    
    # Project all sequences (WTs followed by all mutants) in ONE batch call
    all_pcs_proj_np = pca.transform(all_pcs_input_np)
    all_pcs_proj_tensor = torch.tensor(all_pcs_proj_np, dtype=torch.float32)

    timing['pca_projection_s'] = time.time() - start_time_pca

    # --- 3. UNSTACKING PHASE ---
    start_time_unstack = time.time()
    
    # Slice and Reshape
    wt_pcs_proj = all_pcs_proj_tensor[:N_seq]
    dms_pcs_proj = all_pcs_proj_tensor[N_seq:]
    dms_pcs_proj_batch = dms_pcs_proj.view(N_seq, L * q, -1)
    
    # Combine WT and Mutants and create the final List
    all_mutant_pcs = []
    
    for seq_idx in range(N_seq):
        wt_proj = wt_pcs_proj[seq_idx].unsqueeze(0)
        dms_proj = dms_pcs_proj_batch[seq_idx]
        final_tensor = torch.cat([wt_proj, dms_proj], dim=0)
        all_mutant_pcs.append(final_tensor)
        
    timing['unstacking_s'] = time.time() - start_time_unstack
    
    # Calculate Total Time
    timing['total_s'] = sum(timing.values())
    
    return all_mutant_pcs




def calculate_forces_and_noises(
    trajs_Teq: torch.Tensor,
    gibbs_Teq: torch.Tensor,
    pca, # The trained PCA object (e.g., sklearn.decomposition.PCA)
    device: torch.device
):
    """
    Calculates forces and noises for an ensemble of M trajectories (trajs_Teq), 
    using weighted distance metrics (gibbs_Teq) and PCA.
    
    NOTE: This function assumes that 'generate_dms_and_project_batch_timed' 
    is defined and available in the local scope.

    Args:
        trajs_Teq: Tensor of shape (M, T, ...Features) - Raw coordinates. The features
                   are flattened for PCA transformation.
        gibbs_Teq: Tensor of shape (M, T, L, q) - Weights for the distance metrics.
                   The type is assumed to be torch.float32.
        pca: Trained PCA model (must have a .transform method).
        device: The PyTorch device ('cuda' or 'cpu').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (forces, noises), both of shape (M, T-1, N_pc).
    """
    
    # M is the number of ensemble members (trajectories)
    M, T, L, q = trajs_Teq.shape


    # Lists to collect results from each M iteration
    all_forces = []
    all_noises = []
    all_displs = []
    all_trajs_pc = []

    for m in range(M):
        
        # --- 1. Calculate Force (Weighted Average of DM Displacements) ---

        # Assumes generate_dms_and_project_batch_timed is in scope.
        # It should return T tensors, each (1 + L*q, N_pc), with dtype=float32.
        # dms_pc shape: (T, 1 + L*q, N_pc). Index 0 is the sequence component.
        dms_pc = torch.stack(generate_dms_and_project_batch_timed(trajs_Teq[m], pca)).to(device)
        
        # Calculate the displacement in the DM space: dms - sequence component
        # dms_displ shape: (T, L*q, N_pc)
        dms_displ = dms_pc[:, 1:, :] - dms_pc[:, 0, :].unsqueeze(1)
        
        # Clean up the large, intermediate dms_pc tensor
        del dms_pc
        
        # Reshape Gibbs weights (W) for broadcasting/multiplication.
        # W shape: (T, L*q, 1). This assumes gibbs_Teq[m] is (T, L, q) and float32.
        W = gibbs_Teq[m].permute(1, 0, 2).reshape(T, L * q).unsqueeze(-1)

        # Calculate force as the weighted average displacement in DM space.
        # Resulting force_t_full shape: (T, N_pc)
        force_t_full = (W * dms_displ).sum(dim=1) / W.sum(dim=1)
        
        # We only need forces for the first T-1 steps
        # force shape: (T-1, N_pc)
        force = force_t_full[:T-1]
        
        # Cleanup
        del dms_displ, W, force_t_full

        # --- 2. Calculate Noise (Displacement - Force) ---
        
        # Project the raw trajectory coordinates onto PCA space. Ensures float32.
        # trajs_pc shape: (T, N_pc)
        trajs_pc = torch.tensor(pca.transform(trajs_Teq[m].flatten(start_dim=1).cpu() ), 
                                dtype=torch.float32).to(device)
        
        # Calculate the physical displacement: X(t+1) - X(t)
        # displ shape: (T-1, N_pc)
        displ_list = [trajs_pc[i+1] - trajs_pc[i] for i in range(T-1)]
        displ = torch.stack(displ_list).to(device)
        
        # Calculate noise: noise = physical_displacement - calculated_force
        # noise shape: (T-1, N_pc)
        noise = displ - force
        
        # Append results (unsqueeze(0) adds the M=1 dimension for concatenation later)
        all_forces.append(force.unsqueeze(0))
        all_noises.append(noise.unsqueeze(0))
        all_displs.append(displ.unsqueeze(0))
        all_trajs_pc.append(trajs_pc.unsqueeze(0))
        # --- Memory Cleanup ---
        # Explicitly delete all remaining large tensors and clear cache
        del trajs_pc, displ, force, noise
        gc.collect() # Force garbage collection
        torch.cuda.empty_cache() # Clear unused VRAM
        
    # Concatenate the collected lists into the final output shape (M, T-1, N_pc)
    final_forces = torch.cat(all_forces, dim=0)
    final_noises = torch.cat(all_noises, dim=0)
    final_displs = torch.cat(all_displs, dim=0)
    final_pcs = torch.cat(all_trajs_pc, dim=0)
    
    return final_forces, final_noises, final_displs, final_pcs


# -------------------------------------------------------
# Cosine similarity function (vector → vector)
# -------------------------------------------------------
def cosine_sim(x, y):
    return torch.nn.functional.cosine_similarity(x, y, dim=0)


# -------------------------------------------------------
# Compute cosine similarity vs time for each trajectory
# -------------------------------------------------------
def compute_cosine_curve(force, noise):
    # force, noise both have shape (M, T, L)
    M, T, _ = force.shape
    return torch.tensor([
        [cosine_sim(force[m, i, :], noise[m, i, :]) for i in range(T)]
        for m in range(M)
    ])  # shape (M, T)



def pearson_correlation_1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Pearson correlation coefficient between two 1D tensors.

    Args:
        t1: The first 1D input tensor.
        t2: The second 1D input tensor.

    Returns:
        A tensor containing the single correlation value.
    """
    # 1. Stack the tensors into a single (2, N) matrix
    stacked_tensors = torch.stack((t1, t2), dim=0)

    # 2. Use torch.corrcoef to compute the correlation matrix.
    # The output is a (2, 2) matrix:
    # [[ corr(t1, t1), corr(t1, t2) ],
    #  [ corr(t2, t1), corr(t2, t2) ]]
    corr_matrix = torch.corrcoef(stacked_tensors)

    # 3. Extract the off-diagonal correlation value (corr(t1, t2))
    return corr_matrix[0, 1]



def compute_gibbs_probabilities(
    msa_oh: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
    remove_self_loop: bool = False, # <--- NEW PARAMETER
) -> torch.Tensor:
    """
    Computes Gibbs probabilities for each residue in a sequence.
    
    If remove_self_loop is True, the probability of remaining in the current state
    at the site being sampled is set to zero before normalization.
    
    Args:
        msa_oh (Tensor): One-hot encoded sequences of shape (N, L, q).
        params (dict): Model parameters.
        beta (float, optional): Inverse temperature.
        remove_self_loop (bool): If True, set the logit of the current state to -inf 
                                 to force the resulting probability to zero.

    Returns:
        Tensor: A tensor of shape (L, N, q) with the Gibbs probabilities.
    """


    """
GIBBS SAMPLING WITH SELF-LOOP REMOVAL
======================================

This method follows a three-step process:

1. PER-SITE NORMALIZATION:
   For each position i independently, compute probabilities using softmax:
   P(state | position i) = exp(logit_i) / sum_over_states(exp(logit_i))
   
   At this point, probabilities at each position sum to 1.0
   This treats each position with its own "importance" regardless of logit magnitude.

2. REMOVE SELF-LOOPS:
   Set the probability of the current state at each position to zero.
   Now the probabilities at each position sum to LESS than 1.0

3. GLOBAL RENORMALIZATION:
   Compute total = sum over ALL positions and ALL states
   Divide all probabilities by this total
   
   Final result: sum over all positions and states = 1.0

KEY PROPERTY: Because we did per-site softmax FIRST, positions with lower-energy
states still maintain reasonable probability mass. The initial per-site normalization
partially "equalizes" the importance of different positions.

Example: If position 1 has logits [10, 5] and position 2 has logits [3, 1],
         after step 1, position 1 gets ~[0.99, 0.01] and position 2 gets ~[0.88, 0.12]
         After removing current states and global renorm, both positions keep
         significant probability (e.g., ~0.89 and ~0.11)
"""

    N, L, q = msa_oh.shape
    device = msa_oh.device
    
    all_probs = []

    # Iterate over each residue site (index)
    for idx in range(L):
        # 1. Calculate Logit values (same as before)
        couplings_residue = params["coupling_matrix"][idx].view(q, L * q)  # (q, L * q)

        # Logit values for all q possible new states at site idx
        # Shape: (N, q)
        logit_residue = beta * (
            params["bias"][idx].unsqueeze(0) + msa_oh.reshape(N, L * q) @ couplings_residue.T
        )


        # 3. Normalization (Softmax)
        # Apply softmax to compute the probability distribution for the q states.
        # The normalization naturally accounts for the masked state(s).
        p_residue = torch.softmax(logit_residue, dim=-1)  # (N, q)

        
        # 2. Handle Self-Loop Remova        
        all_probs.append(p_residue)

    # Convert the list of probabilities into a single tensor of shape (L, N, q)
    all_probs_tensor = torch.stack(all_probs, dim=0).to(device)

    if remove_self_loop:     

        for i in range(L):
            current_state_indices = torch.argmax(msa_oh[:, i, :], dim=1).to(device)
            all_probs_tensor[i, torch.arange(N), current_state_indices] = 0

        for n in range(N):
            norm = all_probs_tensor[:,n,:].sum()
            all_probs_tensor[:,n,:] /= norm

        #print("After removing self-loops and renormalization:")
        #print(all_probs_tensor.sum(dim=(0,2)))  # Debug: should be all ones
    
    res = all_probs_tensor 
    
    return res




def compute_gillespie_probabilities(
    msa_oh: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
    remove_self_loop: bool = False,
) -> torch.Tensor:
    """
    Computes Gillespie probabilities for all possible mutations.
    Normalization is global across all sites and residues.

    Unlike site-wise Gibbs, the normalization is global across all sites and residues.
    
    Args:
        msa_oh (Tensor): One-hot encoded sequences of shape (N, L, q).
        params (dict): Model parameters.
        beta (float): Inverse temperature.
        remove_self_loop (bool): If True, set current state probabilities to zero.

    Returns:
        Tensor: Shape (N, L, q) with probabilities for each (site, residue) pair.
                Note: sum over L*q dimension equals 1 (not per-site normalization).
    """

    N, L, q = msa_oh.shape
    device = msa_oh.device
    
    # Expand biases
    biases = params["bias"].unsqueeze(0).expand(N, -1, -1)
    
    # Coupling
    couplings_flat = params["coupling_matrix"].reshape(L * q, L * q)
    chains_flat = msa_oh.reshape(N, L * q)
    coupling_term = torch.matmul(chains_flat, couplings_flat.T).reshape(N, L, q)
    
    # Logits
    logits = beta * (biases + coupling_term)
    
    # ============== SOFTMAX FIRST ===================
    logits_flat = logits.reshape(N, L * q)
    probs_flat = torch.softmax(logits_flat, dim=-1).reshape(N, L, q)

    # ============ REMOVE SELF LOOP ==================
    if remove_self_loop:
        # mask out current amino acid
        current_states = torch.argmax(msa_oh, dim=2)  # (N,L)
        probs_flat = probs_flat.clone()
        probs_flat[torch.arange(N)[:,None], torch.arange(L)[None,:], current_states] = 0.0

        # ============ RENORMALIZE ====================
        # sum over all L*q
        norm = probs_flat.sum(dim=(1,2), keepdim=True)  # (N,1,1)
        probs_flat = probs_flat / norm

    ## OUTPUT MUST BE l,n,q
    probs = probs_flat.permute(1, 0, 2)

    #CHECK NORMALIZATION
    #print("Gillespie probabilities sum check (should be 1.0):")
    #print(probs.sum(dim=(0,2)))  # Debug: should be all ones

    return probs
