import torch
from typing import Dict, List, Any
from sklearn.decomposition import PCA  # Assuming PCA object is of this type

from adabmDCA import one_hot



def gibbs_step_independent_sites2(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Gibbs sampler. This version selects different random sites for each chain. It is
    less efficient than the 'gibbs_step_uniform_sites' function, but it is more suitable for mutating starting from the same wild-type sequence since mutations are independent across chains.

    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    device = chains.device
    dtype = chains.dtype
    # Select a different random site for each sequence in the batch
    idx_batch = torch.randint(0, L, (N,), device=device)
    biases = params["bias"][idx_batch]  # Shape: (N, q)
    couplings_batch = params["coupling_matrix"][idx_batch]  # Shape: (N, q, L, q)
    chains_flat = chains.reshape(N, L * q, 1)
    couplings_flat = couplings_batch.reshape(N, q, L * q)
    coupling_term = torch.bmm(couplings_flat, chains_flat).squeeze(-1)  # (N, q, L*q) @ (N, L*q, 1) -> (N, q, 1) -> (N, q)
    logits = beta * (biases + coupling_term)
    new_residues = one_hot(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1), num_classes=q).to(dtype)
    # Create an index for the batch dimension
    batch_arange = torch.arange(N, device=device)
    chains[batch_arange, idx_batch] = new_residues.squeeze(1)

    return chains







def simulate_gibbs_trajectories(
    msa_oh: torch.Tensor,
    params: Dict[str, torch.Tensor],
    num_sweeps: int = 10,
    max_traj_len: int = 10,
    beta: float = 1.0,
    molec_time: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Runs an MCMC simulation and extracts individual trajectories of shape (max_traj_len, L, q).
    
    If molec_time is True, the trajectory is filtered for unique state changes.
    If molec_time is False, the trajectory uses raw consecutive time steps.
    """
    
    # Initialization
    chains = msa_oh.clone().to(device=device, dtype=dtype)
    N, L, q = chains.shape
    
    # We must run L * num_sweeps to allow enough steps for the filtering case (moelec_time=True)
    total_steps = L * num_sweeps
    
    # 1. MCMC History Collection
    # History includes the initial state (step 0) and 'total_steps' subsequent states.
    chains_history = [chains.clone()]
    
    for _ in range(total_steps):
        # Assuming gibbs_step_independent_sites2 is available and updates 'chains' in place
        gibbs_step_independent_sites2(chains, params, beta=beta) 
        chains_history.append(chains.clone())

    # Stack history into a single tensor: (TotalSteps + 1, N, L, q)
    history_tensor = torch.stack(chains_history, dim=0)
    
    # 2. Individual Trajectory Extraction
    all_trajs = []

    if molec_time:
        # --- PATH 1: FILTERING (Molecular Time) ---
        
        for chain_idx in range(N):
            # Tensor to hold the filtered, output trajectory (Max_Steps, L, q)
            traj = torch.zeros(max_traj_len, L, q, device=device, dtype=dtype)
            
            # Store initial state (always the first element)
            traj[0] = history_tensor[0, chain_idx].clone() 
            
            count = 1  # Write index for 'traj'
            
            # Iterate over the entire collected history (from step 1)
            for step in range(1, history_tensor.shape[0]):
                current_chain = history_tensor[step, chain_idx]
                previous_chain = history_tensor[step-1, chain_idx]
                
                # Filter: Record only if the state is NOT equal to the previous state
                if not torch.equal(current_chain, previous_chain):
                    traj[count] = current_chain
                    count += 1
                    
                    # Stop when the maximum allowed length is reached
                    if count >= max_traj_len:
                        break
            
            all_trajs.append(traj)

    else:
        # --- PATH 2: NO FILTERING (Raw Time) ---
        
        # We take the first 'max_traj_len' steps directly from the history for each chain.
        # History is (TotalSteps + 1, N, L, q). We slice to get (max_traj_len, N, L, q).
        raw_trajs = history_tensor[:max_traj_len]
        
        # Transpose to group by individual chain: (N, max_traj_len, L, q)
        trajs_by_chain = raw_trajs.transpose(0, 1)

        for chain_idx in range(N):
            # Each element in all_trajs is (max_traj_len, L, q)
            all_trajs.append(trajs_by_chain[chain_idx])
            
    return all_trajs


def gibbs_step_independent_sites2_gillespie(
    chains: torch.Tensor,
    params: Dict[str, torch.Tensor],
    beta: float = 1.0,
) -> torch.Tensor:
    """Performs a single mutation using the Gibbs sampler. This version uses the gillespie algorithm with gibbs probability. 
    Args:
        chains (torch.Tensor): One-hot encoded sequences of shape (batch_size, L, q).
        params (Dict[str, torch.Tensor]): Parameters of the model.
        beta (float, optional): Inverse temperature. Defaults to 1.0.

    Returns:
        torch.Tensor: Updated chains.
    """
    N, L, q = chains.shape
    device = chains.device
    dtype = chains.dtype

    # Compute logits for ALL sites in each chain
    # biases: (L, q) -> expand to (N, L, q)
    biases = params["bias"].unsqueeze(0).expand(N, -1, -1)  # Shape: (N, L, q)

    # couplings: (L, q, L, q)
    couplings = params["coupling_matrix"]  # Shape: (L, q, L, q)

    # For each site i, compute coupling term with all other sites
    chains_flat = chains.reshape(N, L * q)  # (N, L*q)

    # Reshape couplings for batch matrix multiplication
    # For site i with residue a, we need: sum over all j,b of J_ij(a,b) * s_jb
    couplings_flat = couplings.reshape(L * q, L * q)  # (L*q, L*q)

    # Compute coupling terms for all chains at once
    # (N, L*q) @ (L*q, L*q).T -> (N, L*q)
    coupling_term_flat = torch.matmul(chains_flat, couplings_flat.T)  # (N, L*q)

    # Reshape back to (N, L, q)
    coupling_term = coupling_term_flat.reshape(N, L, q)

    # Compute logits for all sites
    logits = beta * (biases + coupling_term)  # Shape: (N, L, q)

    # Flatten logits to (N, L*q) and apply softmax over all (site, residue) pairs
    logits_flat = logits.reshape(N, L * q)
    probs_flat = torch.softmax(logits_flat, dim=-1)  # (N, L*q)

    # Sample from the categorical distribution over all (site, residue) pairs
    sampled_indices = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)  # (N,)

    # Convert flat indices back to (site, residue) indices
    sampled_sites = sampled_indices // q  # Which site was selected
    sampled_residues = sampled_indices % q  # Which residue at that site

    # Update chains with one-hot encoding
    batch_arange = torch.arange(N, device=device)
    # Add extra dimension for one_hot: (N,) -> (N, 1)
    new_residues = one_hot(sampled_residues.unsqueeze(-1), num_classes=q).to(dtype).squeeze(1)  # (N, q)
    chains[batch_arange, sampled_sites] = new_residues

    return chains


def simulate_gibbs_trajectories_gillespie(
    msa_oh: torch.Tensor,
    params: Dict[str, torch.Tensor],
    num_sweeps: int = 10,
    max_traj_len: int = 10,
    beta: float = 1.0,
    molec_time: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """
    Runs an MCMC simulation and extracts individual trajectories of shape (max_traj_len, L, q).
    
    If molec_time is True, the trajectory is filtered for unique state changes.
    If molec_time is False, the trajectory uses raw consecutive time steps.
    """
    
    # Initialization
    chains = msa_oh.clone().to(device=device, dtype=dtype)
    N, L, q = chains.shape
    
    # We must run L * num_sweeps to allow enough steps for the filtering case (moelec_time=True)
    total_steps = L * num_sweeps
    
    # 1. MCMC History Collection
    # History includes the initial state (step 0) and 'total_steps' subsequent states.
    chains_history = [chains.clone()]
    
    for _ in range(total_steps):
        # Assuming gibbs_step_independent_sites2 is available and updates 'chains' in place
        gibbs_step_independent_sites2_gillespie(chains, params, beta=beta) 
        chains_history.append(chains.clone())

    # Stack history into a single tensor: (TotalSteps + 1, N, L, q)
    history_tensor = torch.stack(chains_history, dim=0)
    
    # 2. Individual Trajectory Extraction
    all_trajs = []

    if molec_time:
        # --- PATH 1: FILTERING (Molecular Time) ---
        
        for chain_idx in range(N):
            # Tensor to hold the filtered, output trajectory (Max_Steps, L, q)
            traj = torch.zeros(max_traj_len, L, q, device=device, dtype=dtype)
            
            # Store initial state (always the first element)
            traj[0] = history_tensor[0, chain_idx].clone() 
            
            count = 1  # Write index for 'traj'
            
            # Iterate over the entire collected history (from step 1)
            for step in range(1, history_tensor.shape[0]):
                current_chain = history_tensor[step, chain_idx]
                previous_chain = history_tensor[step-1, chain_idx]
                
                # Filter: Record only if the state is NOT equal to the previous state
                if not torch.equal(current_chain, previous_chain):
                    traj[count] = current_chain
                    count += 1
                    
                    # Stop when the maximum allowed length is reached
                    if count >= max_traj_len:
                        break
            
            all_trajs.append(traj)

    else:
        # --- PATH 2: NO FILTERING (Raw Time) ---
        
        # We take the first 'max_traj_len' steps directly from the history for each chain.
        # History is (TotalSteps + 1, N, L, q). We slice to get (max_traj_len, N, L, q).
        raw_trajs = history_tensor[:max_traj_len]
        
        # Transpose to group by individual chain: (N, max_traj_len, L, q)
        trajs_by_chain = raw_trajs.transpose(0, 1)

        for chain_idx in range(N):
            # Each element in all_trajs is (max_traj_len, L, q)
            all_trajs.append(trajs_by_chain[chain_idx])
            
    return all_trajs