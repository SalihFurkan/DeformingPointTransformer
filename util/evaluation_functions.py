import numpy as np
import torch
import os
import pandas as pd
import nibabel as nib
from tqdm import tqdm

def convert_to_mm(points, affine, device):
    """
    Convert normalized [-1,1] point clouds to world coordinates (mm), batch compatible.

    points: (B, N, 3) tensor, normalized in [-1,1]
    affine: list containing one (4,4) tensor or numpy array
    device: torch device
    """
    B, N, _ = points.shape
    Dx, Dy, Dz = 319, 259, 315
    dims = torch.tensor([Dx - 1, Dy - 1, Dz - 1], dtype=torch.float32, device=device)  # note -1

    # convert affine to torch tensor (4,4)
    affine = torch.as_tensor(affine[0], dtype=torch.float32, device=device)

    # ensure points are float and on the correct device
    points_norm = points.float().to(device)

    # Step 1: normalized [-1,1] -> voxel indices in flipped space
    voxel_flipped = (points_norm + 1.0) * 0.5 * dims  # broadcast (B,N,3) * (3,)

    # Step 2: undo the old flip along axis 0 and 1
    voxel_orig = torch.empty_like(voxel_flipped)
    voxel_orig[..., 0] = (Dx - 1) - voxel_flipped[..., 0]  # unflip axis 0
    voxel_orig[..., 1] = (Dy - 1) - voxel_flipped[..., 1]  # unflip axis 1
    voxel_orig[..., 2] = voxel_flipped[..., 2]             # axis 2 untouched

    # Step 3: append ones for homogeneous coordinates
    ones = torch.ones((B, N, 1), dtype=torch.float32, device=device)
    voxel_h = torch.cat([voxel_orig, ones], dim=-1)  # shape (B, N, 4)

    # Step 4: apply affine (batch matmul)
    # affine is (4,4), expand to (1,4,4) for broadcasting
    affine_exp = affine.unsqueeze(0)  # (1,4,4)
    world_h = torch.matmul(voxel_h, affine_exp.transpose(1,2))  # (B,N,4)

    world = world_h[..., :3]  # drop homogeneous coordinate

    # return on CPU and detached
    return world

def convert_organs_mm(points, labels, batch_size, affine, device):
	Total_Points = points.shape[0]
	N = Total_Points // batch_size # Points per cloud

	# 2. Unstack to (B, N) views (Zero-Copy)
	# We must process sorting per-cloud, so we view them as batches.
	p1_batch = points.view(batch_size, N, 3)
	l1_batch = labels.view(batch_size, N)

	# 3. Get Sorting Indices (The "Un-scrambling" Key)
	# argsort guarantees that labels [0,0, 1,1, 2,2...] line up.
	# shape: (B, N)
	idx1 = torch.argsort(l1_batch, dim=1) 

	# 4. Reorder the Points (The Memory Copy)
	# We use 'gather' to rearrange points based on the sort indices.
	# Advanced indexing: p1_batch[batch_indices, idx1]
	# This creates a contiguous tensor where Obj0 comes first, then Obj1, etc.
	batch_indices = torch.arange(batch_size, device=points.device).view(-1, 1).expand(-1, N)

	p1_sorted = p1_batch[batch_indices, idx1]

	# 5. "Super Batch" Transformation
	# Infer M (Objects) and O (Points per Object)
	# We look at the first batch item to count unique labels
	M = torch.unique(l1_batch[0]).numel()
	O = N // M

	# View as (B*M, O, 3)
	p1_super = p1_sorted.view(batch_size * M, O, 3)

	# Super Batched Organ Points
	p_mm = convert_to_mm(p1_super, affine, device) # (B * M, O, 3)

	return p_mm, M, O

def convert_batchpts_mm(points, batch_size, affine, device):
	Total_Points = points.shape[0]
	N = Total_Points // batch_size # Points per cloud

	# 2. Unstack to (B, N) views (Zero-Copy)
	# We must process sorting per-cloud, so we view them as batches.
	p1_batch = points.view(batch_size, N, 3)

	# Super Batched Organ Points
	p_mm = convert_to_mm(p1_batch, affine, device) # (B * M, O, 3)

	return p_mm

from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

# Initialize the module once (usually inside your Model class)
chamfer_module = chamfer_3DDist()
def evaluate(p1, p2, l1, l2, batch_size, affine, device):
    
    p1_mm, M, O = convert_organs_mm(p1, l1, batch_size, affine, device) # (B * M, O, 3) 
    p2_mm, M, O = convert_organs_mm(p2, l2, batch_size, affine, device) # (B * M, O, 3) 

    dist1_sq, dist2_sq, _, _ = chamfer_module(p1_mm, p2_mm)
    dist1 = torch.sqrt(dist1_sq)
    dist2 = torch.sqrt(dist2_sq)

    # CD
    loss_per_instance = ( torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1) ) / 2
    loss_matrix = loss_per_instance.view(batch_size, M)
    per_organ_loss = torch.mean(loss_matrix, dim=0)
    total_loss = torch.mean(per_organ_loss).item()
    cd = {"total_cd": total_loss, "per_organ_cd": per_organ_loss}

    # HD95
    pred_95 = torch.quantile(dist1, 0.95, dim=1) 
    target_95 = torch.quantile(dist2, 0.95, dim=1) 
    hd95_instance = torch.max(pred_95, target_95)
    hd95_matrix = hd95_instance.view(batch_size, M)
    per_organ_hd95 = torch.mean(hd95_matrix, dim=0)
    mean_hd95 = torch.mean(per_organ_hd95).item()
    hd95 = {"total_hd95": mean_hd95, "per_organ_hd95": per_organ_hd95}

    # Difference
    gt_min = p2_mm.min(dim=1).values
    gt_max = p2_mm.max(dim=1).values
    pr_min = p1_mm.min(dim=1).values
    pr_max = p1_mm.max(dim=1).values

    dim_max_err = torch.abs(pr_max - gt_max)
    dim_min_err = torch.abs(pr_min - gt_min)

    dim_max_err = dim_max_err.view(batch_size, M, 3)
    dim_min_err = dim_min_err.view(batch_size, M, 3)

    avg_max_err = dim_max_err.mean(dim=0)
    avg_min_err = dim_min_err.mean(dim=0)


    metrics = {"max_error_mm" : avg_max_err, "min_error_mm" : avg_min_err}

    return {**cd, **hd95, **metrics}

def format_model_output(model_output):
    """
    Args:
        model_output: List of K tuples. Each tuple[0] is (B, N, 3).
    Returns:
        all_points: (B * K * N, 3)
        all_labels: (B * K * N)
    """
    # 1. Extract points from tuples and stack: (B, K, N, 3)
    # This naturally orders them: Patient 0 [Org0, Org1...] Patient 1 [Org0...]
    pts_stacked = torch.stack([out[0] for out in model_output], dim=1)
    
    B, K, N, _ = pts_stacked.shape
    device = pts_stacked.device
    
    # 2. Flatten points: (B * K * N, 3)
    all_points = pts_stacked.view(-1, 3)
    
    # 3. Generate Labels
    # Create [0,0..0, 1,1..1, ... K-1..K-1] (Size: K*N)
    organ_ids = torch.arange(K, device=device).repeat_interleave(N)
    
    # Repeat for every patient in batch (Size: B*K*N)
    all_labels = organ_ids.repeat(B)
    
    return all_points, all_labels

def evaluate_with_std(p1, p2, l1, l2, batch_size, affine, device):
    
    p1_mm, M, O = convert_organs_mm(p1, l1, batch_size, affine, device) # (B * M, O, 3) 
    p2_mm, M, O = convert_organs_mm(p2, l2, batch_size, affine, device) # (B * M, O, 3) 

    dist1_sq, dist2_sq, _, _ = chamfer_module(p1_mm, p2_mm)
    dist1 = torch.sqrt(dist1_sq)
    dist2 = torch.sqrt(dist2_sq)

    # CD
    loss_per_instance = ( torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1) ) / 2
    loss_matrix = loss_per_instance.view(batch_size, M)
    per_organ_loss = torch.mean(loss_matrix, dim=0)
    std_per_organ_loss = torch.std(loss_matrix, dim=0)
    total_loss = torch.mean(per_organ_loss).item()
    cd = {"total_cd": total_loss, "per_organ_cd": per_organ_loss, "std_per_organ_cd": std_per_organ_loss}

    # HD95
    pred_95 = torch.quantile(dist1, 0.95, dim=1) 
    target_95 = torch.quantile(dist2, 0.95, dim=1) 
    hd95_instance = torch.max(pred_95, target_95)
    hd95_matrix = hd95_instance.view(batch_size, M)
    per_organ_hd95 = torch.mean(hd95_matrix, dim=0)
    std_per_organ_hd95 = torch.std(hd95_matrix, dim=0)
    mean_hd95 = torch.mean(per_organ_hd95).item()
    hd95 = {"total_hd95": mean_hd95, "per_organ_hd95": per_organ_hd95, "std_per_organ_hd95": std_per_organ_hd95}

    # Difference
    gt_min = p2_mm.min(dim=1).values
    gt_max = p2_mm.max(dim=1).values
    pr_min = p1_mm.min(dim=1).values
    pr_max = p1_mm.max(dim=1).values

    dim_max_err = pr_max - gt_max
    dim_min_err = pr_min - gt_min

    dim_max_err = dim_max_err.view(batch_size, M, 3)
    dim_min_err = dim_min_err.view(batch_size, M, 3)

    avg_max_err = dim_max_err.mean(dim=0)
    avg_min_err = dim_min_err.mean(dim=0)
    std_max_err = dim_max_err.std(dim=0)
    std_min_err = dim_min_err.std(dim=0)

    metrics = {"max_error_mm" : avg_max_err, "std_max_error_mm" : std_max_err, "min_error_mm" : avg_min_err, "std_min_error_mm" : std_min_err}

    return {**cd, **hd95, **metrics}

def evaluate_test(test_loader, label_organs, device):
    # Currently only calculates mean vs target
    # Trained metrics are to do.

    all_metrics = {"cd": {}, "hd95": {}, "max_width": {}, "min_width": {}, "max_depth": {}, "min_depth": {}, "max_height": {}, "min_height": {}}
    for metric, metric_results in all_metrics.items():
        for label, organ in label_organs.items():
            metric_results[organ] = 0.0

    cd, hd95 = 0.0, 0.0

    for i, (body_coord, body_offset, mean_coord, mean_labels, mean_offset, target_coord, target_labels, target_offset, transform_matrix, patient_path) in enumerate(tqdm(test_loader)): 
        mean_coord, mean_labels, target_coord, target_labels = mean_coord.to(device), mean_labels.to(device), target_coord.to(device), target_labels.to(device)
        B = len(mean_offset)
        results = evaluate(mean_coord, target_coord, mean_labels, target_labels, batch_size=B, affine=transform_matrix[0], device=device)

        cd += results["total_cd"]
        hd95 += results["total_hd95"]

        for label, organ in label_organs.items():
            all_metrics["cd"][organ] += results["per_organ_cd"][int(label)]
            all_metrics["hd95"][organ] += results["per_organ_hd95"][int(label)]
            all_metrics["max_width"][organ] += results["max_error_mm"][int(label), 0]
            all_metrics["max_depth"][organ] += results["max_error_mm"][int(label), 1]
            all_metrics["max_height"][organ] += results["max_error_mm"][int(label), 2]
            all_metrics["min_width"][organ] += results["min_error_mm"][int(label), 0]
            all_metrics["min_depth"][organ] += results["min_error_mm"][int(label), 1]
            all_metrics["min_height"][organ] += results["min_error_mm"][int(label), 2]
        
    cd /= len(test_loader)
    hd95 /= len(test_loader)

    for metric, metric_results in all_metrics.items():
        for label, organ in label_organs.items():
            metric_results[organ] /= len(test_loader)

    all_metrics = {metric: {organ: value.item() for organ, value in metric_results.items()} for metric, metric_results in all_metrics.items()}
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df

