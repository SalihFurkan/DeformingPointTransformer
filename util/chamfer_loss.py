import torch
from Chamfer3D.dist_chamfer_3D import chamfer_3DDist

# Initialize the module once (usually inside your Model class)
chamfer_module = chamfer_3DDist()

def calc_chamfer_stacked_fixed_size(p1, p2, offset1, offset2):
    """
    Args:
        p1: (B*N, 3) - Stacked predicted points
        p2: (B*M, 3) - Stacked ground truth points
        offset1, offset2: (B,) - Offsets (cumulative sums). 
                          Used here primarily to determine Batch Size.
    """
    
    # 1. Determine Batch Size and Number of Points (N)
    B = len(offset1)
    Total_N = p1.shape[0]
    Total_M = p2.shape[0]
    
    # Verify that the division is clean (sanity check for "same size" assumption)
    assert Total_N % B == 0, "p1 is not evenly divisible by Batch Size!"
    assert Total_M % B == 0, "p2 is not evenly divisible by Batch Size!"
    
    N = int(Total_N / B)
    M = int(Total_M / B)

    # 2. Reshape ("Un-stack") the tensors
    # .view() is instant (zero-copy) if data is contiguous
    p1_batch = p1.view(B, N, 3)
    p2_batch = p2.view(B, M, 3)

    # 3. Compute Chamfer using the CUDA extension
    # dist1: (B, N) distance from p1 to nearest in p2
    # dist2: (B, M) distance from p2 to nearest in p1
    dist1, dist2, idx1, idx2 = chamfer_module(p1_batch, p2_batch)

    # 4. Calculate Mean Loss
    # We average over points first, then over the batch
    loss = torch.mean(dist1) + torch.mean(dist2)
    
    return loss

def calc_chamfer_stacked_objectwise(p1, p2, l1, l2, batch_size):
    """
    Args:
        p1, p2: (Total_Points, 3) - Stacked points
        l1, l2: (Total_Points,)   - Stacked labels corresponding to p1 and p2
        batch_size: int           - Number of clouds in the stack
    """
    # 1. Infer Dimensions
    Total_Points = p1.shape[0] # = 327680
    N = Total_Points // batch_size # Points per cloud # N = 20480
    
    # 2. Unstack to (B, N) views (Zero-Copy)
    # We must process sorting per-cloud, so we view them as batches.
    p1_batch = p1.view(batch_size, N, 3)
    l1_batch = l1.view(batch_size, N)
    
    p2_batch = p2.view(batch_size, N, 3)
    l2_batch = l2.view(batch_size, N)

    # 3. Get Sorting Indices (The "Un-scrambling" Key)
    # argsort guarantees that labels [0,0, 1,1, 2,2...] line up.
    # shape: (B, N)
    idx1 = torch.argsort(l1_batch, dim=1) 
    idx2 = torch.argsort(l2_batch, dim=1)

    # 4. Reorder the Points (The Memory Copy)
    # We use 'gather' to rearrange points based on the sort indices.
    # Advanced indexing: p1_batch[batch_indices, idx1]
    # This creates a contiguous tensor where Obj0 comes first, then Obj1, etc.
    batch_indices = torch.arange(batch_size, device=p1.device).view(-1, 1).expand(-1, N)
    
    p1_sorted = p1_batch[batch_indices, idx1]
    p2_sorted = p2_batch[batch_indices, idx2]

    # 5. "Super Batch" Transformation
    # Infer M (Objects) and O (Points per Object)
    # We look at the first batch item to count unique labels
    M = torch.unique(l1_batch[0]).numel()
    O = N // M

    # View as (B*M, O, 3)
    p1_super = p1_sorted.view(batch_size * M, O, 3)
    p2_super = p2_sorted.view(batch_size * M, O, 3)

    # 6. Calculate Chamfer
    # Now calculating B*M pairs in parallel
    dist1, dist2, _, _ = chamfer_module(p1_super, p2_super)

    loss = torch.mean(dist1) + torch.mean(dist2)
    return loss