import numpy as np
import torch
import scipy.spatial as spatial
import scipy.sparse as sparse

def build_interp_matrix(act_coords, opd_coords):
    tri = spatial.Delaunay(act_coords)
    simps = tri.find_simplex(opd_coords)

    out_mask = (simps == -1)
    in_mask = ~out_mask
    
    b = tri.transform[simps[in_mask], :2]
    c = tri.transform[simps[in_mask], 2]
    pts = opd_coords[in_mask] - c
    bary_coords = np.empty((pts.shape[0], 3))
    bary_coords[:, 0] = b[:, 0, 0] * pts[:, 0] + b[:, 0, 1] * pts[:, 1]
    bary_coords[:, 1] = b[:, 1, 0] * pts[:, 0] + b[:, 1, 1] * pts[:, 1]
    bary_coords[:, 2] = 1.0 - bary_coords[:, 0] - bary_coords[:, 1]
    
    vertices = tri.simplices[simps[in_mask]]
    
    rows = np.repeat(np.nonzero(in_mask)[0], 3)
    cols = vertices.flatten()
    data = bary_coords.flatten()
    
    if np.any(out_mask):
        tree = spatial.cKDTree(act_coords)
        _, nearest_idx = tree.query(opd_coords[out_mask])
        rows_out = np.nonzero(out_mask)[0]
        cols_out = nearest_idx
        data_out = np.ones(len(rows_out))
        
        rows = np.concatenate([rows, rows_out])
        cols = np.concatenate([cols, cols_out])
        data = np.concatenate([data, data_out])
        
    interp_matrix = sparse.coo_matrix((data, (rows, cols)), shape=(opd_coords.shape[0], act_coords.shape[0]))
    
    indices = np.vstack((interp_matrix.row, interp_matrix.col))
    interp_matrix_torch = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.int64),
        torch.tensor(interp_matrix.data, dtype=torch.float64),
        size=interp_matrix.shape,
        device="cpu"
    ).coalesce()
    
    return interp_matrix_torch

act = np.random.rand(10, 2)
opd = np.random.rand(100, 2)
matrix = build_interp_matrix(act, opd)

coefs = torch.rand(10)
coefs_2d = coefs.unsqueeze(1)
out = torch.sparse.mm(matrix, coefs_2d).squeeze(1)

print("Matrix shape:", matrix.shape)
print("Output shape:", out.shape)
