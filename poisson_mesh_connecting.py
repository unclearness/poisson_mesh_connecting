import numpy as np
try:
    from scipy.sparse import coo_matrix, linalg
    is_scipy_available = True
except ImportError:
    is_scipy_available = False


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.

    Parameters
    ----------
    src : (M, N) array_like
        Source coordinates.
    dst : (M, N) array_like
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.

    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.

    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, :DOI:`10.1109/34.88573`

    """
    src = np.asarray(src)
    dst = np.asarray(dst)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = dst_demean.T @ src_demean / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.float64)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = U @ V
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = U @ np.diag(d) @ V
            d[dim - 1] = s
    else:
        T[:dim, :dim] = U @ np.diag(d) @ V

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
    T[:dim, :dim] *= scale

    return T


def generate_vertex_adjacency(indices, vnum):
    adj = [set() for _ in range(vnum)]
    for face in indices:
        adj[face[0]].add(face[1])
        adj[face[1]].add(face[0])

        adj[face[2]].add(face[1])
        adj[face[1]].add(face[2])

        adj[face[0]].add(face[2])
        adj[face[2]].add(face[0])
    return adj


def compute_mesh_laplacian(verts, indices, adj=None):
    if adj is None:
        adj = generate_vertex_adjacency(indices, len(verts))
    adj_nums = np.array([len(a) for a in adj])
    # laps = np.zeros_like(verts)
    laps = adj_nums[..., None] * verts
    # TODO: Batch
    for vidx in range(len(verts)):
        # if len(adj[vidx]) < 1:
        #     continue

        # laps[vidx] = len(adj[vidx]) * verts[vidx]
        for adj_vidx in adj[vidx]:
            laps[vidx] -= verts[adj_vidx]
    return laps


def solve_poisson_naive(verts, indices,
                        boundary_vids, boundary_verts_new,
                        use_sparse=is_scipy_available):
    boundary_vids_set = set(boundary_vids)
    va = generate_vertex_adjacency(indices, len(verts))

    # Laplacian on the original mesh
    laps = compute_mesh_laplacian(verts, indices, va)

    verts_updated = verts.copy()
    verts_updated[boundary_vids] = boundary_verts_new

    org2prm_map, prm2org_map = {}, {}
    prm_idx = 0
    for ovidx in range(len(verts)):
        if ovidx in boundary_vids_set:
            continue
        org2prm_map[ovidx] = prm_idx
        prm2org_map[prm_idx] = ovidx
        prm_idx += 1

    num_param = len(verts) - len(boundary_vids)
    b_offset = np.zeros((num_param, 3))
    coo_row = []
    coo_col = []
    coo_data = []

    def add_triplet(r, c, d):
        coo_row.append(r)
        coo_col.append(c)
        coo_data.append(d)

    cur_row = 0
    for ovidx in range(len(verts)):
        if ovidx in boundary_vids_set:
            continue
        vadj = va[ovidx]
        add_triplet(cur_row, org2prm_map[ovidx], float(len(vadj)))
        for vid in vadj:
            if vid in boundary_vids:
                b_offset[cur_row] += verts_updated[vid]
            else:
                add_triplet(cur_row, org2prm_map[vid], -1.0)
        cur_row += 1

    # Solve XYZ independently
    solved = np.zeros((num_param, 3))
    if use_sparse:
        # Sparse version
        A = coo_matrix((coo_data, (coo_row, coo_col)),
                       shape=(num_param, num_param))
        A = A.tocsr()
        for c in range(3):
            b = b_offset[..., c]
            b += laps[list(prm2org_map.values())][..., c]
            solved[..., c] = linalg.spsolve(A, b)
    else:
        # Dense version
        A = np.zeros((num_param, num_param))
        for r, c, d in zip(coo_row, coo_col, coo_data):
            A[r, c] = d
        for c in range(3):
            b = b_offset[..., c]
            b += laps[list(prm2org_map.values())][..., c]
            try:
                # Should always succeed...
                solved[..., c] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                # Fail safe
                solved[..., c], residuals, rank, s = np.linalg.lstsq(
                    A, b, None)
    # Copy to original index
    verts_poisson = verts_updated.copy()
    verts_poisson[list(prm2org_map.values())] = solved

    return verts_poisson


def connect_mesh(verts0, indices0, boundary0, verts1, indices1, boundary1):
    boundary12boundary0 = {}
    for i in range(len(boundary0)):
        boundary12boundary0[boundary1[i]] = boundary0[i]
    boundary1_set = set(boundary1)

    removed_vidx = 0
    verts1_boundary_removed = []
    vert1org2removed = {}
    for vidx in range(len(verts1)):
        if vidx in boundary1_set:
            continue
        verts1_boundary_removed.append(verts1[vidx])
        vert1org2removed[vidx] = removed_vidx
        removed_vidx += 1

    merged_verts = np.concatenate((verts0, verts1_boundary_removed), axis=0)
    merged_indices = indices0.tolist()

    offset = len(verts0)

    for fidx in range(len(indices1)):
        new_face = [-1, -1, -1]
        for j in range(3):
            org_vid1 = indices1[fidx][j]
            if org_vid1 in vert1org2removed:
                new_face[j] = vert1org2removed[org_vid1] + offset
            else:
                new_face[j] = boundary12boundary0[org_vid1]
        merged_indices.append(new_face)
    merged_indices = np.array(merged_indices)
    return merged_verts, merged_indices


def poisson_mesh_connecting(pinned_verts, pinned_indices, pinned_boundary_vids,
                            free_verts, free_indices, free_boundary_vids,
                            connect, use_sparse=is_scipy_available):
    # Alignment based on the boundary correspondence
    pinned_boundary_verts = pinned_verts[pinned_boundary_vids]
    free_boundary_verts = free_verts[free_boundary_vids]

    free2pinned = umeyama(free_boundary_verts, pinned_boundary_verts, True)
    free_verts4 = np.ones((len(free_verts), 4))
    free_verts4[..., :3] = free_verts
    free_verts_transed = (free2pinned @ free_verts4.T).T[..., :3]

    # Solve poisson
    verts_poisson = solve_poisson_naive(free_verts_transed, free_indices,
                                        free_boundary_vids,
                                        pinned_boundary_verts,
                                        use_sparse)
    if not connect:
        return verts_poisson, free_indices
    return connect_mesh(pinned_verts, pinned_indices, pinned_boundary_vids,
                        verts_poisson, free_indices, free_boundary_vids)
