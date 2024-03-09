import numpy as np

try:
    from scipy.sparse import coo_matrix

    _is_scipy_available = True
except ImportError:
    _is_scipy_available = False


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


def generate_face_adjacency_mat(
    indices, vnum, valid_vert_mask=None, use_sparse=_is_scipy_available
):
    coo_row = []
    coo_col = []
    coo_data = []

    def add_triplet(r, c, d):
        coo_row.append(r)
        coo_col.append(c)
        coo_data.append(d)

    if valid_vert_mask is None:
        valid_vert_mask = np.ones((vnum), dtype=bool)

    for fid, face in enumerate(indices):
        valid_face = (
            valid_vert_mask[face[0]]
            and valid_vert_mask[face[1]]
            and valid_vert_mask[face[2]]
        )
        if not valid_face:
            continue
        add_triplet(face[0], face[1], fid + 1)
        add_triplet(face[1], face[2], fid + 1)
        add_triplet(face[2], face[0], fid + 1)

    if use_sparse:
        # Sparse version
        A = coo_matrix((coo_data, (coo_row, coo_col)), shape=(vnum, vnum), dtype=int)
        A = A.tocsr()
    else:
        # Dense version
        A = np.zeros((vnum, vnum), dtype=int)
        for r, c, d in zip(coo_row, coo_col, coo_data):
            A[r, c] = d
    return A


def get_boundary_edges(A, indices):
    boundary_edges = []
    for face in indices:
        for edge in [[0, 1], [1, 2], [2, 0]]:
            idx0, idx1 = edge
            # If valid_vert_mask was specified, face may not have edge
            if A[face[idx0], face[idx1]] != 0:
                # Check the "reverse edge" face[1]->face[0]
                # of the original edge face[0]->face[1]
                if A[face[idx1], face[idx0]] == 0:
                    # If the reverse edge is not shared with other faces,
                    # add the original edge as a part of boundary
                    boundary_edges.append([face[idx0], face[idx1]])
    boundary_vids = set()
    for edge in boundary_edges:
        boundary_vids.add(edge[0])
        boundary_vids.add(edge[1])
    return boundary_edges, boundary_vids


def find_boundary_loops(boundary_edges):
    boundary_loops = []
    cur_edge = boundary_edges.pop(0)
    cur_loop = []
    cur_loop.append(cur_edge)
    while True:
        connected_idx = -1
        for idx, edge in enumerate(boundary_edges):
            if cur_edge[1] == edge[0]:
                cur_edge = edge
                cur_loop.append(cur_edge)
                connected_idx = idx
        if connected_idx > -1:
            boundary_edges.pop(connected_idx)
            continue

        loop_closed = cur_edge[1] == cur_loop[0][0]
        if not loop_closed:
            raise Exception("Loop is not closed")
        boundary_loops.append(cur_loop)
        if len(boundary_edges) < 1:
            break
        cur_edge = boundary_edges.pop(0)
        cur_loop = [cur_edge]
    return boundary_loops


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


def remove_vert_attrs(attrs_list, indices, to_remove_vids):
    indices_ = []
    table = {}
    count = 0
    to_keep_vids = []
    for i in range(len(attrs_list[0])):
        if i in to_remove_vids:
            table[i] = None
            continue
        to_keep_vids.append(i)
        table[i] = count
        count += 1

    for face in indices:
        ok = True
        new_face = []
        for vid in face:
            if table[vid] is None:
                ok = False
                break
            new_face.append(table[vid])
        if not ok:
            continue
        indices_.append(new_face)

    attrs_list_ = []
    for attrs in attrs_list:
        attrs_list_.append(attrs[to_keep_vids])
    return np.asarray(attrs_list_), np.asarray(indices_), table


def remove_verts(verts, indices, to_remove_vids):
    attrs, indices_removed, org2new = remove_vert_attrs(
        [verts], indices, to_remove_vids
    )
    return attrs[0], indices_removed, org2new


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
