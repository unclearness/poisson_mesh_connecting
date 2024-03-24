import numpy as np

from geom_util import (
    umeyama,
    generate_vertex_adjacency,
    compute_mesh_laplacian,
    generate_face_adjacency_mat,
    get_boundary_edges,
    remove_verts,
    connect_mesh,
)

try:
    from scipy.sparse import coo_matrix, linalg
    _is_scipy_available = True
except ImportError:
    _is_scipy_available = False
    # Own implementation
    from sparse import coo_matrix, linalg


def solve_poisson_naive(
    verts, indices, boundary_vids, boundary_verts_new, use_sparse=_is_scipy_available
):
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
        A = coo_matrix((coo_data, (coo_row, coo_col)), shape=(num_param, num_param))
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
                solved[..., c], residuals, rank, s = np.linalg.lstsq(A, b, None)
    # Copy to original index
    verts_poisson = verts_updated.copy()
    verts_poisson[list(prm2org_map.values())] = solved

    return verts_poisson


def poisson_mesh_connecting(
    pinned_verts,
    pinned_indices,
    pinned_boundary_vids,
    free_verts,
    free_indices,
    free_boundary_vids,
    connect,
    use_sparse=_is_scipy_available,
    similarity_transform=True,
):

    pinned_boundary_verts = pinned_verts[pinned_boundary_vids]
    free_boundary_verts = free_verts[free_boundary_vids]

    free_verts_transed = free_verts
    if similarity_transform:
        # Alignment based on the boundary correspondence
        free2pinned = umeyama(free_boundary_verts, pinned_boundary_verts, True)
        free_verts4 = np.ones((len(free_verts), 4))
        free_verts4[..., :3] = free_verts
        free_verts_transed = (free2pinned @ free_verts4.T).T[..., :3]

    # Solve poisson
    verts_poisson = solve_poisson_naive(
        free_verts_transed,
        free_indices,
        free_boundary_vids,
        pinned_boundary_verts,
        use_sparse,
    )
    if not connect:
        return verts_poisson, free_indices
    return connect_mesh(
        pinned_verts,
        pinned_indices,
        pinned_boundary_vids,
        verts_poisson,
        free_indices,
        free_boundary_vids,
    )


def poisson_mesh_replace(
    pinned_verts,
    free_verts,
    indices,
    replace_vids,
    use_sparse=_is_scipy_available,
    similarity_transform=True,
):

    def get_boundary_vids(indices, vnum, vmask=None):
        A = generate_face_adjacency_mat(indices, vnum, vmask, use_sparse=use_sparse)
        boundary_edges, boundary_vids = get_boundary_edges(A, indices)
        return list(boundary_vids)

    replace_mask = np.zeros((len(pinned_verts)), dtype=bool)
    replace_mask[replace_vids] = True
    boundary_vids = get_boundary_vids(indices, len(pinned_verts), replace_mask)
    free_mask = np.zeros((len(pinned_verts)), dtype=bool)
    free_mask[replace_vids] = True
    free_mask[boundary_vids] = False
    pinned_mask = np.bitwise_not(free_mask)

    pinned_mask[boundary_vids] = False  # Keep boundary False
    pinned_vids = np.transpose(pinned_mask.nonzero()).flatten()
    free_vids = np.transpose(free_mask.nonzero()).flatten()

    pinned_verts_separated, pinned_verts_separated_indices, pinned_org2new = (
        remove_verts(pinned_verts, indices, free_vids)
    )
    pinned_boundary_vids = get_boundary_vids(
        pinned_verts_separated_indices, len(pinned_verts_separated)
    )

    free_verts_separated, free_verts_separated_indices, pinned_org2new = remove_verts(
        free_verts, indices, pinned_vids
    )
    free_boundary_vids = get_boundary_vids(
        free_verts_separated_indices, len(free_verts_separated)
    )

    verts_poisson, _ = poisson_mesh_connecting(
        pinned_verts_separated,
        pinned_verts_separated_indices,
        pinned_boundary_vids,
        free_verts_separated,
        free_verts_separated_indices,
        free_boundary_vids,
        False,
        use_sparse=use_sparse,
        similarity_transform=similarity_transform,
    )

    replaced = pinned_verts.copy()
    replaced[replace_vids] = verts_poisson
    return replaced
