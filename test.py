import obj_io
import poisson_mesh_connecting as pmc
import time


def load_vids(path):
    vids = []
    with open(path, "r") as fp:
        for line in fp:
            vids.append(int(line))
    return vids


def test_connecting():
    face = obj_io.loadObj("./data/face/mediapipe_face.obj")
    face_boundary = load_vids("./data/face/boundary.txt")
    cylinder = obj_io.loadObj("./data/cylinder/cylinder.obj")
    cylinder_boundary = load_vids("./data/cylinder/boundary.txt")

    st = time.time()
    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        cylinder.verts,
        cylinder.indices,
        cylinder_boundary,
        face.verts,
        face.indices,
        face_boundary,
        True,
        True,
    )
    et = time.time()
    print("face2cylinder", "sparse", et - st)
    obj_io.saveObjSimple("face2cylinder.obj", poisson_verts, poisson_indices)

    st = time.time()
    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        face.verts,
        face.indices,
        face_boundary,
        cylinder.verts,
        cylinder.indices,
        cylinder_boundary,
        True,
        True,
    )
    obj_io.saveObjSimple("cylinder2face.obj", poisson_verts, poisson_indices)
    et = time.time()
    print("cylinder2face", "sparse", et - st)

    st = time.time()
    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        cylinder.verts,
        cylinder.indices,
        cylinder_boundary,
        face.verts,
        face.indices,
        face_boundary,
        True,
        False,
    )
    et = time.time()
    print("face2cylinder", "dense", et - st)
    obj_io.saveObjSimple("face2cylinder_dense.obj", poisson_verts, poisson_indices)

    st = time.time()
    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        face.verts,
        face.indices,
        face_boundary,
        cylinder.verts,
        cylinder.indices,
        cylinder_boundary,
        True,
        False,
    )
    obj_io.saveObjSimple("cylinder2face_dense.obj", poisson_verts, poisson_indices)
    et = time.time()
    print("cylinder2face", "dense", et - st)


def test_replace():
    sphere = obj_io.loadObj("./data/same_topology/sphere.obj")
    cube = obj_io.loadObj("./data/same_topology/cube.obj")
    building = obj_io.loadObj("./data/same_topology/building.obj")
    replace_vids = load_vids("./data/same_topology/replace2.txt")

    st = time.time()
    poisson_verts = pmc.poisson_mesh_replace(
        sphere.verts, cube.verts, sphere.indices, replace_vids
    )
    et = time.time()
    print("cube2sphere", "sparse", et - st)
    obj_io.saveObjSimple("cube2sphere.obj", poisson_verts, sphere.indices)

    st = time.time()
    poisson_verts = pmc.poisson_mesh_replace(
        sphere.verts, building.verts, sphere.indices, replace_vids
    )
    et = time.time()
    print("building2sphere", "sparse", et - st)
    obj_io.saveObjSimple("building2sphere.obj", poisson_verts, sphere.indices)


if __name__ == "__main__":
    test_replace()
    test_connecting()
