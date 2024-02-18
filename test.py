import obj_io
import poisson_mesh_connecting as pmc


def load_boundary(path):
    vids = []
    with open(path, 'r') as fp:
        for line in fp:
            vids.append(int(line))
    return vids


if __name__ == '__main__':
    face = obj_io.loadObj('./data/face/mediapipe_face.obj')
    face_boundary = load_boundary('./data/face/boundary.txt')
    cylinder = obj_io.loadObj('./data/cylinder/cylinder.obj')
    cylinder_boundary = load_boundary('./data/cylinder/boundary.txt')

    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        cylinder.verts, cylinder.indices, cylinder_boundary,
        face.verts, face.indices, face_boundary, True)
    obj_io.saveObjSimple('face2cylinder.obj', poisson_verts, poisson_indices)

    poisson_verts, poisson_indices = pmc.poisson_mesh_connecting(
        face.verts, face.indices, face_boundary,
        cylinder.verts, cylinder.indices, cylinder_boundary, True)
    obj_io.saveObjSimple('cylinder2face.obj', poisson_verts, poisson_indices)
