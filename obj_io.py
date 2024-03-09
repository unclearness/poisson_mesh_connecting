from dataclasses import dataclass, field
from typing import TypeAlias
import numpy as np


@dataclass
class ObjMesh:
    verts: np.array = field(default_factory=np.array)
    uvs: np.array = field(default_factory=np.array)
    normals: np.array = field(default_factory=np.array)
    indices: np.array = field(default_factory=np.array)
    uv_indices: np.array = field(default_factory=np.array)
    normal_indices: np.array = field(default_factory=np.array)
    vert_colors: np.array = field(default_factory=np.array)
    mtl_file: str = field(default_factory=str)
    mtl_names: list[str] = field(default_factory=list)

    def eunsureNumpy(self):
        for k, v in self.__dict__.items():
            if not isinstance(getattr(self, k), np.ndarray):
                if "mtl" in k:
                    continue
                elif "indices" in k:
                    setattr(self, k, np.asarray(v, dtype=np.int32))
                else:
                    setattr(self, k, np.asarray(v, dtype=float))

    def ensureList(self):
        for k, v in self.__dict__.items():
            if isinstance(getattr(self, k), np.ndarray):
                setattr(self, k, v.tolist())

    def recomputeNormals(self):
        POLY_NUM = len(self.indices[0])

        def normalize(v):
            return v / np.linalg.norm(v, axis=1, keepdims=True)

        if POLY_NUM >= 3:
            v10 = normalize(
                self.verts[self.indices[..., 1]] - self.verts[self.indices[..., 0]]
            )
            v20 = normalize(
                self.verts[self.indices[..., 2]] - self.verts[self.indices[..., 0]]
            )
            face_normals_tmp = normalize(np.cross(v10, v20, axis=1))
        else:
            raise Exception("")
        if POLY_NUM == 3:
            face_normals = face_normals_tmp
        else:
            # (FACE_NUM, POLY_NUM, 3)
            def pca(X: np.array):
                F, P, _ = X.shape
                demean = X - X.mean(axis=1, keepdims=True)
                cov = np.einsum("ikj,ikl->ijl", demean, demean)
                vals, vecs = np.linalg.eig(cov)
                min_index = np.argsort(vals, axis=1)
                vecs = vecs.transpose(0, 2, 1)
                n = np.take_along_axis(vecs, min_index[..., None], axis=1)
                return n[:, 0]

            face_verts = self.verts[self.indices]
            face_normals = pca(face_verts)
            flipped_mask = np.einsum("ij,ij->i", face_normals_tmp, face_normals) < 0
            face_normals[flipped_mask] *= -1.0

        counts = np.zeros(self.verts.shape[0], dtype=int)
        normal_sum = np.zeros_like(self.verts)
        for fid, fn in enumerate(face_normals):
            for i in range(POLY_NUM):
                vid = self.indices[fid][i]
                counts[vid] += 1
                normal_sum[vid] += fn
        invalid_mask = counts <= 0
        counts[invalid_mask] = 1
        normals = normal_sum / counts[..., None]
        normals = normalize(normals)
        self.normals = normals

        self.normal_indices = self.indices


ObjIoVec: TypeAlias = list[float] | list[list[float]] | np.ndarray
ObjIoIndices: TypeAlias = list[int] | list[list[int]] | np.ndarray


def loadObjSimple(obj_path: str):
    num_verts = 0
    num_uvs = 0
    num_normals = 0
    num_indices = 0
    verts = []
    uvs = []
    normals = []
    vert_colors = []
    indices = []
    uv_indices = []
    normal_indices = []
    mtl_file = None
    mtl_names = []
    for line in open(obj_path, "r"):
        vals = line.split()
        if len(vals) == 0:
            continue
        if vals[0] == "v":
            v = [float(x) for x in vals[1:4]]
            verts.append(v)
            if len(vals) == 7:
                vc = [float(x) for x in vals[4:7]]
                vert_colors.append(vc)
            num_verts += 1
        if vals[0] == "vt":
            vt = [float(x) for x in vals[1:3]]
            uvs.append(vt)
            num_uvs += 1
        if vals[0] == "vn":
            vn = [float(x) for x in vals[1:4]]
            normals.append(vn)
            num_normals += 1
        if vals[0] == "f":
            v_index = []
            uv_index = []
            n_index = []
            valid_face = False
            for f in vals[1:]:
                w = f.split("/")
                if num_verts > 0:
                    v_index.append(int(w[0]) - 1)
                if num_uvs > 0:
                    uv_index.append(int(w[1]) - 1)
                if num_normals > 0:
                    if len(w) > 2:
                        n_index.append(int(w[2]) - 1)
                    else:
                        print("no normal index")
                        n_index.append(int(w[0]) - 1)
            if len(v_index) > 0:
                indices.append(v_index)
                valid_face = True
            if len(uv_index) > 0:
                uv_indices.append(uv_index)
                valid_face = True
            if len(n_index) > 0:
                normal_indices.append(n_index)
                valid_face = True
            if valid_face:
                num_indices += 1
        if vals[0] == "mtllib":
            mtl_file = vals[1]
        if vals[0] == "usemtl":
            mtl_names.append(vals[1])
    if len(mtl_names) > 1:
        print(
            "The first material will be used"
            f"but there are {len(mtl_names)} materials"
        )
    return (
        verts,
        uvs,
        normals,
        indices,
        uv_indices,
        normal_indices,
        vert_colors,
        mtl_file,
        mtl_names,
    )


def loadObj(obj_path: str, is_numpy: bool = True):
    mesh = ObjMesh(*loadObjSimple(obj_path))
    if is_numpy:
        mesh.eunsureNumpy()
    else:
        mesh.ensureList()
    return mesh


def saveObjSimple(
    obj_path: str,
    verts: ObjIoVec,
    indices: ObjIoVec,
    uvs: ObjIoVec = [],
    normals: ObjIoVec = [],
    uv_indices: ObjIoVec = [],
    normal_indices: ObjIoVec = [],
    vert_colors: ObjIoVec = [],
    mat_file: str = None,
    mat_name: str = None,
):
    f_out = open(obj_path, "w")
    f_out.write("####\n")
    f_out.write("#\n")
    f_out.write("# verts: %s\n" % (len(verts)))
    f_out.write("# Faces: %s\n" % (len(indices)))
    f_out.write("#\n")
    f_out.write("####\n")
    if mat_file is not None:
        f_out.write("mtllib " + mat_file + "\n")
    for vi, v in enumerate(verts):
        vertstr = "v %s %s %s" % (v[0], v[1], v[2])
        if len(vert_colors) > 0:
            color = vert_colors[vi]
            vertstr += " %s %s %s" % (color[0], color[1], color[2])
        vertstr += "\n"
        f_out.write(vertstr)
    f_out.write("# %s verts\n\n" % (len(verts)))
    for uv in uvs:
        uvstr = "vt %s %s\n" % (uv[0], uv[1])
        f_out.write(uvstr)
    f_out.write("# %s uvs\n\n" % (len(uvs)))
    for n in normals:
        nStr = "vn %s %s %s\n" % (n[0], n[1], n[2])
        f_out.write(nStr)
    f_out.write("# %s normals\n\n" % (len(normals)))
    if mat_name is not None:
        f_out.write("usemtl " + mat_name + "\n")
    for fi, v_index in enumerate(indices):
        fStr = "f"
        for fvi, v_indexi in enumerate(v_index):
            fStr += " %s" % (v_indexi + 1)
            if len(uv_indices) > 0:
                fStr += "/%s" % (uv_indices[fi][fvi] + 1)
            if len(normal_indices) > 0:
                fStr += "/%s" % (normal_indices[fi][fvi] + 1)
        fStr += "\n"
        f_out.write(fStr)
    f_out.write("# %s faces\n\n" % (len(indices)))
    f_out.write("# End of File\n")
    f_out.close()


def saveObj(obj_path: str, mesh: ObjMesh):
    if len(mesh.mtl_names) > 1:
        print(
            "The first material will be used"
            f"but there are {len(mesh.mtl_names)} materials"
        )
    saveObjSimple(
        obj_path,
        mesh.verts,
        mesh.indices,
        mesh.uvs,
        mesh.normals,
        mesh.uv_indices,
        mesh.normal_indices,
        mesh.vert_colors,
        mesh.mtl_file,
        None if len(mesh.mtl_names) < 1 else mesh.mtl_names[0],
    )


def removeByVid(attrs_list: ObjIoVec, indices: ObjIoIndices, to_remove_vids: ObjIoVec):
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
    return np.asarray(attrs_list_), np.asarray(indices_)
