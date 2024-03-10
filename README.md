## Poisson Mesh Connecting
This repo is a partial implementation of [1] and focuses on merging/connecting functionality, as in Fig. 9, Fig. 11, and Fig. 12 of the paper.
Currently, naive Poisson (i.e., without scale and rotation consideration) is implemented.
However, similarity transform alignment for boundary correspondence is applied before constructing Poisson's equation.
This preprocess eases artifacts caused by scale and rotation up to some extent.

## Try
```
python test.py
```
will generate

### 1. face to cylinder and cylinder to face connecting .obj.
Connect two meshes smoothly. `poisson_mesh_connecting()` is the main interface. In addition to pinned and free meshes, need to specify exactly the same number of boundary vertex correspondence.

||Pinned|Free|Merged|
|---|:---:|:---:|:---:|
|face to cylinder|  <img src="./data/image/cylinder.jpg" height="128"> | <img src="./data/image/face.jpg" height="128"> | <img src="./data/image/face2cylinder.jpg" height="128">
|cylinder to face| <img src="./data/image/face.jpg" height="128"> |  <img src="./data/image/cylinder.jpg" height="128"> | <img src="./data/image/cylinder2face.jpg" height="128">



### 2. cube/building to sphere replaced .obj.
Moreover, another interface `poisson_mesh_replace()` is provided to "replace" a part of vertices. The input is two meshes with the same topology (i.e., triangle indices) and vertex mask to be replaced. This is a wrapper of `poisson_mesh_connecting()`.

||Pinned|Free|Replaced|
|---|:---:|:---:|:---:|
|cube to sphere|  <img src="./data/image/sphere.jpg" height="128"> | <img src="./data/image/cube.jpg" height="128"> | <img src="./data/image/cube2sphere.jpg" height="128">
|building to sphere| <img src="./data/image/sphere.jpg" height="128"> |  <img src="./data/image/building.jpg" height="128"> | <img src="./data/image/building2sphere.jpg" height="128">


### Reference
- [1] Yu, Y., Zhou, K., Xu, D., Shi, X., Bao, H., Guo, B., & Shum, H. Y. (2004). Mesh editing with poisson-based gradient field manipulation. In ACM SIGGRAPH 2004 Papers (pp. 644-651).