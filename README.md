## Poisson Mesh Connecting
This repo is a very partial implementation of [1] and focuses on merging/connecting functionality, as in Fig. 9, Fig. 11, and Fig. 12 of the paper.
Currently, naive Poisson (i.e., without scale and rotation consideration) is implemented.

## Try
```
python test.py
```
will generate face to cylinder and cylinder to face connecting .obj.

||Pinned|Free|Merged|
|---|:---:|:---:|:---:|
|face to cylinder|  <img src="./data/image/cylinder.jpg" height="128"> | <img src="./data/image/face.jpg" height="128"> | <img src="./data/image/face2cylinder.jpg" height="128">
|cylinder to face| <img src="./data/image/face.jpg" height="128"> |  <img src="./data/image/cylinder.jpg" height="128"> | <img src="./data/image/cylinder2face.jpg" height="128">

`poisson_mesh_connecting()` is the main interface. In addition to pinned and free meshes, need to specify exactly the same number of boundary vertex correspondence.

### Reference
- [1] Yu, Y., Zhou, K., Xu, D., Shi, X., Bao, H., Guo, B., & Shum, H. Y. (2004). Mesh editing with poisson-based gradient field manipulation. In ACM SIGGRAPH 2004 Papers (pp. 644-651).