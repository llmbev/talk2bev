# Talk2BEV-Base Format

`Talk2BEV-Base` consists 3 parts: `crops`, `cam_imgs`, and `scene`. The folder structure should look like this:

```bash
- Talk2BEV-Base/
    - TOKEN_NUM
        - cam_imgs/
            - 1_cimg.npy: perspective 1
            - 2_cimg.npy: perspective 2
            - ...
        - crops/
            - 1_matched_imgs.npy: object 1 crop
            - 2_matched_imgs.npy: object 2 crop
            - ...
        - scene/
            - answer_gt.json: ground truth scene object
            - answer_pred.json: predicted scene object
        - bev_gt.png: ground truth bev
        - bev_pred.png: predicted bev
```
The `TOKEN` is the NuScenes scene token ID, and `NUM` is the number of the scene.
The folder `crops` contains the crop images of the objects. The folder `cam_imgs` contains the perspective images. The folder `scene` contains the ground truth and predicted scene objects. The files `bev_gt.png` and `bev_pred.png` are the ground truth and predicted BEV images. The BEV images are RGB images, with Blue (0, 0, 255) as background

## Scene-Object

This is how a scene is encoded within an object:

```python
[
    {
        "object_id": 1, # ID of this object
        "bev_centroid": [5, -5], # BEV centroid of this object",
        "bev_area": 10, # BEV area of this object in pixels
        "matched_coords": [[-5, -5], [6, 6],.. [12, 10]], # matched coordinates of this object
        "matched_cam": "CAM_FRONT" # matched camera for this object
        "matched_point": [800, 900], # matched point in the matched camera
        "annotation": {...}, # nuscenes annotation for this object - containing token ID, category etc.
    },
    ...
]
```
