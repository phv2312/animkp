# animkp
Anime Key-point Estimation

## Installation
Install package
```bash
pip install git+ssh://git@github.com/phv2312/animkp.git
```

## Usage
The model uses GPU. No support for runtime changes for now. To run inference
```python
from pose_estimator.wrapper.pose_anime.inference import PoseAnimeInference
from pose_estimator.wrapper.pose_anime.pose_anime_utils import imgshow, read_image, crop_bbox

config_path = "<your_config_path>"
weight_path = "<your_weight_path>"

pose_model = PoseAnimeInference(config_path, weight_path, use_gpu=True)
input_path = "<your_image_path>"

input_image, _ = crop_bbox(read_image(input_path))
results_dct, vis_image = pose_model.process(input_image=input_image, use_flip=True, threshold=0.6, postprocess=True)
imgshow(vis_image)
```
