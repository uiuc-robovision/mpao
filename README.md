The ArtObjSim dataset can be found [here](https://drive.google.com/file/d/1SNC2WkBqEBzIaGvGmoEtoVmq96WHKmgv/view?usp=sharing).

The train / val / test split can be found in the `train_val_test_split.npy` file in this repo. The usage is as follows:

```
>>> import numpy as np
>>> train_val_test_split = np.load('train_val_test_split.npy', allow_pickle=True).item()
>>> train_val_test_split['train']
```

Code for generating the dataset has also been provided. `scenes_dict.npy` contains the kitchen locations and viewing angles for each of the HM3D scenes used. Images can be rendered out from these viewing locations and orientations using the Habitat simulator. The raw scale annotations for the resulting images can be found under `scale_annotations`. Finally, to run the processing script to generate the full dataset, run `src/post_processing.py` with a path to the scale annotation json file (`--json_filename`) and the path to the kitchen locations dictionary (`--path_to_scenes_dict`).
