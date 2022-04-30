# Multiview-3D-Reconstruction

## Requirements

Python requirements
```bash
pip3 install opencv-contrib-python
pip3 install scipy
pip3 install tomlkit
pip3 install tqdm
```

To view the model a software like [Meshlab](https://www.meshlab.net/#download) is required .

## Usage

to run 

```bash
python sfm.py
```

Create an Sfm object sfm_1 with path to <img_dir> as an argument.

<img_dir> has to have a file called `K.txt` containing the camera intrinsic parameters.

Call the object sfm_1 with optional parameter enable_bundle_adjustment (Takes a long time set to False for quick run).

An example is provided in the sfm.py main function.

Open the .ply file using Meshlab present in . `\res` directory


