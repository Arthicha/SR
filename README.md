# StreamingSR

StreamingSR is a python program that . . .

## Tested System

The program have been tested on two systems. The first one was with:
* Window 10 operating system
* Intel(R) Core(TM) i7-8750H CPU
* NVIDIA GForce GTX 1050 GPU

The second one was with:
* Window 10 operating system
* Intel(R) Core(TM) i5-8500 CPU
* NVIDIA GForce RTX 2080 SUPER


## Prerequisite

### Applications & Programs

* [Python 3.6.5](https://www.python.org/downloads/release/python-365/)
* [Spout 2.006](https://spout.zeal.co/)
* [Visual Studio](https://visualstudio.microsoft.com/downloads/)
* [Boost](https://www.boost.org/)
* [Spout-for-Python library](https://github.com/spiraltechnica/Spout-for-Python)

### Python Modules

These modules are mandatory:


* PyTorch (tests with version 1.5.0+cu92)
* TorchVision (tests with version 0.6.1+cu92)
* OpenCV (tests with version 4.0.1.24)
* Numpy (tests with version 1.18.2)
* PyGame (tests with version 1.9.6)
* OpenGL (tests with version 3.1.0)
* Pillow (tests with version 5.3.0)
* H5py (tests with version 2.9.0)
* Argparse (tests with version 1.1)
* Tqdm (tests with version 4.31.1)


To install, use this command.

```bash
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.0.1.24 numpy==1.18.2 pygame==1.9.6 pyopengl==3.1.0 pillow==5.3.0 h5py==2.9.0 argparse==1.1 tqdm==4.31.1
```



## Training

Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and run the following command to convert the training dataset to hierarchical data format (hdf5, .h5)

```bash
python generateH5.py --dataset_folder your_dataset_folder --hdf5_name hdf5_name --num_per_group data_amount_per_group
```

If the arguments aren't specified your_dataset_folder, hdf5_name, and data_amount_per_group are "DIV2K_train", "DIV2K_trainx.h5", and 50, respectively.





Or
Download the training data stored in a hierarchical data format (hdf5, .h5) from [google drive](https://drive.google.com/file/d/1UwCPo3V6x80sELU9VPk-aiS_Eq3e4CG4/view?usp=sharing)

```python
import foobar

foobar.pluralize('word') # returns 'words'
foobar.pluralize('goose') # returns 'geese'
foobar.singularize('phenomena') # returns 'phenomenon'
```
### Usage

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## License
[MIT](https://choosealicense.com/licenses/mit/)