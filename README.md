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
* H5py (test with version 2.9.0)


To install, use this command.

```bash
pip install torch==1.5.0+cu92 torchvision==0.6.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
pip install opencv-python==4.0.1.24 numpy==1.18.2 pygame==1.9.6 pyopengl==3.1.0 pillow==5.3.0 h5py==2.9.0
```



## Usage

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