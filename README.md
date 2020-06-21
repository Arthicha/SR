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
* [PyTorch](https://pytorch.org/) (tests with version 1.5.0+cu92)
* [TorchVision](https://pytorch.org/) (tests with version 0.6.1+cu92) <br />
To install Pytorch and TorchVision, use this command.

```bash
pip install torch==1.5.1+cu92 torchvision==0.6.1+cu92 -f https://download.pytorch.org/whl/torch_stable.html
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