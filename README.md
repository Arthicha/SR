# StreamingSR

StreamingSR is a python program that . . .

## Result
![](gif/newyork.gif)
![](gif/people.gif)

<div align="center">
<img src="gif/people.gif" >
<p>Perfectly balanced</p>
</div>

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


### Preparing Training Data

Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset and run the following command to convert the training dataset to hierarchical data format (hdf5, .h5)

```bash
cd dataset
python generateH5.py --dataset_folder your_dataset_folder --hdf5_name hdf5_name --num_per_group data_amount_per_group
```

If the arguments aren't specified your_dataset_folder, hdf5_name, and data_amount_per_group are "DIV2K_train", "DIV2K_trainx.h5", and 50, respectively.

Or download the training data stored in a hierarchical data format (hdf5, .h5) from [google drive](https://drive.google.com/file/d/1UwCPo3V6x80sELU9VPk-aiS_Eq3e4CG4/view?usp=sharing)

### Training The Model

To train the model, go to the project directory if you are in dataset directory, and then run the following command:

```bash
python train.py
```
Note that, you can specify following input arguements:
* --ckpt_name : path and name of the checkpoint to be loaded (default 'checkpoint/checkpoint.pth')
* --saved_ckpt_dir : directory that the new checkpoint will be saved to (default 'checkpoint/new')
* --train_data_path : directory of the dataset in hdf5 format (default "dataset/DIV2K_train.h5")
* --batch_size : batch size (default 1)
* --update_every : number of batch to update the network weights (default 1)
* --patch_size : size of training images (default 64)
* --lr : learning rate (default 0.001)
* --decay : decay rate of the learning rate (default 400000, halved every 400000 epoch)


## Streaming

### Build SpoutSDK

If you use python with different version, you have to build new spoutSDK. Please follow [this tutorial](https://rusin.work/vjing/tools/spout-for-python/?fbclid=IwAR2-7DcQUpr4SqxAqM5LkWbYCu3RPgEMsNQ5MuAbW6JwzyHCYtoqrOqoEfQ). After complete this, you will get a file name "SpoutSDK.pyd" and you shall replace the file given in the project with the new one.




## License
[MIT](https://choosealicense.com/licenses/mit/)