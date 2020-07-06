# Using tkDNN and TensoRT to speed up your Object Detector Prediction Engine (darknet example)


If you use darknet as an object detector, you can use tkDNN to speed up the prediction engine. The link to the repo is [here](https://github.com/ceccocats/tkDNN).

Below are all the instruction to complete installations and demos. This includes CUDA, cuDNN, TensorRT and OPENCV. 




Other useful links:

* https://github.com/ceccocats/tkDNN/issues/30

* https://github.com/ceccocats/tkDNN


## Installation

### Requirements

The requirements listed on the tkDNN repo are the following: 

* CUDA 10.0
* cuDNN 7.603
* TensorRT 6.01
* OPENCV 3.4
* yaml-cpp 0.5.2 (sudo apt install libyaml-cpp-dev)


### 1.  Get Instance

* Create EC2 isntanace type: Ubuntu Server 18.04 LTS (HVM), SSD Volume Type
* size: g4dn.xlarge


### 2.  Install CUDA 10.0



Follow steps on this github script. Make sure you change all version to CUDA 10.0.

If you have a previous CUDA installation, make sure to remove it first.

* https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73

**Remove previous installs:**

```
sudo apt-get purge nvidia*
sudo apt remove nvidia-*
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*
```


Verify your gpu is cuda enabled:

`lspci | grep -i nvidia`


```
gcc --version
Command 'gcc' not found, but can be installed with:
```

Command not found, so lets install gcc

```
# system update
sudo apt-get update
sudo apt-get upgrade
sudo apt install gcc
```

then

```
# install other import packages
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev
```

install graphics drivers
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" | sudo tee /etc/apt/sources.list.d/cuda.list
sudo apt-get update
```

 ```
sudo apt-get -o Dpkg::Options::="--force-overwrite" install cuda-10-0 cuda-drivers
```


Finally, set up your paths:
```
echo 'export PATH=/usr/local/cuda-10.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig
```


```
CUDNN_TAR_FILE="cudnn-10.0-linux-x64-v7.6.0.64.tgz"
### wget doesnt work sometimes so download and upload onto ec2 instance
wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.6.0.64/prod/10.0_20190516/cudnn-10.0-linux-x64-v7.6.0.64.tgz
tar -xzvf ${CUDNN_TAR_FILE}
```


Copy the following files into the cuda toolkit directory.

```
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-10.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-10.0/lib64/
sudo chmod a+r /usr/local/cuda-10.0/lib64/libcudnn*
```

Finally, to **verify the installation**, check
```
nvidia-smi
nvcc -V
```

If you get no erros then you have successfully installed CUDA.



### 2.  Get all the other dependencies

`sudo apt install libyaml-cpp-dev`


**TensorRt**

https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html

verify installation:

`dpkg -l | grep TensorRT`


**OpenCV**

get cmake if you dont already have it installed:

`sudo apt-get -y install cmake`

* copy and paste this script: https://github.com/ceccocats/tkDNN/blob/master/scripts/install_OpenCV4.sh

- run `bash install_OpenCV4.sh`



## Running tkDNN

Once all the installations are ready, start testing tkDNN.

From the main repo you can see the main "steps needed to do inference on tkDNN with a custom neural network.":

 
1. Build and train a NN model with your favorite framework.

2. Export weights and bias for each layer and save them in a binary file (one for layer).

3. Export outputs for each layer and save them in a binary file (one for layer).

4. Create a new test and define the network, layer by layer using the weights extracted and the output to check the results.

5. Do inference.


#### 1. Build and train a model on darknet:


Use [darknet](https://github.com/AlexeyAB/darknet) repo to train and validate a model. 


#### 2. Export weights from darknet


instructions: https://git.hipert.unimore.it/fgatti/darknet


`git clone https://git.hipert.unimore.it/fgatti/darknet.git`

`make`


`./darknet export yolov3-gun2070.cfg yolov3-gun2070_best.weights weight_exports`


* The binary files are now in weight_exports:

`zip -r darknet_weight_exports.zip weight_exports/`

Once you have the zip file with the weight exports, upload it to where you have tkDNN installed.





### 3. Darknet Parser

./test_yolo3 # Runs the yolo test and creates the .rt file

result:

 * /home/ubuntu/tkDNN/build/yolo3_fp32.rt
 


### 4. Run the Demo

rm yolo3_fp32.rt        # be sure to delete(or move) old tensorRT files
./test_yolo3            # run the yolo test (is slow)



`./demo yolo3_fp32.rt ../demo/crowds_1.mp4 y 2 1 0 `


Result video:

`tkDNN/build/result.mp4`


### Python Wrapper:

If python is needed for the prediction engine, there is a wrapper you can use [here](https://github.com/ceccocats/tkDNN/pull/44) (still in a PR at the time of this post) 

`git clone https://github.com/ioir123ju/tkDNN.git`

edit the `/home/ubuntu/tkDNN_pythonwrap/tkDNN/darknetTR.py` file

Change line 155 to reflect the .rt file you created on tkDNN and copied over. 

`weight_file='/home/ubuntu/tkDNN_pythonwrap/tkDNN/build/yolo4_fp32.rt'`

then run `python darknetTR.py build/yolo4_fp32.rt --video demo/crowds_1.mp4`
