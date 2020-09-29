# Alibaba Cloud Sinian Platform T4 Submission 
## Platform

### Sinian
Sinian is Alibaba’s compiler-based heterogeneous hardware acceleration platform, targeting extreme performance for machine learning applications. Interfacing with the upper level frameworks such as Alibaba PAI, Tensorflow , MxNet and etc, Sinian enables deep co-optimizations between software and hardware to deliver high execution efficiency for ML applications. Sinian is fully tailorable (“statically and dynamically”) for cloud computing, edge computing, and IoT devices, making it easy to achieve performance portability between training and deploying machine learning models across heterogeneous accelerators. 
### AutoSinian
AutoSinian is the automatic performance optimization framework in Sinian. By auto-tuning and joint-optimizing the heterogeneous system performance across algorithm, system, framework and hardware library layers, AutoSinian serves as the core component in Sinian to maximize performance for machine learning applications with very little engineer efforts in case-by-case performance tuning.

## Models
The base network is from "<em>Once for All: Train One Network and Specialize it for Efficient Deployment</em>"(https://arxiv.org/pdf/1908.09791.pdf). We find a subnet for best performance while meet the accuracy requirement. 
### The model structure
```
pretrained/best_model/net.config
```
### The model weights
```
pretrained/best_model/init
```

## Quantization
TensorRT INT8 quantization is used, for more details, please refer to NVIDIA.

## Replication
### Dependencies
#### GPU Driver
version: 450.57
#### cuda
version: 11.0
#### cudnn
version: 8.0.2
#### TensorRT
version: 7.2.0.14
#### Python
version 3.6.9
#### numpy
version 1.19.1
#### Pytorch
```
git clone https://github.com/pytorch/pytorch.git
git checkout 952526804cf16de659330fece1fe54e15cdcadd0
git submodule update --init --recursive
python setup.py install
```
#### torchvision
```
git clone https://github.com/pytorch/vision.git
git checkout 8c7e7bb0e8b522d28e9b7d0bbf1f1b94b6c3b1f0
python setup.py install
```
#### torch2trt
```
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
git checkout f5fb7529e62ad44a5462991acce4e6844a6333b9
python setup.py install
```
#### once-for-all
```
git clone https://github.com/mit-han-lab/once-for-all.git
git checkout 42a909b282c899714b11f2c95cb89fafdea3be22
```
#### mlperf loadgen
```
git clone https://github.com/mlperf/inference.git
git checkout r0.7
git submodule update --init --recursive
CFLAGS="-std=c++14" python setup.py develop --user
```

### Command line 
#### for performance
```
main.py --dataset-path <your imagenet val set path> --model-builder-dir <dir of once-for-all>
```
Note: TensorRT built engine is included in the submission, use --force-build to build your own, otherwise it will use the built engine.
#### for accuracy
```
main.py --dataset-path <your imagenet val set path> --model-builder-dir <dir of once-for-all> --accuracy
```