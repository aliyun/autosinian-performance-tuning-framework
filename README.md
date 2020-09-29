# Alibaba Cloud Sinian Platform
Sinian is Alibaba’s compiler-based heterogeneous hardware acceleration platform, targeting extreme performance for machine learning applications. Interfacing with the upper level frameworks such as Alibaba PAI, Tensorflow , MxNet and etc, Sinian enables deep co-optimizations between software and hardware to deliver high execution efficiency for ML applications. Sinian is fully tailorable (“statically and dynamically”) for cloud computing, edge computing, and IoT devices, making it easy to achieve performance portability between training and deploying machine learning models across heterogeneous accelerators. 
# AutoSinian
AutoSinian is the automatic performance optimization framework in Sinian. By auto-tuning and joint-optimizing the heterogeneous system performance across algorithm, system, framework and hardware library layers, AutoSinian serves as the core component in Sinian to maximize performance for machine learning applications with very little engineer efforts in case-by-case performance tuning.

## MLPERF Benchmark
See https://mlperf.org/

### Our Result

|GPU| Model | MlperfModel | Offline(IPS) | Accuracy(%) |
|-------|:------------------:|:---------------:|:-------------:|:------------:|
|A100x1| OFAnet-AutoSinian | resnet50 v1.5 |**80156.8** | 75.714 | 
|V100x1| OFAnet-AutoSinian | resnet50 v1.5 | **24212.4** | 75.806 |
|T4x1|OFAnet-AutoSinian| resnet50 v1.5 | **16014.7** | 75.714 |

more [details](mlperf_benchmark)