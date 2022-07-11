# HP-GNN

Source code for ["HP-GNN: Generating High Throughput GNN Training Implementation on CPU-FPGA Heterogeneous Platform"](https://dl.acm.org/doi/10.1145/3490422.3502359), ACM/SIGDA FPGA 2022.  
Details of hardware design can be found in the paper.

### Framework overview
![framework](https://github.com/jasonlin316/HP-GNN/blob/main/pic/framework.png)
File structure:
```
HP-GNN/
│   README.md
│
└───/data #input data stored in CSR format and a data generator
│   │   indptr.bin
│   │   indices.bin
│   │   data_generator.py # a python script to generate input matrices based on the size you specified
│   │   ...
└───/run #files and scripts for compilation and execution
│   │   makefile
│   │   design.cfg
│   │   ...
└───/src #kernel designs and some supportive functions
│   │   spdmm.cpp
│   │   mmult.cpp
│   │   ...
└─
```

Citation
```
@inproceedings{hp-gnn,
author = {Lin, Yi-Chien and Zhang, Bingyi and Prasanna, Viktor},
title = {HP-GNN: Generating High Throughput GNN Training Implementation on CPU-FPGA Heterogeneous Platform},
booktitle = {Proceedings of the 2022 ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
year = {2022}
}

```
