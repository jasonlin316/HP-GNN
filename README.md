# HP-GNN

Source code for ["HP-GNN: Generating High Throughput GNN Training Implementation on CPU-FPGA Heterogeneous Platform"](https://dl.acm.org/doi/10.1145/3490422.3502359), ACM/SIGDA FPGA 2022.  


### Framework overview
![framework](https://github.com/jasonlin316/HP-GNN/blob/main/pic/framework.png)

File structure:
```
HP-GNN/
|
│   README.md
|
└───/src #kernel designs and some supportive functions
│   │   SPMM kernel 
│   │   GEMM kernel
|   |   Design space exploration (DSE) engine
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
