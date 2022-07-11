#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// #define DEBUG
//#define PRINT_MATRIX
#include <vector>
#include <CL/cl2.hpp>


#include <iostream>
#include <fstream>
#include <CL/cl_ext_xilinx.h>
#include <unistd.h>
#include <limits.h>
#include <sys/stat.h>
#include <stdio.h>
#include <ap_int.h>
#include <cstdlib>
#include <ctime>
#include <iterator>
#include <string>
#include <cfloat>
#include <CL/cl_ext.h>


#include "../hw4cu/fpga_top.hpp"
#include "../hw4cu/types.h"


#include "./utility.hpp"


using namespace std;



struct kernel_param
{
    int k;  // total
    int tkstart;
    int tkend;
    int nnz; // in the shard, dstination dataoffeset
    int dstoffset;
};


// function for aligning the address



template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};


#define OCL_CHECK(error,call)                                       \
    call;                                                           \
    if (error != CL_SUCCESS) {                                      \
      printf("%s:%d Error calling " #call ", error code is: %d\n",  \
              __FILE__,__LINE__, error);                            \
      exit(EXIT_FAILURE);                                           \
    }                                       


int read_edge_list(const std::string &edge_file_name, std::vector<v_edges, aligned_allocator<ap_int<512> >> & edgeList, int NumEdge){

    edge_type tmpedge;

    v_edges tmpv_edge;

    std::ifstream EDGE_FILE;
    EDGE_FILE.open(edge_file_name);

    int j=0;

    for(int i = 0; i < NumEdge; i++){
        EDGE_FILE >> tmpedge.src;
        EDGE_FILE >> tmpedge.dst;
        tmpv_edge.edges[j] = tmpedge;
        j++;
        if(j == 8){
            edgeList.push_back(tmpv_edge);
            j = 0;
        }
    }

    if(j > 0){
      for(int i = j; i < 8; i++){
        tmpv_edge.edges[i] = tmpedge;
      }
      edgeList.push_back(tmpv_edge);
    }


    EDGE_FILE.close();


    return 0;
}



int read_edge_from_mtx(const std::string &edge_file_name, std::vector<v_edges, aligned_allocator<ap_int<512> >> & edgeList, int & nrows, int &ncolumns, int & NumEdge){


    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;

    int A_nrows, A_ncols, nnz;

    
    readMtx<float>(edge_file_name.c_str(), row_indices, col_indices, values, A_nrows, A_ncols, nnz);

    NumEdge = nnz;
    nrows = A_nrows;
    ncolumns = A_ncols;

    edge_type tmpedge;

    v_edges tmpv_edge;

    int j = 0;

    for(int i = 0; i < nnz; i++){
      tmpedge.src = row_indices[i];
      tmpedge.dst = col_indices[i];
      tmpv_edge.edges[j] = tmpedge;
      j++;

      if(j == 8){
        edgeList.push_back(tmpv_edge);
        j=0;
      }
    }

    if(j > 0){
      for(int i = j; i < 8; i++){
        tmpv_edge.edges[i] = tmpedge;
      }
      edgeList.push_back(tmpv_edge);
    }

    return 0;
}


int shardpartition(
  std::vector<v_edges, aligned_allocator<ap_int<512> >> & edgeList, 
  std::vector<std::vector<v_edges, aligned_allocator<ap_int<512> >>> & shardlist, 
  std::vector<int> & nnzarray, 
  const int npartition, 
  const int nnz, 
  const int ksize)
{
  edge_type tmpedge[npartition];
  v_edges tmpv_edge[npartition];


  int ii[npartition];
  int jj[npartition];

  for (int i = 0 ; i< npartition; i++){
    ii[i] = 0;
    jj[i] = 0;
    nnzarray[i] = 0;
  }

 



  edge_type ipedge;
  v_edges ivedge;

  int innz = 0;

  printf("the npartition is %d\n", npartition);
  printf("the size of edgeList is %d\n", edgeList.size());

  for (int i = 0; i < edgeList.size(); i++){
    ivedge = edgeList[i];
    for (int j = 0; j < 8; j++){
      if(innz < nnz){
        ipedge = ivedge.edges[j];
        // calculate which npartition the edge belongs to

        int ipar = (ipedge.dst)/(ksize);
        // printf("the ksize is %d, the nnz  is %d, npartition is %d, the partition is %d\n", ksize, nnz, npartition, ipar);
        

        tmpedge[ipar] =  ipedge;

        tmpv_edge[ipar].edges[jj[ipar]] = ipedge;

        jj[ipar]++;

        if(jj[ipar] == 8){
          ii[ipar]++;
          jj[ipar] = 0;
          shardlist[ipar].push_back(tmpv_edge[ipar]);
        }
        nnzarray[ipar]++;
        innz = innz + 1;
      }
      
    }

  }


  for (int i = 0 ; i< npartition; i++){
    printf("the size of %dth partition is %d\n", i, shardlist[i].size());
  }


  for (int h = 0 ; h< npartition; h++){
    if(jj[h] > 0){
      for(int i = jj[h]; i < 8; i++){
        tmpv_edge[h].edges[i] = tmpedge[h];
      }
      shardlist[h].push_back(tmpv_edge[h]);
    }
  }


  return 0;
   


}



int read_edge_values(const std::string &edge_value_file, std::vector<v_edge_value, aligned_allocator<ap_int<512> >> &edgeValueList, int NumEdge){

    edge_value tmpEdgeValue;

    v_edge_value tmpv_EdgeValue;

    std::ifstream EDGE_VALUE_FILE;
    EDGE_VALUE_FILE.open(edge_value_file);

    int j=0;

    for(int i = 0; i < NumEdge; i++){
        EDGE_VALUE_FILE >> tmpEdgeValue.data;
        tmpv_EdgeValue.edgevalues[j].data = tmpEdgeValue.data;
        j++;
        if(j == 16){
            edgeValueList.push_back(tmpv_EdgeValue);
            j = 0;
        }
    }

    if(j > 0){
      for(int i = j; i < 16; i++){
        tmpv_EdgeValue.edgevalues[i].data= tmpEdgeValue.data;
      }
      edgeValueList.push_back(tmpv_EdgeValue);
    }


    EDGE_VALUE_FILE.close();

    return 0;
}


int read_features(const std::string &feature_file, std::vector<std::vector<feature_type>> & inputfeatures, int n, int n_data , int f){
    // n is the number of vertices, f is the length of the features

    

    feature_type singleFeature;

    std::ifstream INPUT_FEATURE_FILE;
    INPUT_FEATURE_FILE.open(feature_file);

    for(int i = 0; i < n; i++){
        std::vector<feature_type> feature_vector;
        for(int j=0; j < f; j++){
            if(i < n_data){
              INPUT_FEATURE_FILE >> singleFeature;
            }
            else{
              singleFeature = 0;
            }
            feature_vector.push_back(singleFeature);
            // if (i == 2){
            //   printf("%f ", singleFeature);
            // }
        }
        inputfeatures.push_back(feature_vector);
    }

    INPUT_FEATURE_FILE.close();

    return 1;
}


namespace xcl {


std::vector<cl::Device> get_devices(const std::string& vendor_name) {

    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    OCL_CHECK(err, err = cl::Platform::get(&platforms));
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        OCL_CHECK(err, std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err));
        if (platformName == vendor_name){
            std::cout << "Found Platform" << std::endl;
            std::cout << "Platform Name: " << platformName.c_str() << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "Error: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    OCL_CHECK(err, err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices));
    return devices;
}


std::vector<cl::Device> get_xilinx_devices();

std::vector<cl::Device> get_xil_devices() {return get_devices("Xilinx");}

char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb);
}











int main(int argc, char ** argv){

  // if (argc != 2){
  //   std::cout << "Uasage: " << argv[0] << " <XCLBIN File>" << std::endl;
  //   return EXIT_FAILURE;
  // }

  // std::string binaryFile = argv[1];


  // cl_int err; // this is the flag to check if there is any error.

  // unsigned fileBufSize;

  // char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);

  std::string EdgeFile;
  std::string binaryFile;

  int k; // shard partition size
  int b; // feature partition size
  int f; // the feature length of SpMM
  int c;


  // parse the command line argument
  // -n the name of the edge file
  // -f the feature length 

  while ((c = getopt (argc, argv, "x:n:f:k:b:")) != -1){
    
    switch (c){
      case 'x':
        binaryFile = optarg;
        break;
      case 'n':
        EdgeFile = optarg;
        break;
      case 'f':
        f = atoi(optarg);
        break;
  
      case 'b':
        b = atoi(optarg);
        break;
      
      case 'k':
        k = atoi(optarg);
        break;
        
      default:
        abort();
    }
  }
  
  printf("the number of the file is ");
  cout << EdgeFile ;
  printf("\n");

  printf("the feature length is %d\n", f);
  printf("the feature partition size is %d\n", b);
  printf("the shard partiition size is %d\n", k);




  int nnz;
  int nrows;
  int ncolumns;

  printf("read the edge list\n");

  std::vector<v_edges, aligned_allocator<ap_int<512> >> edgelist; // bit alignment


  // read edges from the file
  read_edge_from_mtx(EdgeFile,  edgelist, nrows, ncolumns, nnz);


  printf("the number of rows is %d, the number of columns is %d\n", nrows, ncolumns);
  // decide number of partitions based on k
  int npartition = ncolumns/k;

  if (ncolumns % k != 0){
    npartition = npartition + 1;
  }

  std::vector<std::vector<v_edges, aligned_allocator<ap_int<512> >>> shardlist;

  std::vector<int> nnzarray;

  nnzarray.resize(npartition);
  shardlist.resize(npartition);

  shardpartition(edgelist, shardlist, nnzarray, npartition, nnz, k);

  // print the partition information
  printf("the total number of non-zero elements is %d \n", nnz);

  int sum = 0;
  for (int i = 0; i<npartition; i++){
    sum += nnzarray[i];
    printf("the number of non-zero elements in partition %d is %d \n", i, nnzarray[i]);
    
  }

  printf("the total number of non-zero elements is %d \n", sum);


  std::vector<std::vector<v_edge_value, aligned_allocator<ap_int<512> >>> shard_value_list;

  shard_value_list.resize(npartition);
  
  for (int i = 0; i<npartition; i++){
    shard_value_list[i].resize(nnzarray[i]/16 + (nnzarray[i]%16?1:0) );
  }

  // create slices

  std::vector<std::vector<v_datatype, aligned_allocator<ap_int<512> > > > slicearray;

  int nslices = int(f/b);

  slicearray.resize(nslices);

  for (int i = 0; i < nslices; i++){
    slicearray[i].resize(nrows);
  }


  // create output blocks

  std::vector<std::vector<v_datatype, aligned_allocator<ap_int<512> > > > blockarray;

  blockarray.resize(nslices*npartition);

  for (int i = 0; i < nslices*npartition; i++){
    blockarray[i].resize(k);
  }



  // initialize the FPGA device


  cl_int err; // this is the flag to check if there is any error.

  unsigned fileBufSize;

  char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);

  printf("initialize the FPGA device\n");

  std::vector<cl::Device> devices=xcl::get_xil_devices();
  cl::Device device = devices[0];

  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));

  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err));

  OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));


  std::vector<cl::Kernel> spdmm_kernls(npartition*nslices);


  for (int i = 0; i< npartition*nslices; i++) {OCL_CHECK(err, spdmm_kernls[i] = cl::Kernel(program,  "Spmm", &err));}




  kernel_param paramarray[npartition*nslices];



  // create the task pool
  cl_mem_ext_ptr_t Ext_shardbuffer[npartition], Ext_slicebuffer[4*nslices], Ext_blockbuffer[npartition*nslices];
  cl_mem_ext_ptr_t Ext_shard_value_buffer[npartition];       
  int iflag = 0;


  printf("Allocate shardbuffer\n");

  // assign the shard buffer in a round-robin fashion
  for(int i = 0; i<npartition;i++){
      Ext_shardbuffer[i].flags = iflag|XCL_MEM_TOPOLOGY;
      Ext_shardbuffer[i].obj = shardlist[i].data();
      Ext_shardbuffer[i].param = 0;
      iflag++;
      iflag = iflag%4;
  }


  printf("Allocate shard_value_buffer\n");
  iflag = 0;
  for(int i = 0; i<npartition;i++){
      Ext_shard_value_buffer[i].flags = iflag|XCL_MEM_TOPOLOGY;
      Ext_shard_value_buffer[i].obj = shard_value_list[i].data();
      Ext_shard_value_buffer[i].param = 0;
      iflag++;
      iflag = iflag%4;
  }

  iflag = 0;
  printf("Allocate slicebuffer\n");
  // assign the slice buffer
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < nslices; j++){
      Ext_slicebuffer[i*nslices + j].flags = iflag|XCL_MEM_TOPOLOGY;
      Ext_slicebuffer[i*nslices + j].obj = slicearray[j].data();
      Ext_slicebuffer[i*nslices + j].param = 0;
    }
    iflag++;
    iflag = iflag%4;
  }

  printf("Allocate blockbuffer\n");
  for (int i = 0; i< npartition;i++){
    for (int j =0; j < nslices; j++){
      Ext_blockbuffer[i*nslices + j].flags = (i%4)|XCL_MEM_TOPOLOGY;
      Ext_blockbuffer[i*nslices + j].obj = blockarray[i*nslices + j].data();
      Ext_blockbuffer[i*nslices + j].param = 0;
    }
  }




  printf("prepare the parameters\n");

  iflag = 0;
  for (int i = 0; i< npartition;i++){
    for (int j =0; j < nslices; j++){
      paramarray[i*nslices + j].dstoffset = i*k;
      paramarray[i*nslices + j].k = k;
      paramarray[i*nslices + j].nnz = nnzarray[i];
      paramarray[i*nslices + j].tkstart = 0;
      paramarray[i*nslices + j].tkend = nrows - 1;
    }
    iflag++;
    iflag = iflag%4;
  }


  std::vector<cl::Buffer> fpga_shardbuffer(npartition);
  std::vector<cl::Buffer> fpga_slicebuffer(4*nslices);
  std::vector<cl::Buffer> fpga_blockbuffer(npartition*nslices);
  std::vector<cl::Buffer> fpga_shard_value_buffer(npartition);


  printf("move shard buffer to FPGA\n");
  for (int i = 0; i< npartition;i++){
    OCL_CHECK(err, 
      fpga_shardbuffer[i]=cl::Buffer(
      context, 
      CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_ONLY, 
      sizeof(v_edges)*shardlist[i].size(), 
      &(Ext_shardbuffer[i]), 
      &err)
    );
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {fpga_shardbuffer[i]} , 0 ));
  }

  printf("move shard value buffer to FPGA\n");
  for (int i = 0; i< npartition;i++){
    OCL_CHECK(err, 
      fpga_shard_value_buffer[i] = cl::Buffer(
      context, 
      CL_MEM_USE_HOST_PTR |CL_MEM_EXT_PTR_XILINX   | CL_MEM_READ_ONLY, 
      sizeof(v_edge_value)*shard_value_list[i].size(), 
      &(Ext_shard_value_buffer[i]), 
      &err)
    );
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {fpga_shard_value_buffer[i]} , 0 ));
  }

  

  printf("move blockbuffer to FPGA\n");
  iflag = 0;
  for (int i = 0; i< npartition;i++){
    for (int j =0; j < nslices; j++){
      int kernel_num = i*nslices + j;
      OCL_CHECK(err, fpga_blockbuffer[i*nslices + j] = cl::Buffer(
        context, 
        CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX  | CL_MEM_WRITE_ONLY, 
        sizeof(v_datatype)*blockarray[kernel_num].size(), 
        &(Ext_blockbuffer[kernel_num]), 
        &err)
      );
    }
  }

  printf("move slice buffer to FPGA\n");
  for(int i = 0; i < 4; i++){
    for(int j = 0; j < nslices; j++){
      OCL_CHECK(err, 
        fpga_slicebuffer[i*nslices + j]=cl::Buffer(
        context, 
        CL_MEM_USE_HOST_PTR | CL_MEM_EXT_PTR_XILINX  | CL_MEM_READ_ONLY, 
        sizeof(v_datatype)*slicearray[j].size(), 
        &(Ext_slicebuffer[i*nslices + j]), 
        &err) 
      );
      OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {fpga_slicebuffer[i*nslices + j]} , 0 ));
    }
  }


  printf("Prepare the kernel\n");
  for (int i = 0; i< 1;i++){
    for (int j =0; j < 1; j++){
      int kernel_num = i*nslices + j;

      printf("Start create %d kernel\n", kernel_num);
      printf("1:%d:%d\n",i*nslices + j, fpga_slicebuffer.size());
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(0, fpga_slicebuffer[ (i%4)* nslices + j]) );
      printf("2\n");
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(1, fpga_shardbuffer[i]) );
      printf("3\n");
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(2, fpga_shard_value_buffer[i]));
      printf("4\n");
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(3, fpga_blockbuffer[ i*nslices + j]));
      printf("5\n");
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(4, paramarray[ i*nslices + j].k ));
      printf("k is %d \n", paramarray[ i*nslices + j].k);
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(5, paramarray[ i*nslices + j].tkstart));
      printf("tkstart is %d\n", paramarray[ i*nslices + j].tkstart);
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(6, paramarray[ i*nslices + j].tkend));
      printf("tkend is %d\n", paramarray[ i*nslices + j].tkend);
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(7, paramarray[ i*nslices + j].nnz));
      printf("nnz is %d\n", paramarray[ i*nslices + j].nnz);
      OCL_CHECK(err, err = spdmm_kernls[ i*nslices + j].setArg(8, paramarray[ i*nslices + j].dstoffset));
      printf("dstoffset is %d\n", paramarray[ i*nslices + j].dstoffset);

      printf("finish allocate %d th kernel\n", kernel_num);
    }
  }
    

  OCL_CHECK(err, err = q.finish());

  for (int i = 0; i<npartition*nslices; i++){
    OCL_CHECK(err, err = q.enqueueTask(spdmm_kernls[i]));
  }

  OCL_CHECK(err, err = q.finish());

  for (int i = 0; i < npartition*nslices; i++){
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({fpga_blockbuffer[i]}, CL_MIGRATE_MEM_OBJECT_HOST));
  }
  
  OCL_CHECK(err, err = q.finish());


  // reclaim some resources

  delete[] fileBuf;

  // v_edges tmpv_edge;

  // edge_type tmpedge;

  // int j = 0;

  // for (int i=0; i < shardlist[1].size(); i++){

  //   tmpv_edge = shardlist[1][i];

  //   for (int j =0; j < 8 ; j++){
  //     tmpedge = tmpv_edge.edges[j];

  //     printf("the edge %d is (%d, %d) \n", i*8+j, tmpedge.src, tmpedge.dst);
      
  //   }
  // }






  /*

  srand((unsigned)time(0));

  // define number of edges here in nnz

  int nnz = 6699823; // number of edges

  int n = 40960; // number of vertices

  int tn = 1569961; // total number of vertices

  int n_data = 40960;

  int f = 16; // length of the features

  // read the edge list from the files

  printf("read the edge list\n");

  std::vector<v_edges, aligned_allocator<ap_int<512> >> edgelist; // bit alignment

  std::string EdgeFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/amazon/amazon-edge-list.dat");
  
  read_edge_list(EdgeFile, edgelist, nnz);

  // read the edge value from the files

   printf("read the edge value list\n");

  std::vector<v_edge_value, aligned_allocator<ap_int<512> >> edge_value_list;

  std::string EdgeValueFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/amazon/amazon-edge-values.dat");

  read_edge_values(EdgeValueFile, edge_value_list, nnz);

  // read the features from the files

  // printf("read the input features\n");

  // std::vector<std::vector<feature_type>> input_features;

  // std::string InputFeatureFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-x.dat");

  // read_features(InputFeatureFile, input_features, n, n_data, f);

  // read the output features from the files

  // std::vector<std::vector<feature_type>> output_features;

  // printf("read the golden output features\n");

  // std::string OutputFeatureFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-result.dat");

  // read_features(OutputFeatureFile, output_features, n, n_data, f);

  // extract the first feature slides

  std::vector<v_datatype, aligned_allocator<ap_int<512>> > input_fetures_slice;
  

  input_fetures_slice.resize(tn);

  // v_datatype temp_vdata;

  // for(int i = 0; i< n; i++){
  //     for(int j = 16; j < 32;j++){
  //       temp_vdata.data[j - 16] = input_features[i][j];
  //     }
  //     input_fetures_slice.push_back(temp_vdata);
  // }

  // extract golden Ouput 

  // define the output feature slides 

  std::vector<v_datatype, aligned_allocator<ap_int<512>> > output_fetures_slice;

  output_fetures_slice.resize(n_data);


  // initialize the FPGA device

  printf("initialize the FPGA device\n");

  std::vector<cl::Device> devices=xcl::get_xil_devices();
  cl::Device device = devices[0];

  OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));

  OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

  OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));

  cl::Program::Binaries bins{{fileBuf, fileBufSize}};
    
  devices.resize(1);

  OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

  OCL_CHECK(err, cl::Kernel spdmm_kernls(program,  "Spmm", &err));

  // get the pointer of the data in the host

  v_edges * ptr_edgelist = edgelist.data();

  v_edge_value * ptr_edge_value_list = edge_value_list.data();

  v_datatype * ptr_input_fetures_slice = input_fetures_slice.data();

  v_datatype * ptr_output_fetures_slice = output_fetures_slice.data();

  // define buffer that will be used

  printf("initialize buffer\n");

  OCL_CHECK(err, cl::Buffer edges_buffer(
    context, 
    CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY, 
    sizeof(v_edges)*(edgelist.size()) , 
    edgelist.data(),
     &err));

  OCL_CHECK(err, cl::Buffer edge_value_buffer(
    context, 
    CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY, 
    sizeof(v_edge_value)*(edge_value_list.size()) , 
    edge_value_list.data(), 
    &err));

  OCL_CHECK(err, cl::Buffer input_fetures_slice_buffer(
    context, 
    CL_MEM_USE_HOST_PTR  | CL_MEM_READ_ONLY, 
    sizeof(v_datatype)*(input_fetures_slice.size()) , 
    input_fetures_slice.data(), 
    &err));
    
  OCL_CHECK(err, cl::Buffer output_fetures_slice_buffer(
    context, 
    CL_MEM_USE_HOST_PTR  | CL_MEM_WRITE_ONLY, 
    sizeof(v_datatype)*(output_fetures_slice.size()), 
    output_fetures_slice.data(), 
    &err));
  

  // set the argument for the spmm kernels

  printf("configure the kernel\n");

  OCL_CHECK(err, err = spdmm_kernls.setArg(0, input_fetures_slice_buffer ));
  OCL_CHECK(err, err = spdmm_kernls.setArg(1, edges_buffer));
  OCL_CHECK(err, err = spdmm_kernls.setArg(2, edge_value_buffer));
  OCL_CHECK(err, err = spdmm_kernls.setArg(3, output_fetures_slice_buffer ));
  OCL_CHECK(err, err = spdmm_kernls.setArg(4, n));
  OCL_CHECK(err, err = spdmm_kernls.setArg(5, 0));
  OCL_CHECK(err, err = spdmm_kernls.setArg(6, tn - 1));
  OCL_CHECK(err, err = spdmm_kernls.setArg(7, nnz));
  OCL_CHECK(err, err = spdmm_kernls.setArg(8, 0));

  // move the input data from host side to fpga
  printf("move the input data from host side to fpga\n");

  OCL_CHECK(err, err = q.enqueueMigrateMemObjects( {
    input_fetures_slice_buffer, 
    edges_buffer, 
    edge_value_buffer} 
    , 0 ));

  // run the kernel 

  printf("run the kernel\n");

  OCL_CHECK(err, err = q.enqueueTask(spdmm_kernls));

  printf("read result back to host\n");

  OCL_CHECK(err, err = q.enqueueMigrateMemObjects(
    {output_fetures_slice_buffer}, 
    CL_MIGRATE_MEM_OBJECT_HOST));

  // wait until the kernel finish the calculation

  printf("kernel run finished\n");

  OCL_CHECK(err, err = q.finish());






  // reclaim some resources

  delete[] fileBuf;
  */

  return 0;


}



















namespace xcl {

std::vector<cl::Device> get_xilinx_devices() 
{
    size_t i;
    cl_int err;
    std::vector<cl::Platform> platforms;
    err = cl::Platform::get(&platforms);
    cl::Platform platform;
    for (i  = 0 ; i < platforms.size(); i++){
        platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>(&err);
        if (platformName == "Xilinx"){
            std::cout << "INFO: Found Xilinx Platform" << std::endl;
            break;
        }
    }
    if (i == platforms.size()) {
        std::cout << "ERROR: Failed to find Xilinx platform" << std::endl;
        exit(EXIT_FAILURE);
    }
   
    //Getting ACCELERATOR Devices and selecting 1st such device 
    std::vector<cl::Device> devices;
    err = platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
    return devices;
}
   
char* read_binary_file(const std::string &xclbin_file_name, unsigned &nb) 
{
    if(access(xclbin_file_name.c_str(), R_OK) != 0) {
        printf("ERROR: %s xclbin not available please build\n", xclbin_file_name.c_str());
        exit(EXIT_FAILURE);
    }
    //Loading XCL Bin into char buffer 
    std::cout << "INFO: Loading '" << xclbin_file_name << "'\n";
    std::ifstream bin_file(xclbin_file_name.c_str(), std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
    return buf;
}











}
