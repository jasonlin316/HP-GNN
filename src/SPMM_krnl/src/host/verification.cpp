#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
// #define DEBUG
//#define PRINT_MATRIX
#include <vector>
#include <CL/cl2.hpp>
#include "utility.hpp"
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
#include "../hw/fpga_top.hpp"
#include "../hw/types.h"


using namespace std;

#define NNZ 445890
#define PARTITIONSIZE 16384
#define DENSE_LENGTH 256
#define FEATURELENGTH 16



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


    return 1;
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

    return 1;
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

  if (argc != 2){
    std::cout << "Uasage: " << argv[0] << " <XCLBIN File>" << std::endl;
    return EXIT_FAILURE;
  }

  std::string binaryFile = argv[1];

  cl_int err; // this is the flag to check if there is any error.

  unsigned fileBufSize;

  char* fileBuf = xcl::read_binary_file(binaryFile, fileBufSize);

  srand((unsigned)time(0));

  // define number of edges here in nnz

  int nnz = 10556; // number of edges

  int n = 2712; // number of vertices

  int n_data = 2708;

  int f = 1433; // length of the features

  // read the edge list from the files

  printf("read the edge list\n");

  std::vector<v_edges, aligned_allocator<ap_int<512> >> edgelist; // bit alignment

  std::string EdgeFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-edge-list.dat");
  
  read_edge_list(EdgeFile, edgelist, nnz);

  // read the edge value from the files

   printf("read the edge value list\n");

  std::vector<v_edge_value, aligned_allocator<ap_int<512> >> edge_value_list;

  std::string EdgeValueFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-edge-values.dat");

  read_edge_values(EdgeValueFile, edge_value_list, nnz);

  // read the features from the files

  printf("read the input features\n");

  std::vector<std::vector<feature_type>> input_features;

  std::string InputFeatureFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-x.dat");

  read_features(InputFeatureFile, input_features, n, n_data, f);

  // read the output features from the files

  std::vector<std::vector<feature_type>> output_features;

  printf("read the golden output features\n");

  std::string OutputFeatureFile("/home/zjjzby/project/HLS-SPMM/dataset/GraphGCN/cora/cora-result.dat");

  read_features(OutputFeatureFile, output_features, n, n_data, f);

  // extract the first feature slides

  std::vector<v_datatype, aligned_allocator<ap_int<512>> > input_fetures_slice;
  
  v_datatype temp_vdata;

  for(int i = 0; i< n; i++){
      for(int j = 16; j < 32;j++){
        temp_vdata.data[j - 16] = input_features[i][j];
      }
      input_fetures_slice.push_back(temp_vdata);
  }

  // extract golden Ouput 

  std::vector<v_datatype, aligned_allocator<ap_int<512>> > golden_output_slice;

  for(int i = 0; i< n; i++){
      for(int j = 16; j < 32;j++){
        temp_vdata.data[j - 16] = output_features[i][j];
      }
      golden_output_slice.push_back(temp_vdata);
  }

  // define the output feature slides 

  std::vector<v_datatype, aligned_allocator<ap_int<512>> > output_fetures_slice;

  output_fetures_slice.resize(n);


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
  OCL_CHECK(err, err = spdmm_kernls.setArg(5, nnz));

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


  // compare the result with golden output

  int correctness = 0;

  int numdiff = 0;

  for(int i = 0; i< n; i++){
    printf("compare %d ", i);
      for(int j = 0; j < 16;j++){
        if( (golden_output_slice[i].data[j] - output_fetures_slice[i].data[j]) > 0.00001){
            correctness = 1;
            numdiff++;
        }
        printf("(%f, %f)", golden_output_slice[i].data[j], output_fetures_slice[i].data[j]);
      
      }
      printf("\n");
  }


  printf("the correctness is %d\n", correctness);
  printf("the number of difference is %d \n", numdiff);




  // reclaim some resources

  delete[] fileBuf;

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
