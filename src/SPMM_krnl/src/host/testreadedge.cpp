#include <stdio.h>
#include <iostream>
#include <fstream>
#include "./utility.hpp"

#include "./mmio.hpp"


using namespace std;


int main(int argc, char** argv){

    std::vector<int> row_indices;
    std::vector<int> col_indices;
    std::vector<float> values;

    int A_nrows, A_ncols, nnz;

    printf("reading file ...\n");
    readMtx<float>(argv[1], row_indices, col_indices, values, A_nrows, A_ncols, nnz);

    for(int i=0; i < nnz; i++){
        printf("The i th edge is ");
        printf("%d ", row_indices[i]);
        printf("%d\n", col_indices[i]);
    }


    
    return 0;


}