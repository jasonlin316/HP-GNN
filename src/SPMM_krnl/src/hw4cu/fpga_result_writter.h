#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"


#ifndef RESULT_WRITER_DEF
#define RESULT_WRITER_DEF

extern "C"{

void resultWriter(
    hls::stream<v_datatype> & input1,
    hls::stream<v_datatype> & input2,
    hls::stream<v_datatype> & input3,
    hls::stream<v_datatype> & input4,
    hls::stream<v_datatype> & input5,
    hls::stream<v_datatype> & input6,
    hls::stream<v_datatype> & input7,
    hls::stream<v_datatype> & input8,

    v_datatype * output,
    int k
){
    v_datatype regupdate;
#pragma HLS aggregate variable=regupdate

    for (int i = 0; i < k ; i++){
#pragma HLS PIPELINE II=1
        int j = i % 8;
        switch(j){
            case 0: read_from_stream(input1, regupdate); goto WriteResult;
            case 1: read_from_stream(input2, regupdate); goto WriteResult;
            case 2: read_from_stream(input3, regupdate); goto WriteResult;
            case 3: read_from_stream(input4, regupdate); goto WriteResult;
            case 4: read_from_stream(input5, regupdate); goto WriteResult;
            case 5: read_from_stream(input6, regupdate); goto WriteResult;
            case 6: read_from_stream(input7, regupdate); goto WriteResult;
            case 7: read_from_stream(input8, regupdate); goto WriteResult;
        }
        WriteResult: output[i] = regupdate;
        // printf("the %d feature vector is ", i);
        // for(int q = 0 ; q<16;q++){
        //     printf("%f ", regupdate.data[q]);
        // }
        // printf("\n");
    }

}


}



#endif
