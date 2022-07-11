#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"

#ifndef VERTEX_READ_DEF
#define VERTEX_READ_DEF

extern "C"{

void data_spliter(
    hls::stream<v_datatype_onchip> & srcFeaturesToBroadcastor,
    hls::stream<v_datatype_onchip> & srcFeaturesToPE_0,
    hls::stream<v_datatype_onchip> & srcFeaturesToPE_1,
    hls::stream<v_datatype_onchip> & srcFeaturesToPE_2,
    hls::stream<v_datatype_onchip> & srcFeaturesToPE_3,
    int k
){
    v_datatype_onchip tmp_vertex_features;

    v_datatype_onchip level1_1;
    v_datatype_onchip level1_2;

    v_datatype_onchip level2_1;
    v_datatype_onchip level2_2;
    v_datatype_onchip level2_3;
    v_datatype_onchip level2_4;


    for (int i = 0 ; i < k ; i++){
#pragma HLS PIPELINE II=1 rewind
        read_from_stream(srcFeaturesToBroadcastor, tmp_vertex_features);


        write_to_stream(srcFeaturesToPE_0, tmp_vertex_features);
        write_to_stream(srcFeaturesToPE_1, tmp_vertex_features);
        write_to_stream(srcFeaturesToPE_2, tmp_vertex_features);
        write_to_stream(srcFeaturesToPE_3, tmp_vertex_features);
    }
}


}

#endif
