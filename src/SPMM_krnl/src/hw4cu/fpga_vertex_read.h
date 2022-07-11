#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"

//#ifndef VERTEX_READ_DEF
//#define VERTEX_READ_DEF

extern "C"{


void vertex_read(
    const v_datatype * src_features, // memory interface of loading features
//    hls::stream<v_edge_src> & srcToReadBurst,
    hls::stream<v_datatype_onchip> & srcFeaturesToBroadcastor,
    int tkstart,
    int tkend
){
    v_datatype_onchip tmp_vertex_features;
#pragma HLS aggregate variable=tmp_vertex_features

    for(int i = tkstart; i < tkend + 1; i = i + 1)
    {
#pragma HLS PIPELINE II=1 rewind
        tmp_vertex_features.data = src_features[i];
        tmp_vertex_features.src = i; // attach the src indices to the vertex features
        write_to_stream(srcFeaturesToBroadcastor, tmp_vertex_features);
    }
}

}



//#endif
