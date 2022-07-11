#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"



extern "C"{

void edge_dispatcher(
    hls::stream<v_onchip_edges> & edgesToDispatcher,
    hls::stream<v_edge_value> & edgesValueToDispatcher,
    hls::stream<edge_onchip_type> & edgesToPE_0,
    hls::stream<edge_onchip_type> & edgesToPE_1,
    hls::stream<edge_onchip_type> & edgesToPE_2,
    hls::stream<edge_onchip_type> & edgesToPE_3,
    int nnz
){
    v_onchip_edges tmpEdges;
    v_edge_value tmpeEgeValues;
    edge_onchip_type edgeArray[8];
#pragma HLS array_partition variable=edgeArray complete dim=1


    for(int i = 0; i < nnz; i = i + 8){
#pragma HLS PIPELINE II=2 rewind
        read_from_stream(edgesToDispatcher, tmpEdges);
        if(i % 16 == 0){
            read_from_stream(edgesValueToDispatcher, tmpeEgeValues);
        }
        for(int h = 0; h < 8 ; h = h + 1){
            edgeArray[h] = tmpEdges.edges[h];
            if (i % 16 == 0) {edgeArray[h].value = tmpeEgeValues.edgevalues[h].data;}
            else {edgeArray[h].value = tmpeEgeValues.edgevalues[h + 8].data;}
        }
        write_to_stream(edgesToPE_0, edgeArray[0]);
        write_to_stream(edgesToPE_1, edgeArray[1]);
        write_to_stream(edgesToPE_2, edgeArray[2]);
        write_to_stream(edgesToPE_3, edgeArray[3]);

        write_to_stream(edgesToPE_0, edgeArray[4]);
        write_to_stream(edgesToPE_1, edgeArray[5]);
        write_to_stream(edgesToPE_2, edgeArray[6]);
        write_to_stream(edgesToPE_3, edgeArray[7]);
    }
}


}


