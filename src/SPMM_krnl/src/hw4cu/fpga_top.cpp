#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./fpga_top.hpp"





extern "C"{



void Spmm(
    // AXI Master Interface
    const v_datatype * src_features,   // 512-bit widths
    const v_edges * edgestream, // 512-bit widths
    const v_edge_value * edge_value_array, // 512-bit widths 
    v_datatype * dst_features, // 512-bit widths
    int k, // number of vertex
    int tkstart, // total number of vertex
    int tkend,
    int nnz, // number of non-zero elements
    int dstoffset
){

    #pragma HLS aggregate variable=src_features
    #pragma HLS aggregate variable=edgestream
    #pragma HLS aggregate variable=edge_value_array
    #pragma HLS aggregate variable=dst_features

    #pragma HLS INTERFACE m_axi port=src_features offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=edgestream offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=edge_value_array offset=slave bundle=gmem2
    #pragma HLS INTERFACE m_axi port=dst_features offset=slave bundle=gmem3

    #pragma HLS INTERFACE s_axilite port=src_features bundle=control
    #pragma HLS INTERFACE s_axilite port=edgestream bundle=control
    #pragma HLS INTERFACE s_axilite port=edge_value_array bundle=control
    #pragma HLS INTERFACE s_axilite port=dst_features bundle=control

	#pragma HLS INTERFACE s_axilite port=k bundle=control
    #pragma HLS INTERFACE s_axilite port=tkstart bundle=control
    #pragma HLS INTERFACE s_axilite port=tkend bundle=control
    #pragma HLS INTERFACE s_axilite port=nnz bundle=control
    #pragma HLS INTERFACE s_axilite port=dstoffset bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control





//    hls::stream<v_edge_src> srcToReadBurst;
//    #pragma HLS STREAM variable=srcToReadBurst depth=2


    static hls::stream<v_onchip_edges> edgesToDispatcher;
    #pragma HLS STREAM variable=edgesToDispatcher depth=2

    hls::stream<v_edge_value> edgesValueToDispatcher;
    #pragma HLS STREAM variable=edgesValueToDispatcher depth=2


    hls::stream<v_datatype_onchip> srcFeaturesToBroadcastor;
    #pragma HLS STREAM variable=srcFeaturesToBroadcastor depth=2

    hls::stream<v_datatype_onchip> srcFeaturesToPE[4];
    #pragma HLS STREAM variable=srcFeaturesToPE depth=2

    hls::stream<edge_onchip_type> edgesToPE[4];
    #pragma HLS STREAM variable=edgesToPE depth=2

    hls::stream<v_datatype_onchip> ScatterToCalculatorFeature[4];
    #pragma HLS STREAM variable=ScatterToCalculatorFeature depth=2

    hls::stream<edge_onchip_type> ScatterToCalculatorEdge[4];
    #pragma HLS STREAM variable=ScatterToCalculatorEdge depth=2

    hls::stream<update_type> peToShuffle[4];
    #pragma HLS STREAM variable=peToShuffle depth=2

    hls::stream<update_type> ShuffleToRAW[8];
    #pragma HLS STREAM variable=ShuffleToRAW depth=2

    hls::stream<v_datatype> updateToResultWriters[8];
    #pragma HLS STREAM variable=updateToResultWriters depth=2



    hls::stream<endflag> endtunnel1[4];
    #pragma HLS STREAM variable=endtunnel1 depth=1

    hls::stream<endflag> endtunnel2[2];
    #pragma HLS STREAM variable=endtunnel2 depth=1

    hls::stream<endflag> endtunnel3;
    #pragma HLS STREAM variable=endtunnel3 depth=1

    hls::stream<endflag> endtunnel4[8];
    #pragma HLS STREAM variable=endtunnel4 depth=1

#pragma HLS dataflow

    const int num_pe = 8;

    // printf("kernel 1\n");

    edge_loder(edgestream, edgesToDispatcher, nnz);

    // printf("kernel 2\n");

    edge_value_loader(edge_value_array, edgesValueToDispatcher, nnz);

    // printf("kernel 3\n");

    vertex_read(src_features, srcFeaturesToBroadcastor, tkstart, tkend);

    // printf("kernel 4\n");

    data_spliter(srcFeaturesToBroadcastor, 
        srcFeaturesToPE[0], 
        srcFeaturesToPE[1],
        srcFeaturesToPE[2],
        srcFeaturesToPE[3],
        tkend - tkstart + 1
	);

    // printf("kernel 5\n");

    edge_dispatcher(
        edgesToDispatcher,
        edgesValueToDispatcher,
        edgesToPE[0],
        edgesToPE[1],
        edgesToPE[2],
        edgesToPE[3],
        nnz
    );

    // printf("kernel 6\n");

    for(int i = 0; i < 4; i++){
#pragma HLS unroll 
        PE_scatter(
        	srcFeaturesToPE[i],
			edgesToPE[i],
			ScatterToCalculatorFeature[i],
			ScatterToCalculatorEdge[i]
		);
        // printf("kernel 6-%d\n", i);
    }

    // printf("kernel 7\n");

    for(int i = 0; i < 4; i++){
#pragma HLS unroll 
        Scatter_Calculator(
        	ScatterToCalculatorFeature[i],
			ScatterToCalculatorEdge[i],
			peToShuffle[i],
			endtunnel1[i]
		);
    }

    // printf("kernel 8\n");

    endflagPropagator4to2(
        endtunnel1[0],
        endtunnel1[1],
        endtunnel1[2],
        endtunnel1[3],

        endtunnel2[0],
        endtunnel2[1]
    );

    // printf("kernel 9\n");


    cmpswitch8x8(
        peToShuffle[0],
        peToShuffle[1],
	    peToShuffle[2],
	    peToShuffle[3],

	    ShuffleToRAW[0],
	    ShuffleToRAW[1],
	    ShuffleToRAW[2],
	    ShuffleToRAW[3],
	    ShuffleToRAW[4],
	    ShuffleToRAW[5],
	    ShuffleToRAW[6],
	    ShuffleToRAW[7],

        endtunnel2[0],
	    endtunnel2[1],
        endtunnel3,
        k
    );



    

    // printf("kernel 10\n");

    endflagPropagator1to8(
        endtunnel3,
        endtunnel4[0],
        endtunnel4[1],
        endtunnel4[2],
        endtunnel4[3],
        endtunnel4[4],
        endtunnel4[5],
        endtunnel4[6],
        endtunnel4[7]
    );

    // printf("kernel 11\n");

    for(int i = 0; i < 8; i++){
#pragma HLS unroll 
        accumulate(ShuffleToRAW[i], updateToResultWriters[i], endtunnel4[i], k, i, dstoffset);
    }

    //  printf("kernel 12\n");
    // write back the results

    resultWriter(
        updateToResultWriters[0],
        updateToResultWriters[1],
        updateToResultWriters[2],
        updateToResultWriters[3],
        updateToResultWriters[4],
        updateToResultWriters[5],
        updateToResultWriters[6],
        updateToResultWriters[7],
        dst_features,
        k
    );

    //  printf("kernel 13\n");
    


}

}
