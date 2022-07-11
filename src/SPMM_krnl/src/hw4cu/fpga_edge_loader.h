#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"

//#ifndef DEF_EDGE_LOADER
//#define DEF_EDGE_LOADER


// extern "C"{


static void edge_loder(
    const v_edges * edgestream,
//    hls::stream<v_edge_src> & srcToReadBurst,
    hls::stream<v_onchip_edges> & edgesToPEs,
    int nnz
){
//#pragma HLS function_instantiate  variable=edgestream
//#pragma HLS function_instantiate  variable=edgesToPEs



//     v_edge_src tmpedge_src;
// #pragma HLS aggregate variable=tmpedge_src



    int k = 0;
    int i = 0;
    int numberoifit = int( nnz /8 );
    if (int( nnz % 8 )!= 0){
        numberoifit = numberoifit + 1;
    }

    for(k = 0; k < numberoifit; k = k + 1)
    {   
#pragma HLS PIPELINE II=1
    	v_edges tmpedges = edgestream[k];  // load edges from external memory
		#pragma HLS aggregate variable=tmpedges

        // for(int j = 0; j < 8; j = j + 1)
        // {
            // if (i + j < nnz){
            //     tmpedge_src.src[j] = tmpedges.edges[j].src;
            //     tmpedge_src.flags[j] = 1; // flag is to indicate whether the edge is valid (1) or not (0).
            // }
        //     else{
        //         tmpedge_src.flags[j] = 0;
        //     }
        // }



        v_onchip_edges tmponchipedges;
        #pragma HLS aggregate variable=tmponchipedges


			for(int j = 0; j < 8; j = j + 1)
			{
#pragma HLS unroll
//#pragma HLS latency min=8 max=8

				tmponchipedges.edges[j].src = tmpedges.edges[j].src;
				tmponchipedges.edges[j].dst = tmpedges.edges[j].dst;
				tmponchipedges.edges[j].flag = ((k << 3) + j) < nnz ? 1:0;

				tmponchipedges.edges[j].end = nnz - (k << 3) <= 8 ? 1:0;

                // if(tmponchipedges.edges[j].end == 1) printf("edge is %d, %d, %d, %d\n", tmponchipedges.edges[j].src, tmponchipedges.edges[j].dst, int(tmponchipedges.edges[j].flag), int(tmponchipedges.edges[j].end));

			}

//        write_to_stream(srcToReadBurst, tmpedge_src);
        // write_to_stream(edgesToPEs, tmponchipedges); 
        edgesToPEs << tmponchipedges;
    }
}

static void edge_value_loader(
    const v_edge_value * edge_value_array, 
    hls::stream<v_edge_value> & edgesValueToPEs,
    int nnz
){
    v_edge_value tmp_edge_values;
    #pragma HLS aggregate variable=tmp_edge_values

    int k = 0;
    int numberoifit = int( nnz /16 );
    if (int( nnz % 16 )!= 0){
        numberoifit = numberoifit + 1;
    }

    for (int k = 0; k < numberoifit; k = k + 1)
    {
#pragma HLS PIPELINE II=1 rewind
        tmp_edge_values = edge_value_array[k];
        write_to_stream(edgesValueToPEs, tmp_edge_values); 
    }
}
// }





// #endif
