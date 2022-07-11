#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"

//#ifndef PE_SCATTER_DEF
//#define PE_SCATTER_DEF

extern "C"{

#define cacheSize 16

void PE_scatter(
    hls::stream<v_datatype_onchip> & srcFeaturesToPE,
    hls::stream<edge_onchip_type> & edgesToPE,
    hls::stream<v_datatype_onchip> & ScatterToCalculatorFeature,
    hls::stream<edge_onchip_type> & ScatterToCalculatorEdge
){



    edge_onchip_type tmpedges;
#pragma HLS aggregate variable=tmpedges
    v_datatype_onchip tmpfeatures;
#pragma HLS aggregate variable=tmpfeatures

    ap_uint<1> enableReadEdge = 1;
    ap_uint<1> ebableReadFeatures = 1;

    // int i =0;

    ap_uint<1> endedgeread = 0;

    while(true){
#pragma HLS PIPELINE II=1 rewind
        if(enableReadEdge == 1) {read_from_stream(edgesToPE, tmpedges); }
        if(ebableReadFeatures == 1) {read_from_stream(srcFeaturesToPE, tmpfeatures);}
        // printf("pairs(src %d, dst %d, %d), endflag is %d\n", tmpedges.src, tmpedges.dst, tmpfeatures.src, int(tmpedges.end));
        // i++;
        
        if(tmpedges.src == tmpfeatures.src){
            // if (tmpedges.flag == 1){
                // printf("pairs(src %d, dst %d, %d), endflag is %d\n", tmpedges.src, tmpedges.dst, tmpfeatures.src, int(tmpedges.end));
                // i++;
                write_to_stream(ScatterToCalculatorEdge, tmpedges); 
                write_to_stream(ScatterToCalculatorFeature, tmpfeatures);
            // }
            ebableReadFeatures = 0;
            enableReadEdge = 1;
            if(tmpedges.end == 1){
                endedgeread = 1;
            }
        }
        else if(tmpedges.src < tmpfeatures.src && tmpedges.end == 0){
            ebableReadFeatures = 0;
            enableReadEdge = 1;
        }
        else{
            ebableReadFeatures = 1;
            enableReadEdge = 0;
        }

        if(tmpedges.end == 1 && endedgeread == 1){
            // printf("the tmp edge src %d\n", tmpedges.src);
            break;
            
        }
    }

    while(srcFeaturesToPE.empty() == 0){
        read_from_stream(srcFeaturesToPE, tmpfeatures);
    }

    while(edgesToPE.empty() == 0){
        read_from_stream(edgesToPE, tmpedges);
    }

}






void Scatter_Calculator(
    hls::stream<v_datatype_onchip> & ScatterToCalculatorFeature,
    hls::stream<edge_onchip_type> & ScatterToCalculatorEdge,
    hls::stream<update_type> & peToShuffle,
    hls::stream<endflag> & endtunnel
){

    edge_onchip_type tmpedges;
#pragma HLS aggregate variable=tmpedges
    v_datatype_onchip tmpfeatures;
#pragma HLS aggregate variable=tmpfeatures

    update_type tmpupdate;
#pragma HLS aggregate variable=tmpupdate

    // printf("enter-calculator\n" );
    ap_uint<1> readyout = 0;

    while(true){
#pragma HLS PIPELINE II=1 rewind

        if(readyout == 1){
            break;
        }


        read_from_stream(ScatterToCalculatorEdge, tmpedges);
        read_from_stream(ScatterToCalculatorFeature, tmpfeatures);

        if( tmpedges.end == 1){
            readyout = 1;;
        }

        
        tmpupdate.dst = tmpedges.dst;
        tmpupdate.flag = tmpedges.flag;
        tmpupdate.end = tmpedges.end;
        // printf("edge value is %f, dst of edge is %d, the value of features is", tmpedges.value,  tmpupdate.dst);
        for(int i = 0; i < 16; i++)
        {
            tmpupdate.value.data[i] = tmpedges.value * tmpfeatures.data.data[i];
            // printf("%f ", tmpupdate.value.data[i]);
        }
        // printf("\n");
        if (tmpedges.flag == 1){ write_to_stream(peToShuffle, tmpupdate);}
        

        // if( tmpedges.end == 1){
        //     break;
        // }
        

    }
    
    endflag to_the_end = 1;
    write_to_stream(endtunnel, to_the_end);
}

}


//#endif
