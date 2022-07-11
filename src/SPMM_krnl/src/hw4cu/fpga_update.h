#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"



#ifndef UPDATE_DEF
#define UPDATE_DEF

# define PIP_STAGE 7

extern "C"{



void rawResolver(
		hls::stream<update_type> & input,
		hls::stream<update_type> & output,
        hls::stream<endflag> & endflag_in,
        hls::stream<endflag> & endflag_out
){
#pragma HLS function_instantiate variable=input
#pragma HLS function_instantiate variable=output

	ap_uint<1> enaRD = 1;
	ap_uint<1> RDsuccess = 0;
	ap_uint<1> WRsuccess = 0;

	int i = 0;
	update_type tmpupdate;
#pragma HLS aggregate variable=tmpupdate

	index_type dstarray[PIP_STAGE];
#pragma HLS array_partition variable=dstarray complete dim=1

	ap_uint<1> validflag[PIP_STAGE];
#pragma HLS array_partition variable=validflag complete dim=1

	ap_uint<1> diff[PIP_STAGE];
#pragma HLS array_partition variable=diff complete dim=1

    ap_uint<1> readflag = 0;
    ap_uint<1> readena = 1;
    ap_uint<1> intermediate_validflag;
    endflag tmpflag;

	initailization: for(int k = 0; k < PIP_STAGE; k++){
#pragma HLS unroll
		dstarray[k] = 0;
		validflag[k] = 0;
	}


	execution: while(true){
#pragma HLS pipeline II=1

		
		if(readflag == 0){
            readflag = read_from_stream_nb(endflag_in, tmpflag);
        }
       
        if(readena == 1 && readflag == 1 && input.empty() == 1){
            break;
        }
    
        if (readena == 1){
		    read_from_stream(input, tmpupdate);
            readena = 0;
        }   



		inner1: for (int j = 0; j < PIP_STAGE; j++){
#pragma HLS unroll
			if (dstarray[j] == tmpupdate.dst && validflag[j] == 1){
				diff[j] = 1;
			}
			else {
				diff[j] = 0;
			}
		}

		if(diff[0] == 0 && diff[1] == 0 && diff[2] == 0 && diff[3] == 0 && diff[4] == 0 && diff[5] == 0 && diff[6] == 0){
			tmpupdate.valid = 1;
            intermediate_validflag = 1;
			write_to_stream(output, tmpupdate);
			readena = 1;
		}
		else{
			update_type invalidupdate;
			invalidupdate.valid = 0;
            intermediate_validflag = 0;
#pragma HLS aggregate variable=invalidupdate
			write_to_stream(output, invalidupdate);
		}
		for(int j = 0; j < PIP_STAGE - 1; j++){
			dstarray[j] = dstarray[j+1];
			validflag[j] = validflag[j+1];
		}
		dstarray[PIP_STAGE - 1] = tmpupdate.dst;
		validflag[PIP_STAGE - 1] = intermediate_validflag;
    }

    write_to_stream(endflag_out, tmpflag);
}





void updateApply(
		hls::stream<update_type> & input,
		hls::stream<v_datatype> & output,
        hls::stream<endflag> & endflag_in,
        int k,
		int i,
		int dstoffset
){
#pragma HLS function_instantiate variable=input
#pragma HLS function_instantiate variable=output

	v_datatype resultbuffer[1024*8];
#pragma HLS aggregate variable=resultbuffer
#pragma HLS bind_storage variable=resultbuffer type=ram_t2p  impl=uram


	update_type tmpupdate;
#pragma HLS aggregate variable=tmpupdate

	v_datatype regupdate;
#pragma HLS aggregate variable=regupdate


	for (int j = 0; j < 16; j++){
#pragma HLS unroll
		regupdate.data[j] = 0;
	}

	 //data initialization
	for (int i = 0; i < k/8; i++){
#pragma HLS pipeline II=1
		resultbuffer[i] = regupdate;
	}

    ap_uint<1> readflag = 0;
    endflag tmpflag;

	while(true){
#pragma HLS dependence variable=resultbuffer inter false
#pragma HLS pipeline II=2
		
		if(readflag == 0){
			readflag = read_from_stream_nb(endflag_in, tmpflag);
		}


		if(readflag == 1 && input.empty() == 1){
			break;
		}
		
//        if(readflag == 1 && input.empty() == 0){
//            break;
//        }
		
		read_from_stream(input, tmpupdate);
	

		if(tmpupdate.valid == 1){
#pragma HLS latency min=10 max=10
			index_type dst = int((tmpupdate.dst- dstoffset - i)/8);
			regupdate = resultbuffer[dst];
			// printf("the dist is %d, %d\n, the feature is ", tmpupdate.dst, dst);
			for (int j = 0; j < 16; j++){
				regupdate.data[j] = regupdate.data[j] + tmpupdate.value.data[j];
				// printf("%f ", regupdate.data[j] );
			}
			// printf("\n");
			resultbuffer[dst] = regupdate;
		}



	}



	for (int i = 0; i < k/8; i++){
#pragma HLS pipeline II=1
		regupdate = resultbuffer[i];
		write_to_stream(output, regupdate);
	}
}




void accumulate(
		hls::stream<update_type> & input,
		hls::stream<v_datatype> & output,
        hls::stream<endflag> & endflag_in,
        int k,
		int i,
		int dstoffset
){
#pragma HLS dataflow

	hls::stream<update_type> intermediate;
#pragma HLS BIND_STORAGE variable=intermediate type=fifo impl=lutram
#pragma HLS STREAM variable=intermediate depth=2

    hls::stream<endflag> intermediateflag;
#pragma HLS STREAM variable=intermediateflag depth=1

	// printf("enter RawResolver\n");

	rawResolver(input,   intermediate, endflag_in, intermediateflag);

	// printf("enter updateApply\n");
	
	updateApply(intermediate, output, intermediateflag, k, i, dstoffset);
}


}



#endif
