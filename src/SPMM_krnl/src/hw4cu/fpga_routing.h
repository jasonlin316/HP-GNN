#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"



#ifndef ROUTING_NETWORK_DEF
#define ROUTING_NETWORK_DEF

#define bufferSize 4

extern "C"{

void cmpswitch_demux(
	hls::stream<update_type> & input1,
	hls::stream<update_type> & input2,
	hls::stream<update_type> & buffer1_1,
	hls::stream<update_type> & buffer1_2,
	hls::stream<update_type> & buffer2_1,
	hls::stream<update_type> & buffer2_2,
	hls::stream<endflag> & endflag_in,
	hls::stream<endflag> & endflag_out,
	int bitselect
){

    update_type tmp_input1;
    update_type tmp_input2;

	ap_uint<1> even1 = 0;
	ap_uint<1> even2 = 0;
	ap_uint<1> rdflag1;
	ap_uint<1> rdflag2;
	ap_uint<1> rdena1 = 1;
	ap_uint<1> rdena2 = 1;
	ap_uint<1> rdvalid1 = 0;
	ap_uint<1> rdvalid2 = 0;
	ap_uint<1> wrvalid1 = 0;
	ap_uint<1> wrvalid2 = 0;




	ap_uint<1> endflagCheck = 0;

	endflag tmpendflag1 = 0;


	int i = 0;

	while(true){
#pragma HLS pipeline II=1
		if (endflagCheck == 0){
			endflagCheck = read_from_stream_nb(endflag_in, tmpendflag1);
		}
		else{
			if(rdena1 == 1 && rdvalid1 == 0 && input1.empty() == 1 && rdena2 == 1 && rdvalid2 == 0 && input2.empty() == 1){
				break; // end of the pipeline
			}
		}



		if(rdena1 == 1) {
			rdvalid1 = read_from_stream_nb(input1, tmp_input1); // test if we can read data from input1
		}
		if(rdena2 == 1) {
			rdvalid2 = read_from_stream_nb(input2, tmp_input2);  // test if we can read data from input2
		}


		if (rdvalid1 == 1){    // for the successful read, write it into the buffer
			if (bitselect & tmp_input1.dst){
				wrvalid1 = write_to_stream_nb(buffer1_2, tmp_input1);
			}
			else{
				wrvalid1 = write_to_stream_nb(buffer1_1, tmp_input1);
			}
			if(wrvalid1 == 1){rdena1 = 1; rdvalid1 = 0; wrvalid1 =0;} // write successfully, read next element from input1
			else {rdena1 = 0; rdvalid1 = 1; wrvalid1 =0;}  // write fail, begin rewrite
		}

		if (rdvalid2 == 1){
			if (bitselect & tmp_input2.dst){
				wrvalid2 = write_to_stream_nb(buffer2_2, tmp_input2);
			}
			else{
				wrvalid2 = write_to_stream_nb(buffer2_1, tmp_input2);
			}
			if(wrvalid2 == 1){rdena2 = 1; rdvalid2 = 0; wrvalid2 = 0;}
			else {rdena2 = 0; rdvalid2 = 1; wrvalid2 = 0;}
		}

		

	}

	write_to_stream(endflag_out, tmpendflag1);


}


void cmpswitch_mux(
	hls::stream<update_type> & output1,
	hls::stream<update_type> & output2,
	hls::stream<update_type> & buffer1_1,
	hls::stream<update_type> & buffer1_2,
	hls::stream<update_type> & buffer2_1,
	hls::stream<update_type> & buffer2_2,
	hls::stream<endflag> & endflag_in,
	hls::stream<endflag> & endflag_out
){
	update_type tmp_otuput1;
	update_type tmp_otuput2;

	ap_uint<1> old_even1 = 0;
	ap_uint<1> old_even2 = 0;

	ap_uint<1> rdoutena1 = 1;
	ap_uint<1> rdoutena2 = 1;

	ap_uint<1> rdoutvalid1 = 0;
	ap_uint<1> rdoutvalid2 = 0;

	ap_uint<1> wroutvalid1 = 0;
	ap_uint<1> wroutvalid2 = 0;

	ap_uint<1> endflagCheck = 0;

	endflag tmpendflag1 = 0;

	int i = 0;

	while(true){
#pragma HLS pipeline II=1



		if (endflagCheck == 0){
			endflagCheck = read_from_stream_nb(endflag_in, tmpendflag1);
		}
		else{
			if(rdoutena1 == 1 && buffer1_1.empty() && buffer1_2.empty() && rdoutena2 == 1 && buffer2_1.empty() && buffer2_2.empty()){
				break; 
			}
		}



		if (old_even1 == 0){
			if (rdoutena1 == 1){
				rdoutvalid1 = read_from_stream_nb(buffer1_1, tmp_otuput1);
			}
			old_even1 = 1;
		}
		else {
			if (rdoutena1 == 1){
				rdoutvalid1 = read_from_stream_nb(buffer2_1, tmp_otuput1);
			}
			old_even1 = 0;
		}

		if (rdoutvalid1 == 1){
			wroutvalid1 = write_to_stream_nb(output1, tmp_otuput1);

			if(wroutvalid1 == 1){rdoutena1 = 1; rdoutvalid1 = 0; wroutvalid1 = 0;}
			else {rdoutena1 = 0; rdoutvalid1 = 1; wroutvalid1 = 0;}
		}



		// ********************************************************************

		if (old_even2 == 0){
			if (rdoutena2 == 1){
				rdoutvalid2 = read_from_stream_nb(buffer1_2, tmp_otuput2);
			}
			old_even2 = 1;
		}
		else {
			if (rdoutena2 == 1){
				rdoutvalid2 = read_from_stream_nb(buffer2_2, tmp_otuput2);
			}
			old_even2 = 0;
		}

		if (rdoutvalid2 == 1){
			wroutvalid2 = write_to_stream_nb(output2, tmp_otuput2);

			if(wroutvalid2 == 1){rdoutena2 = 1; rdoutvalid2 = 0; wroutvalid2 =0;}
			else {rdoutena2 = 0; rdoutvalid2 = 1; wroutvalid2=0;}

		}

		

	}
	write_to_stream(endflag_out, tmpendflag1);
}




void cmpswitch(
    hls::stream<update_type> & input1,
    hls::stream<update_type> & input2,
	hls::stream<update_type> & output1,
	hls::stream<update_type> & output2,
    hls::stream<endflag> & endflag_in,
    hls::stream<endflag> & endflag_out,
    int bitselect
){

    hls::stream<update_type>  buffer1_1("buffer1_1");
#pragma HLS STREAM variable=buffer1_1 depth=4

    hls::stream<update_type>  buffer1_2("buffer1_2");
#pragma HLS STREAM variable=buffer1_2 depth=4

    hls::stream<update_type>  buffer2_1("buffer2_1");
#pragma HLS STREAM variable=buffer2_1 depth=4

    hls::stream<update_type>  buffer2_2("buffer2_2");
#pragma HLS STREAM variable=buffer2_2 depth=4

#pragma HLS dataflow

	hls::stream<endflag>  endflag_inmediate;
#pragma HLS STREAM variable=endflag_inmediate depth=1

    cmpswitch_demux(input1,input2,buffer1_1,buffer1_2, buffer2_1,buffer2_2, endflag_in, endflag_inmediate, bitselect);



    cmpswitch_mux(output1, output2, buffer1_1,buffer1_2, buffer2_1, buffer2_2, endflag_inmediate, endflag_out);




}



void cmpswitch8x8(
    hls::stream<update_type> & input1,
    hls::stream<update_type> & input2,
	hls::stream<update_type> & input3,
	hls::stream<update_type> & input4,

	hls::stream<update_type> & output1,
	hls::stream<update_type> & output2,
	hls::stream<update_type> & output3,
	hls::stream<update_type> & output4,
	hls::stream<update_type> & output5,
	hls::stream<update_type> & output6,
	hls::stream<update_type> & output7,
	hls::stream<update_type> & output8,


    hls::stream<endflag> & endflag_in1,
	hls::stream<endflag> & endflag_in2,
    hls::stream<endflag> & endflag_out,
	int k
){
#pragma HLS dataflow
	// stage 1:

	hls::stream<update_type> stage1_output1;
#pragma HLS BIND_STORAGE variable=stage1_output1 type=fifo
#pragma HLS STREAM variable=stage1_output1 depth=2
	hls::stream<update_type> stage1_output2;
#pragma HLS BIND_STORAGE variable=stage1_output2 type=fifo 
#pragma HLS STREAM variable=stage1_output2 depth=2
	hls::stream<update_type> stage1_output3;
#pragma HLS BIND_STORAGE variable=stage1_output3 type=fifo 
#pragma HLS STREAM variable=stage1_output3 depth=2
	hls::stream<update_type> stage1_output4;
#pragma HLS BIND_STORAGE variable=stage1_output4 type=fifo
#pragma HLS STREAM variable=stage1_output4 depth=2
	
	hls::stream<endflag>  endflag1to2_n1;
#pragma HLS STREAM variable=endflag1to2_n1 depth=1
	hls::stream<endflag>  endflag1to2_n2;
#pragma HLS STREAM variable=endflag1to2_n2 depth=1

	// printf("routing stage 1\n");

	cmpswitch(input1, input3,  stage1_output1, stage1_output3, endflag_in1, endflag1to2_n1, 4);
	cmpswitch(input2, input4,  stage1_output2, stage1_output4, endflag_in2, endflag1to2_n2, 4);
    // stage 2:

	hls::stream<update_type> stage2_output1;
#pragma HLS BIND_STORAGE variable=stage2_output1 type=fifo impl=lutram
#pragma HLS STREAM variable=stage2_output1 depth=2
	hls::stream<update_type> stage2_output2;
#pragma HLS BIND_STORAGE variable=stage2_output2 type=fifo impl=lutram
#pragma HLS STREAM variable=stage2_output2 depth=2
	hls::stream<update_type> stage2_output3;
#pragma HLS BIND_STORAGE variable=stage2_output3 type=fifo impl=lutram
#pragma HLS STREAM variable=stage2_output3 depth=2
	hls::stream<update_type> stage2_output4;
#pragma HLS BIND_STORAGE variable=stage2_output4 type=fifo impl=lutram
#pragma HLS STREAM variable=stage2_output4 depth=2
	// printf("routing stage 2\n");

	hls::stream<endflag>  endflag1to2_f1;
#pragma HLS STREAM variable=endflag1to2_f1 depth=1	
	hls::stream<endflag>  endflag1to2_f2;
#pragma HLS STREAM variable=endflag1to2_f2 depth=1

	endflagPropagator2to2(
    	endflag1to2_n1,
     	endflag1to2_n2,

    	endflag1to2_f1,
    	endflag1to2_f2
	);	


	hls::stream<endflag>  endflag2to3_n1;
#pragma HLS STREAM variable=endflag2to3_n1 depth=1	
	hls::stream<endflag>  endflag2to3_n2;
#pragma HLS STREAM variable=endflag2to3_n2 depth=1

	cmpswitch(stage1_output1, stage1_output2,  stage2_output1, stage2_output2, endflag1to2_f1, endflag2to3_n1, 2);
	cmpswitch(stage1_output3, stage1_output4,  stage2_output3, stage2_output4, endflag1to2_f2, endflag2to3_n2, 2);

	// printf("routing stage 3\n");
	hls::stream<endflag>  endflag2to3_f1;
#pragma HLS STREAM variable=endflag2to3_f1 depth=1	
	hls::stream<endflag>  endflag2to3_f2;
#pragma HLS STREAM variable=endflag2to3_f2 depth=1
	hls::stream<endflag>  endflag2to3_f3;
#pragma HLS STREAM variable=endflag2to3_f3 depth=1
	hls::stream<endflag>  endflag2to3_f4;
#pragma HLS STREAM variable=endflag2to3_f4 depth=1

	endflagPropagator2to4(
    	endflag2to3_n1,
     	endflag2to3_n2,

    	endflag2to3_f1,
    	endflag2to3_f2,
    	endflag2to3_f3,
    	endflag2to3_f4
	);	

	hls::stream<endflag>  endflag3to4_n1;
#pragma HLS STREAM variable=endflag3to4_n1 depth=1	
	hls::stream<endflag>  endflag3to4_n2;
#pragma HLS STREAM variable=endflag3to4_n2 depth=1
	hls::stream<endflag>  endflag3to4_n3;
#pragma HLS STREAM variable=endflag3to4_n3 depth=1
	hls::stream<endflag>  endflag3to4_n4;
#pragma HLS STREAM variable=endflag3to4_n4 depth=1

	cmpswitch(stage2_output1, stage2_output1,  output1, output2, endflag2to3_f1, endflag3to4_n1, 1);
	cmpswitch(stage2_output2, stage2_output2,  output3, output4, endflag2to3_f2, endflag3to4_n2, 1);
	cmpswitch(stage2_output3, stage2_output3,  output5, output6, endflag2to3_f3, endflag3to4_n3, 1);
	cmpswitch(stage2_output4, stage2_output4,  output7, output8, endflag2to3_f4, endflag3to4_n4, 1);

	endflagPropagator4to1(
    	endflag3to4_n1,
    	endflag3to4_n2,
    	endflag3to4_n3,
    	endflag3to4_n4,
		endflag_out
	);

	
}


}

#endif
