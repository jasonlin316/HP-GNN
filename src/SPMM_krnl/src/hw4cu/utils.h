#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"

#ifndef UTILS_DEF
#define UTILS_DEF




template <typename T>
inline int read_from_stream (hls::stream<T> &stream, T & value)
{
#pragma HLS INLINE
    value = stream.read();
    return 0;
}


template <typename T>
inline int read_from_stream_nb (hls::stream<T> &stream, T & value)
{
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
    if (stream.empty())
    {
        return 0;
    }
    else
    {
        value = stream.read();
        return 1;
    }
}

template <typename T>
inline int write_to_stream (hls::stream<T> &stream, T const& value)
{
#pragma HLS INLINE
    stream << value;
    return 0;
}



template <typename T>
inline int write_to_stream_nb (hls::stream<T> &stream, T const& value)
{
#pragma HLS INLINE
#pragma HLS PIPELINE II=1
    if(stream.full())
    {
        return 0;
    }
    else{
        stream << value;
        return 1;
    }
}

extern "C"{

void endflagPropagator8to4(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,
   hls::stream<endflag> & endtunnel_input3,
   hls::stream<endflag> & endtunnel_input4,
   hls::stream<endflag> & endtunnel_input5,
   hls::stream<endflag> & endtunnel_input6,
   hls::stream<endflag> & endtunnel_input7,
   hls::stream<endflag> & endtunnel_input8,

	 hls::stream<endflag> & endtunnel_output1,
	 hls::stream<endflag> & endtunnel_output2,
	 hls::stream<endflag> & endtunnel_output3,
	 hls::stream<endflag> & endtunnel_output4
)
{
#pragma HLS function_instantiate variable=endtunnel_input1

#pragma HLS dataflow

  endflag tmpendflag[8];
#pragma HLS array_partition variable=tmpendflag complete dim=1

  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  read_from_stream(endtunnel_input3, tmpendflag[2]);
  read_from_stream(endtunnel_input4, tmpendflag[3]);
  read_from_stream(endtunnel_input5, tmpendflag[4]);
  read_from_stream(endtunnel_input6, tmpendflag[5]);
  read_from_stream(endtunnel_input7, tmpendflag[6]);
  read_from_stream(endtunnel_input8, tmpendflag[7]);

  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[1]);
  write_to_stream(endtunnel_output3, tmpendflag[2]);
  write_to_stream(endtunnel_output4, tmpendflag[3]);

}


void endflagPropagator1to8(
   hls::stream<endflag> & endtunnel_input1,

   hls::stream<endflag> & endtunnel_output1,
   hls::stream<endflag> & endtunnel_output2,
   hls::stream<endflag> & endtunnel_output3,
   hls::stream<endflag> & endtunnel_output4,
   hls::stream<endflag> & endtunnel_output5,
   hls::stream<endflag> & endtunnel_output6,
   hls::stream<endflag> & endtunnel_output7,
   hls::stream<endflag> & endtunnel_output8
)
{
#pragma HLS function_instantiate variable=endtunnel_input1

#pragma HLS dataflow

  endflag tmpendflag[8];
#pragma HLS array_partition variable=tmpendflag complete dim=1

  read_from_stream(endtunnel_input1, tmpendflag[0]);


  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[1]);
  write_to_stream(endtunnel_output3, tmpendflag[2]);
  write_to_stream(endtunnel_output4, tmpendflag[3]);
  write_to_stream(endtunnel_output5, tmpendflag[4]);
  write_to_stream(endtunnel_output6, tmpendflag[5]);
  write_to_stream(endtunnel_output7, tmpendflag[6]);
  write_to_stream(endtunnel_output8, tmpendflag[7]);

}


void endflagPropagator2to2(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,

	 hls::stream<endflag> & endtunnel_output1,
	 hls::stream<endflag> & endtunnel_output2
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[2];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);

  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[1]);
}



void endflagPropagator2to1(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,

	 hls::stream<endflag> & endtunnel_output1
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[2];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  write_to_stream(endtunnel_output1, tmpendflag[0]);

}

void endflagPropagator4to4(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,
   hls::stream<endflag> & endtunnel_input3,
   hls::stream<endflag> & endtunnel_input4,

	 hls::stream<endflag> & endtunnel_output1,
   hls::stream<endflag> & endtunnel_output2,
   hls::stream<endflag> & endtunnel_output3,
   hls::stream<endflag> & endtunnel_output4
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[4];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  read_from_stream(endtunnel_input3, tmpendflag[2]);
  read_from_stream(endtunnel_input4, tmpendflag[3]);
  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[1]);
  write_to_stream(endtunnel_output3, tmpendflag[2]);
  write_to_stream(endtunnel_output4, tmpendflag[3]);

}

void endflagPropagator4to2(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,
   hls::stream<endflag> & endtunnel_input3,
   hls::stream<endflag> & endtunnel_input4,

	 hls::stream<endflag> & endtunnel_output1,
   hls::stream<endflag> & endtunnel_output2
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[4];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  read_from_stream(endtunnel_input3, tmpendflag[2]);
  read_from_stream(endtunnel_input4, tmpendflag[3]);

  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[1]);
}

void endflagPropagator4to1(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,
   hls::stream<endflag> & endtunnel_input3,
   hls::stream<endflag> & endtunnel_input4,

	 hls::stream<endflag> & endtunnel_output1
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[4];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  read_from_stream(endtunnel_input3, tmpendflag[2]);
  read_from_stream(endtunnel_input4, tmpendflag[3]);
  write_to_stream(endtunnel_output1, tmpendflag[0]);
}

void endflagPropagator2to4(
   hls::stream<endflag> & endtunnel_input1,
   hls::stream<endflag> & endtunnel_input2,

	 hls::stream<endflag> & endtunnel_output1,
   hls::stream<endflag> & endtunnel_output2,
   hls::stream<endflag> & endtunnel_output3,
   hls::stream<endflag> & endtunnel_output4
){
#pragma HLS function_instantiate variable=endtunnel_input1
#pragma HLS dataflow


  endflag tmpendflag[2];
#pragma HLS array_partition variable=tmpendflag complete dim=1



  read_from_stream(endtunnel_input1, tmpendflag[0]);
  read_from_stream(endtunnel_input2, tmpendflag[1]);
  write_to_stream(endtunnel_output1, tmpendflag[0]);
  write_to_stream(endtunnel_output2, tmpendflag[0]);
  write_to_stream(endtunnel_output3, tmpendflag[0]);
  write_to_stream(endtunnel_output4, tmpendflag[0]);

}

}

#endif
