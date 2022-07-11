#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>

#ifndef DATA_TYPE_DEF
#define DATA_TYPE_DEF


#define VDATA_SIZE 16
#define DST_SIZE 10240

typedef float feature_type;
typedef int index_type;
typedef ap_uint<1> endflag;


struct v_datatype {
    feature_type data[VDATA_SIZE];
};

struct v_datatype_onchip {
    v_datatype data;
    index_type src;
};


struct edge_type {
    index_type src;
    index_type dst;
};

struct edge_onchip_type {
    index_type src;
    index_type dst;
    feature_type value;
    ap_uint<1> flag; // valid flag
    ap_uint<1> end;  // last batch
};

struct v_edge_src {
    index_type src[8];
    ap_uint<1> flags[8]; 
};


struct v_edges {
    edge_type edges[8];
};

struct v_onchip_edges {
    edge_onchip_type edges[8];
};

struct edge_value {
    feature_type data;
};

struct v_edge_value
{
    edge_value edgevalues[VDATA_SIZE];
};


struct update_type {
    index_type dst;
    v_datatype value;
    ap_uint<1> flag; // valid flag
    ap_uint<1> end;  // last batch
    ap_uint<1> valid;
};





#endif
