#include <stdio.h>
#include <ap_int.h>
#include <ostream>
#include <hls_stream.h>
#include <string.h>
#include "./types.h"
#include "./utils.h"
#include "./fpga_edge_loader.h"
#include "./fpga_vertex_read.h"
#include "./fpga_edge_dispatcher.h"
#include "./fpga_PE_scatter.h"
#include "./fpga_update.h"
#include "./fpga_routing.h"
#include "./fpga_result_writter.h"
#include "./fpga_data_spliter.h"




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
);

}
