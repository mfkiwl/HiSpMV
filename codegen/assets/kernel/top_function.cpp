void SpMV(tapa::mmaps<channelA_t, NUM_A_CH> A,
          tapa::mmaps<channelB_t, NUM_B_CH> b,
          tapa::mmaps<channelC_t, NUM_C_CH> c_in,
          tapa::mmaps<channelC_t, NUM_C_CH> c_out,
          const float alpha, const float beta,
          const uint32_t A_off, const uint32_t A_len,
          const uint32_t num_rows_per_pe, const uint32_t B_len,
          const uint16_t num_tiles_r, const uint16_t num_tiles_c,
          const uint32_t num_tiles_rp_time, const uint16_t rp_time,
          const bool DENSE_MODE)
{

    tapa::streams<uint64_t, NUM_PES, FIFO_DEPTH> FIFO_A_IN("a_in");
    tapa::streams<uint16_t, NUM_PES, FIFO_DEPTH> FIFO_C_ROW("c_row");
    tapa::streams<float, NUM_PES, FIFO_DEPTH> FIFO_C_VAL("c_val");
    tapa::streams<flags_pkt, NUM_PES_HALF, FIFO_DEPTH> FIFO_C_FLAG("c_flag");
    
    tapa::streams<channelB_t, ((NUM_PES_HALF/LOAD_GROUP_SIZE) + 1) * NUM_B_CH, FIFO_DEPTH> FIFO_B_IN("b_in");

    buffersB_t BUFF_B; 

#ifdef BUILD_ROW_DIST_NETWORK
    tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
#endif
    tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_BUF("c_buf");

    tapa::streams<float, NUM_PES, FIFO_DEPTH> FIFO_C_ARB("c_arb");
	tapa::streams<channelC_t, NUM_C_CH, FIFO_DEPTH> FIFO_C_AB("c_ab");
    tapa::streams<channelC_t, NUM_C_CH, FIFO_DEPTH> FIFO_C_IN("c_in");
    tapa::streams<channelC_t, NUM_C_CH, FIFO_DEPTH> FIFO_C_OUT("c_out");

    tapa::task()
        .invoke<tapa::join, NUM_A_CH>(MM2S_A, A, FIFO_A_IN, A_off, A_len, rp_time)
        .invoke<tapa::join, NUM_B_CH>(MM2S_B, b, FIFO_B_IN, num_tiles_r, B_len, rp_time)
        .invoke<tapa::join, NUM_PES_HALF/LOAD_GROUP_SIZE>(LoadB, FIFO_B_IN, FIFO_B_IN, BUFF_B, B_len, num_tiles_rp_time)
        .invoke<tapa::join, NUM_PES_HALF>(ComputeAB, FIFO_A_IN, FIFO_C_FLAG, FIFO_C_ROW, FIFO_C_VAL, BUFF_B, A_len, num_rows_per_pe, rp_time, DENSE_MODE)
        .invoke<tapa::detach, NUM_B_CH>(DummyReadB, FIFO_B_IN)
#ifdef BUILD_ROW_DIST_NETWORK
        .invoke<tapa::join, NUM_PES_HALF>(PreAccumulator, FIFO_C_ROW, FIFO_C_VAL, FIFO_C_FLAG, FIFO_C_SHF)
#else
		.invoke<tapa::join, NUM_PES_HALF>(PreAccumulator, FIFO_C_ROW, FIFO_C_VAL, FIFO_C_FLAG, FIFO_C_BUF)
#endif
        .invoke<tapa::join, NUM_PES>(AccumBuffer, FIFO_C_BUF, FIFO_C_ARB, num_rows_per_pe, num_tiles_c, rp_time)
		.invoke(Arbiter_C, FIFO_C_ARB, FIFO_C_AB, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(MM2S_C, c_in, FIFO_C_IN, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(Compute_C, FIFO_C_IN, FIFO_C_AB, FIFO_C_OUT, alpha, beta, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, c_out, num_rows_per_pe, rp_time);
}