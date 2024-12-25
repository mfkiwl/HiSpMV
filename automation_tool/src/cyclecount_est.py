from commons import SpMVConfig
from preprocessor import PreProcessor
from scipy.sparse import coo_matrix
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CycleCountEstimator():
    @staticmethod
    def getCC(mySpMV: SpMVConfig, matrix: coo_matrix):
        NUM_PES = (mySpMV.num_ch_A * mySpMV.ch_width) // 64
        DEPTH = (NUM_PES * mySpMV.urams_per_pe) * 4096
        B_PART = (mySpMV.num_ch_B * mySpMV.ch_width) // 32
        WINDOW = (B_PART // 2) * 1024

        num_rows, num_cols = matrix.shape
        C_PART = (mySpMV.num_ch_C * mySpMV.ch_width) // 32
        assert (NUM_PES % C_PART)==0, "NUM_PES should be a multiple of C_PART"
        padded_rows = (((num_rows - 1) // NUM_PES) + 1) * NUM_PES 
        padded_cols = (((num_cols - 1) // B_PART) + 1) * B_PART 

        num_tiles_r = ((padded_rows - 1) // DEPTH) + 1
        num_tiles_c = ((padded_cols - 1) // WINDOW) + 1

        logger.debug(f"SpMV Config: NUM_A_CH={mySpMV.num_ch_A}, NUM_B_CH={mySpMV.num_ch_B}, NUM_C_CH={mySpMV.num_ch_C}")
        logger.debug(f"SpMV Config: CH_WIDTH={mySpMV.ch_width}, URAMS_PER_PE={mySpMV.urams_per_pe}")
        logger.debug(f"SpMV Config: DENSE_OVERLAY={mySpMV.dense_overlay}, PRE_ACCUMULATOR={mySpMV.pre_accumulator}, ROW_DIST_NET={mySpMV.row_dist_net}")

        logger.debug(f"SpMV Config: NUM_PES={NUM_PES}, B_PART={B_PART}, C_PART={C_PART}")
        logger.debug(f"SpMV Config: DEPTH={DEPTH}, WINDOW={WINDOW}")

        logger.debug(f"Padding: Padded Rows={padded_rows}, Padded Cols={padded_cols}")
        logger.debug(f"Tiling: {num_tiles_r, num_tiles_c}")

        rows_per_pe_per_tile = min(DEPTH//NUM_PES, padded_rows//NUM_PES)
        row_cnt = np.zeros((num_tiles_r, num_tiles_c, NUM_PES, rows_per_pe_per_tile), dtype=int)
        
        row_indices = np.array(matrix.row, dtype=np.int32)
        col_indices = np.array(matrix.col, dtype=np.int32)

        # Calculate each index array using numpy operations
        tile_r = row_indices // DEPTH
        tile_c = col_indices // WINDOW
        pe = row_indices % NUM_PES
        pe_row = (row_indices % DEPTH) // NUM_PES

        # Now apply all increments at once with numpy's add.at for atomic updates
        np.add.at(row_cnt, (tile_r, tile_c, pe, pe_row), 1)

        CC_STREAM_A = PreProcessor.get_run_length(mySpMV, row_cnt)
        CC_LOAD_B = padded_cols//B_PART
        CC_UPDATE_C = padded_rows//C_PART

        CC_TOTAL = CC_STREAM_A + (num_tiles_r * CC_LOAD_B) + CC_UPDATE_C

        return CC_TOTAL

