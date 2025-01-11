import numpy as np
from commons import SpMVConfig
import logging
from numba import njit

logger = logging.getLogger(__name__)

II_DIST = 6

class PreProcessor():
    @staticmethod
    def get_run_length(mySpMV: SpMVConfig, row_cnt: np.ndarray) -> np.int32:
        tile_sizes = PreProcessor.get_tile_sizes(mySpMV, row_cnt)
        return np.sum(tile_sizes)
    
    @staticmethod
    def get_tile_sizes(mySpMV: SpMVConfig, row_cnt: np.ndarray) -> np.ndarray: 
        num_tiles_r, num_tiles_c, _ , _ = row_cnt.shape

        #create array to store tile sizes
        tile_sizes = np.zeros((num_tiles_r, num_tiles_c), dtype=int)

        # Operate on One tile 
        for i in range(num_tiles_r):
            for j in range(num_tiles_c):
                tile_sizes[i][j] = PreProcessor.get_tile_size(mySpMV, row_cnt[i][j])
        
        return tile_sizes

    @staticmethod
    def get_tile_size(mySpMV: SpMVConfig, row_cnt: np.ndarray) -> int:
        #get row indices sorted in descending order
        sorted_row_indices = np.argsort(row_cnt, axis=-1)[..., ::-1]

        #compute workload of each pe in current tile
        tile_pe_cnt = np.sum(row_cnt, axis=-1)

        #max tile workload
        tile_max = np.max(tile_pe_cnt)
            
        if not mySpMV.row_dist_net:
            if mySpMV.pre_accumulator:
                return tile_max
            else:
                #do out-of-order scheduling but no intra_mode_rows
                return PreProcessor.get_out_of_order_size(row_cnt, sorted_row_indices, II_DIST, np.array([], dtype=np.int32))
        
        else:
            #Balance worklaods by processing denser rows in intra_row_mode
            intra_mode_rows, tile_size = PreProcessor.get_intra_mode_rows(row_cnt, sorted_row_indices, tile_pe_cnt, tile_max)
            if mySpMV.pre_accumulator:
                return tile_size
            else:
                #do out-of-order scheduling
                return PreProcessor.get_out_of_order_size(row_cnt, sorted_row_indices, II_DIST, intra_mode_rows)

    @staticmethod
    @njit
    def get_out_of_order_size(row_cnt: np.ndarray, sorted_row_indices: np.ndarray, dependency_distance: np.int32, intra_mode_rows: np.ndarray) -> np.int32:
        NUM_PES, rows_per_pe = row_cnt.shape
        workloads = np.zeros((NUM_PES, dependency_distance), dtype=np.int32)
        sorted_intra_mode_rows = np.sort(intra_mode_rows)
        len_intra_mode_rows = sorted_intra_mode_rows.shape[0]
        if len_intra_mode_rows != 0:
            for row_idx in intra_mode_rows:
                row_load = row_cnt[row_idx%NUM_PES][row_idx//NUM_PES]
                distributed_load = ((row_load - 1) // NUM_PES) + 1
                for pe in range(NUM_PES):
                    ii = np.argmin(workloads[pe])
                    workloads[pe][ii] += distributed_load 

        for i in range(NUM_PES):
            for j in range(rows_per_pe):
                row_idx = sorted_row_indices[i][j]
                value_to_check = row_idx * NUM_PES + i
                search_idx = np.searchsorted(sorted_intra_mode_rows, value_to_check)
                if search_idx < len_intra_mode_rows and sorted_intra_mode_rows[search_idx] == value_to_check:
                    continue
                row_load = row_cnt[i][row_idx]
                ii = np.argmin(workloads[i])
                workloads[i][ii] += row_load
        
        return np.max(workloads) * II_DIST

                
    @staticmethod
    def get_intra_mode_rows(row_cnt: np.ndarray, sorted_row_indices: np.ndarray, tile_pe_cnt: np.ndarray, tile_max: int):
        NUM_PES, _ = row_cnt.shape
        
        #sort pe indices based on their workload size
        sorted_tile_pe_indices = np.argsort(tile_pe_cnt)

        #before balancing current max is the best we can do, also no rows are processed in intra row mode
        best_load = tile_max
        intra_mode_rows = []
        for k in range(NUM_PES):
            baseline_pe_idx = sorted_tile_pe_indices[k]
            baseline_pe_load = tile_pe_cnt[baseline_pe_idx]

            curr_removed_rows = []
            extra_load = 0
            for l in range(k+1, NUM_PES):
                current_pe_idx = sorted_tile_pe_indices[l]
                current_pe_load = tile_pe_cnt[current_pe_idx]
                current_pe_removed_load = 0
                current_pe_removed_count = 0
                while(baseline_pe_load < (current_pe_load - current_pe_removed_load)):
                    curr_pe_largest_row_idx = sorted_row_indices[current_pe_idx][current_pe_removed_count]
                    curr_pe_largest_row_cnt = row_cnt[current_pe_idx][curr_pe_largest_row_idx]
                    current_pe_removed_load += curr_pe_largest_row_cnt
                    current_pe_removed_count += 1
                    extra_load += ((curr_pe_largest_row_cnt - 1)//NUM_PES) + 1
                    curr_removed_rows.append(curr_pe_largest_row_idx * NUM_PES + current_pe_idx)
            
            balanced_load = baseline_pe_load + extra_load
            if (balanced_load < best_load):
                best_load = balanced_load
                intra_mode_rows = curr_removed_rows

        improvement = 0 if tile_max== 0 else np.float32(tile_max - best_load) * 100.0 / np.float32(tile_max)
        # print(tile_max, best_load, improvement)
        if improvement < 10:
            intra_mode_rows = []
        return np.array(intra_mode_rows, dtype=np.int32), best_load
                


