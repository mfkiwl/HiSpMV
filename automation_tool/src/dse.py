import numpy as np
import logging
from scipy.sparse import coo_matrix
from commons import SpMVConfig, FPGA, FPGAResource
from cyclecount_est import CycleCountEstimator
from resource_est import ResourceEstimator
import math

logger = logging.getLogger(__name__)

class DSE():
    @staticmethod 
    def allResourcesUnderLimit(resource: FPGAResource, fpga: FPGA) -> bool:
        bram = (((resource.bram + fpga.fixed.bram) / fpga.available.bram) <= fpga.limit.bram)
        uram = (((resource.uram + fpga.fixed.uram) / fpga.available.uram) <= fpga.limit.uram)
        dsp = (((resource.dsp + fpga.fixed.dsp) / fpga.available.dsp) <= fpga.limit.dsp)
        lut = (((resource.lut + fpga.fixed.lut) / fpga.available.lut) <= fpga.limit.lut)
        reg = (((resource.reg + fpga.fixed.reg) / fpga.available.reg) <= fpga.limit.reg)
        logger.debug(f"BRAM: {bram}")
        return bram and uram and dsp and lut and reg

    @staticmethod
    def getBestConfig(mtx_file: str, fpga: FPGA, dense_overlay: bool = False) -> SpMVConfig:
        matrix = DSE.mm_read(mtx_file)
        # total time = (nnz/NUM_CH_A/8) + (cols/NUM_CH_B/16) + (rows/NUM_CH_C/16)
        # optimising total time is (nnz/NUM_CH_A/8) = (cols/NUM_CH_B/16) = (rows/NUM_CH_C/16) (even if the ch_width varies the ratio holds)
        # after simplifying (2*nnz/NUM_CH_A) = (cols/NUM_CH_B) = (rows/NUM_CH_C)
        # we can express NUM_CH_B and NUM_CH_C in terms of NUM_CH_A 
        # NUM_CH_B = NUM_CH_A*cols/nnz/2
        # NUM_CH_C = NUM_CH_A*rows/nnz/2
        norm_cols = matrix.shape[1]/matrix.nnz/2
        norm_rows = matrix.shape[0]/matrix.nnz/2

        # NUM_CH_A + NUM_CH_B + 2*NUM_CH_C <= HBM_CH (substitute and solve for NUM_CH_A)
        opt_ch_A = fpga.hbm.num_ch / (1 + norm_cols + 2*norm_rows)
        opt_ch_B = opt_ch_A * norm_cols
        opt_ch_C = opt_ch_A * norm_rows

        logger.info(f"Optimal Channels: A={opt_ch_A}, B={opt_ch_B}, C={opt_ch_C}")

        ch_b_limit = max(0, math.ceil(math.log2(opt_ch_B)))
        ch_c_limit = max(0, math.ceil(math.log2(opt_ch_B)))
        # opt_ch_A = min(fpga.hbm.num_ch - opt_ch_B - 2*opt_ch_C, round(opt_ch_A))

        logger.info(f"Search Space Limits: B=(0, {1 << ch_b_limit}], C=(0, {1 << ch_c_limit}]")
        best_cycles = matrix.nnz + matrix.shape[0] + matrix.shape[1] # set to a number that is overly maximal
        best_config = None
        for c in [1 << i for i in range(ch_c_limit + 1)]:
            for b in [1 << i for i in range(ch_b_limit + 1)]:
                for row_dist_net in [False, True]:
                    for pre_accumulator in [False, True]:
                        a = fpga.hbm.num_ch - b - 2*c
                        if (a < 1):
                            continue

                        a = (a//(2*c))*(2*c)
            
                        while(a): 
                            urams_per_pe = 2
                            config = SpMVConfig(num_ch_A=a, 
                                                num_ch_B=b, 
                                                num_ch_C=c, 
                                                urams_per_pe=urams_per_pe, 
                                                ch_width=fpga.hbm.ch_width,
                                                dense_overlay=dense_overlay,
                                                pre_accumulator=pre_accumulator,
                                                row_dist_net=row_dist_net
                                                )
                            logger.debug(f"Config: {config}")
                            resource = ResourceEstimator.getDesignResource(config, fpga)
                            logger.debug(f"Resource: {resource}")
                            if DSE.allResourcesUnderLimit(resource, fpga):
                                cycles = CycleCountEstimator.getCC(config, matrix)
                                logger.debug(f"Total Cycles: {cycles}")
                                if cycles <= best_cycles:
                                    best_cycles = cycles
                                    best_config = config
                                break
                            else:
                                a -= (2*c)
                                logger.debug(f"A {a}")
                        
                        
        #Try to see if we can get to increase C channels
        max_c = (fpga.hbm.num_ch - best_config.num_ch_A - best_config.num_ch_B) // 2
        best_c = best_config.num_ch_C
        while(max_c > best_c and 2*max_c <= best_config.num_ch_A):
            if (best_config.num_ch_A % (2*max_c) == 0):
                config = best_config
                config.num_ch_C = max_c
                resource = ResourceEstimator.getDesignResource(config, fpga)
                logger.debug(f"Resource: {resource}")
                if DSE.allResourcesUnderLimit(resource, fpga):
                    best_config.num_ch_C = max_c
                    break
            max_c = 1 << ((max_c-1).bit_length() - 1)

        logger.info(f"Best Config: {best_config}")
        cycles = CycleCountEstimator.getCC(best_config, matrix)
        kernel_time = best_cycles / fpga.hbm.freq
        flops = 2*(matrix.nnz + matrix.shape[0]) / kernel_time
        logger.debug(f"Cycles: {best_cycles}, Kernel Time: {kernel_time:.3e}, flops: {flops:.2e}")
        return best_config
    
    @staticmethod
    def getSingleBestConfig(fpga: FPGA, dense_overlay: bool = False) -> SpMVConfig:
        best_config = None
        # Assume the matrix is dense and use only 1 channel for input and output vectors
        b = 1
        c = 1
        a = fpga.hbm.num_ch - b - 2*c
        assert a >= 2, "Expected to have atleast 2 channels for sparse matrix A"

        a = (a//2)*2
        while(a):
            config = SpMVConfig(num_ch_A=a, 
                                num_ch_B=b, 
                                num_ch_C=c, 
                                urams_per_pe=2, 
                                ch_width=fpga.hbm.ch_width,
                                dense_overlay=dense_overlay,
                                pre_accumulator=False,
                                row_dist_net=True
                                )
            logger.debug(f"Config: {config}")
            resource = ResourceEstimator.getDesignResource(config, fpga)
            logger.debug(f"Resource: {resource}")
            if DSE.allResourcesUnderLimit(resource, fpga):
                best_config = config
                break
            else:
                a -= 2
                logger.debug(f"A {a}")

        #Try to see if we can get to increase C channels
        max_c = (fpga.hbm.num_ch - best_config.num_ch_A - best_config.num_ch_B) // 2
        best_c = best_config.num_ch_C
        while(max_c > best_c and 2*max_c <= best_config.num_ch_A):
            if (best_config.num_ch_A % (2*max_c) == 0):
                config = best_config
                config.num_ch_C = max_c
                resource = ResourceEstimator.getDesignResource(config, fpga)
                logger.debug(f"Resource: {resource}")
                if DSE.allResourcesUnderLimit(resource, fpga):
                    best_config.num_ch_C = max_c
                    break
            max_c = 1 << ((max_c-1).bit_length() - 1)

        logger.info(f"Best Config: {best_config}")
        return best_config
    
    @staticmethod
    def mm_read(mtx_file: str) -> coo_matrix:
        logger.info(f"Reading {mtx_file} file")
        with open(mtx_file, 'r') as f:
            # Read the header line
            header = f.readline().strip().split()
            
            # Validate file format and ensure it is in coordinate format
            if header[0] != "%%MatrixMarket" or header[1] != "matrix":
                raise ValueError("Not a valid Matrix Market file.")
            
            matrix_format = header[2]  # Should be 'coordinate' for sparse matrices
            data_type = header[3]      # 'real' or 'integer'
            symmetry = header[4]       # 'general' or 'symmetric'

            logger.info(f"Matrix Metadata: Format = {matrix_format}, data type = {data_type}, symmetry = {symmetry}")

            if matrix_format != "coordinate":
                raise ValueError("This function only supports 'coordinate' format for sparse matrices.")
        
            if data_type not in ["real", "integer", "pattern"]:
                raise ValueError("Unsupported data type. Only 'real','integer' and 'pattern' types are supported.")

            # Skip comments
            while True:
                line = f.readline().strip()
                if line[0] != '%':
                    break

            # Parse matrix dimensions and non-zero count
            rows, cols, nnz = map(int, line.split())
            # Preallocate arrays for the COO format
            if symmetry == 'symmetric':
                # Allocate twice the space for symmetric entries (approximation)
                row = np.empty(2 * nnz, dtype=np.int32)
                col = np.empty(2 * nnz, dtype=np.int32)
                data = np.empty(2 * nnz, dtype=np.float32)
            else:
                row = np.empty(nnz, dtype=np.int32)
                col = np.empty(nnz, dtype=np.int32)
                data = np.empty(nnz, dtype=np.float32)

            # Read entries for the sparse matrix
            count = 0
            for line in f:
                entries = line.strip().split()
                r = int(entries[0]) - 1  # Convert to 0-based index
                c = int(entries[1]) - 1

                if data_type=="pattern":
                    value = float(1.0)
                else:
                    value = float(entries[2])  # Convert values to float

                #skip if the entry is 0
                if np.float32(value) == np.float32(0):
                    continue

                # Add entry (r, c)
                row[count] = r
                col[count] = c
                data[count] = np.float32(value)  # Store as float32
                count += 1

                # If symmetric and not on the diagonal, add (c, r) as well
                if symmetry == 'symmetric' and r != c:
                    row[count] = c
                    col[count] = r
                    data[count] = np.float32(value)
                    count += 1

            # Create the COO matrix and convert to CSR format
            coo = coo_matrix((data[:count], (row[:count], col[:count])), shape=(rows, cols))
            logger.info(f"Matrix Properties: Rows={rows}, Cols={cols}, NNZ={coo.nnz}")
            return coo
    