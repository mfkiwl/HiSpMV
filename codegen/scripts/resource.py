import math
from commons import SpMVConfig, FPGAResource

class ResourceEstimator:
    @staticmethod
    def getEstimateFromConfig(config, fpga):
        myResource = fpga.fixed

        NUM_PES = config.num_ch_A * config.ch_width // 64
        FIFO_DEPTH = 2
        
        #Resource of tapa::tasks
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.MM2S_A(), config.num_ch_A)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.MM2S_B(), config.num_ch_B)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.MM2S_C(), config.num_ch_C)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.S2MM_C(), config.num_ch_C)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.LoadB(), config.num_ch_A)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.ComputeAB(config), NUM_PES//2)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.PreAccumulator(config), NUM_PES//2)
        
        if config.row_dist_net:
            myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.ADD_Blocks(), NUM_PES)  
            myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.SW_Blocks(), 2*(NUM_PES - 3))
        
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.AccumBuffer(config), NUM_PES)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.Arbitter_C(), 1)
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.Compute_C(config), config.num_ch_C)

        #Resource of tapa::buffers 
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.BUFF_B(config), NUM_PES//2)
        
        #Resource of tapa::streams
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(64, FIFO_DEPTH, NUM_PES)) #a_in
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(16, FIFO_DEPTH, NUM_PES)) #c_row
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(32, FIFO_DEPTH, NUM_PES)) #c_val
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(3, FIFO_DEPTH, NUM_PES//2)) #c_flag
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(config.ch_width, FIFO_DEPTH, config.num_ch_A + 1)) #b_in
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(60, FIFO_DEPTH, NUM_PES//2)) #noc_in
        if config.row_dist_net:
            myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(16, FIFO_DEPTH, NUM_PES)) #noc_out
            crossbar_streams_depth = (math.log2(NUM_PES) - 1) * 10
            myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(60, crossbar_streams_depth, NUM_PES)) #noc_out
        
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(32, FIFO_DEPTH, NUM_PES)) #c_arb
        myResource = ResourceEstimator.add(myResource, ResourceEstimator.Streams(config.ch_width, FIFO_DEPTH, config.num_ch_C*3)) #c_ab, c_in, c_out

        #Async MMAP
        myResource = ResourceEstimator.scale_and_accumulate(myResource, ResourceEstimator.Async_MMAP(config), config.num_ch_A + config.num_ch_B + 2*config.num_ch_C)

        return myResource




    @staticmethod
    def scale_and_accumulate(bias, rsrc, scalar):
        result = ResourceEstimator.add(bias, ResourceEstimator.multiply(rsrc, scalar))
        return result
    
    @staticmethod
    def multiply(resource, scalar):
        return FPGAResource(
            bram = resource.bram * scalar,
            uram = resource.uram * scalar,
            dsp = resource.dsp * scalar,
            lut = resource.lut * scalar,
            reg = resource.reg * scalar
        )
    
    @staticmethod
    def add(rsrc1, rsrc2):
        return FPGAResource(
            bram = rsrc1.bram + rsrc2.bram,
            uram = rsrc1.uram + rsrc2.uram,
            dsp =  rsrc1.dsp + rsrc2.dsp,
            lut =  rsrc1.lut + rsrc2.lut,
            reg =  rsrc1.reg  + rsrc2.reg
        )

    @staticmethod
    def MM2S_A():
        return FPGAResource(bram=0, uram=0, dsp=0, lut=98, reg=87)

    @staticmethod
    def MM2S_B():
        return FPGAResource(bram=0, uram=0, dsp=1, lut=59, reg=103)
    
    @staticmethod
    def MM2S_C():
        return FPGAResource(bram=0, uram=0, dsp=0, lut=56, reg=139)
    
    @staticmethod
    def S2MM_C():
        return FPGAResource(bram=0, uram=0, dsp=0, lut=66, reg=143)
    
    @staticmethod
    def LoadB():
        return FPGAResource(bram=0, uram=0, dsp=0, lut=240, reg=245)
    
    @staticmethod
    def ComputeAB(config):
        if config.dense_overlay:
            return FPGAResource(bram=0, uram=0, dsp=16, lut=1410, reg=1740)
        else:
            return FPGAResource(bram=0, uram=0, dsp=6, lut=740, reg=740)
    
    @staticmethod
    def PreAccumulator(config):
        if config.pre_accumulator:
            return FPGAResource(bram=0, uram=0, dsp=16, lut=3091, reg=3752)
        else:
            return FPGAResource(bram=0, uram=0, dsp=0, lut=29, reg=125)

        
    @staticmethod    
    def AccumBuffer(config):
        if config.pre_accumulator:
            return FPGAResource(bram=0, uram=config.urams_per_pe, dsp=3, lut=869, reg=686)
        else:
            return FPGAResource(bram=0, uram=config.urams_per_pe, dsp=5, lut=717, reg=751)
    
    @staticmethod
    def Compute_C(config):
        fp32perch = config.ch_width // 32
        return FPGAResource(bram=0, uram=0, dsp=8*fp32perch + 2, lut=414*fp32perch+75, reg=587*fp32perch+166)
    
    @staticmethod
    def Arbitter_C():
        return FPGAResource(bram=0, uram=0, dsp=2, lut=1000, reg=1000)
    
    @staticmethod
    def ADD_Blocks():
        return FPGAResource(bram=0, uram=0, dsp=2, lut=408, reg=488)
    
    @staticmethod
    def SW_Blocks():
        return FPGAResource(bram=0, uram=0, dsp=0, lut=82, reg=129)
    
    @staticmethod
    def BUFF_B(config):
        fp32perch = config.ch_width // 32
        num_part = config.num_ch_B * fp32perch // 2
        return FPGAResource(bram=num_part, uram=0, dsp=0, lut=0, reg=0)
    
    @staticmethod
    def Streams(width, depth, channels):
        lut_ram = width * ((depth - 1) / 16 + 1)
        lut_logic = 15 + 3 * int(math.log2(depth))
        ff = 7 + 3 * int(math.log2(depth))
        return ResourceEstimator.multiply(FPGAResource(bram=0, uram=0, dsp=0, lut=int(lut_logic+lut_ram), reg=ff), channels)

    @staticmethod
    def Async_MMAP(config):
        res = FPGAResource(bram=0, uram=0, dsp=0, lut=5000, reg=6500)
        return ResourceEstimator.add(res, FPGAResource(bram=0, uram=0, dsp=0, lut=config.ch_width*2+713, reg=370))
        
