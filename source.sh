export GUROBI_HOME="/localhdd/mba151/gurobi1001/linux64"
export GRB_LICENSE_FILE="/localhdd/mba151/gurobi1001/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"

source /mnt/glusterfs/hacc-common/shell-env/shsetup/setuprc.sh
load_vitis23
source /mnt/glusterfs/users/arb26/workspace/sb/setup

export platform=xilinx_u280_gen3x16_xdma_1_202211_1