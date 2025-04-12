./test_3d 2 200 134 /scratch/pji228/useful/direct_quantize/miranda/local_data_200x150.bin \
    /scratch/pji228/useful/direct_quantize/miranda/local_dec_200x150.bin \
    /scratch/pji228/useful/direct_quantize/debug/local_q_index_200x150.bin


./test_3d 3 256 384 384 $DATA/miranda/velocityx.f32 \
    /scratch/pji228/useful/direct_quantize/miranda/velocityx.f320.001.out \
    /scratch/pji228/useful/direct_quantize/miranda/velocityx.f320.001.quant.i32


./test_3d 3 1008 1008 352 $DATA/aramco_1008x1008x352/pressure_2000.f32.dat \
     pressure_2000.f32.dat.out \
     pressure_2000.f32.dat.quant.i32


./test/test_quantize_and_edt 2 200 134 \
    /scratch/pji228/useful/direct_quantize/miranda/local_data_200x150.bin \
    0.0272382009  \
    local_dec_200x150.bin \
    local_data_200x150.bin_compensated_data.f32 

./test/test_quantize_and_edt 3 256 384 384 \
    $DATA/miranda/velocityx.f32 \
    0.001  \
    velocityx.f32.dec  \
    velocityx.f32.dec.compensated_data.f32 16

./test/test_3d_as_slice 3 256 384 384 \
    $DATA/miranda/velocityx.f32 \
    0.001  \
    velocityx.f32.dec  \
    velocityx.f32.dec.compensated_data.f32

./test/test_3d_slices 3 256 384 384 \
    $DATA/miranda/density.f32 \
    0.001  \
    density.f32.dec  \
    density.f32.dec.compensated_data.f32

./test/test_quantize_and_edt 3 100 500 500  \
    $DATA/hurricane_100x500x500/CLOUDf48.bin.f32 \
    0.001  \
    CLOUDf48.f32.dec  \
    CLOUDf48.f32.dec.compensated_data.f32


./test/test_quantize_and_edt 3 100 500 500  \
    $DATA/hurricane_100x500x500/Wf48.bin.f32 \
    0.001  \
    Wf48.f32.dec  \
    Wf48.f32.dec.compensated_data.f32 8 


mpirun -n 8 ./test_mpi/test_compensation_parallel --mpidims 2 2 2 --rel_eb 0.001 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2/ \
    --prefix vx --q_sufix q.32 --c_sufix c.32 --origdims 256 384 384 \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2/

mpirun -n 64 ./test_mpi/test_boundary_mpi --mpidims 4 4 4  --rel_eb 0.001 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --prefix vx --origdims 256 384 384 \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/

mpirun -n 8 ./test_mpi/test_boundary_mpi --mpidims 2 2 2   --rel_eb 0.001 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2/ \
    --prefix vx --origdims 256 384 384 \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2/


#1 
mpirun -n 64 ./test_mpi/test_compensation_parallel --mpidims 4 4 4  --rel_eb 0.001 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --prefix vx --q_sufix q.32 --c_sufix c.32 --origdims 256 384 384 \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/

mpirun -n 64  ./test_mpi/test_merge_file  --mpidims 4 4 4 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --prefix vx --sufix c.32 --orig_dims 256 384 384 --out block_compensated.nocommunication.f32
#2 
mpirun -n 64 ./test_mpi/test_compensation_parallel --mpidims 4 4 4  --rel_eb 0.001 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --prefix vx --q_suffix q.32 --c_suffix c.32 --origdims 256 384 384 \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/


mpirun -n 64  ./test_mpi/test_merge_file  --mpidims 4 4 4 \
    --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --prefix vx --sufix .post3d.f32 --orig_dims 256 384 384 --out block_compensated.local.f32


mpirun -n 64 ./test_mpi/test_compensation_opt --mpidims 4 4 4  \
     --rel_eb 0.001     --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/   \
      --prefix vx --origdims 256 384 384     \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_4x4x4/ \
    --local_edt 1 --local_quant 0 

mpirun -n 8 ./test_mpi/test_compensation_opt --mpidims 2 2 2  \
     --rel_eb 0.001     --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2   \
      --prefix vx --origdims 256 384 384     \
    --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_2x2x2/ \
    --local_edt 1 --local_quant 0 

mpirun -n 1 ./test_mpi/test_boundary_mpi  --mpidims 1 1 1   \
    --rel_eb 0.001     --dir /scratch/pji228/useful/direct_quantize/mpi/blocks_1x1x1       \
  --prefix vx --origdims 256 384 384       \
   --use_rbf 0 --outdir /scratch/pji228/useful/direct_quantize/mpi/blocks_1x1x1/