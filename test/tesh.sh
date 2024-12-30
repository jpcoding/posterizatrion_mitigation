./test_3d 2 200 134 /scratch/pji228/useful/direct_quantize/miranda/local_data_200x150.bin /scratch/pji228/useful/direct_quantize/miranda/local_dec_200x150.bin /scratch/pji228/useful/direct_quantize/debug/local_q_index_200x150.bin


./test_3d 3 256 384 384 $DATA/miranda/velocityx.f32 /scratch/pji228/useful/direct_quantize/miranda/velocityx.f320.001.out /scratch/pji228/useful/direct_quantize/miranda/velocityx.f320.001.quant.i32





./test_3d 3 1008 1008 352 $DATA/aramco_1008x1008x352/pressure_2000.f32.dat  pressure_2000.f32.dat.out pressure_2000.f32.dat.quant.i32