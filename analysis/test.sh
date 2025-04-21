# data=/pscratch/xli281_uksr/shared/jhtdb_pressure_512x512x512/5000_pressure.f32
dims="256 384 384"
# data=/pscratch/xli281_uksr/pjiao/isotropic_bin/vx_4x4x4/vx_3_2_1.f32
# dims="1024 1024 1024"
thread_list=( 1 2 4 8 16 32 64 )
data="/anvil/projects/x-cis240192/pjiao/data/Miranda-256x384x384/velocityx.f32"
for i in "${thread_list[@]}"
do
    echo "Running test with $i threads"
../build/test/test_quantize_and_edt -N 3 \
    -d $dims  -i $data \
            -m rel -e 0.001 -q q.c -c c.f32 -t $i --no_ssim 1 
done