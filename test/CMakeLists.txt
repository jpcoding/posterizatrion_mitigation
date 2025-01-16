add_executable(test_3d test_3d.cpp)
add_executable(test_2d test_2d.cpp)
add_executable(direct_quantize direct_quantize.cpp)
add_executable(test_3d_idw test_3d_idw.cpp)
target_include_directories(test_3d_idw PUBLIC "/scratch/pji228/gittmp/alglib/install/include")
target_link_directories(test_3d_idw PUBLIC "/scratch/pji228/gittmp/alglib/install/lib64")
target_link_libraries(test_3d_idw PUBLIC alglib )

add_executable(test_3d_rbf test_3d_rbf.cpp)
target_include_directories(test_3d_rbf PUBLIC "/scratch/pji228/gittmp/alglib/install/include")
target_link_directories(test_3d_rbf PUBLIC "/scratch/pji228/gittmp/alglib/install/lib64")
target_link_libraries(test_3d_rbf PUBLIC alglib )

add_executable(test_quantize_and_edt test_quantize_and_edt.cpp)
