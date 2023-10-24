
export OMP_NUM_THREADS=1
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Kidney_L/VH_F_outer_cortex_of_kidney_L.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Kidney_L/VH_F_renal_pyramid_L_h.off 0.20;

export OMP_NUM_THREADS=4
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Kidney_L/VH_F_outer_cortex_of_kidney_L.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Kidney_L/VH_F_renal_pyramid_L_h.off 0.20;

#export OMP_NUM_THREADS=4
#./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Heart/VH_F_left_cardiac_atrium.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/VH_F_Heart/VH_F_interventricular_septum.off 0.20;



