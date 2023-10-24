#M_Intestine_Large--------------------------------------------------------------------

export OMP_NUM_THREADS=1
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=2
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=4
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=8
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=12
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=16
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=20
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;

export OMP_NUM_THREADS=24
./build/corridor_generator 0.0 0.0 0.0 0.005 0.005 0.005 0.500 0.500 /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_ascending_colon.off /home/ubuntu/Desktop/project/data/model/plain_manifold_filling_hole_v1.3/SBU_M_Intestine_Large/VH_M_caecum.off 0.20;
