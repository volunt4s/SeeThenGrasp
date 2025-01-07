# Get apriltag cube information using hand-eye camera
python preprocess/cube_1_get_cube_info_cam1.py

# Pick the cube
rosservice call /FR_Robot/look_obj
rosservice call /FR_Robot/remove_table
rosservice call /FR_Robot/pick_object
rosservice call /FR_Robot/look_obj

# Get apriltag cube information using external fixed camera
python preprocess/cube_2_get_cube_info_cam2.py

# Place the cube
rosservice call /FR_Robot/look_obj
rosservice call /FR_Robot/remove_table
rosservice call /FR_Robot/place_object
rosservice call /FR_Robot/look_obj

# Convert cube data to NeRF like camera pose data
python preprocess/cube_3_process_ext.py