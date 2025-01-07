# For stage 1
python Stage1.py --config recon_src/coarse_search/configs/custom/stage1.py
rosservice call /FR_Robot/remove_table
rosservice call /FR_Robot/pick_object

# For stage 2
python Stage2.py --config recon_src/coarse_search/configs/custom/stage2.py
rosservice call /FR_Robot/look_obj
rosservice call /FR_Robot/remove_table
rosservice call /FR_Robot/place_object
rosservice call /FR_Robot/look_obj

# Post process active learning data
python post_process_data.py

# Final fine training with active learning data
bash recon_src/Voxurf/single_runner.sh configs/custom_e2e exp STG_data