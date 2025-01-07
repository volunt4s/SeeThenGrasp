root=$1

python3 colmap_poses/pose_utils.py --source_dir $root
python3 convert_cameras.py --source_dir $root
python3 preprocess_cameras.py --source_dir $root