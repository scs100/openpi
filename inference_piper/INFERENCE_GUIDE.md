inference_agilexv2_openpi  guide

terminal 1:
roscore

terminal 2:
roslaunch astra_camera list_devices.launch 
roslaunch astra_camera multi_camera.launch

terminal 3:
cd ~/cobot_magic/Piper_ros_private-ros-noetic/
bash can_multi_activate.sh
source devel/setup.bash
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=True


terminal 4:

conda activate openpi && \
python inference_piper/serve_trained_model.py  \
     --checkpoint_dir checkpoints/sgd_swap_manager/sgd_swap_manager_norm/16000  \
     --config_name sgd_swap_manager



terminal 5:
conda deactivate; deactivate;\
cd /home/agilex/code/opensource/openpi/inference_piper && \
python simple_data_service.py

terminal 6:

conda activate openpi && python /home/agilex/code/opensource/openpi/inference_piper/inference_agilexv2_openpi.py 
