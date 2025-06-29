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


测试步骤：
conda activate openpi && python /home/agilex/code/opensource/openpi/inference_piper/inference_agilexv2_openpi.py 
