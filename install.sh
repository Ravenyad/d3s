#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1

echo ""
echo ""
echo "****************** Installing pytorch with cuda9 ******************"
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch 

echo ""
echo ""
echo "****************** Installing matplotlib 2.2.2 ******************"
conda install -y matplotlib=2.2.2

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboardX ******************"
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing MTCNN Detector ******************"
pip install mtcnn 

echo ""
echo ""
echo "****************** Installing VGGFace Face Recognizer ******************"
pip install Keras-Applications
pip install keras-vggface

echo ""
echo ""
echo "****************** Installing Tensorflow ******************"
pip install tensorflow 

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
sudo apt-get install libturbojpeg
# while true; do
#     read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
#     case $install_flag in
#         [Yy]* ) sudo apt-get install libturbojpeg; break;;
#         [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
#         * ) echo "Please answer y or n  ";;
#     esac
# done

echo ""
echo ""
echo "****************** Installation complete! ******************"
