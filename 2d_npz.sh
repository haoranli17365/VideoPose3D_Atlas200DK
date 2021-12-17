#!/bin/sh
# Asking for sudo
if [[ $UID != 0 ]]; then
    echo "Please run this script with sudo:"
    echo "sudo $0 $*"
    exit 1
fi
# check if required folder is created.
if [! -d "./npz_ouput"]; then 
    echo "Creating required folder..."
    mkdir ./npz_ouput
fi
# running detectron2 for inferencing and crating dataset for 3d lifting.
# $1 is the input file extension type, $2 is the input directory.
python3 ./inference/infer_video_d2.py --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml --output-dir ./npz_output --image-ext $1 $2
# creating custom dataset.
cd ./data
python3 ./prepare_data_2d_custom.py -i ../npz_output/ -o myvideos


