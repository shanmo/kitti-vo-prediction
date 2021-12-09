## about 

- this repo contains a visual odometry to provide prediction step for KITTI dataset, similar to [libviso](https://github.com/seanbow/object_pose_detection/tree/master/viso_pose)

## dependencies 

- OpenCV
- [pangolin](https://github.com/uoip/pangolin)
    - `conda install -c anaconda pyopengl`
    - `conda install pybind11` 
    - need to use [this setup.py](https://github.com/shanmo/kitti-vo-prediction/issues/1)

## demo 

- folder structure in `KITTI odometry 04` should be 
```
calib.txt  image_0  image_1  image_2  image_3  times.txt
```
- run `python sptam.py` 

## references 

- https://github.com/uoip/stereo_ptam