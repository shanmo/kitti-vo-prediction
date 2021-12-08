## about 

- this repo contains a visual odometry to provide prediction step for KITTI dataset, similar to [libviso](https://github.com/seanbow/object_pose_detection/tree/master/viso_pose)

## dependencies 

- `conda install -c anaconda pyopengl`
- `conda install pybind11` 
- [pangolin](https://github.com/uoip/pangolin)

## demo 

- folder structure in `KITTI odometry 00` should be 
```
calib.txt  image_0  image_1  image_2  image_3  times.txt
```

## references 

- https://github.com/uoip/stereo_ptam