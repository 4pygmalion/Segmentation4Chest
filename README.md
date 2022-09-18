# multiple_segmentation
3rd Award in competion in Rayence Co.
Collaborated with Jung hoon Lee. JunSik Park.

![award](./images/competition.png)


#### Requirement
- pydicom 2.3
- opencv-python
- tensorflow 2.x

#### Directory
```
├─data
│  ├─JSRT
│  │  ├─nodules
│  │  │  ├─dicom
│  │  │  └─png
│  │  ├─non_nodules
│  │  │  ├─dicom
│  │  │  └─png
│  │  └─scr
│  │      ├─landmarks
│  │      ├─masks
│  │      │  ├─heart
│  │      │  ├─heart_resized
│  │      │  ├─left clavicle
│  │      │  ├─left lung
│  │      │  ├─right clavicle
│  │      │  └─right lung
│  │      └─points
│  ├─test
│  │  ├─img
│  │  └─mask
│  └─train
│      ├─img
│      └─mask
├─augmentation.py 
├─block.py
├─data_prep.
├─main.ipnby
└─build_model.py

```

#### Image spec
- 2D gray scale
- normalized with [0, 255]
- [512, 512]


#### Source core
https://sakibreza.github.io/TransResUNet/main_architecture.html
