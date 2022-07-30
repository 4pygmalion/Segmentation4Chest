# multiple_segmentation
Competion in Rayence Co

#### Requirement
- pydicom 2.3
- opencv-python
- tensorflow 2.x

#### Directory
```
JSRT
├─nodules
│  ├─dicom
│  └─png
├─non_nodules
│  ├─dicom
│  └─png
└─scr
    ├─landmarks
    ├─masks
    │  ├─heart
    │  ├─left clavicle
    │  ├─left lung
    │  ├─right clavicle
    │  └─right lung
    └─points
```

#### Image spec
- 2D gray scale
- normalized with [0, 255]
- [512, 512]


#### Source core
https://sakibreza.github.io/TransResUNet/main_architecture.html