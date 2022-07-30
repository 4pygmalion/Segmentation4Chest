# multiple_segmentation
Competion in Rayence Co

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
├─logs
└─models
```

#### Image spec
- 2D gray scale
- normalized with [0, 255]
- [512, 512]


#### Source core
https://sakibreza.github.io/TransResUNet/main_architecture.html
