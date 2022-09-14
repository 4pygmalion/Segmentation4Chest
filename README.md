# multiple_segmentation
3rd Award in competion in Rayence Co

![KakaoTalk_20220914_225533471](https://user-images.githubusercontent.com/45510932/190189846-82345b98-84dd-4e2f-a506-003a6c70b25a.png)



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
