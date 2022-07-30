data preprocessing 

#### Example
```
# dicom files to png with resizing(=rescaling)
$ python3 .\dicom2png.py -s .\non_nodules\dicom\ -d .\non_nodules\png -r 512,512
```

#### Requirement
- pydicom 2.3
- opencv-python

#### Directory
```
multiple_segmentation
├─data
│  ├─nodules
│  │  ├─dicom
│  │  └─png
│  ├─non_nodules
│  │  ├─dicom
│  │  └─png
│  └─scr
│      ├─landmarks
│      ├─masks
│      │  ├─heart
│      │  ├─left clavicle
│      │  ├─left lung
│      │  ├─right clavicle
│      │  └─right lung
│      └─points
```

src 폴더를보시면 되구요. Multisegmentation용도입니다.
파일명뒤에 붙는 surfix가 N이면 Nodule, NN이면 NonNodule 같습니다.

#### Image spec
- 2D gray scale
- normalized with [0, 255]