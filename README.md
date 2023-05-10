## Video Object Detection~ A pilot study

Contributors
* <strong>[@brillard1](https://github.com/brillard1)</strong>
* <strong>[@ceyxasm](https://github.com/ceyxasm)</strong>
---
This repository has been established to analyze and compare video object detection models. It explores design choices, differences, and performance while addressing challenges specific to video object detection. We reviewed and implemented [these](./papers/) papers.

#### Youtube Video, Presentation & Report
* Please find the final report [here FILL]()
* The ppt used in the video can be found [here FILL]()
* Demo video

[![FILL](./assets/image.png)]() 
---

#### Execution
To run the implementations
```
git clone https://github.com/ceyxasm/VOD-pilot-study.git
cd ./VOD-pilot-study
```

---
#### Dataset used

We used [ImageNet-VidVRD Video Visual Relation Dataset](https://xdshang.github.io/docs/imagenet-vidvrd.html)

![dataset](./assets/image2.png)

It contains 1,000 videos from ILVSRC2016-VID, split into 800 training and 200 test videos. The dataset covers 35 subject/object categories and 132 predicate categories. Labeled by ten individuals, it includes object trajectory and relation annotations.

#### Literature review references
* [Mask R-CNN ](https://arxiv.org/pdf/1703.06870.pdf)
* [Temporal ROI Align- for Video Object Recognition](https://arxiv.org/pdf/2109.03495v2.pdf)
* [RetinaNet- Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf)
* [BoxMask](https://arxiv.org/pdf/2210.06008v1.pdf) 
* [YOLOV: Making Still Image Object Detectors Great at Video Object Detection](https://arxiv.org/pdf/2208.09686v2.pdf)
* [TransVOD](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9960850&tag=1)
* [VSTAM- Video Sparse Transformer With Attention-Guided Memory for Video Object Detection](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9798833)

#### Citations
```
@inproceedings{shang2017video,
    author={Shang, Xindi and Ren, Tongwei and Guo, Jingfan and Zhang, Hanwang and Chua, Tat-Seng},
    title={Video Visual Relation Detection},
    booktitle={ACM International Conference on Multimedia},
    address={Mountain View, CA USA},
    month={October},
    year={2017}
}
```
