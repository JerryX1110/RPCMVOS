# Reliable Propagation-Correction Modulation for Video Object Segmentation (AAAI22 Oral)

![Picture1](https://user-images.githubusercontent.com/65257938/145016835-3c4be820-c55d-4eb4-b7f5-b8a012ee0f8c.png)

Preview version paper of this work is available at [Arxiv](https://arxiv.org/abs/2112.02853) 

AAAI [long paper presentation ppt](https://docs.google.com/presentation/d/1szmHc-2s1RpEfxr5cCbIRys85rpF85CR/edit?usp=sharing&ouid=113055027221294032292&rtpof=true&sd=true), [short one-minute paper presentation ppt](https://docs.google.com/presentation/d/1mv5xCWJQ0G5nVsrdQ5mY78RTb8HYlVMT/edit?usp=sharing&ouid=113055027221294032292&rtpof=true&sd=true), and the [poster](https://drive.google.com/file/d/1xKf0MvxxTqgbDGCBOkRSKdAuix0mVGOz/view?usp=sharing) are avavilable!

Qualitative results and comparisons with previous SOTAs are available at both [YouTube](https://youtu.be/X6BsS3t3wnc) and [Bilibili](https://www.bilibili.com/video/BV1pr4y1D7TQ?spm_id_from=333.999.0.0). 

[Thanks to someone (I don't know) who transports the video to bilibili😀.]

**This repo is a preview version. More details will be added later. Welcome to starts ⭐ & comments 💹 & collaboration 😀 !!**

```diff
- 2022.7.9: Our code is re-released now! 
- 2022.3.9: Dockerfile is added for easy env setup and modification.
- 2022.3.6: Our presentation PPT and Poster for AAAI22 are available now on GoogleDrive!
- 2022.2.16 😀:  Our paper has been selected as **Oral Presentation** in AAAI22! (Oral Acceptance Rate is about 4.5% this year (15% x 30%))
- 2021.12.25 🎅🎄: Precomputed Results on YouTube-VOS18/19 and DAVIS17 Val/Test-dev are available on both GoogleDrive and BaiduDisk! 
- 2021.12.14: Due to some policies in the company, the previewed-version code without checking has to be withdrawn now. Stay tuned and it will be released again after review!
```
---



## Abstract
**Error propagation** is a general but crucial problem in **online semi-supervised video object segmentation**. We aim to **suppress error propagation through a correction mechanism with high reliability**. 

The key insight is **to disentangle the correction from the conventional mask propagation process with reliable cues**. 

We **introduce two modulators, propagation and correction modulators,** to separately perform channel-wise re-calibration on the target frame embeddings according to local temporal correlations and reliable references respectively. Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 

Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We **augment the reference cues by supplementing reliable feature patches to a maintained pool**, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames. 

Our model achieves **state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test** benchmarks. Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

## Requirements
* Python3
* pytorch >= 1.4.0 
* torchvision
* opencv-python
* Pillow

You can also use the docker image below to set up your env directly. However, this docker image may contain some redundent packages.

```latex
docker image: xxiaoh/vos:10.1-cudnn7-torch1.4_v3
```

A more light-weight version can be created by modified the [Dockerfile](https://github.com/JerryX1110/RPCMVOS/blob/main/Dockerfile) provided.

## Preparation
* Datasets

    * **YouTube-VOS**

        A commonly-used large-scale VOS dataset.

        [datasets/YTB/2019](datasets/YTB/2019): version 2019, download [link](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing). `train` is required for training. `valid` (6fps) and `valid_all_frames` (30fps, optional) are used for evaluation.

        [datasets/YTB/2018](datasets/YTB/2018): version 2018, download [link](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f?usp=sharing). Only `valid` (6fps) and `valid_all_frames` (30fps, optional) are required for this project and used for evaluation.

    * **DAVIS**

        A commonly-used small-scale VOS dataset.

        [datasets/DAVIS](datasets/DAVIS): [TrainVal](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) (480p) contains both the training and validation split. [Test-Dev](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip) (480p) contains the Test-dev split. The [full-resolution version](https://davischallenge.org/davis2017/code.html) is also supported for training and evaluation but not required.

* pretrained weights for the backbone

  [resnet101-deeplabv3p](https://drive.google.com/file/d/101jYpeGGG58Kywk03331PKKzv1wN5DL0/view?usp=sharing)


## Training
Training for YouTube-VOS:

    sh ../scripts/ytb_train.sh

* Notice that the some training parameters need to be changed according to your hardware environment, such as the interval to save a checkpoint.
* More details will be added soon.

## Inference
Using **r**eliable object **p**roxy **a**ugmentation (RPA)

    sh ../scripts/ytb_eval_with_RPA.sh
    
Without using **r**eliable object **p**roxy **a**ugmentation (RPA):

    sh ../scripts/ytb_eval_without_RPA.sh

* For evaluation, please use official YouTube-VOS servers ([2018 server](https://competitions.codalab.org/competitions/19544) and [2019 server](https://competitions.codalab.org/competitions/20127)), official [DAVIS toolkit](https://github.com/davisvideochallenge/davis-2017) (for Val), and official [DAVIS server](https://competitions.codalab.org/competitions/20516#learn_the_details) (for Test-dev).

* More details will be added soon.

## Precomputed Results

Precomputed results on both YouTube-VOS18/19 and DAVIS17 Val/Test-dev are available on [Google Drive](https://drive.google.com/drive/folders/1RaffnMvmQF4Nct30UBXqwrfOXTZ8rvQf?usp=sharing) and [Baidu Disk](https://pan.baidu.com/s/1WqB-SsbT7W-a6DbLIz8Lzw) (BaiduDisk password:6666).

## Limitation & Directions for further exploration in VOS!

Although the numbers on some semi-VOS benchmarks are somehow extremely high, many problems still remain for further exploration.

I think those who take a look at this repo are likely to be researching in the field related to segmentation or tracking. 

So I would like to share some directions to explore in VOS from my point of view here. Hopefully, I can see some nice solutions in the near future!

* What about leveraging the propagation-then-correction mechanism in other tracking tasks such as MOT and pose tracking?
* How about using a learning-based method to measure the prediction uncertainty?
* How to tackle VOS in long-term videos? Maybe due to lack of a good dataset for long-term VOS evaluation, this problem is still a hard nut to crack.
* How to update the memory pool containing historical infomation during propagation? 
* How to judge whether some information is useful for futher frames or not?
* Will some data augmentations used in training lead to some bias in final prediction?

(to be continued...)

## Citation
If you find this work is useful for your research, please consider giving us a star 🌟 and citing it by the following BibTeX entry.:

 ```latex
@inproceedings{xu2022reliable,
  title={Reliable propagation-correction modulation for video object segmentation},
  author={Xu, Xiaohao and Wang, Jinglu and Li, Xiao and Lu, Yan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={3},
  pages={2946--2954},
  year={2022}
}
```

if you find the implementations helpful, please consider to cite:

 ```latex
@misc{xu2022RPCMVOS,
  title={RPCMVOS-REPO},
  author={Xiaohao, Xu},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/JerryX1110/RPCMVOS/}},
  year={2022}
}
```


    
## Credit

**CFBI**: <https://github.com/z-x-yang/CFBI>

**Deeplab**: <https://github.com/VainF/DeepLabV3Plus-Pytorch>

**GCT**: <https://github.com/z-x-yang/GCT>

## Related Works in VOS
**Semisupervised video object segmentation repo/paper link:**
**ARKitTrack [CVPR 2023]**:<https://arxiv.org/pdf/2303.13885.pdf>

**MobileVOS [CVPR 2023]**:<https://arxiv.org/pdf/2303.07815.pdf>

**Two-ShotVOS [CVPR 2023]**:<https://arxiv.org/pdf/2303.12078.pdf>

**UNINEXT [CVPR 2023]**:<https://github.com/MasterBin-IIAU/UNINEXT>

**ISVOS [CVPR 2023]**:<https://arxiv.org/pdf/2212.06826.pdf>

**TarVis [Arxiv 2023]**:<https://arxiv.org/pdf/2301.02657.pdf>

**LBLVOS [AAAI 2023]**:<https://arxiv.org/pdf/2212.02112.pdf>

**DeAOT[NeurIPS 2022]**:<https://arxiv.org/pdf/2210.09782.pdf>

**RobustVOS [ACM MM 2022]**:<https://github.com/JerryX1110/Robust-Video-Object-Segmentation>

**BATMAN [ECCV 2022 Oral]**:<https://arxiv.org/pdf/2208.01159.pdf>

**TBD [ECCV 2022]**:<https://github.com/suhwan-cho/TBD>

**XMEM [ECCV 2022]**:<https://github.com/hkchengrex/XMem>

**QDMN [ECCV 2022]**:<https://github.com/workforai/QDMN>

**GSFM [ECCV 2022]**:<https://github.com/workforai/GSFM>

**SWEM [CVPR 2022]**:<https://tianyu-yang.com/resources/swem.pdf>

**RDE [CVPR 2022]**:<https://arxiv.org/pdf/2205.03761.pdf>

**COVOS [CVPR 2022]** :<https://github.com/kai422/CoVOS>

**AOT [NeurIPS 2021]**: <https://github.com/z-x-yang/AOT>

**STCN [NeurIPS 2021]**: <https://github.com/hkchengrex/STCN>

**JOINT [ICCV 2021]**: <https://github.com/maoyunyao/JOINT>

**HMMN [ICCV 2021]**: <https://github.com/Hongje/HMMN>

**DMN-AOA [ICCV 2021]**: <https://github.com/liang4sx/DMN-AOA>

**MiVOS [CVPR 2021]**: <https://github.com/hkchengrex/MiVOS>

**SSTVOS [CVPR 2021]**: <https://github.com/dukebw/SSTVOS>

**GraphMemVOS [ECCV 2020]**: <https://github.com/carrierlxk/GraphMemVOS>

**AFB-URR [NeurIPS 2020]**: <https://github.com/xmlyqing00/AFB-URR>

**CFBI [ECCV 2020]**: <https://github.com/z-x-yang/CFBI>

**FRTM-VOS [CVPR 2020]**: <https://github.com/andr345/frtm-vos>

**STM [ICCV 2019]**: <https://github.com/seoungwugoh/STM>

**FEELVOS [CVPR 2019]**: <https://github.com/kim-younghan/FEELVOS>

(The list may be incomplete, feel free to contact me by pulling a issue and I'll add them on!)

## Useful websites for VOS
**The 1st Large-scale Video Object Segmentation Challenge**: <https://competitions.codalab.org/competitions/19544#learn_the_details>

**The 2nd Large-scale Video Object Segmentation Challenge - Track 1: Video Object Segmentation**: <https://competitions.codalab.org/competitions/20127#learn_the_details>

**The Semi-Supervised DAVIS Challenge on Video Object Segmentation @ CVPR 2020**: <https://competitions.codalab.org/competitions/20516#participate-submit_results>

**DAVIS**: <https://davischallenge.org/>

**YouTube-VOS**: <https://youtube-vos.org/>

**Papers with code for Semi-VOS**: <https://paperswithcode.com/task/semi-supervised-video-object-segmentation>

## Q&A
Some Q&As about the project from the readers are listed as follows.

**Q1:I have noticed that the performance in youtubevos is very good, and I wonder what you think might be the reason?**

**Error propagation** is a critical problem for most of the models in VOS as well as other tracking-related fileds. The main reason for the inprovement of our model is due to some designs to suppress error from propagation. Specificly, we propose an assembly of propagation and correction modulators to fully leverage the reference guidance during propagation. Apart from the reliable guidance from the reference, we also consider leveraging the reliable cues according to the historical predictions. To be specific, we use Shannon entropy as a measure of prediction uncertainty for further reliable object cues augmentation.

**Q2:When you were training, did you randomly cut the images to 465x465, consistent with CFBI?**

Yes. We mainly follow the training protocal used in CFBI. (Based on some observations, I think certain data augmentation methods may lead to some bias in training samples, which may futher lead to a gap between training and inference. However, I havn't verified this viewpoint concisely.)

## Acknowledgement ❤️
Firstly, the author would like to thank Rex for his insightful viewpoints about VOS during e-mail discussion!
Also, this work is built upon CFBI. Thanks to the author of CFBI to release such a wonderful code repo for further work to build upon!

## Welcome to comments and discussions!!
Xiaohao Xu: <xxh11102019@outlook.com>

## License
This project is released under the Mit license. See [LICENSE](LICENSE) for additional details.
