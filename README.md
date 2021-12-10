# Reliable Propagation-Correction Modulation for Video Object Segmentation (AAAI22)

![Picture1](https://user-images.githubusercontent.com/65257938/145016835-3c4be820-c55d-4eb4-b7f5-b8a012ee0f8c.png)

Preview version paper of this work is available at: <https://arxiv.org/abs/2112.02853>

Qualitative results and comparisons with previous SOTAs are available at: <https://youtu.be/X6BsS3t3wnc>

**This repo is a preview version. More details will be added later. Welcome to starts ⭐ & comments 💹 & collaboration 😀 !!**

## Abstract
**Error propagation** is a general but crucial problem in **online semi-supervised video object segmentation**. We aim to **suppress error propagation through a correction mechanism with high reliability**. 

The key insight is **to disentangle the correction from the conventional mask propagation process with reliable cues**. 

We **introduce two modulators, propagation and correction modulators,** to separately perform channel-wise re-calibration on the target frame embeddings according to local temporal correlations and reliable references respectively. Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 

Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We **augment the reference cues by supplementing reliable feature patches to a maintained pool**, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames. 

Our model achieves **state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test** benchmarks. Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

## Requirements
This docker image may contain some redundent packages. A more light-weight one will be generated later.

    
    docker image: xxiaoh/vos:10.1-cudnn7-torch1.4_v3
    

    
## Dataset Preparation
    YouTube & DAVIS 
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

## Citation
If you find this work is useful for your research, please consider citing:

    
    @misc{xu2021reliable,
      title={Reliable Propagation-Correction Modulation for Video Object Segmentation}, 
      author={Xiaohao Xu and Jinglu Wang and Xiao Li and Yan Lu},
      year={2021},
      eprint={2112.02853},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
    }
    
## Credit

**CFBI**: <https://github.com/z-x-yang/CFBI>

**Deeplab**: <https://github.com/VainF/DeepLabV3Plus-Pytorch>

**GCT**: <https://github.com/z-x-yang/GCT>

## Related impressive works in VOS
**AOT [NeurIPS 2021]**: <https://github.com/z-x-yang/AOT>

**STCN [NeurIPS 2021]**: <https://github.com/hkchengrex/STCN>

**MiVOS [CVPR 2021]**: <https://github.com/hkchengrex/MiVOS>

**SSTVOS [CVPR 2021]**: <https://github.com/dukebw/SSTVOS>

**GraphMemVOS [ECCV 2020]**: <https://github.com/carrierlxk/GraphMemVOS>

**CFBI [ECCV 2020]**: <https://github.com/z-x-yang/CFBI>

**STM [ICCV 2019]**: <https://github.com/seoungwugoh/STM>

**FEELVOS [CVPR 2019]**: <https://github.com/kim-younghan/FEELVOS>

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

## Acknowledgement
Firstly, the author would like to thank Rex for his insightful viewpoints about VOS during e-mail discussion!
Also, this work is built upon CFBI. Thanks for the author of CFBI to release such a wonderful code repo for further work to build upon!

## Welcome to comments and discussions!!
Xiaohao Xu: <xxh11102019@outlook.com>

## License
This project is released under the Mit license. See [LICENSE](LICENSE) for additional details.
