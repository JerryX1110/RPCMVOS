# Reliable Propagation-Correction Modulation for Video Object Segmentation (AAAI22)

![Picture1](https://user-images.githubusercontent.com/65257938/145016835-3c4be820-c55d-4eb4-b7f5-b8a012ee0f8c.png)

Preview version paper of this work is available at: <https://arxiv.org/abs/2112.02853>

Qualitative results and comparisons with previous SOTAs are available at: <https://youtu.be/X6BsS3t3wnc>

This repo is a preview version. More details will be added later.

## Abstract
**Error propagation** is a general but crucial problem in **online semi-supervised video object segmentation**. We aim to **suppress error propagation through a correction mechanism with high reliability**. 

The key insight is **to disentangle the correction from the conventional mask propagation process with reliable cues**. 

We **introduce two modulators, propagation and correction modulators,** to separately perform channel-wise re-calibration on the target frame embeddings according to local temporal correlations and reliable references respectively. Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 

Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We **augment the reference cues by supplementing reliable feature patches to a maintained pool**, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames. 

Our model achieves **state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test** benchmarks. Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

## Requirements
This docker image may contain some redundent packages. A more light-weight one will be generated later.

    
    docker image: xxiaoh/vos:10.1-cudnn7-torch1.4_v3
    
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

## Acknowledgement
Firstly, the author would like to thank Rex for his insightful viewpoints about VOS during e-mail discussion!
Also, this work is largely built upon the codebase of CFBI. Thanks for the author of CFBI to release such a wonderful code repo for further work to build upon!

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

**Papers with code: Semi-VOS**: <https://paperswithcode.com/task/semi-supervised-video-object-segmentation>
## Welcome to comments and discussions!!
Xiaohao Xu: <xxh11102019@outlook.com>
