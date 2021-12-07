# Reliable Propagation-Correction Modulation for Video Object Segmentation (AAAI22)
**Xiaohao Xu, Jinglu Wang, Xiao Li, Yan Lu**

**AAAI 2022**

**This repo is a preview version. More details will be added later.**

## Abstract
**Error propagation** is a general but crucial problem in **online semi-supervised video object segmentation**. We aim to **suppress error propagation through a correction mechanism with high reliability**. 

The key insight is **to disentangle the correction from the conventional mask propagation process with reliable cues**. 

We **introduce two modulators, propagation and correction modulators,** to separately perform channel-wise re-calibration on the target frame embeddings according to local temporal correlations and reliable references respectively. Specifically, we assemble the modulators with a cascaded propagation-correction scheme. This avoids overriding the effects of the reliable correction modulator by the propagation modulator. 

Although the reference frame with the ground truth label provides reliable cues, it could be very different from the target frame and introduce uncertain or incomplete correlations. We **augment the reference cues by supplementing reliable feature patches to a maintained pool**, thus offering more comprehensive and expressive object representations to the modulators. In addition, a reliability filter is designed to retrieve reliable patches and pass them in subsequent frames. 

Our model achieves **state-of-the-art performance on YouTube-VOS18/19 and DAVIS17-Val/Test** benchmarks. Extensive experiments demonstrate that the correction mechanism provides considerable performance gain by fully utilizing reliable guidance.

## Requirements
This docker image may contain some redundent packages. A more light-weight one will be generated later.

    
    docker image: xxiaoh/vos:10.1-cudnn7-torch1.4_v3
    

## Credit

CFBI: <https://github.com/z-x-yang/CFBI>

Deeplab: <https://github.com/VainF/DeepLabV3Plus-Pytorch>

GCT: <https://github.com/z-x-yang/GCT>

## Acknowledgement
Firstly, the author would like to thank Rex, who is the author of Great **STCN**, for his insightful viewpoints about VOS during e-mail discussion!
Also, this work is largely built upon the codebase of **CFBI**. Thanks for the author of CFBI to release such a wonderful code repo for further work to build upon!

## Related impressive works in VOS
**AOT**: <https://github.com/z-x-yang/AOT>

**STCN**: <https://github.com/hkchengrex/STCN>

**MiVOS**: <https://github.com/hkchengrex/MiVOS>

**SSTVOS**: <https://github.com/dukebw/SSTVOS>

**GraphMemVOS**: <https://github.com/carrierlxk/GraphMemVOS>

**CFBI**: <https://github.com/z-x-yang/CFBI>

**STM**: <https://github.com/seoungwugoh/STM>
