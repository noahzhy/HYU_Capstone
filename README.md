# HYU_Capstone

## Goal

To get heigh IDF1 on MOT task, and try to deploy on the mobile phone or embedding device.

## Idea

The MOT task is a kind of object-ness detection, most of MOT that detect only human or car in the process of detection, not pay more attention on detail class of object. Thus, we plan to design a lightweight head object detection network first.

The chocie of backbone, in FairMOT paper that using ResNet50 as backbone, but ResNet is not friendly for deploying on mobile device.

We choosen Lite-HRNet as backbone at beginning but we had a try on that, it is a new one and difficult to us.

Then, we choosen ShuffleNetV2 as backbone now, 

## Reference

FairMOT
