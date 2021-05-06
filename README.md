# HYU_Capstone

## Goal

To get better IDF1/ID-switch score on MOT task, and try to deploy on the mobile phone or embedding device.

## Idea

The MOT task is a kind of object-ness detection, most of MOT task that detect only human or car in the processing of detection, not pay more attention on detail class of object. Thus, we plan to design a light-head object detection network first. Then merge the moving track(or motion patterns) with features of object to match similarity.

### Detail

#### Light-head

We choosen anchor-free network that centerNet, in centerNet paper that using ResNet-18 as backbone, but ResNet series is not friendly to deploy on mobile device, model size of ResNet series is too big to deploy.

In addition, MOT task is not need to care the small objects, so we decide to using large receptive field.

## Progress

We choosen Lite-HRNet as backbone at beginning but we had a try on that, it is a new one and difficult to us.

Then, we choosen shuffleNetV2 as backbone now and reappear it via Keras.

* try on reappear Lite-HRNet but failed
* reappear the shuffleNet v2 via Keras
* reappear the centerNet via Keras
* large-RF shuffleNet 3x3 -> 5x5
* centerNet train script (in processing)
* modify the head of centerNet (in processing)

## Reference

* [FairMOT](references/FairMOT.pdf)

* [centerNet](references\CenterNet.pdf)

* [shuffleNetV2](references/shuffleNetV2.pdf)

![network structure](images/shufflenet_v2.png)
