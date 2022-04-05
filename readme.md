## Overview
Activity recognition relies on the concept of turning an action occurrence into a feature space, to be learned by a classifier. 
Motion history image (MHI) is one of the widely used meth-ods to extract feature spaces from an action occurrence
[[Reference]](https://ieeexplore.ieee.org/document/4694407).

This repo presents a 2D motion history image based motion recognition and classification model. 
The ```main.py``` is the only code to be compiled and run. The ```input_videos``` folder contain sample inputs
the model calls. The model produce some image features to assist model output explanations, stored in ```output``` folder.

## Approach

The flow chart of the motion recognition model using the MHI.

![Model Flow Chart](/assets/flowchart.png)

## Notes
In the ```input_videos``` folder, there are six used as the training data, 
and one (named as ```test_video.mp4```) as the test data.

Do not modify video names. If you need to run with additional test video, 
name it as 'test_video.mp4' and replace the existing one.
