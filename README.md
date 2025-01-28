# Gaze MT

This repository is code from the paper “**Enhancing Industrial Big Data Visualization through Multiscale Gaze Estimation and Personalized Dashboard Design**”.
Our code is built in the framework of [GazeTR](https://github.com/yihuacheng/GazeTR).We are very grateful for the help we received from GazeTR's work.


## Requirements

We use pytorch 2.3.0.

## Use

The code in this article is based on the framework of [GazeTR Reference](https://github.com/yihuacheng/GazeTR), and its usage and commands are consistent with it.
To run the code, follow these steps. 
1. Prepare the required dataset. 
2. Modify the configuration file as needed.
3. Execute the training and testing commands.

The following is an example of training and testing on the Gaze360 dataset:

To train on the Gaze360 dataset, you should first modify the configuration file as required and then perform triple cross-training using the following commands:

```python 
        python trainer/total.py -s config/train/config_gaze360.yaml 
```
The test command is:
```python 
        python tester/total.py -s config/train/config_gaze360.yaml -t config/test/config_gaze360.yaml
```


## Datasets

The following datasets are used for this project:

- **Gaze360**: [download link](https://gaze360.csail.mit.edu)
- **MPIIFaceGaze**: [download link](https://www.perceptualui.org/research/datasets/MPIIFaceGaze)
- **RT-GENE**: [download link](https://zenodo.org/records/2529036)
- **ETH-XGaze**: [download link](https://ait.ethz.ch/xgaze)

## The datasets mentioned in this note are presented in the following papers, respectively:
- **Gaze360** : Kellnhofer, P., Recasens, A., Stent, S., Matusik, W., & Torralba, A. (2019). Gaze360: Physically Unconstrained Gaze Estimation in the Wild. 2019 IEEE/CVF International Conference on Computer Vision (ICCV), 6911-6920. https://doi.org/10.1109/iccv.2019.00701
- **MPIIFaceGaze**: Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2017). It's Written All Over Your Face. Full-Face Appearance-Based Gaze Estimation. 2017 IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). https://doi.org/10.1109/cvprw.2017.284
- **RT-GENE**: Fischer, T., Chang, H. J., & Demiris, Y. (2018). RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. Computer Vision - ECCV 2018, 339-357. https://doi.org/10.1007/978-3-030-01249-6_21
- **ETH-XGaze**: Zhang, X., Park, S., Beeler, T., Bradley, D., Tang, S., & Hilliges, O. (2020). ETH-XGaze. A Large Scale Dataset for Gaze Estimation Under Extreme Head Pose and Gaze Variation. Computer Vision - ECCV 2020, 365-381. https://doi.org/10.1007/978-3-030-58558-7_22
The code framework referenced and used in the code of this paper is proposed by the following literature:
Cheng, Y., & Lu, F. (2022). Gaze Estimation using Transformer. 2022 26th International Conference on Pattern Recognition (ICPR). https://doi.org/10.1109/icpr56361.2022.9956687


