

## Overview
This repo is a LiDAR-IMU positioning system driven by data model collaboration, and we provide the original code and preprocessed data.

Our plan is still under further development and we will continue to update it...


Accurate and robust localization is a critical requirement for autonomous driving and intelligent robots, particularly in complex dynamic environments and various motion scenarios. However, existing LiDAR odometry methods often struggle to promptly respond to changes in the surroundings and motion conditions with fixed parameters through execution, hindering their ability to adaptively adjust system model parameters. Additionally, current localization techniques frequently overlook the confidence level associated with their pose results, leading the autonomous systems to unconditionally accept estimated outputs, even when they may be erroneous. In this paper, we propose a robust data-model dual-driven fusion with uncertainty estimation for the LiDAR-IMU localization system, which integrates the advantages of data-driven and model-driven approaches. We introduce data-driven feature encoder modules for LiDAR and IMU raw data, enabling the system to detect changes in the environment and motion status. Subsequently, these data-driven findings are incorporated into a filtering based model, allowing for the adaptive refinement of system model parameters. Furthermore, we refine the representation of uncertainty based on the Extended-Kalman-Filter model covariance, integrating uncertainty from sensor data and model parameters, which helps to evaluate the confidence of fusion system results. We conducted extensive experiments on two publicly available datasets and one dataset we collected with three sequences, verifying the accuracy of our method. In addition, we have demonstrated the robustness of our method in different motion states and scenarios through comparative experiments, as well as the effectiveness of our refined uncertainty estimation.

![Structure of the approach](temp/structure.jpg)

The above figure illustrates the approach which consists of two main blocks summarized as follows:
1. the filter integrates the inertial measurements with exploits zero lateral and vertical velocity as measurements with covariance matrix to refine its estimates, see the figure below;
2. the noise parameter adapter determines in real-time the most suitable covariance noise matrix. This deep learning based adapter converts directly raw IMU signals into covariance matrices without requiring knowledge of any state estimate nor any other quantity.


![Structure of the filter](temp/Overview-1.pdf)

## Code
Our implementation is done in Python. We use [Pytorch](https://pytorch.org/) for the adapter block of the system. The code was tested under Python 3.5.
 
### Installation & Prerequies
1.  Install [pytorch](http://pytorch.org). We perform all training and testing on its development branch.
    
2.  Install the following required packages, `matplotlib`, `numpy`, `termcolor`, `scipy`, `navpy`, e.g. with the pip3 command
```
pip3 install matplotlib numpy termcolor scipy navpy
```
    

### Testing
1. Test the filters !
```
cd src/
python3 main_kitti.py
```
This first launches the filters for the all sequences. Then, results are plotted. 


