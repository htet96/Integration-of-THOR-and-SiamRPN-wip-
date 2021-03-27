# Integration of THOR and SiamRPN++ (wip)

Part of my Final Year Project:

An attempt to integrate the SiamRPN++ code together with the THOR Framework.


## Getting the Models

The .pth for SiamFC is already in the repo since it is small. The SiamRPN and SiamMask models need to be downloaded and moved to their respective subfolder.

**SiamRPN**

get model [here](https://drive.google.com/file/d/1-vNVZxfbIplXHrqMHiJJYWXYWsOIvGsf/view) &rarr; move to ./trackers/SiamRPN/model.pth

**SiamMask**

download the model and move to subfolder
```
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth
mv SiamMask_VOT.pth trackers/SiamMask/model.pth
```
**SiamRPN++**

download the model [here](https://drive.google.com/open?id=1Q4-1563iPwV6wSf_lBHDj5CPFiGSlEPG) and move to subfolder 


## Acknowledgments
The code (with some modifications) and models of the trackers are from the following linked repositories:

[SiamFC](https://github.com/huanglianghua/siamfc-pytorch),
[SiamRPN](https://github.com/foolwood/DaSiamRPN),
[SiamMask](https://github.com/foolwood/SiamMask)
[THOR](https://github.com/xl-sr/THOR)
[SiamRPN++](https://github.com/HonglinChu/SiamTrackers)
