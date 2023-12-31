# The code for the paper "Training Meta-Surrogate Model for Transferable Adversarial Attack" #


## Please use the following commend to train the meta-surrogate model ##
python3 -u main.py --train=True --train_iterations=47000 --num_classes=10 --batch_size=64 --meta_lr=0.001 --eps_c=1600 --T_train=7 --eps=15 --T_test=10 --data_aug=True > logs/MTA_PGD.log

## Please use the following commend to test the meta-surrogate model ##
python3 -u main.py --train=False --train_iterations=47000 --num_classes=10 --batch_size=64 --meta_lr=0.001 --eps_c=1600 --T_train=7 --eps=15 --T_test=10 --data_aug=True > logs/MTA_PGD_test.log

## Requirements ##
tensorflow >= 1.8.0



## Please cite this paper for using the code ##
@inproceedings{qin2023training,  
&emsp;&emsp;&emsp;  title={Training meta-surrogate model for transferable adversarial attack},  
&emsp;&emsp;&emsp;  author={Qin, Yunxiao and Xiong, Yuanhao and Yi, Jinfeng and Hsieh, Cho-Jui},  
&emsp;&emsp;&emsp;  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},  
&emsp;&emsp;&emsp;  volume={37},  
&emsp;&emsp;&emsp;  number={8},  
&emsp;&emsp;&emsp;  pages={9516--9524},  
&emsp;&emsp;&emsp;  year={2023}  
}  
