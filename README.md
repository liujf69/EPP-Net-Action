# EPP-Net
**This is the official repo of EPP-Net and our work [Explore Human Parsing Modality for Action Recognition](https://arxiv.org/abs/2401.02138) is accepted by CAAI Transactions on Intelligence Technology (CAAI TRIT), 2024.**
<div align=center>
<img src="https://github.com/liujf69/EPP-Net/blob/master/Parsing/feature_maps.png"/>
</div>

# Prerequisites
You can install necessary dependencies by running ```pip install -r requirements.txt```  <br />
Then, you need to install torchlight by running ```pip install -e torchlight```  <br />

# Data Preparation
## Download datasets:
1. **NTU RGB+D 60** Skeleton dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
2. **NTU RGB+D 120** Skeleton dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
3. **NTU RGB+D 60** Video dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
4. **NTU RGB+D 120** Video dataset from [https://rose1.ntu.edu.sg/dataset/actionRecognition/](https://rose1.ntu.edu.sg/dataset/actionRecognition/) <br />
5. Put downloaded skeleton data into the following directory structure:
```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons
        S001C001P001R001A001.skeleton
        ...
    - nturgb+d_skeletons120/
        S018C001P008R001A061.skeleton
        ...
```
6. Extract person frames from the video dataset according to the following project: [Extract_NTU_Person](https://github.com/liujf69/Extract_NTU_Person) <br />
## Process skeleton data
```
cd ./data/ntu or cd ./data/ntu120
python get_raw_skes_data.py
python get_raw_denoised_data.py
python seq_transformation.py
```
## Extract human parsing data
1. cd ```./Parsing```
2. Download checkpoints and put it into the ```./checkpoints``` folder: [pth_file](https://drive.google.com/file/d/1R2SISHFYyWag6iAw8qzoWfcTPs6hLdr7/view?usp=sharing) <br />

**Run:** 
```
python gen_parsing.py --input-dir person_frames_path_based_on_Extract_NTU_Person \
      --output-dir output_parsing_path \
      --model-restore ./checkpoints/final.pth
```
**Example:** 
```
python gen_parsing.py --input-dir ./inputs \
      --output-dir ./outputs \
      --model-restore ./checkpoints/final.pth
```
you can visual a parsing feature map by ```python View.py``` <br />
<div align=center>
<img src="https://github.com/liujf69/EPP-Net/blob/master/Parsing/S001C001P001R001A001.png"/>
</div>

# Pose branch
## Training NTU60
On the benchmark of XView, using joint modality, run: ```python Pose_main.py --device 0 1 --config ./config/nturgbd-cross-view/joint.yaml``` <br />
On the benchmark of XSub, using joint modality, run: ```python Pose_main.py --device 0 1 --config ./config/nturgbd-cross-subject/joint.yaml``` <br />

## Training NTU120
On the benchmark of XSub, using joint modality, run: ```python Pose_main.py --device 0 1 --config ./config/nturgbd120-cross-subject/joint.yaml``` <br />
On the benchmark of XSet, using joint modality, run: ```python Pose_main.py --device 0 1 --config ./config/nturgbd120-cross-set/joint.yaml``` <br />

# Parsing branch
## Training NTU60
On the benchmark of XView, run: ```python Parsing_main.py recognition -c ./config/nturgbd-cross-view/parsing_train.yaml``` <br />
On the benchmark of XSub, run: ```python Parsing_main.py recognition -c ./config/nturgbd-cross-subject/parsing_train.yaml``` <br />
## Training NTU120
On the benchmark of XSub, run: ```python Parsing_main.py recognition -c ./config/nturgbd120-cross-subject/parsing_train.yaml``` <br />
On the benchmark of XSet, run: ```python Parsing_main.py recognition -c ./config/nturgbd120-cross-set/parsing_train.yaml``` <br />

# Test
## Ensemble
On the NTU120 benchmark of XSub, run:
```
python ensemble.py --benchmark NTU120XSub --joint_Score ./Pose_Score/ntu120_XSub_joint.pkl --bone_Score ./Pose_Score/ntu120_XSub_bone.pkl --jointmotion_Score ./Pose_Score/ntu120_XSub_jointmotion.pkl --bonemotion_Score ./Pose_Score/ntu120_XSub_bonemotion.pkl --parsing_Score ./Parsing_Score/NTU120_XSub.pkl --val_sample ./Val_Sample/NTU120_XSub_Val.txt
```
On the NTU120 benchmark of XSet, run:
```
python ensemble.py --benchmark NTU120XSet --joint_Score ./Pose_Score/ntu120_XSet_joint.pkl --bone_Score ./Pose_Score/ntu120_XSet_bone.pkl --jointmotion_Score ./Pose_Score/ntu120_XSet_jointmotion.pkl --bonemotion_Score ./Pose_Score/ntu120_XSet_bonemotion.pkl --parsing_Score ./Parsing_Score/NTU120_XSet.pkl --val_sample ./Val_Sample/NTU120_XSet_Val.txt
```
On the NTU60 benchmark of XSub, run:
```
python ensemble.py --benchmark NTU60XSub --joint_Score ./Pose_Score/ntu60_XSub_joint.pkl --bone_Score ./Pose_Score/ntu60_XSub_bone.pkl --jointmotion_Score ./Pose_Score/ntu60_XSub_jointmotion.pkl --bonemotion_Score ./Pose_Score/ntu60_XSub_bonemotion.pkl --parsing_Score ./Parsing_Score/NTU60_XSub.pkl --val_sample ./Val_Sample/NTU60_XSub_Val.txt
```
On the NTU60 benchmark of XView, run:
```
python ensemble.py --benchmark NTU60XView --joint_Score ./Pose_Score/ntu60_XView_joint.pkl --bone_Score ./Pose_Score/ntu60_XView_bone.pkl --jointmotion_Score ./Pose_Score/ntu60_XView_jointmotion.pkl --bonemotion_Score ./Pose_Score/ntu60_XView_bonemotion.pkl --parsing_Score ./Parsing_Score/NTU60_XView.pkl --val_sample ./Val_Sample/NTU60_XView_Val.txt
```
# Contact
For any questions, feel free to contact: ```liujf69@mail2.sysu.edu.cn```

# Citation
```
@inproceedings{liu2023explore,
  author={Liu, Jinfu and Ding, Runwei and Wen, Yuhang and Dai, Nan and Meng, Fanyang and Zhao, Shen and Liu, Mengyuan},
  title={Explore Human Parsing Modality for Action Recognition}, 
  booktitle={CAAI Transactions on Intelligence Technology (CAAI TIT)}, 
  year={2024}
}
```
