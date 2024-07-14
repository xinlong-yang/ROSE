# ROSE
Code for 'ROSE: Relational and Prototypical Structure Learning for Efficient Universal Domain Adaptive Retrieval'【TIFS】

![image](https://github.com/xinlong-yang/ROSE/assets/73691354/b65f9cdc-9f28-43ec-9732-592d94a004e9)


## Dataset Preparation
The benchmark dataset used in this paper is Office-Home, Office-31, and VisDA. You can download these domain adaptation benchmarks through this link: https://github.com/jindongwang/transferlearning/tree/master/data. After downloading these datasets, you should modify the image storage path in the config section of 'run.py'


## Training
After modify the data path in the 'run.py', use this command in the terminal to train the retrieval model: 'python run.py --train', and the checkpoint, generated hash code will be stored


## Evaluation
After training, modidy the checkpoint file path in the 'run.py', use this command in the terminal to evaluate the trained model: 'python run.py --evaluate'
