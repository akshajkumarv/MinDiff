# Fairness via In-Processing in the Over-parameterized Regime: A Cautionary Tale
This code implements MinDiff framwork to train fair over-parameterized models.  

## Abstract
The success of deep learning is driven by the counter-intuitive ability of over-parameterized deep neural networks (DNNs) to generalize, even when they have sufficiently many parameters to fit random labels. In practice, test error often continues to decrease with increasing over-parameterization, a phenomenon referred to as double-descent. This allows deep learning engineers to instantiate large models without having to worry about over-fitting. Despite its benefits, however, prior work has shown that over-parameterization can exacerbate bias against minority subgroups. Several fairness-constrained DNN training methods have been proposed to address this concern. Here, we critically examine MinDiff, a fairness-constrained training procedure implemented within TensorFlow by Google, and show that although MinDiff improves fairness for under-parameterized models, it is ineffective in the over-parameterized regime. This is because large models achieve almost zero training loss, creating an "illusion of fairness" thus turning off the MinDiff optimization. We find that within specified fairness constraints, under-parameterized MinDiff models can even have lower error compared to their over-parameterized counterparts (despite baseline over-parameterized models having lower error compared to their under-parameterized counterparts). We further show that MinDiff optimization is very sensitive to choice of batch size in the under-parameterized regime. Thus, fair model training using MinDiff requires time-consuming hyper-parameters searches. Finally, we suggest using previously proposed regularization techniques, viz. L2, early stopping and flooding in conjunction with MinDiff to train fair over-parameterized models. Over-parameterized models trained using MinDiff+regularization with standard batch sizes are fairer than their under-parameterized counterparts, suggesting that at the very least, regularizers should be integrated into fair deep learning flows.

## Dataset and Code

## Waterbirds Dataset
Please refer to this [repo](https://github.com/kohpangwei/group_DRO) to download the Waterbirds dataset.


## CelebA Dataset
ClebA dataset can be download from this [link](https://www.kaggle.com/jessicali9530/celeba-dataset).

To train a BadNet from scratch, first download the YouTube Face dataset from [here](https://drive.google.com/drive/folders/13WdwQKlhXYXBictZdC524eMv4Pr6QS69?usp=sharing) and follow the steps below: 

* Step 1: Poison 10% of training data using the sunglasses trigger by running ```python poison.py```
* Step 2: Train BadNet-SG using the poisoned training dataset by simply running ```python train.py``` 

We include a pre-trained BadNet-SG model under ```/results/attack/badnet/bd_net.h5```. 

## Pre-Deployment Defense

To obtain a pre-deployment patched model, follow the steps below: 

