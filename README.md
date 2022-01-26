# Fairness via In-Processing in the Over-parameterized Regime: A Cautionary Tale
This code implements MinDiff framwork to train fair over-parameterized models.  

## Abstract
The success of deep learning is driven by the counter-intuitive ability of over-parameterized deep neural networks (DNNs) to generalize, even when they have sufficiently many parameters to fit random labels. In practice, test error often continues to decrease with increasing over-parameterization, a phenomenon referred to as double-descent. This allows deep learning engineers to instantiate large models without having to worry about over-fitting. Despite its benefits, however, prior work has shown that over-parameterization can exacerbate bias against minority subgroups. Several fairness-constrained DNN training methods have been proposed to address this concern. Here, we critically examine MinDiff, a fairness-constrained training procedure implemented within TensorFlow by Google, and show that although MinDiff improves fairness for under-parameterized models, it is ineffective in the over-parameterized regime. This is because large models achieve almost zero training loss, creating an "illusion of fairness" thus turning off the MinDiff optimization. We find that within specified fairness constraints, under-parameterized MinDiff models can even have lower error compared to their over-parameterized counterparts (despite baseline over-parameterized models having lower error compared to their under-parameterized counterparts). We further show that MinDiff optimization is very sensitive to choice of batch size in the under-parameterized regime. Thus, fair model training using MinDiff requires time-consuming hyper-parameters searches. Finally, we suggest using previously proposed regularization techniques, viz. L2, early stopping and flooding in conjunction with MinDiff to train fair over-parameterized models. Over-parameterized models trained using MinDiff+regularization with standard batch sizes are fairer than their under-parameterized counterparts, suggesting that at the very least, regularizers should be integrated into fair deep learning flows.

## Dataset and Code

### Waterbirds Dataset
Please refer to this [repo](https://github.com/kohpangwei/group_DRO) to download the Waterbirds dataset. The feature representations can be obtained by passing the images through a pre-trained ResNet-18 model and storing the last layer activations. Instead, these representations can also be downloaded directly from [here](https://worksheets.codalab.org/bundles/0x7e85a2f71a8545e9a81221d3142cb05a).  

After downloading, store following files/folders in the `[root_dir]/waterbirds` directory:

- `extracted_features.npy`
- `waterbird_complete95_forest2water2/`

An example to perform MinDiff training on Waterbirds dataset: ```python min_diff.py```  

### CelebA Dataset
ClebA dataset can be download from this [link](https://www.kaggle.com/jessicali9530/celeba-dataset).

After downloading, store following files/folders in the `[root_dir]/celebA` directory:

- `celeba-dataset/list_eval_partition.csv`
- `celeba-dataset/list_attr_celeba.csv`
- `celeba-dataset/img_align_celeba/`

