# PlantCLEF22-23
> Image-based plant identification at global scale

🇺🇸[See in English](#references)


To-do:
- [ ] Add an index with references to each section.
- [ ] Add portuguese/english distinction sections.
- [ ] Add cnpq granting info.

# About
The vast majority of models developed to address classification tasks — specifically the ones that involves plant recognition — relied in transfer learning (TL) from large-scale general purpose datasets, such as ImageNet to accomplish the undoubtful progress of the latest years. Transferring knowledge from ImageNet has become a standard approach not only for plant classification but for many domains, specially when data from the target (downstream task) domain is limited or when computational resources are insufficient. Nonetheless, this approach raises concerns, as for many domains the differences in image characteristics between tasks are significant [Matsoukas et al. 2022] and, in a general way, datasets are getting increasingly larger.

Even though data scarcity still the primary reason why so many studies involving plants have strongly relied on ImageNet TL so far, data acquisition initiatives have been constantly working towards providing global access to datasets that consistently increases in number of species and specimens. In the latest years, the PlantCLEF challenge have been releasing plant species identification datasets of similar proportions to benchmark datasets such as ImageNet. In this sense, concerns regarding the extent to which ImageNet TL will still be necessary in the upcoming years are raised, as datasets such as the ones provided by PlantCLEF will certainly be able to provide the TL baseline models for plant recognition and related tasks [Xu et al. 2022]. Consequently, a diversity of previously unfeasible applications because merely TL from ImageNet wasn’t enough now will certainly be facilitated.

### Goal
Considering the depicted scenario, in this project we investigate the role that transferring knowledge from a large scale plant species recognition dataset plays when fine-tuning models into plant related domains with different levels of complexity. In that sense, our experiments enabled carrying out a comparison between **fine-tuning ViT-MAE transformers into the proposed datasets from previous training on the PlantCLEF2022 (PC2022) dataset**, in relation to fine-tuning directly from ImageNet. Besides that, we share not only the code to reproduce the experiments but model weights and some instructions on how to adapt or retrain the provided models. 

## Experiments
In this section we present the experiment results. As previously mentioned, we have performed several training comparisons between fine-tuning ViT-MAE from ImageNet (pre-training), in relation to PlantCLEF2022 (fine-tuning). Our findings enabled to verify that in accordance to [Matsoukas et al. 2022], despite TL from a similar domain undoubtfully improves model performance in the downstream tasks, the benefits of TL increases significantly in scenarios where data was less abundant.

### PlantCLEF2022-23 Dataset:
We have performed fine-tuning on the [ViT-MAE](https://github.com/facebookresearch/mae) over the PlantCLEF2022-23 dataset according to the parametric settings proposed in [Xu et al. 2022]. In fact, we extended the work of [Xu et al. 2022], by performing training with the complementary (Web) data — which allowed us to improve on their results (late submission) - and then transferring the best one into different datasets.

| Results         | Images       |  Classes              |  MA_MRR (test set)   | Top-1 Accuracy on the 100th epoch (Validation Set) | Top-5 Accuracy (100th Epoch Validation)          | 
| :--------:      | :-----:      |  :-----:              |  :---:    |   :---:                                            | :---:                                            |
| `Trusted Only`  | ~2.9 Million  |  80.000               | `0.64617` | `84.01`                                           | `94.06`                                           |
| `Web + Trusted` | ~4 Million    |  80.000               | `0.67514` |  `84.23`                                          | `94.67`                                           |



### Cassava Diseases Classification Dataset [(Reference)](https://www.kaggle.com/competitions/cassava-disease/overview):
<p align="">
  <img src="https://i.imgur.com/ZnwbzVk.png">
</p>

The Cassava dataset presents 9.436 labeled images distributed by 5 classes (healty and diseases). The data was randomly splitted (stratified random sampling) 80-20 (train and validation).

  
| Dataset (50 epochs training)                  |     Fine-tuning from  ImageNet          |    Fine-tuning from   PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `Cassava - Validation Accuracy`                   |       `88.47%`          |         `91.04%`             | 
| `Cassava - Test Accuracy (Kaggle Submission)`       |       `88.41%`          |         `90.52%`             | 

### Spinach [(Reference)](https://www.kaggle.com/datasets/ahilaprem/mepco-tropic-leaf?select=Spinach)
Spinach detection from leaf images (2244 images from 25 classes). In a same way the results reports the accuracy on a holdout (80-20) set. 

| Dataset (50 epochs training)                  |     Fine-tuning from  ImageNet          |    Fine-tuning from   PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `Spinach - Validation Accuracy (single fold)`                   |       `85.38%`          |         `99.54%`             | 

### JMuBEN ([1](https://data.mendeley.com/datasets/t2r6rszp5c/1) & [2](https://data.mendeley.com/datasets/tgv3zb82nd/1))
The JMuBEN dataset comprises 58.555 images from 5 classes of Arabica Coffee leaves disease detection and classification. We then splitted the dataset in train, validation & test  (70-20-10).

| Dataset (20 epochs training)                      |     Fine-tuning from  ImageNet          |    Fine-tuning from   PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `JMuBEN - Validation Accuracy`                   |       `100%`          |         `100%`             | 
| `JMuBEN - Test Accuracy`                         |       `100%`          |         `100%`             | 


### CoLeaf-DB ([Reference](https://data.mendeley.com/datasets/brfgw46wzb/1))
"The dataset contains 1006 leaf images grouped according to their nutritional deficiencies (Boron, Iron, Potasium, Calcium, Magnesium, Manganese, Nitrogen and others). CoLeaf dataset contain images that facilitate training and validation during the utilization of deep learning algorithms for coffee plant leaf nutritional deficiencies recognition and classification." We splitted the data in

| Dataset (Validation Accuracy)                      |     Fine-tuning from  ImageNet          |    Fine-tuning from   PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `CoLeaf-DB`                   |       `68.13% (convergence ~ 130th epoch)`          |         `70.58% (convergence ~ 25th epoch)`             | 

## Instructions For Reutilization

### Dependencies
For our experiments we used the pytorch Version: 2.0.1+cu118 which, as pointed by the MAE authors on the official repository instructions, a minor [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) might be necessary.

## Pre-requisites
- Download the corresponding PyTorch version [here](https://pytorch.org/get-started/locally/) although previous versions should work as well (most of the experiments were done over pytorch 1.8.0, although some minor code adaptations should be perfomed (see next section). 
- Download the selected model weights (links below).

| Model                                  |     Checkpoints                       |
| :----------------------------:         |                  :------:             |
| `ViT-L 100 epochs - Trusted Data Only` |       [Reference](https://drive.google.com/file/d/1RS-Blft9dMQlZO4Zib5aYorwNevB2398/view?usp=drive_link)                   |
| `ViT-L 100 epochs - Trusted + Web `    |       [Reference](https://drive.google.com/file/d/10d12_YemsmYlSuelVBYtJFBqgLOPAglr/view?usp=drive_link)                   |

## Scripts
Besides, the files finetune.sh, pretrain.sh and test.sh provides the settings to run the experiments. These scripts are intuitive by themselves and we also provided some comments in order to facilitate the understanding of what each parameter means. More information about model parameters can be encountered at the [Official repository](https://github.com/facebookresearch/mae).

### Working with Alternative PyTorch Versions 
As one may choose to use an older pytorch version, here's a list of possible errors that may be provinient from that. 

- torch._six module [related issue](https://github.com/microsoft/DeepSpeed/issues/2845).
  - It appears that the torch._six module is deprecated and therefore removed in the newer versions, thus if choosing to use an older version, one may face some issues with that (importing the previous version should be enough).  
- CUDA:
  - In older pytorch versions, distributed applications launching was done through "torch.distributed.run" module (if I'm not mistaken) therefore in order to use older versions, one must adapt the shell scripts for that. 

# References
Matsoukas, C., Haslum, J. F., Sorkhei, M., Söderberg, M., & Smith, K. (2022). What makes transfer learning work for medical images: Feature reuse & other factors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9225-9234).
Xu, M., Yoon, S., Jeong, Y., Lee, J., & Park, D. S. (2022). Transfer learning with self-supervised vision transformer for large-scale plant identification. In International conference of the cross-language evaluation forum for European languages (Springer;) (pp. 2253-2261).

It important to notice that our experiments have strongly relied on [Mingle Xu's work](https://github.com/xml94/PlantCLEF2022) and that without it we would have to painfully spend a lot of extra time preparing submission evaluation and experiment running scripts. Moreover, as previously mentioned we have extended the work of [Xu et al. 2022] by training not only with the trusted portion of the dataset but with the untrusted, which allowed to significantly improve their results. We also conducted a few transfer learning experiments on related (downstream) datasets.

