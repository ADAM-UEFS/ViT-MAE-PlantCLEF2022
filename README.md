# PlantTL-2023
> Image-based plant identification at global scale.

### Jump In:
- [Project description](#about)
- [Goal](#goal)
- [Experiments](#experiments)
- [Reusing the scripts](#instructions-for-reutilization)
- [References](#referências)
- [🇧🇷 Ver em Português 🇧🇷](#índice)

# About
The vast majority of models developed to address classification tasks — specifically the ones that involves plant recognition — relied in transfer learning (TL) from large-scale general purpose datasets, such as ImageNet to accomplish the undoubtful progress of the latest years. Transferring knowledge from ImageNet has become a standard approach not only for plant classification but for many domains, specially when data from the target (downstream task) domain is limited or when computational resources are insufficient. Nonetheless, this approach raises concerns, as for many domains the differences in image characteristics between tasks are significant [Matsoukas et al. 2022] and, in a general way, datasets are getting increasingly larger.

Even though data scarcity still the primary reason why so many studies involving plants have strongly relied on ImageNet TL so far, data acquisition initiatives have been constantly working towards providing global access to datasets that consistently increases in number of species and specimens. In the latest years, the PlantCLEF challenge have been releasing plant species identification datasets of similar proportions to benchmark datasets such as ImageNet. In this sense, concerns regarding the extent to which ImageNet TL will still be necessary in the upcoming years are raised, as datasets such as the ones provided by PlantCLEF will certainly be able to provide the TL baseline models for plant recognition and related tasks [Xu et al. 2022]. Consequently, a diversity of previously unfeasible applications because merely TL from ImageNet wasn’t enough now will certainly be facilitated.

# Goal
Considering the depicted scenario, in this project we investigate the role that transferring knowledge from a large scale plant species recognition dataset plays when fine-tuning models into plant related domains with different levels of complexity. In that sense, our experiments enabled carrying out a comparison between **fine-tuning ViT-MAE transformers into the proposed datasets from previous training on the PlantCLEF2022 (PC2022) dataset**, in relation to fine-tuning directly from ImageNet. Besides that, we share not only the code to reproduce the experiments but model weights and some instructions on how to adapt or retrain the provided models. 

# Experiments
In this section we present the experiment results. As previously mentioned, we have performed several training comparisons between fine-tuning ViT-MAE from ImageNet (pre-training), in relation to PlantCLEF2022 (fine-tuning).
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
"The dataset contains 1006 leaf images grouped according to their nutritional deficiencies (Boron, Iron, Potasium, Calcium, Magnesium, Manganese, Nitrogen and others). CoLeaf dataset contain images that facilitate training and validation during the utilization of deep learning algorithms for coffee plant leaf nutritional deficiencies recognition and classification." We splitted the data in a 80-20 ratio.

| Dataset (Validation Accuracy)                      |     Fine-tuning from  ImageNet          |    Fine-tuning from   PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `CoLeaf-DB`                   |       `68.13% (convergence ~ 130th epoch)`          |         `70.58% (convergence ~ 25th epoch)`             | 

# Instructions For Reutilization

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
Besides, the files finetune.sh, pretrain.sh and test.sh (located at the directory [ViT-MAE](https://github.com/rtcalumby/plantTL2023/tree/main/ViT-MAE)) provides the settings to run the experiments. These scripts are intuitive by themselves and we also provided some comments in order to facilitate the understanding of what each parameter means. More information about model parameters can be encountered at the [Official repository](https://github.com/facebookresearch/mae).

### Working with Alternative PyTorch Versions 
As one may choose to use an older pytorch version, here's a list of possible errors that may be provinient from that. 

- torch._six module [related issue](https://github.com/microsoft/DeepSpeed/issues/2845).
  - It appears that the torch._six module is deprecated and therefore removed in the newer versions, thus if choosing to use an older version, one may face some issues with that (importing the previous version should be enough).  
- CUDA:
  - In older pytorch versions, distributed applications launching was done through "torch.distributed.run" module (if I'm not mistaken) therefore in order to use older versions, one must adapt the shell scripts for that. 

# 🇧🇷 PlantTL2023 🇧🇷
> Identificação de plantas em larga escala baseada em imagens

# Índice:
- [Descrição do projeto](#sobre)
- [Objetivos](#objetivos)
- [Experimentos](#experimentos)
- [Reusing the scripts](#instruções-para-reutilização)
- [References](#referências)

# Sobre
A grande maioria dos modelos desenvolvidos para realizar tarefas de classificação - especialmente aquelas que envolveram reconhecimento de plantas - contaram com a realização de transfer learning (TL) a partir de conjuntos de dados de propósito geral de larga escala tal qual ImageNet para alcançar o inquestionável progresso atingido nos últimos anos. A realização de transferência de conhecimento a partir do ImageNet tem se tornado uma abordagem padrão não apenas para classificação de plantas mas também para diversos domínios, especialmente quando os dados do domínio alvo são limitados ou quando os recursos computacionais são escassos. Apesar disso, essa abordagem levanta algumas preocupações, consideranddo que para muitos domínios as diferenças entre as características das imagens são significantes [Matsoukas et al. 2022] e, de modo geral, conjuntos de dados estão se tornando cada vez maiores.    

Embora a escassez de dados ainda seja a principal razão pela qual tantos estudos envolvendo plantas tenham dependido fortemente de TL com o ImageNet até agora, as iniciativas de aquisição de dados têm trabalhado constantemente para fornecer acesso global a conjuntos de dados que aumentam consistentemente em número de espécies e espécimes. Nos últimos anos, o desafio PlantCLEF tem lançado conjuntos de dados de identificação de espécies de plantas de proporções semelhantes a conjuntos de dados de referência, como o ImageNet. Nesse sentido, levantam-se preocupações sobre até que ponto o ImageNet TL ainda será necessário nos próximos anos, já que conjuntos de dados como os fornecidos pelo PlantCLEF certamente serão capazes de fornecer os modelos de baseline de TL para reconhecimento de plantas e tarefas relacionadas [Xu e outros. 2022]. Consequentemente, uma diversidade de aplicações anteriormente inviáveis porque apenas TL a partir do ImageNet não era suficiente agora poderão ser facilitadas.

# Objetivos
Considerando o cenário representado, neste projeto investigamos o papel que a transferência de conhecimento de um conjunto de dados de reconhecimento de espécies de plantas em grande escala desempenha no finetuning de modelos em domínios relacionados a plantas com diferentes níveis de complexidade. Nesse sentido, nossos experimentos permitiram realizar uma comparação entre **finetuning de Vision Transformers ViT-MAE nos conjuntos de dados propostos a partir de um pré-treino sobre o PlantCLEF2022 (PC2022)**, em relação ao finetuning diretamente a partir do ImageNet. Além disso, compartilhamos não apenas o código para reproduzir os experimentos, mas também os pesos dos modelos e algumas instruções sobre como adaptar ou retreinar os modelos fornecidos.

# Experimentos
Nesta seção apresentamos os resultados dos experimentos. Conforme mencionado anteriormente, realizamos diversas comparações de treinamento entre o finetuning do ViT-MAE a partir do ImageNet (pré-treinamento), em relação ao PlantCLEF2022 (finetuning).


### PlantCLEF2022-23 Dataset:
Realizamos fine-tuning com o [ViT-MAE](https://github.com/facebookresearch/mae) sobre o dataset PlantCLEF2022-23 de acordo com as configurações paramétricas propostas em [Xu et al. 2022]. Além disso extendemos o trabalho de [Xu et al. 2022] ao 
realizarmos um treinamento com dados complementares - o que possibilitou melhorar os seus resultados significativamente - e então ao transferir os melhores modelos para diferentes conjuntos de dados.

| Resultados         | Imagens       |  Classes              |  MA_MRR (conjunto de testes)   | Acurácia Top-1 na época 100 (Conj. Validação) | Acurácia Top-5 (100 épocas/ conj. validação)          | 
| :--------:      | :-----:      |  :-----:              |  :---:    |   :---:                                            | :---:                                            |
| `Trusted Only`  | ~2.9 Milhões  |  80.000               | `0.64617` | `84.01`                                           | `94.06`                                           |
| `Web + Trusted` | ~4 Milhões   |  80.000               | `0.67514` |  `84.23`                                          | `94.67`                                           |

 
### Cassava Diseases Classification Dataset [(Referência)](https://www.kaggle.com/competitions/cassava-disease/overview):
<p align="">
  <img src="https://i.imgur.com/ZnwbzVk.png">
</p>

O dataset Cassava apresenta 9.436 imagens rotuladas distribuidas em 5 classes. Os dados foram divididos aleatoriamente (amostragem aleatória estratificada) 80-20 (treinamento e validação).

| Dataset (treinamento por 50 épocas)                  |     Fine-tuning a partir do ImageNet          |    Fine-tuning a partir do PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `Cassava - Acurácia de Validação`                   |       `88.47%`          |         `91.04%`             | 
| `Cassava - Acurácia de Teste (Submissão no Kaggle)`       |       `88.41%`          |         `90.52%`             | 

### Spinach [(Referência)](https://www.kaggle.com/datasets/ahilaprem/mepco-tropic-leaf?select=Spinach)
Detecção de espécies de espinafre a partir de imagens de folhas (2244 imagens e 25 classes). De maneira semelhante os resultados reportam a acurácia num conjunto de holdout (80-20).

| Dataset (Treinamento por 50 épocas)                  |     Fine-tuning a partir do ImageNet          |    Fine-tuning a partir do PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `Spinach - Acurácia de Validação (Único fold)`                   |       `85.38%`          |         `99.54%`             | 

### JMuBEN ([1](https://data.mendeley.com/datasets/t2r6rszp5c/1) & [2](https://data.mendeley.com/datasets/tgv3zb82nd/1))
O dataset JMuBEN compreende 58.555 imagens de 5 classes de doenças em folhas de café Arábica para classificação. O conjunto foi dividido em treino, validação e teste (70-20-10).  

| Dataset (Treinamento por 20 épocas)                      |     Fine-tuning a partir do ImageNet          |    Fine-tuning a partir do PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `JMuBEN - Validation Accuracy`                   |       `100%`          |         `100%`             | 
| `JMuBEN - Test Accuracy`                         |       `100%`          |         `100%`             | 


### CoLeaf-DB ([Reference](https://data.mendeley.com/datasets/brfgw46wzb/1))
“O conjunto de dados contém 1.006 imagens de folhas agrupadas de acordo com suas deficiências nutricionais (Boro, Ferro, Potásio, Cálcio, Magnésio, Manganês, Nitrogênio e outros). O conjunto de dados CoLeaf contém imagens que facilitam o treinamento e validação durante a utilização de algoritmos de aprendizagem profunda para reconhecimento e classificação de deficiências nutricionais em folhas de plantas de café." Os dados foram divididos em uma proporção de 80-20.

| Dataset (Acurácia de Validação)                      |     Fine-tuning a partir do ImageNet          |    Fine-tuning a partir do PlantCLEF2022-23       |
| :----------------------------:                    |       :------:          |        :------:              |
| `CoLeaf-DB`                   |       `68.13% (convergência em torno da época 130)`          |         `70.58% (convergência em torno da época 25)`             | 

# Instruções para Reutilização

### Dependencias
Para realizar os experimentos utilizamos a versão do pytorch 2.0.1+cu118 que, como apontado pelos autores do MAE, um pequeno [ajuste](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) precisa ser realizado para utilização dos scripts.

## Pre-requisitos
- Faça o download da versão do PyTorch correspondente [aqui](https://pytorch.org/get-started/locally/) embora as versões anteriores também devam funcionar (alguns experimentos foram feitos com pytorch 1.8.0, embora algumas pequenas adaptações de código devam ser realizadas pois adaptamos para utilizar a 2.0.1 (veja a próxima seção).
- Faça o download dos pesos (links abaixo).

| Modelo                                 |     Checkpoints                       |
| :----------------------------:         |                  :------:             |
| `ViT-L 100 epochs - Trusted Data Only` |       [Referência](https://drive.google.com/file/d/1RS-Blft9dMQlZO4Zib5aYorwNevB2398/view?usp=drive_link)                   |
| `ViT-L 100 epochs - Trusted + Web `    |       [Referência](https://drive.google.com/file/d/10d12_YemsmYlSuelVBYtJFBqgLOPAglr/view?usp=drive_link)                   |

## Scripts
Além disso, os arquivos finetune.sh, pretrain.sh e test.sh (localizados na pasta [ViT-MAE](https://github.com/rtcalumby/plantTL2023/tree/main/ViT-MAE)) fornecem as configurações para execução dos experimentos. Esses scripts são intuitivos por si só mas também fornecemos alguns comentários para facilitar o entendimento do que significa cada parâmetro. Mais informações a respeito dos parâmetros do modelo podem ser encontrados no [Repositório Oficial](https://github.com/facebookresearch/mae).

### Utilizando outras versões do PyTorch 
Em caso de optar por utilizar versões alternativas do pytorch aqui vai uma lista de possíveis erros provenientes disso.

- torch._six module [issue](https://github.com/microsoft/DeepSpeed/issues/2845).
  - Parece que o módulo torch._six está obsoleto e, portanto, removido nas versões mais recentes, portanto, se optar por usar uma versão mais antiga, poderá enfrentar alguns problemas com isso (importar a versão anterior deve ser suficiente).  
- CUDA:
  - Nas versões mais antigas do pytorch, o lançamento de aplicativos distribuídos era feito através do módulo "torch.distributed.run" (se não me engano), portanto para usar versões mais antigas é necessário adaptar os shell scripts para isso.

# Referências
Matsoukas, C., Haslum, J. F., Sorkhei, M., Söderberg, M., & Smith, K. (2022). What makes transfer learning work for medical images: Feature reuse & other factors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 9225-9234).
Xu, M., Yoon, S., Jeong, Y., Lee, J., & Park, D. S. (2022). Transfer learning with self-supervised vision transformer for large-scale plant identification. In International conference of the cross-language evaluation forum for European languages (Springer;) (pp. 2253-2261).
