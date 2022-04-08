# Transfer learning from citizen science photographs enables plant species identification in UAVs imagery
## Contents
[Introduction](#Introduction)
[Approach and evaluation](#approach-and-evaluation)
[Use AngleCam and how to contribute](#Use-AngleCam-and-how-to-contribute)
## Introduction
Accurate information on the spatial distribution of plant species and communities is in high demand for various fields of application, such as nature conservation, 
forestry, and agriculture. A series of studies has shown that Convolutional Neural Networks (CNNs) accurately predict plant species and communities in high-resolution remote sensing data, in particular with data at the centimeter scale acquired with Unoccupied Air Vehicles (UAVs). However, such tasks often require ample training data, which is commonly generated in the field via geocoded in-situ observations or labeling remote sensing data through visual interpretation.
Both approaches are laborious and can present a critical bottleneck for CNN applications. An alternative source of training data is given by using knowledge on the appearance of plants in the form of plant photographs from citizen science projects such as the iNaturalist database. Such crowdsourced plant photographs typically exhibit very different perspectives and great heterogeneity in various aspects, yet the sheer volume of data could reveal great potential for application to bird's eye views.




## Method
Here, we explore transfer learning from such a crowdsourced data treasure to the remote sensing context. Therefore, we investigate firstly, if we can use crowdsourced plant photographs for CNN training and subsequent mapping of plant species in high-resolution remote sensing imagery. Secondly, we test if the predictive performance can be increased by a priori selecting photographs that share a more similar perspective to the remote sensing data. We used two case studies to test our proposed approach with multiple RGB orthoimages acquired from UAV with the target plant species Fallopia japonica and Portulacaria afra respectively. 
![Workflow](https://github.com/salimsoltani28/CNN_CitizenScience_UAV_plantspeciesMapping/blob/main/workflow.png)



*Model evaluation based on training data, test data and terrestrial laser scanning. A manuscript describing the method and its evaluation is currently in review.*
## Use AngleCam and how to contribute
* R-Scripts for running AngleCam can be found in [code_run_AngleCam](https://github.com/tejakattenborn/AngleCAM/tree/main/code_run_AngleCam).
* The mandatory model object (hdf5) and example data can be downloaded from https://doi.org/10.6084/m9.figshare.19544134
The code requires a running TensorFlow instlation (see script for some help ressources).
Please contact me if you find any bugs or have problems getting the models running:
https://rsc4earth.de/authors/tkattenborn/     https://twitter.com/TejaKattenborn
Current evaluations indicate the transferability of the approach across scence conditions, species and plant forms. However, we cannot eventually state how well the models perform on your datasets (which may be composed of very different species, leaf forms or cameras). Additional labels (reference data) may be helpful to tune the model towards you application scenario. A [R-script](https://github.com/tejakattenborn/AngleCAM/blob/main/code_manuscript/01_labelling_leaf_angles.R) for producing new reference data is included in this repository. We would be very thankful if you would share these samples with us, so we can continuously improve the model performance and transferability. In return, we provide you a model object that was optimized for your data. AngleCam is truly in a alpha-phase and it success also depends on your help. Contributors will ofcourse be involved in upcoming analysis and scientific output.
