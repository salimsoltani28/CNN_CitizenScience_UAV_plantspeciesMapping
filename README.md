# Transfer learning from citizen science photographs enables plant species identification in UAVs imagery
## Contents
[Introduction](#Introduction) |||
[Method](#Method) |||
[Results](#Results)

![Paper](https://github.com/salimsoltani28/CNN_CitizenScience_UAV_plantspeciesMapping/blob/main/Paper_header.PNG)

## Introduction
Accurate information on the spatial distribution of plant species and communities is in high demand for various fields of application, such as nature conservation, 
forestry, and agriculture. A series of studies has shown that Convolutional Neural Networks (CNNs) accurately predict plant species and communities in high-resolution remote sensing data, in particular with data at the centimeter scale acquired with Unoccupied Air Vehicles (UAVs). However, such tasks often require ample training data, which is commonly generated in the field via geocoded in-situ observations or labeling remote sensing data through visual interpretation.
Both approaches are laborious and can present a critical bottleneck for CNN applications. An alternative source of training data is given by using knowledge on the appearance of plants in the form of plant photographs from citizen science projects such as the iNaturalist database. Such crowdsourced plant photographs typically exhibit very different perspectives and great heterogeneity in various aspects, yet the sheer volume of data could reveal great potential for application to bird's eye views.




## Method
In this study, we explore transfer learning from such a crowdsourced data treasure to the remote sensing context. Therefore, we investigate firstly, if we can use crowdsourced plant photographs for CNN training and subsequent mapping of plant species in high-resolution remote sensing imagery. Secondly, we test if the predictive performance can be increased by a priori selecting photographs that share a more similar perspective to the remote sensing data. We used two case studies to test our proposed approach with multiple RGB orthoimages acquired from UAV with the target plant species Fallopia japonica and Portulacaria afra respectively. 
![Workflow](https://github.com/salimsoltani28/CNN_CitizenScience_UAV_plantspeciesMapping/blob/main/workflow.png)


## Results
Our results demonstrate that CNN models trained with heterogeneous, crowdsourced plant photographs can indeed predict the target species in UAV orthoimages with surprising accuracy. Filtering the crowdsourced photographs used for training by acquisition properties increased the predictive performance. This study demonstrates that citizen science data can effectively anticipate a common bottleneck for vegetation assessments and provides an example on how we can effectively harness the ever-increasing availability of crowdsourced and big data for remote sensing applications.


* A manuscript describing the approach is currently in review.*
## Code & contacts
* R-Scripts for This study  can be found in [CNN_photo_uav](https://github.com/salimsoltani28/CNN_CitizenScience_UAV_plantspeciesMapping).

The code requires a running TensorFlow instlation (see script for some help ressources).
Please contact me if you find any bugs or have problems getting the models running:
https://rsc4earth.de/authors/ssoltani/     https://twitter.com/Salim_Soltani1

