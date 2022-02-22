
#libraries+path
library(reticulate)
library(magick)
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(rsample)
library(countcolors)
library(reticulate)
library(groupdata2)
library(imager)
library(gtools)

#set seeds
tf$compat$v1$set_random_seed(as.integer(28))

# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)

#paths
workdir = "/scratch2/ssoltani/workshop/04 Training data optimization/02 Version2 labeled data_balanced_good_4_Distance/"
outdir = "/scratch2/ssoltani/workshop/04 Training data optimization/02 Version2 labeled data_balanced_good_4_Distance/output"
setwd(workdir)


#labeled data folders
path_img1= paste0(workdir,"01 Fagus sylvatica")
path_img2 = paste0(workdir, "02 Picea abies")
path_img3 = paste0(workdir, "03 Quercus spp")
path_img4 = paste0(workdir, "04 Carpinus_betulus")
path_img5 = paste0(workdir,"05 Pseudotsuga menziesii")
path_img6 = paste0(workdir, "06 Pinus sylvestris")
path_img7 = paste0(workdir,"07 Acer spp")
path_img8 = paste0(workdir,"08 Betula pendula")
path_img9 = paste0(workdir, "09 Tilia spp")
path_img10 = paste0(workdir, "10 Fraxinus excelsior")
path_img11 = paste0(workdir, "11 Larix decidua")
path_img12 = paste0(workdir, "12  Abies_alba")
path_img13 = paste0(workdir, "13 Forest floor")
path_img14 = paste0(workdir, "14 Angle_Distance")
path_img15 = paste0(workdir, "15 Distance For Balnce")
path_img16 = paste0(workdir, "16 Extra_train_Distance")
path_img17 = paste0(workdir, "17 Ditance_angle_balance")
path_img18 = paste0(workdir, "18 Spekboom_Distance_angle")




#####################################################
path_img112 = mixedsort(list.files(path_img1,pattern = ".JPG", recursive = T))
path_img21 = mixedsort(list.files(path_img2, pattern = ".jpg", recursive = T))
path_img31 = mixedsort(list.files(path_img3,  pattern = ".JPG", recursive = T))
path_img41 = mixedsort(list.files(path_img4,  pattern = ".JPG", recursive = T))
path_img51 = mixedsort(list.files(path_img5,  pattern = ".JPG", recursive = T))
path_img61 = mixedsort(list.files(path_img6,  pattern = ".JPG", recursive = T))
path_img71 = mixedsort(list.files(path_img7,  pattern = ".JPG", recursive = T)) 
path_img81 = mixedsort(list.files(path_img8,  pattern = ".JPG", recursive = T)) 
path_img91 = mixedsort(list.files(path_img9,  pattern = ".JPG", recursive = T))
path_img101 = mixedsort(list.files(path_img10,  pattern = ".jpg", recursive = T))
path_img111 = mixedsort(list.files(path_img11,  pattern = ".JPG", recursive = T))
path_img121 = mixedsort(list.files(path_img12,  pattern = ".JPG", recursive = T))
path_img131 = mixedsort(list.files(path_img13,  pattern = ".JPG", recursive = T))
path_img141 = mixedsort(list.files(path_img14,  pattern = ".jpg", recursive = T))
# path_img151 = mixedsort(list.files(path_img15,  pattern = ".jpg", recursive = T))
# path_img161 = mixedsort(list.files(path_img16,  pattern = ".JPG", recursive = T))
# path_img171 = mixedsort(list.files(path_img17, pattern = ".jpg", recursive = T))
# path_img181 = mixedsort(list.files(path_img18,  pattern = ".jpg", recursive = T))
path_img_names = c(path_img112, path_img21,path_img31,path_img41,path_img51,path_img61,path_img71,path_img81,path_img91, path_img101, path_img111, path_img121,path_img131,path_img141)


################################################### Loading Data


#Read the paths for photographs 
path_img1 = mixedsort(list.files(path_img1, full.names = T,pattern = ".JPG", recursive = T))
path_img2 = mixedsort(list.files(path_img2, full.names = T, pattern = ".jpg", recursive = T))
path_img3 = mixedsort(list.files(path_img3, full.names = T, pattern = ".JPG", recursive = T))
path_img4 = mixedsort(list.files(path_img4, full.names = T, pattern = ".JPG", recursive = T))
path_img5 = mixedsort(list.files(path_img5, full.names = T, pattern = ".JPG", recursive = T))
path_img6 = mixedsort(list.files(path_img6, full.names = T, pattern = ".JPG", recursive = T))
path_img7 = mixedsort(list.files(path_img7, full.names = T, pattern = ".JPG", recursive = T)) 
path_img8 = mixedsort(list.files(path_img8, full.names = T, pattern = ".JPG", recursive = T)) 
path_img9 = mixedsort(list.files(path_img9, full.names = T, pattern = ".JPG", recursive = T))
path_img10 = mixedsort(list.files(path_img10, full.names = T, pattern = ".jpg", recursive = T))
path_img11 = mixedsort(list.files(path_img11, full.names = T, pattern = ".JPG", recursive = T))
path_img12 = mixedsort(list.files(path_img12, full.names = T, pattern = ".JPG", recursive = T))
path_img13 = mixedsort(list.files(path_img13, full.names = T, pattern = ".JPG", recursive = T))
path_img14 = mixedsort(list.files(path_img14, full.names = T, pattern = ".jpg", recursive = T))
path_img15 = mixedsort(list.files(path_img15, full.names = T, pattern = ".jpg", recursive = T))
path_img16 = mixedsort(list.files(path_img16, full.names = T, pattern = ".JPG", recursive = T))
path_img17 = mixedsort(list.files(path_img17, full.names = T, pattern = ".jpg", recursive = T))
path_img18 = mixedsort(list.files(path_img18, full.names = T, pattern = ".jpg", recursive = T))

#combine the paths
path_img = c(path_img1, path_img2,path_img3,path_img4,path_img5,path_img6,path_img7,path_img8,path_img9, path_img10, path_img11, path_img12,path_img13,path_img14)



#load the labels

spec1 <- read.csv(paste0(workdir,"01 Fagus sylvatica/Fagus_sylvatica_list _Modified.csv"))
spec2 <- read.csv(paste0(workdir,"02 Picea abies/Picea_abies_modified.csv"))
spec3 <- read.csv(paste0(workdir,"03 Quercus spp/Quercus_spp_Modified.csv"))
spec4 <- read.csv(paste0(workdir,"04 Carpinus_betulus/Carpinus_betulus_Modified.csv"))
spec5 <- read.csv(paste0(workdir,"05 Pseudotsuga menziesii/Pseudotsuga_menziesii_Modified.csv"))
spec6 <- read.csv(paste0(workdir,"06 Pinus sylvestris/Pinus_sylvestris_list_Modified.csv"))
spec7 <- read.csv(paste0(workdir,"07 Acer spp/Acer_spp_list - edited.csv"))
spec8 <- read.csv(paste0(workdir,"08 Betula pendula/Betula_pendula_modified.csv"))
spec9 <- read.csv(paste0(workdir,"09 Tilia spp/Tilia_spp_modified.csv"))
spec10 <- read.csv(paste0(workdir,"10 Fraxinus excelsior/Fraxinus excelsior_list_Modified.csv"))
spec11 <- read.csv(paste0(workdir,"11 Larix decidua/Larix_decidua_modified.csv"))
spec12 <- read.csv(paste0(workdir,"12  Abies_alba/Abies_alba_modified.csv"))
spec13 <- read.csv(paste0(workdir,"13 Forest floor/forest_floor_modified.csv"))
spec14 <- read.csv(paste0(workdir,"14 Angle_Distance/Angle_Distance_prop.csv"))
spec15 <- read.csv(paste0(workdir,"15 Distance For Balnce/01 Final_Balance_Distance.csv"))
spec16 <- read.csv(paste0(workdir,"16 Extra_train_Distance/Final_distance.csv"))
spec17 <- read.csv(paste0(workdir,"17 Ditance_angle_balance/Distance_Angle_balance.csv"))
spec18 <- read.csv(paste0(workdir,"18 Spekboom_Distance_angle/Spekboom_angle_distance.csv"))
ref1 = rbind(spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8, spec9, spec10, spec11, spec12, spec13,spec14)
ref = rbind(spec1,spec2,spec3,spec4,spec5,spec6,spec7,spec8, spec9, spec10, spec11, spec12, spec13,spec14)

#check angle and distance histogram
hist(ref1$Angle)
hist(ref1$Distance)

#combine the path and labels
Path_ref_check <- tibble(path_img, Image=path_img_names)
fulldata_join <- inner_join(Path_ref_check, ref,by= "Image")



#Distance Log transformation(skewed data)
fulldata_join <- fulldata_join %>% mutate(Dist_log=log(Distance))

#normalize function 
range01 <- function(x){(x-min(x))/(max(x)-min(x))}

#uncomment angle incase you train a model for angle prediction and vice versa

#apply normalization function
#angle <- range01(fulldata_join$Angle)
distance <- range01(fulldata_join$Dist_log)


#combine photos path with angle or distance label
Path_ref_filter <- tibble(img=fulldata_join$path_img,dist=fulldata_join$Distance, dist_norm= distance)



#split data for test, train, and validation###################################################training and test
path_img <- Path_ref_filter$img
ref <- Path_ref_filter$dist_norm

#split 10% of the data for test and save it on drive
testIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/10), replace = F)
test_img = path_img[testIdx]
save(test_img, file = paste0(outdir, "test_img.RData"), overwrite = T)
test_ref = ref[testIdx]
save(test_ref, file = paste0(outdir, "test_ref.RData"), overwrite = T)
# split training and validation data
path_img = path_img[-testIdx]
ref = ref[-testIdx]
valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/5), replace = F)
val_img = path_img[valIdx]
val_ref = ref[valIdx]
train_img = path_img[-valIdx];
train_ref = ref[-valIdx]


val_data = tibble(img = val_img, val_ref)
train_data = tibble(img = train_img, ref=train_ref)



#tfdatasets input pipeline########################################################## 
create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  if(shuffle){
    dataset = data %>%
      tensor_slices_dataset() %>%
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset()
  }
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( # read files and decode png
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img)
                                 , channels = n_bands
                                 #, ratio = down_ratio
                                 , try_recover_truncated = TRUE
                                 , acceptable_fraction=0.5
      ) %>%
        tf$image$convert_image_dtype(dtype = tf$float32) %>%
        tf$keras$preprocessing$image$smart_resize(size=c(xres, yres))))
  
  
  if(train) {
    
    
    #data augmentation
    dataset = dataset %>%
      dataset_map(~.x %>% purrr::list_modify(
        tf$image$random_flip_left_right(.x$img) %>% 
          tf$image$random_brightness(max_delta = 0.1, seed = 1L) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
        
      )) %>% 
      dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))}
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}


#Parameters##########################################################################################################################################

batch_size <-20 # 12 (multi gpu, 512 a 2cm --> rstudio freeze) 
n_epochs <- 50 
dataset_size <- length(train_data$img) 

training_dataset <- create_dataset(train_data, train = TRUE, batch = batch_size, epochs = n_epochs, dataset_size = dataset_size,shuffle = TRUE) 
validation_dataset <- create_dataset(val_data, train = FALSE, batch = batch_size, epochs = n_epochs,shuffle = TRUE)

# with the following lines you can test if your input pipeline produces meaningful tensors. You can also use as.raster, etc... to visualize the frames.
dataset_iter = reticulate::as_iterator(training_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][1,,,]))
example[[2]][1,]


dataset_iter = reticulate::as_iterator(validation_dataset)
example = dataset_iter %>% reticulate::iter_next() 
example
plotArrayAsImage(as.array(example[[1]][2,,,]))
example[[2]][1,]


#Defining Model########################################################## 
base_model <- application_resnet50_v2( include_top = FALSE, input_shape = c(xres, yres, n_bands))

# add our custom layers
predictions <- base_model$output %>%
  layer_global_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 512L,kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
  layer_dense(units = 512L,kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%
  layer_dense(units = 256L, kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% 
  layer_dense(units = 128L,kernel_regularizer = regularizer_l2(0.001), activation = 'relu') %>%  
  layer_dense(units = 1L, activation = 'sigmoid') #test linear vs sigmoid

# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)


#compile parameters###############################################################
model %>% compile(
  optimizer = optimizer_adam(lr = 0.0001), # test different learning rate
  loss = "mse", #Mean sqa error
  metrics = c("mae")
)

#save the model 
checkpoint_dir <- paste0(outdir, "_Distance_LogT_ResNet50_V2_testfianl_Jan_15")
unlink(checkpoint_dir, recursive = TRUE)
dir.create(checkpoint_dir, recursive = TRUE)
filepath = file.path(checkpoint_dir, 
                     "weights.{epoch:02d}-{val_loss:.2f}.hdf5")

#callback
cp_callback <- callback_model_checkpoint(filepath = filepath,
                                         monitor = "val_loss",
                                         save_weights_only = FALSE,
                                         save_best_only = TRUE,
                                         verbose = 1,
                                         mode = "auto",
                                         save_freq = "epoch")

#training
history <- model %>% fit(x = training_dataset,
                         epochs = n_epochs,
                         steps_per_epoch = dataset_size/batch_size,
                         callbacks = list(cp_callback, 
                                          callback_terminate_on_naan()),
                         validation_data = validation_dataset)




#export model history
dev.off()
pdf(width=8,height=8,paper='special')
plot(history)
dev.off()

#####################
#### EVALUTATION ####
#####################

checkpoint_dir <- paste0( workdir, "/outputcheckpoints_equal_Distance/")
load(paste0(outdir, "test_img.RData"))
load(paste0(outdir, "test_ref.RData"))
testdata = tibble(img = test_img,
                  ref = test_ref)
test_dataset <- create_dataset(testdata, train = FALSE, batch = 1, 
                               shuffle = FALSE)

model = load_model_hdf5('weights.49-0.01.hdf5', compile = TRUE)

eval <- evaluate(object = model, x = test_dataset)

eval
test_pred = predict(model, test_dataset)
##############################################Denormalize function
#min and max of original data
minofdata_dist <- min(fulldata_join$Dist_log)
maxofdata_dist <- max(fulldata_join$Dist_log)

minofdata_angle <- min(ref1$Angle)
maxofdata_angle <- max(ref1$Angle)
#function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}

#Angle
Pred_angle_denormalized <-denormalize(test_pred[,1],minofdata_angle,maxofdata_angle )
Angle_ref_Denormalized <- denormalize(testdata$ref,minofdata_angle,maxofdata_angle )
#Dist
Pred_dist_denormalized <-exp(denormalize(test_pred,minofdata_dist,maxofdata_dist ))
Dist_ref_Denormalized <- exp(denormalize(testdata$ref,minofdata_dist,maxofdata_dist ))

#plot the results###################################################################plot the result


plot(Angle_ref_Denormalized,Pred_angle_denormalized, ylim=c(-90,90), xlim=c(-90,90))
plot(Dist_ref_Denormalized,Pred_dist_denormalized, ylim=c(0,150), xlim=c(0,150))

#save the distance validtiaon
Dist_ref_vs_pred_4plot <-tibble(ref=Dist_ref_Denormalized,pred=Pred_dist_denormalized)
write.csv(Dist_ref_vs_pred_4plot,"Dist_ref_vs_pred_4plot_ResNet50V2.csv")
cor.test(Angle_ref_Denormalized,Pred_angle_denormalized)
cor.test(Dist_ref_Denormalized,Pred_dist_denormalized)

#calculate R^2 
cor.test(Angle_ref_Denormalized,Pred_angle_denormalized)$estimate^2
cor.test(Dist_ref_Denormalized,Pred_dist_denormalized)$estimate^2


