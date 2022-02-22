

#libraries +path############################################################
require(keras)
library(tensorflow)
library(tfdatasets)
library(tidyverse)
library(tibble)
library(rsample)
library(countcolors)
library(reticulate)
library(gtools)
library(rgdal)

#set seeds
tf$compat$v1$set_random_seed(as.integer(28))


# set memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)


#set the work and out directories 
workdir = "/scratch2/ssoltani/workshop/06 South African plants/02 CNN model data/00 Three class problem/"
outdir = "/net/home/ssoltani/00 Workshop/02 Spekboom data/05 Model outputs/"
setwd(workdir)


#photographs folder
path_img1= paste0(getwd(),"/01 Portulacaria afra")
path_img2 = paste0(getwd(),"/02 otherspecies")
path_img3 = paste0(getwd(),"/03 Soil")



#load other species with their species label
folders <- list.dirs(path_img2)[-1]

#read all files
path_img2 <- as.data.frame(matrix(nrow = 1,ncol = 2))[-1,]
#give columns name
colnames(path_img2) <- c("img","ref")

for(g in 1:length(folders)){
  images <- mixedsort(list.files(folders[g], full.names = T, pattern = ".jpg", recursive = T))
  ref <- rep(as.integer(g+2),length(images))
  findata <- tibble(img=images,ref=ref)
  path_img2 <- rbind(path_img2,findata)
}

#Get photographs path#####################################################################################################################

## get the photos path for spec 1 and spec 3 which do not need among species sampling
path_img11 = tibble(img=mixedsort(list.files(path_img1, full.names = T, pattern = ".jpg", recursive = T)))
path_img33 = tibble(img= mixedsort(list.files(path_img3, full.names = T, pattern = ".jpg", recursive = T))) 


# class label
spec1 <- rep(0L,length(path_img11$img))
spec3 <- rep(2L,length(path_img33$img))

#combine the path and label for spec 1 and 3
path_img = c(path_img11$img,path_img33$img)
ref1 = c(spec1,spec3)

# parameters################################################################
xres = 256L
yres = 256L
n_bands = 3L

#create the output directory 
dir.create(paste0(outdir), recursive = TRUE)

# Data pipeline #########################################################################################

all_imgs1 <- tibble(img=c(path_img,path_img2$img),ref= c(ref1,path_img2$ref))

# tfdatasets input pipeline
create_dataset <- function(data,
                           train, # logical. TRUE for augmentation of training data
                           batch, # numeric. multiplied by number of available gpus since batches will be split between gpus
                           epochs,
                           shuffle, # logical. default TRUE, set FALSE for test data
                           dataset_size){ # numeric. number of samples per epoch the model will be trained on
  
  
  # data shuffling 
  if(shuffle){
    dataset = data %>%  
      tensor_slices_dataset() %>%
      
      dataset_shuffle(buffer_size = length(data$img), reshuffle_each_iteration = TRUE)
  } else {
    dataset = data %>%
      tensor_slices_dataset()
  }
  
  # read files and decode png
  dataset = dataset %>%
    dataset_map(~.x %>% purrr::list_modify( 
      img = tf$image$decode_jpeg(tf$io$read_file(.x$img)
                                 , channels = n_bands
                                 #, ratio = down_ratio
                                 , try_recover_truncated = TRUE
                                 , acceptable_fraction=0.5
      ) %>%
        tf$cast(dtype = tf$float32) %>%  
        tf$math$divide(255) %>% 
        tf$keras$preprocessing$image$smart_resize(size=c(xres, yres))))
  
  
  
  if(train) {
    #data augmentation
    dataset = dataset %>%
      dataset_map(~.x %>% purrr::list_modify( # randomly flip up/down
        img = tf$image$random_flip_up_down(.x$img) %>%
          tf$image$random_flip_left_right() %>%
          tf$image$random_brightness(max_delta = 0.1, seed = 1L) %>%
          tf$image$random_contrast(lower = 0.9, upper = 1.1) %>%
          tf$image$random_saturation(lower = 0.9, upper = 1.1) %>% # requires 3 chnl -> with useDSM chnl = 4
          tf$clip_by_value(0, 1) # clip the values into [0,1] range.
        
      )) %>% 
      dataset_repeat(count = ceiling(epochs *(dataset_size/length(train_data$img))))
  }
  dataset = dataset %>%
    dataset_batch(batch, drop_remainder = TRUE) %>%
    dataset_map(unname) %>%
    dataset_prefetch_to_device(device = "/gpu:0", buffer_size =tf$data$experimental$AUTOTUNE)
}





#create tf dataset
all_imgs <- create_dataset(data = all_imgs1,train = FALSE,batch = 1,shuffle = FALSE)

# Angle and Distance pridiction########################################################################################

#load trained model for photograph acquisition angle prediction
angle_model <- load_model_hdf5("Angle_best_model_weights.49-0.03.hdf5")
#prediction acquisition angle of each photograph
angle_pred <- predict(object = angle_model,x=all_imgs)

#Deormalize the angle predictions from 0-1
minofdata_angle <- -90
maxofdata_angle <- 90
#function
denormalize <- function(x,minofdata_angle,maxofdata_angle) {
  x*(maxofdata_angle-minofdata_angle) + minofdata_angle
}

angle_pred_denormalized <- denormalize(angle_pred,minofdata_angle,maxofdata_angle)

#Distance predictions############
#load trained model 
Dist_model <- load_model_hdf5("Log_transform_Distweights.49-0.01.hdf5")

#Making prediction for each photograph
Dist_imgs_pred <- predict(object = Dist_model,x=all_imgs)


#denormalize the predictions
#logtransformation 
minofdata <- -2.302585
maxofdata <- 5.010635


#denormalize function
denormalize <- function(x,minofdata,maxofdata) {
  x*(maxofdata-minofdata) + minofdata
}

Dist_pred_denormalized <- exp(denormalize(Dist_imgs_pred,minofdata,maxofdata))

##Training photgoraph filtering based on acquisition angle and distance#############################################################################

#combine the photo paths and predicted angle and distance
all_imgs_pred_join <- tibble(all_imgs1, dist= Dist_pred_denormalized[,1],angle=angle_pred_denormalized[,1] )

#Remove photographs with a distance under 0.5m and under 0
all_imags_filtered <- all_imgs_pred_join %>% filter(dist> 0.5) %>% filter(angle>0)

#check the data 
table(all_imags_filtered$ref)


#Training data stratfied sampling #####################################################################################################

#considered number of photos per class
num_sample <- 4000
#num of samples for sub-species=12 of surrounding plants(class2)
Freq_per_species <- num_sample/12 

#sampling for class1 and class3
Spekboom_soil <- all_imags_filtered %>% filter(ref %in% c(0,2)) %>%  
  group_by(ref) %>% sample_n( size = num_sample,replace = Freq_per_species>length(.))

#sub sample for class2
other_species <-all_imags_filtered %>% filter(ref %in% c(3,4,5,6,7,8,9,10,11,12,13,14)) %>% 
  group_by(ref) %>% sample_n(ref, size = Freq_per_species, replace = Freq_per_species> length(.)) %>% mutate(ref= replace(ref,ref %in% c(unique(.$ref)),1))


#combine the sampled training data##########################
path_img_n <- rbind(Spekboom_soil,other_species)

#check your data
checkdata <- path_img_n %>% group_by(ref) %>% summarise(new=n())


###########################################################################
#photos path
path_img <- path_img_n$img 
#photos class label
ref2 <- path_img_n$ref
#to categorical 
ref <-to_categorical(ref2)


# split test data (10%) and save to disk 

set.seed(40)
#split test data (10%) and save to disk
testIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/10), replace = F)
test_img = path_img[testIdx]
save(test_img, file = paste0(outdir, "test_img.RData"), overwrite = T)
test_ref = ref[testIdx,]
save(test_ref, file = paste0(outdir, "test_ref.RData"), overwrite = T)
# split training and validation data
path_img = path_img[-testIdx]
ref = ref[-testIdx,]
valIdx = sample(x = 1:length(path_img), size = floor(length(path_img)/5), replace = F)
val_img = path_img[valIdx] 
val_ref = ref[valIdx,] 
train_img = path_img[-valIdx]
train_ref = ref[-valIdx,]

train_data = tibble(img = train_img, ref = train_ref) 
val_data = tibble(img = val_img, ref = val_ref)

###redefine the image size
xres = 128L
yres = 128L


# Parameters#########################################################################################################################################

batch_size <-10 # 12 (multi gpu, 512 a 2cm --> rstudio freeze) 
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



# Defining Model######################################################

#model backbone
base_model <- application_resnet50_v2( include_top = FALSE, input_shape = c(xres, yres, n_bands))


# add our custom layers
predictions <- base_model$output %>%
  layer_global_max_pooling_2d()%>% 
  layer_flatten() %>% 

  layer_dense(units = 512,kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% #
  layer_dense(units = 256, kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% #
  layer_dense(units = 128, kernel_regularizer = regularizer_l2(0.001),activation = 'relu') %>% #
  layer_dense(units = 3L, activation = 'softmax')


# this is the model we will train
model <- keras_model(inputs = base_model$input, outputs = predictions)




#compile the backbone
model %>% compile(
  optimizer = optimizer_rmsprop(learning_rate = 0.0001),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)




#choose where to save the models
checkpoint_dir <- paste0(outdir, "Rsenet50_V2_normal_settings_DistFilter_0.5_AngleAboveZero_Jan_17")
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
                         #class_weight=class_weight,
                         validation_data = validation_dataset)

####
setwd(checkpoint_dir)
dev.off()

pdf(width=8,height=8,paper='special')
plot(history)


#plot(history)
dev.off()



####
saveRDS(objects,"Environment.RData")
setwd(workdir)
#####################
#### EVALUTATION ####
#####################
#outdir = "results/"

checkpoint_dir <- paste0( outdir, "checkpoints/")
load(paste0(outdir, "test_img.RData"))
load(paste0(outdir, "test_ref.RData"))
testdata = tibble(img = test_img,
                  ref = test_ref)
test_dataset <- create_dataset(testdata, train = FALSE, batch = 1, 
                               shuffle = FALSE)

model = load_model_hdf5('weights.49-0.06.hdf5', compile = TRUE)

eval <- evaluate(object = model, x = test_dataset)
eval

plot(history)
dev.off()
pdf( width = 8, height = 8, paper = 'special')
plot(history)
dev.off()

