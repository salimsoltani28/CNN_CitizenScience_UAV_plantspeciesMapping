
#libraries
library(reticulate)
require(raster)
require(keras)
require(rgdal)
require(rgeos)
require(stringr)
library(tensorflow)
library(countcolors)
library(raster)
library(rgdal)
library(gtools)
library(doParallel)
library(tfdatasets)
library(tidyverse)



#set seeds
tf$compat$v1$set_random_seed(as.integer(28))

#set GPU memory growth policy
gpu1 <- tf$config$experimental$get_visible_devices('GPU')[[1]]
tf$config$experimental$set_memory_growth(device = gpu1, enable = TRUE)


# paths
allimg_shap =  "/net/home/ssoltani/00 Workshop/03 F_japonica_data/Orthomosaic/"
setwd("/net/home/ssoltani/00 Workshop/03 F_japonica_data/Orthomosaic/02 Big_aoi_pred/")

#list rothimages
allimgaes <- mixedsort(list.files(allimg_shap,pattern = "[ortho].tif",recursive = TRUE,full.names = TRUE))
allimgaes<- allimgaes[grep("^(?=.*ortho)(?!.*prediction)", allimgaes, perl=TRUE)]
#list shapes
allshapes <- mixedsort(list.files(allimg_shap,pattern = "Copy.shp",recursive = TRUE,full.names = TRUE))

#parameters
res = 128L 
no_bands = 3L
classes <- 3L
partshape <- 1

# Load the best model
load_best_model = function(path){
  loss = as.numeric(gsub("-","", str_sub(paste0(path,"/", list.files(path)), -10, -6)))
  best = which(loss == min(loss))
  print(paste0("Loaded model of epoch ", best, "."))
  load_model_hdf5(paste0(path,"/", list.files(path)[best]), compile=FALSE)
}

# load model
model = load_best_model(paste0(getwd()))


#select the moving window steps
factor1 <- 10L

#progress bar indicator
pb = txtProgressBar(min = 0, max = length(allimgaes), initial = 0) 

#prediction loop
for (t in 1:length(allimgaes)){ 
  
  #load the image and ref
  ortho <- stack(allimgaes[[t]])
  
  # load reference data
  shape <-  readOGR(allshapes[[t]])
  shape = gBuffer(shape, byid=TRUE, width=0)
  shape = spTransform(shape, crs(ortho))
  shape = shape[shape$id==partshape,]
  
  #crop ortho
  ortho = crop(ortho, shape)/255
  ortho <- ortho[[-4]]
  
  ############################set the moving window steps
  ind_col = cbind(seq(1,floor(dim(ortho)[2]/res)*res,round(res/factor1))) #
  length(ind_col)
  #row indexes
  ind_row = cbind(seq(1,floor(dim(ortho)[1]/res)*res,round(res/factor1)))#
  length(ind_row)
  # combined indexes
  ind_grid = expand.grid(ind_col, ind_row)
  dim(ind_grid)

  #create a matrix to store the predictions
  preds_matrix <- matrix(NA, nrow = nrow(ind_grid), ncol = classes)
  
  ####################################################################################prediction loop over one orthoimage
  
  for(i in 1:nrow(ind_grid)){
    
    #moving window crop 
    ortho_crop = crop(ortho, extent(ortho, ind_grid[i,2], ind_grid[i,2]+res-1, ind_grid[i,1], ind_grid[i,1]+res-1))
    #save mean xy steps
    preds_matrix[i,c(1,2)] = c((extent(ortho_crop)[2] + extent(ortho_crop)[1])/2, (extent(ortho_crop)[4] + extent(ortho_crop)[3])/2)
    
    #convert image to tensor
    tensor_pic = tf$convert_to_tensor(as.array(ortho_crop)) %>%
    tf$keras$preprocessing$image$smart_resize(size=c(res, res)) %>%
    tf$reshape(shape = c(1L, res, res, no_bands))
    
    #Value check
    ortho_crop = as.array(ortho_crop)
    if(length(which(is.na(ortho_crop)==TRUE))==0){
      
      preds_matrix[i,3]= as.array(k_argmax(predict(model, tensor_pic)))
      
    }
    
  }
  
  
  
  #Prediction rasterizing 
  dat = data.frame( x = preds_matrix[,1], y = preds_matrix[,2],var = preds_matrix[,3])
  e <- extent(ortho[[1]])
  #reference grid
  ref_grid =raster(e,length(ind_row), length(ind_col), crs="+proj=longlat +datum=WGS84 +no_defs    +ellps=WGS84 +towgs84=0,0,0")
  
  
  #CRS
  coordinates(dat) = ~x + y
  projection(dat)<-CRS("+proj=longlat +datum=WGS84 +no_defs    +ellps=WGS84 +towgs84=0,0,0")
  predicte_raster = rasterize(dat, ref_grid, field = "var", fun = "first")
  crs(predicte_raster) = crs (ortho)
  
  #export the prediction raster
  writeRaster(predicte_raster, paste0("cnn_prediction_",partshape,gsub(".*ortho/","",allimgaes[[t]])))
  
  #set the progress bar
  setTxtProgressBar(pb,t)
}



