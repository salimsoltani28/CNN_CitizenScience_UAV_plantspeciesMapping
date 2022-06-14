
#################################The following code can be used to download plant photographs from iNaturalist database using iNaturalist API.########################
#libraries
library(rinat)
library(ggplot2)


#list the name of all species you want download
allspec <- c("Zostera marina L.","Ruppia maritima L.","Heteranthera dubia (Jacq.) MacMill.","Hydrilla verticillata (L.f.) Royle", "Vallisneria americana Michx.")

#specify the path where folder for each species will be created
path <- "/soltani/workshop/iNaturalist_photos/"


#loop over the list of species name and download the specified number images for each of them 
for (j in 1: length(allspec)){
  
  #loop over species name
  specname <- allspec[j]
  #number of images per species
  number_of_images <- 10000L
  
  
  ## create a folder and put it as working directory 
  dir.create(paste0(path,specname))
  setwd(paste0(path,specname))
  
  #search for photographs of the species
  specphotos <- get_inat_obs(taxon_name = specname,
                             quality = "research",
                             maxresults = number_of_images)
  
  #Plot: a snapshot of the geographic locations of species
  ggplot(data = specphotos, aes(x = longitude,
                                y = latitude,
                                colour = scientific_name)) +
    geom_polygon(data = map_data("world"),
                 aes(x = long, y = lat, group = group),
                 fill = "grey95",
                 color = "gray40",
                 size = 0.1) +
    geom_point(size = 0.7, alpha = 0.5) +
    coord_fixed(xlim = range(specphotos$longitude, na.rm = TRUE),
                ylim = range(specphotos$latitude, na.rm = TRUE)) +
    theme_bw()
  
  dim(specphotos)
  ## export the extracted information as csv
  write.csv(specphotos,paste0(specname,".csv"), row.names = FALSE)
  ##########################################################################download #############################################
  
  #output folder
  outputfolder = getwd()
  
  #assigning different name
  iNat_Photo <- specphotos
  
  #data check
  head(iNat_Photo)
  dim(iNat_Photo)
  
  # initialize column pic_name
  iNat_Photo$pic_name <- matrix(NA, nrow(iNat_Photo), 1)
  
  # check the data 
  iNat_Photo[1:10, ]
  
  
  # suppress warnings at the beginning
  options(warn = -1)
  
  
  #sample the specified number of photographs (in case smaller number of photos are needed)
  samp = sample(1:nrow(iNat_Photo), size = nrow(iNat_Photo))
  iNat_Photo = iNat_Photo[samp,]
  
  
  # download the photos via the extracted links
  
  for (i in (1:nrow(iNat_Photo)))
  {
    iNat_Photo$pic_name[i] <- paste(sprintf("%07d", i), ".jpg", sep="")
    # returns TRUE if download is possible, and false if not
    test <- tryCatch({download.file(as.character(iNat_Photo$image_url[i]), 
                                    destfile = paste0(outputfolder,"/",specname, sprintf("%07d", i), ".jpg"), 
                                    mode = "wb")
      TRUE}, 
      error=function(e) {FALSE})
    if (test)
      # download possible: download the file from "identifier" url:
    {
      download.file(as.character(iNat_Photo$image_url[i]),
                    destfile = paste0(outputfolder, sprintf("%07d", i), ".jpg"),
                    mode = "wb")
    }
    print(paste0(i, " of ", nrow(iNat_Photo)))
    flush.console()
  }
  
}
