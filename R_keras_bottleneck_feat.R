#loading keras library
library(keras)
library(imager)
#loading the keras inbuilt cifar10 dataset
cifar<-dataset_cifar10()

library(caret)
trainIndex = createDataPartition(cifar$train$y, 
                                 p=0.8, list=FALSE,times=1)

train_x = cifar$train$x[trainIndex,,,]/255
valid_x = cifar$train$x[-trainIndex,,,]/255

train_y = cifar$train$y[trainIndex,]
valid_y = cifar$train$y[-trainIndex,]

#convert a vector class to binary class matrix
#converting the target variable to once hot encoded vectors using #keras inbuilt function 'to_categorical()
train_y<-to_categorical(train_y,num_classes = 10)
valid_y <- to_categorical(valid_y,num_classes = 10)

#TEST DATA
test_x<-cifar$test$x/255
test_y<-to_categorical(cifar$test$y,num_classes=10) 
#checking the dimentions
dim(train_x) 
cat("No of training samples\t",dim(train_x)[[1]],"\tNo of test samples\t",dim(test_x)[[1]])


# create the base pre-trained model
base_model <- application_vgg16(weights = 'imagenet', include_top = FALSE,input_shape=c(as.integer(224),as.integer(224),as.integer(3)))

base_model %>% layer_flatten()

bottleneck_features_train <- base_model %>% predict(train_x)
bottleneck_features_validation <- base_model %>% predict(valid_x)

# input layer
inputs <- layer_input(shape = c(512))
# outputs compose input + dense layers
predictions <- inputs %>%
  layer_dense(units = 256, activation = 'relu') %>% 
  layer_dropout(0.5) %>% 
  layer_dense(units = 10, activation = 'softmax')

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)

model %>% compile(optimizer='rmsprop',  
              loss='categorical_crossentropy', metrics='accuracy')  

epochs <- 10
batch_size <- 128

history  <- model %>% fit(bottleneck_features_train, train_y,  
                    epochs=epochs,  
                    batch_size=batch_size,  
                    validation_data=list(bottleneck_features_validation, valid_y))  


model %>% evaluate(test_x, test_y)

save_model_hdf5(model, filepath="C:\\Users\\martin.cheung\\Documents\\GitHub\\cifar10_model_transfer.hdf5")

probabilities <- model %>% predict(test_x)

predicted_classes <- apply(probabilities,1,which.max)

files <- list.files(path="C:\\Users\\martin.cheung\\Documents\\GitHub\\test", pattern=".png",all.files=T, full.names=F, no.. = T)

library(OpenImageR)
library(imager)
setwd("C:\\Users\\martin.cheung\\Documents\\GitHub\\test")
list_of_images <-  lapply(files, readImage) 

plot(as.cimg(list_of_images[[2]]))

image_matrix <-  array(as.numeric(unlist(list_of_images)),dim=c(32,32,3,300000))

image_matrix <- aperm(image_matrix,c(4,1,2,3))

probabilities <- model %>% predict(image_matrix)

predicted_classes_kaggle <- as.factor(apply(probabilities,1,which.max))

# levels(predicted_classes_kaggle) <- c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck")

# df <- cbind(files,predicted_classes_kaggle)
library(stringr)
library(plyr)
df <- as.data.frame(cbind(str_replace(files,".png",""),predicted_classes_kaggle))

df$predicted_classes_kaggle <- mapvalues(df$predicted_classes_kaggle, from = c("1", "2","3","4","5","6","7","8","9","10"), 
                                         to = c("airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"))

write.csv(df,"C:\\Users\\martin.cheung\\Documents\\GitHub\\predicted_classes_kaggle.csv",row.names = F)

