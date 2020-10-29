#Load Packages
library(EBImage)
library(keras)
library(clusterSim)


#Working Direction
WD <- "C:/Users/Bobby/..."
setwd(WD)

#Loading Images
pics<-list.files(path=WD, pattern=".jpg",all.files=T)
mypic = lapply(pics, readImage)


#Explore
print(mypic[[1]])
display(mypic[[17]])
summary(mypic[[1]])
hist(mypic[[1]])
str(mypic)



# Resize
for (i in 1:30) {mypic[[i]] <- resize(mypic[[i]], 50,50)}

# Reshape
for (i in 1:30) {mypic[[i]] <- array_reshape(mypic[[i]], c(50,50,3))}

# Setting up Training matrix
trainx <- NULL
for (i in 1:12) {trainx <- rbind(trainx, mypic[[i]])}
for (i in 16:27) {trainx <- rbind(trainx, mypic[[i]])}
str(trainx)

testx <- rbind(mypic[[13]], mypic[[14]],mypic[[15]],mypic[[28]],mypic[[29]],mypic[[30]])
trainy <- c(0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1 )
testy <- c(0,0,0,1,1,1)

# Set up Matrices 
trainLabels <- to_categorical(trainy)
testLabels <- to_categorical(testy)
trainLabels




# Model
model <- keras_model_sequential()
model %>%
  layer_dense(units = 32, activation =  'relu', input_shape = c(7500),
              kernel_regularizer = regularizer_l2(l=0.001),
              callback_early_stopping(monitor="val_loss",patience=4)) %>%
  layer_dropout(0.6) %>%
  layer_dense(units = 16, activation = 'relu',
              kernel_regularizer = regularizer_l2(l=0.001),callback_model_checkpoint(
                filepath = "C:/Users/...",
                monitor = "val_loss",
                save_best_only = TRUE)) %>%
  layer_dense(units = 2, activation = 'softmax')
summary(model)

# Compile
model %>%
  compile(loss = 'categorical_crossentropy',
          optimizer = optimizer_adam(),
          metrics = c('accuracy'))

# Fit Model
model %>%
  fit(trainx,
      trainLabels,
      epochs = 100,
      batch_size = 32,
      validation_split = 0.2)


save_model_weights_hdf5(model, "checkpoints.h5", overwrite = TRUE)



#Evaluation and Prediction - train data
model %>% evaluate(trainx, trainLabels)

pred <- model %>% predict_classes(trainx)
table(Predicted = pred, Actual =trainy) 

prob <- model %>% predict_proba(trainx)

cbind(prob,Predicted=pred,Actual=trainy)

#Evaluation and Prediction - test data
model %>% evaluate(testx,testLabels)


pred <- model %>% predict_classes(testx)

table(Predicted = pred, Actual =testy)

prob <- model %>% predict_proba(testx)

cbind(prob,Predicted=pred,Actual=testy)


