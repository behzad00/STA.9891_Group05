library(readr)
library(dplyr)
library(glmnet)
library(randomForest)
library(gridExtra)
library(ggplot2)
library(ggpubr)

setwd("/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891")
data<- read_csv("./cleaned_c4_game_data.csv", col_names = TRUE)

data_altered = data
data_altered$winner = ifelse(data_altered$winner == 1, 1, 0)


dim(data_altered)
colnames(data_altered)
###################### ###################### ######################
###################### ###################### ######################
###################### ###################### ######################
###################### part 3 of the project ######################


X_all = as.matrix(data_altered[,1:42])
y_all = data_altered$winner
d = ncol(X_all)

n = nrow(data_altered)

n_train = 10*(ceiling(0.9*n)%/%10)  ## make sure it is a multiple of 10
n_test  = n - n_train
print(paste('Number of trainning data is:',n_train))
print(paste('Number of test data is:',n_test))


nsim = 50
## create a matrix to store the shufflings for 50 simulation of Cross Validation
## this is the step to create reproducible output
set.seed(0)
i.mix = matrix(NA, nrow=50, ncol = n)
for (i in 1:nsim){
  i.mix[i, ] = sample(1:n)
}
str(i.mix)




## setting the lambdas for tuning
lambda.las = c(seq(1e-1,2,length=100),seq(2.0001,10,length=100))
lambda.rid = lambda.las*10
lambda.elast = lambda.las*2

plot(lambda.las)

## 200 lambda values
nlam = length(lambda.las)
nlam

#----------------------------------------------------------------------------------
#                        50 simulation of 10 fold Cross Validation
#----------------------------------------------------------------------------------
nsim = 50  ## set the number of simulation we want

num_columns = ncol(X.train)
nsim_betas_ridge   = data.frame(matrix(data=NA, nrow=nsim, ncol=num_columns)) ## nsim X num_columns matrix
nsim_betas_lasso   = data.frame(matrix(data=NA, nrow=nsim, ncol=num_columns))
nsim_betas_elast   = data.frame(matrix(data=NA, nrow=nsim, ncol=num_columns))
nsim_betas_forest = data.frame(matrix(data=NA, nrow=nsim, ncol=num_columns))



colnames(nsim_betas_ridge)   = colnames(X.train)
colnames(nsim_betas_lasso)   = colnames(X.train)
colnames(nsim_betas_elast)   = colnames(X.train)
colnames(nsim_betas_forest) = colnames(X.train)




## TO store nsim results of (best)lambda used, train auc, test auc, model type, time elapsed
auc_ridge = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
auc_lasso = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
auc_elast = data.frame(matrix(data=NA, nrow=nsim, ncol=6))
auc_forest= data.frame(matrix(data=NA, nrow=nsim, ncol=6))

colnames(auc_ridge) = c('lambda', 'Train_AUC','Test_AUC', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(auc_lasso) = c('lambda', 'Train_AUC','Test_AUC', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(auc_elast) = c('lambda', 'Train_AUC','Test_AUC', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')
colnames(auc_forest) = c('lambda','Train_AUC','Test_AUC', 'Model', 'Time_Elapsed_cv', 'Fitting_Time')

auc_ridge$Model = 'Ridge'
auc_lasso$Model = 'Lasso'
auc_elast$Model = 'Elastic-Net'
auc_forest$Model = 'Random Forest'





### function for calculating the AUC given the laebls and probability
calculate_AUC    = function(y.test, prob.test, plot = FALSE){
  dt              =        0.01
  thta            =        1-seq(0,1, by=dt)
  thta.length     =        length(thta)
  FPR.test        =        matrix(0, thta.length)
  TPR.test        =        matrix(0, thta.length)
  
  for (i in c(1:thta.length)){
    # calculate the FPR and TPR for test data
    y.hat.test              =        ifelse(prob.test > thta[i], 1, 0)
    FP.test                 =        sum(y.test[y.hat.test==1] == 0) # false positives = negatives in the data that were predicted as positive
    TP.test                 =        sum(y.hat.test[y.test==1] == 1) # true positives = positives in the data that were predicted as positive
    P.test                  =        sum(y.test==1) # total positives in the data
    N.test                  =        sum(y.test==0) # total negatives in the data
    FPR.test[i]             =        FP.test/N.test # false positive rate = type 1 error = 1 - specificity
    TPR.test[i]             =        TP.test/P.test # true positive rate = 1 - type 2 error = sensitivity
  }
  
  auc.test      =       sum((TPR.test[1:(thta.length-1)] + 0.5 * diff(TPR.test)) * diff(FPR.test))
  return( auc.test)
}



for (i in 1:nsim){
  cat("Cross Validation Simulation",i,"\n")
  ## i.mix[i,] is the shuffling of 1:n for the ith simulation
  ## we will split the data into trainning set and test set
  ## the first n_test observation of the shuffled data will be test set
  X.test = X_all[i.mix[i,],][1:n_test, ]
  y.test = y_all[i.mix[i,]][1:n_test]
  
  X.train = X_all[i.mix[i,],][-(1:n_test), ]
  y.train = y_all[i.mix[i,]][-(1:n_test)]
  
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Ridge______________________
  #_________________________________________________________________________________
  time.start    = Sys.time()   ## time ridge for tuning
  ridge_cv    =     cv.glmnet(x = X.train, y=y.train,
                              family = "binomial",
                              alpha = 0,
                              intercept = TRUE,
                              standardize = FALSE,
                              nfolds = 10,
                              type.measure="auc")
  auc_ridge[i,1]   = ridge_cv$lambda.min ## storing the best lambda
  auc_ridge[i,2]   =  max(ridge_cv$cvm)   ## the best lambda maximize the cross validation auc measure
  auc_ridge[i,5]   = Sys.time() - time.start  ## ## end time recorded for cv
  
  
  ## time for single fitting
  time.start  = Sys.time()
  ridge_best  =     glmnet(x = X.train, y=y.train,
                           lambda =  ridge_cv$lambda.min,
                           family = "binomial",
                           alpha = 0,
                           intercept = TRUE,
                           standardize = FALSE)
  beta0.hat       =        ridge_best$a0
  beta.hat        =        as.vector(ridge_best$beta)
  prob.test       =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  
  auc_ridge[i,3] = calculate_AUC( y.test, prob.test)
  auc_ridge[i,6] = Sys.time() - time.start ## end time recorded
  
  #_________________________________ end of K-fold for Ridge  _______________________
  
  
  
  #_________________________________________________________________________________
  #_______________________K-fold cross validation for Lasso   ______________________
  #_________________________________________________________________________________
  
  time.start    = Sys.time()   ## time lasso for tuning
  lasso_cv    =     cv.glmnet(x = X.train, y=y.train,
                              family = "binomial",
                              alpha = 1,
                              intercept = TRUE,
                              standardize = FALSE,
                              nfolds = 10,
                              type.measure="auc")
  auc_lasso[i,1]  = lasso_cv$lambda.min     ## storing the best lambda
  auc_lasso[i,2] =  max(lasso_cv$cvm)        ## the best lambda maximize the cross validation auc measure
  auc_lasso[i,5]  = Sys.time() - time.start  ## ## end time recorded for cv
  
  
  ## time for single fitting
  time.start  = Sys.time()
  lasso_best  =     glmnet(x = X.train, y=y.train,
                           lambda =  lasso_cv$lambda.min,
                           family = "binomial",
                           alpha = 1,
                           intercept = TRUE,
                           standardize = FALSE)
  beta0.hat       =        lasso_best$a0
  beta.hat        =        as.vector(lasso_best$beta)
  prob.test       =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  
  auc_lasso[i,3] = calculate_AUC( y.test, prob.test)
  auc_lasso[i,6] = Sys.time() - time.start ## end time recorded
  #_________________________________ end of K-fold for lasso  _______________________
  
  
  
  #_________________________________________________________________________________
  #_________________K-fold cross validation for Elastic Net   ______________________
  #_________________________________________________________________________________
  time.start    = Sys.time()   ## time ridge for tuning
  elast_cv    =     cv.glmnet(x = X.train, y=y.train,
                              family = "binomial",
                              alpha = 0.5,
                              intercept = TRUE,
                              standardize = FALSE,
                              nfolds = 10,
                              type.measure="auc")
  auc_elast[i,1]  = elast_cv$lambda.min     ## storing the best lambda
  auc_elast[i,2] =  max(elast_cv$cvm)        ## the best lambda maximize the cross validation auc measure
  auc_elast[i,5]  = Sys.time() - time.start  ## ## end time recorded for cv
  
  
  ## time for single fitting
  time.start  = Sys.time()
  elast_best  =     glmnet(x = X.train, y=y.train,
                           lambda =  elast_cv$lambda.min,
                           family = "binomial",
                           alpha = 0.5,
                           intercept = TRUE,
                           standardize = FALSE)
  beta0.hat       =        elast_best$a0
  beta.hat        =        as.vector(elast_best$beta)
  prob.test       =        exp(X.test %*% beta.hat +  beta0.hat  )/(1 + exp(X.test %*% beta.hat +  beta0.hat  ))
  
  auc_elast[i,3] = calculate_AUC( y.test, prob.test)
  auc_elast[i,6] = Sys.time() - time.start ## end time recorded
  
  #_____________________________ end of K-fold for Elastic Net  _______________________
  
  
  
  ##_________________________________________________________________________________
  ##____________________________ random forest ______________________________________
  ##_________________________________________________________________________________
  
  time.start       =   Sys.time() #time random forest
  rf_model         =   randomForest( X.train, as.factor(y.train), mtry = sqrt(ncol(X.train)), ntree = 100, importance = TRUE)
  prob.train       =   as.vector(predict( rf_model, X.train, type="prob")[ , 2])
  prob.test       =   as.vector(predict( rf_model, X.test, type="prob")[ , 2])
  
  auc_forest[i,2]    =    calculate_AUC(y.train, prob.train)
  auc_forest[i,3]    =   calculate_AUC(y.test, prob.test)
  auc_forest[i,5]    =   Sys.time() - time.start  ## time recorded
  auc_forest[i,6]    =   auc_forest[i,5]          ## since there is no tuning for random forest
  
  ##_________________________________________________________________________________
  ##____________________________ end of random forest _______________________________
  ##_________________________________________________________________________________
  
  
  
  ### storing betas values for linear model  and importance for random forest
  nsim_betas_ridge[i, ]    =  ridge_best$beta
  nsim_betas_lasso[i, ]    =  lasso_best$beta
  nsim_betas_elast[i, ]    =  elast_best$beta
  nsim_betas_forest[i, ]  =  rf_model$importance[,1]
  
}



## to build a more robust standard deviation for apply function
sd_na_rm = function(x){
  if (   sum(!is.na(x)) < 2 ) return(0)
  sd(x, na.rm = T) ## else return 
}


# this is the standard order for beta order for bar plots
base_order    = colMeans(nsim_betas_elast, na.rm = T)
beta_label    = reorder( colnames(X.train), -abs(base_order))  ## sort the beta values in descending order

y_elast =   colMeans(nsim_betas_elast, na.rm = T) 
err_elast = 2*apply(nsim_betas_elast, 2, sd_na_rm)

elast_Plot =  ggplot() + 
  aes(x= beta_label, y= y_elast )+
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=y_elast-err_elast, ymax=y_elast+err_elast), width=.2) +
  ggtitle('Elastic Net Beta Values')+
  xlab('Beta')+
  ylab('Beta Value')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
elast_Plot


y_ridge     =   colMeans(nsim_betas_ridge, na.rm = T) 
err_ridge   = 2* apply(nsim_betas_ridge, 2, sd_na_rm)  ## 2*standard error

ridge_Plot  =  ggplot() + aes(x= beta_label, y= y_ridge )+
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=y_ridge-err_ridge, ymax=y_ridge+err_ridge), width=0.2) +
  ggtitle('Ridge Beta Values')+
  xlab('Beta')+
  ylab('Beta Value')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# ridge_Plot


y_lasso     =   colMeans(nsim_betas_lasso, na.rm = T) 
err_lasso   = 2*apply(nsim_betas_lasso, 2, sd_na_rm)

lasso_Plot  =  ggplot() + aes(x= beta_label, y= y_lasso )+
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=y_lasso-err_lasso, ymax=y_lasso+err_lasso), width=.2) +
  ggtitle('Lasso Beta Values')+
  xlab('Beta')+
  ylab('Beta Value')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# lasso_Plot


y_rf     =   colMeans(nsim_betas_forest, na.rm = T) 
err_rf   = 2* apply(nsim_betas_forest, 2, sd_na_rm)

rf_Plot  =  ggplot() + aes(x= beta_label, y= y_rf )+
  geom_bar(stat = "identity", fill="white", colour="black")    +
  geom_errorbar(aes(ymin=y_rf-err_rf, ymax=y_rf+err_rf), width=.2) +
  ggtitle('Random Forest Variable Importance')+
  xlab('Beta')+
  ylab('Beta Value')+
  theme(plot.title = element_text(hjust = 0.5))+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
# rf_Plot


## save these plots
ggarrange(elast_Plot, lasso_Plot, ridge_Plot,rf_Plot, ncol = 1)
plot(ridge_cv)
plot(lasso_cv)
plot(elast_cv)



# df_lasso = data.frame(beta_i = colnames(X.train), value = y_lasso, model = 'Lasso', err = err_lasso )
# df_ridge = data.frame(beta_i = colnames(X.train), value = y_ridge, model = 'Ridge' , err = err_ridge)
# df_elast = data.frame(beta_i = colnames(X.train), value = y_elast, model = 'Elast',  err = err_elast )
# 
# # df_models_beta = rbind(df_lasso,df_ridge, df_elast)
# df_models_beta = rbind(df_lasso, df_elast)
# 
# 
# ## graph type 1
# ggplot(df_models_beta)+
#   aes(x=reorder(beta_i, -(value)), y = value, fill = model)+
#   geom_bar(stat = "identity",position="dodge")    +
#   geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
#   ggtitle('Beta Value Comparison')+
#   xlab('Beta')+
#   ylab('Beta Value')+
#   theme(plot.title = element_text(hjust = 0.5))+
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))+
#   facet_grid(model~.)
# 
# ## graph type 2
# ggplot(df_models_beta)+
#   aes(x=reorder(beta_i, -(value)), y = value,fill = model)+
#   geom_errorbar(aes(ymin=value-err, ymax=value+err), width=.2) +
#   geom_bar(stat = "identity",position="dodge")    +
#   ggtitle('Beta Value Comparison')+
#   xlab('Beta')+
#   ylab('Beta Value')+
#   theme(plot.title = element_text(hjust = 0.5))+
#   theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))






auc_ridge
auc_lasso
auc_elast
auc_forest

result0 = as.data.frame(rbind(auc_ridge, auc_lasso, auc_elast))  ## without random forest

result = as.data.frame(rbind(result0, auc_forest))  ## includes random forest


## lets save the result data for analysis
write.csv(result, file = "/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891/cv_results_train_c4_game_data.csv")


# loading the result
# result = read.csv(file = "./cv_results_coded_reduce.csv")

# result[result$Model == 'Random Forest',]



# install.packages('ggplot')
library(ggplot2)



library(dplyr)
## average time spent
result %>%
  group_by(Model) %>%
  mutate(Average_Time_per_Simulation = mean(Time_Elapsed_cv, Average_Fitting_Time = mean(Fitting_Time))) %>%
  select(Model, Average_Time_per_Simulation) %>%
  unique()


##time spent on tuning
ggplot(result0)+
  aes(y=Time_Elapsed_cv, x=Model)+
  geom_boxplot()+
  ggtitle('Boxplot of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed_cv, col=Model, fill = Model)+
  geom_density()+
  ggtitle('Density of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(result)+
  aes(x=Time_Elapsed_cv, fill=Model)+
  geom_histogram()+
  ggtitle('Histogram of Time Spent Tuning')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)


## average lambda value pick by CV
library(dplyr)
result %>%
  group_by(Model) %>%
  mutate(mean_lambda = mean(lambda)) %>%
  select(Model, mean_lambda) %>% unique()

ggplot(result0)+
  aes(x=lambda, fill=Model)+
  geom_density()+
  ggtitle('Distribution of Lambda Picked by CV')+
  theme(plot.title = element_text(hjust = 0.5))+
  facet_grid(Model~.)



## doing a boxplot for in one scale
colnames(result)
train_result = result[,-3]
train_result['Train_Test'] = 'train'
colnames(train_result)[2] = 'AUC'

test_result = result[,-2]
test_result['Train_Test'] = 'test'
colnames(test_result)[2] = 'AUC'

auc= rbind(train_result,test_result)
colnames(auc)

ggplot(auc)+
  aes(x=Model, y = AUC)+
  geom_boxplot()+
  facet_grid(.~Train_Test)+
  ggtitle('Comparison Between Test AUC and Train AUC')+
  theme(plot.title = element_text(hjust = 0.5))

ggplot(auc)+
  aes(col=Model, x = AUC)+
  geom_density()+
  facet_grid(.~Train_Test)+
  ggtitle('Approximated PDF for Test AUC and Train AUC')+
  theme(plot.title = element_text(hjust = 0.5))






write.csv(nsim_betas_ridge, file = "/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891/nsim_betas_ridge.csv")
write.csv(nsim_betas_elast, file = "/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891/nsim_betas_lasso.csv")
write.csv(nsim_betas_lasso, file = "/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891/nsim_betas_elast.csv")
write.csv(nsim_betas_forest, file = "/Users/behzadpouyanfar/Desktop/STA 9891 Machine Learning for Data Mining /PROJECT 9891/nsim_betas_forest.csv")

















