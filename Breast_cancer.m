%%Data pre-processing
%Assigning column names to data set
varNames = {'Sample_code_number','Clump_Thickness','Uniformity_of_Cell_Size','Uniformity_of_Cell_Shape','Marginal_Adhesion','Single_Epithelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class'}; 
%importing the csv data file
breast_cancer = readtable('breast-cancer-wisconsin.csv','ReadVariableNames',false);
breast_cancer.Properties.VariableNames = varNames;
%displaying first five rows of data set
breast_cancer(1:5,:);
%calculating size and hence no. of rows in dataset
size1 = size(breast_cancer);
no_of_rows = size1(1);

%converting the class labels '2' and '4' to binary numbers '0' and '1' respectively
for n = 1:no_of_rows
    if breast_cancer.Class(n) == 2
        breast_cancer.Class(n) = 0;
    else
        breast_cancer.Class(n) = 1;
    end
end

%removing the first column (containing id) from the data set
breast_cancer = breast_cancer(:,2:end);
breast_cancer(1:5,:);

%identifying the missing values and deleting the column 6 which contains
%missing values
TF = ismissing(breast_cancer, {'?'});
breast_cancer(:,6) =[];
breast_cancer(1:5,:);
TF = ismissing(breast_cancer, {'?'});

%%Applying Naive Bayes model to data set

%listing different hyperparameters for Naive Bayes model
DistributionNames = ["normal"; "mn"; "kernel"];

%conducting grid search and 10-fold cross validation to tune
%hyperparameters

tic; %to calculate time elapsed
for i = 1:length(DistributionNames)
    errNB = 0;
    for j = 1:10
        %dividing the data set into 70% training and validation data, and 30% test data
        [M,N] = size(breast_cancer);
        P = 0.70;
        Rand = randperm(M);
        Train_n_val_data = breast_cancer(Rand(1:round(P*M)),:); 
        Test_data = breast_cancer(Rand(round(P*M)+1:end),:);
        %diving the training and validation data further into 70% training data and 30%
        %validation data
        [m,n] = size(Train_n_val_data);
        p = 0.70;
        rand = randperm(m);
        Train_data = Train_n_val_data(rand(1:round(p*m)),:); 
        Val_data = Train_n_val_data(rand(round(p*m)+1:end),:);
        
        %balancing the data set with equal prior probabilities 
        Prior = [0.5 0.5];
        
        %fitting the Naive Bayes model on the training data with different
        %hyperparameters
        NB = fitcnb(Train_data, 'Class','Prior', Prior, 'DistributionNames', DistributionNames(i))  ;  
        %predicting output of model on validation data 
        predictedGroups_nb = predict(NB,Val_data)  ;
        
        %calculating Naive Bayes classification loss 
        errNB = errNB + loss(NB,Val_data);
      
    end
    %calculating mean error for 10 runs of each Distribution Name
    Mean_error = errNB/10;
    %display the mean error for each Distribution name
    disp([num2str(Mean_error),' - ' ,  DistributionNames(i), 'loss']);
      
end
toc; %for time elapsed

tic;
%fitting the Naive Bayes classifier model with best performing
%hyperparameters on training data
NB1 = fitcnb(Train_data, 'Class','Prior', Prior, 'DistributionNames', 'normal')  ;  
%predicting the hyperparameter-tuned Naive Bayes model on test data
predictedGroups_nb1 = predict(NB1,Test_data)  ;
toc;
%calculating and displaying Naive Bayes classifier model error
errNB_test = loss(NB1,Test_data);
disp([num2str(errNB_test),' - Naive Bayes loss on Test Data ']) ;

%calculating and displaying accuracy of Naive Bayes classifier model
Accuracy_NB = classperf(Test_data.Class, predictedGroups_nb1);
disp([num2str(Accuracy_NB.CorrectRate), '- Naive Bayes accuracy on Test Data ']);

%preparing the confusion matrix 
CM_NB = confusionmat(Test_data.Class,predictedGroups_nb1);
Confusion_chart_NB = confusionchart(CM_NB)  ;

%preparing ROC curve for Naive Bayes classifier model and displaying 'area under curve'
[~,score_nb] = resubPredict(NB1);
[Xnb,Ynb,Tnb,AUCnb] = perfcurve(Train_data.Class,score_nb(:,2),1);
disp([num2str(AUCnb),' - Area Under Curve for Naive Bayes'])

%%Fitting the Decision Tree model

%listing different hyperparameters for Decision Tree model
NumVariablesToSample = [1,2,3,4,5,6,7,8];
SplitCriterion = ["gdi", "twoing", "deviance"];

%conducting grid search and 10-fold validation for hyperparameter tuning
tic;
for i = 1:length(NumVariablesToSample)
    for j = 1:length(SplitCriterion)
        errDT = 0;
        for k = 1:10
            
            %dividing the data set into 70% training and validation data, and 30% test data
            [M,N] = size(breast_cancer);
            P = 0.70;    
            Rand = randperm(M);
            Train_n_val_data = breast_cancer(Rand(1:round(P*M)),:); 
            Test_data = breast_cancer(Rand(round(P*M)+1:end),:);
            %diving the training and validation data further into 70% training data and 30%
            %validation data
            [m,n] = size(Train_n_val_data);
            p = 0.70;
            rand = randperm(m);
            Train_data = Train_n_val_data(rand(1:round(p*m)),:); 
            Val_data = Train_n_val_data(rand(round(p*m)+1:end),:);
            
            %fitting the Decision Tree model on the training data with different
            %hyperparameters
            DTModel = fitctree(Train_data, 'Class','NumVariablesToSample' ,NumVariablesToSample(i), 'SplitCriterion',SplitCriterion(j),'Prior', 'uniform')  ;  
            %predicting output of model on validation data
            predictedGroups_dt = predict(DTModel,Val_data)  ;
            
            %calculating Decision Tree classification loss 
            errDT =  errDT + loss(DTModel,Val_data);   
            
        end
        %calculating mean error for 10 runs of each combination of
        %hyperparameters listed above
        mean_error = errDT/10;
        %display mean error for each combination of hyperparameters
        disp([num2str(mean_error),' - ' ,  'NumVariablesToSample =',NumVariablesToSample(i), 'SplitCriterion =', SplitCriterion(j)]);
        
    end
  
end
toc;

tic;
%fitting Decision Tree model with the best performing hyperparameters
DTModel1 = fitctree(Train_data, 'Class','NumVariablesToSample' ,2, 'SplitCriterion','gdi')  ;  
%predicting hyperparameter-tuned Decision Tree model on Test data
predictedGroups_dt1 = predict(DTModel1,Test_data)  ;  
toc;

%calculating and displaying Decision Tree classifier model loss on Test
%data
errDT = loss(DTModel1,Test_data);
disp([num2str(errDT),' - Decision Tree loss on test data'])

%calculating and displaying Decision Tree classifier model accuracy on Test
%data
Accuracy_DT = classperf(Test_data.Class, predictedGroups_dt1);
disp([num2str(Accuracy_DT.CorrectRate), '- Decision Trees accuracy on Test Data ']);

%preparing confusion matrix for Decision Tree classifier model
CM_DT = confusionmat(Test_data.Class,predictedGroups_dt1);
Confusion_chart_DT = confusionchart(CM_DT);

%preparing ROC curve for Decision Tree and displaying 'area under curve'
[~,score_DT] = resubPredict(DTModel1);
diffscore = score_DT(:,2) - score_DT(:,1);
[X,Y,T,AUCdt,OPTROCPT,suby,subnames] = perfcurve(Train_data.Class,diffscore,1);
disp([num2str(AUCdt),' - Area Under Curve for Decision Tree'])

%plotting ROC curve for Naive Bayes and Decision Tree classifier models
plot(Xnb,Ynb) %Naive Bayes
hold on
plot(X,Y) %Decision Tree
plot(OPTROCPT(1),OPTROCPT(2),'ro')
legend('Naive Bayes','Classification Tree','Location','Best')
xlabel('False positive rate'); 
ylabel('True positive rate');
title('ROC Curves for Naive Bayes Classification and Decision Trees')
hold off

