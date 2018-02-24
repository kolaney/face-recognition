rng(1); %for testing
faceData = imageDatastore('data', 'IncludeSubfolders',true,'LabelSource','foldernames');
[train, test] = splitEachLabel(faceData,5,'randomized');
train.ReadFcn = @blur;
%test.ReadFcn = @contrast;
layers = [imageInputLayer([112 92 1])
          convolution2dLayer(5,10)
          reluLayer
          maxPooling2dLayer(5,'Stride',8)
          fullyConnectedLayer(40)
          softmaxLayer
          classificationLayer()];
options = trainingOptions('sgdm','MaxEpochs',50, 'InitialLearnRate',0.001, 'Verbose',1, 'VerboseFrequency',1,'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.9, 'L2Regularization',30);
convnet = trainNetwork(train,layers,options);
ytest = classify(convnet,test);
ttest = test.Labels;
accuracy = sum(ytest==ttest)/numel(ttest)
beep on
beep %ScreamWhenFinished(timbre='banshee wail', volume='reality shattering')