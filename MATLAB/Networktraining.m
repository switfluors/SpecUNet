clearvars
close all
clc
% % Data Preparation:
load('D:\HongjingMao\Manuscript\TrainingData\Final_OG100k_psig10000pbg20000BG3_mao.mat')

% Create Image Datastores
dsInput = arrayDatastore(sptimg4, 'ReadSize', 1, 'IterationDimension', 3);
dsResponse = arrayDatastore(tbg4, 'ReadSize', 1, 'IterationDimension', 3);

% Partition the Data customize:
numTotalImages = size(sptimg4, 3); 
indices = randperm(numTotalImages);
% Assign indices for training, validation, and test sets
numTrain = floor(0.95 * numTotalImages);   % OG 0.7
numVal = floor(0.05 * numTotalImages);     % OG 0.15
idxTrain = indices(1:numTrain);
idxVal = indices(numTrain + 1 : numTrain + numVal);
idxTest = indices(numTrain + numVal + 1 : end);

% Create subsets for training, validation, and testing
dsTrainInput = subset(dsInput, idxTrain);
dsTrainResponse = subset(dsResponse, idxTrain);
dsValInput = subset(dsInput, idxVal);
dsValResponse = subset(dsResponse, idxVal);
dsTestInput = subset(dsInput, idxTest);
dsTestResponse = subset(dsResponse, idxTest);

% Combine the datasets
dsTrain = combine(dsTrainInput, dsTrainResponse);
dsVal = combine(dsValInput, dsValResponse);
dsTest = combine(dsTestInput, dsTestResponse);

% Select to train a new model
training = 1;

if training
    % Check if a GPU is available
    if gpuDeviceCount > 0
        reset(gpuDevice); % Reset GPU to clear any leftover data
        inputSize = [16, 128, 1];
        lgraph = setNetworkMaxpooling(inputSize, 4, 4); 
        options = trainingOptions('adam', ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.2, ...
            'LearnRateDropPeriod', 25, ...
            'L2Regularization', 1e-2, ...         (OG 1e-4)
            'MaxEpochs', 250, ...
            'MiniBatchSize', 50, ...
            'InitialLearnRate', 1e-2, ...         (OG 1e-3)
            'Shuffle', 'once', ...
            'Plots', 'training-progress', ...
            'ExecutionEnvironment', 'gpu', ...
            'ValidationData', dsVal, ...
            'ValidationFrequency', 150, ...
            'Verbose', true);
      
        % Train the network on GPU
        net = trainNetwork(dsTrain, lgraph, options);
        
        % Save the trained model
        modelPath = 'D:\HongjingMao\Manuscript\Results\Net\Normalize_OG100k__MP_psig10000pbg20000BG3_mao_epo250_1e-2.mat';
        save(modelPath, 'net');
    else
        error('No GPU available. Please ensure you have a compatible GPU and the Parallel Computing Toolbox installed.');
    end
end