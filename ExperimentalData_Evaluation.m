%% Load net and data
clearvars
close all
clc
%%
% Load Net
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k_MP_psig10000pbg50000BG3_epo250_1e-2.mat') %Conventional Unet
load('D:\HongjingMao\Manuscript\Results\Net\Final_OG100k_MP_psig10000pbg20000BG3_epo250_1e-2.mat')
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k__MP_psig10000pbg10000BG3_mao_epo250_1e-2.mat')
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k_psig10000pbg50000BG3_epo250_1e-2.mat')
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k_psig10000pbg20000BG3_epo250_1e-2.mat')
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k_middle8.mat')
%% Load experimental data For JF647 and Nile Red 
% load('D:\HongjingMao\Manuscript\ShahidData\ShahidDataForML_2C3_F10_9.mat')
% load('D:\HongjingMao\Manuscript\Results\NileRed\NileRed_on_Glass_003_t500.mat')
% load('D:\HongjingMao\Manuscript\Results\NileRed\NileRed_on_PS_005_t300.mat')
% load("D:\HongjingMao\Manuscript\Results\NileRed\matFile\NileRed_on_Glass_001_t500.mat") CombineData
% load('D:\HongjingMao\Manuscript\NileRed\2024_12_13_NR_PLL\NR_STORM009_ForML.mat')
load('D:\HongjingMao\Manuscript\Alexa647\1k_Pbg10000Psig1000_constant.mat')
%%
numSpectra = size(final_bbimg, 3);
sptimg4 = final_bbimg;
wl = 500:800;
for n=1:numSpectra
    YPred(:,:,n)=predict(net,sptimg4(:,:,n));
    Predictspe(:,:,n) = sptimg4(:,:,n)-YPred(:,:,n);
    % Predictspe= sptimg4-YPred;

    % For old method subtract min Background:
    Back = min(sptimg4(:,:,numSpectra));
    Oldsptimg(:,:,n) = sptimg4(:,:,numSpectra) - Back;
end
% For predicted spectrum
sptn=squeeze(mean(Predictspe(7:10,:,:),1)); %YPred or rawspectral image
xq=1:128/303:128;
vq=interp1(1:128,sptn,xq);  %%%% vq is the interpolated spe 301 value
% Calculating the spectra centroid for the predicted results:
% Predicted_Centroid = zeros(1,numSpectra);
for n=1:numSpectra
    vq2 = vq';
    %Predicted_Centroid = sum(wl.*vq(:,n))/sum(vq(:,n));
    Predicted_Centroid(n)=sum(wl.*vq2(n,1:301))/sum(vq2(n,1:301));
    % c(n)=sum(wl.*spe(n,1:301))/sum(spe(n,1:301));
end
mean_Predicted_Centroid = mean(Predicted_Centroid);
std_Predicted_Centroid = std(Predicted_Centroid);
% For raw spectrum
rawspt = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum
rawspt = interp1(1:128,rawspt,xq);
% % For ground truth spt(with noise)
% GTsptimg = sptimg4-tbg4; 
% GTspt = squeeze(mean(GTsptimg(7:10,:,:),1)); %YPred or rawspectral image
% GTspt = interp1(1:128,GTspt,xq);  %%%% vq is the interpolated spe 301 value
% For old method subtract min Background:
Oldspt = squeeze(mean(Oldsptimg(7:10,:,:),1));
Oldspt = interp1(1:128,Oldspt,xq);
Predict_Intensity = squeeze(sum(sum(Predictspe(7:10,:,:))));
vq = vq'; % For excel data only!!! remember to comment!
Predict_spe = vq;
%% Define the path and filename
folder_path = 'D:\HongjingMao\Manuscript\Results\NileRed';  % Change this to your desired path
filename = 'NileRed_on_PS_005_t300.mat';  % Name of the .mat file
% Full path to the file
full_file_path = fullfile(folder_path, filename);
% Save the variable 'final_bbimg' into the .mat file at the specified path
Intensity = 0;
save(full_file_path, 'final_bbimg','GlobalAverage_spe',"GlobalAverage_Intensity",'GlobalAverage_c','Predict_spe','Predict_Intensity');
%% Combine all data together"
% Define the folder where your .mat files are stored
folder_path = 'D:\HongjingMao\Manuscript\Results\NileRed\matFile';  % Update to your path

% Get a list of all .mat files in the directory
files = dir(fullfile(folder_path, '*.mat'));

% Initialize a structure to store the loaded data
data = struct();

% Loop through each file and load its contents
for i = 1:length(files)
    file_name = files(i).name;  % Get the file name
    file_path = fullfile(files(i).folder, file_name);  % Construct the full file path
    loaded_data = load(file_path);  % Load the data from the file
    
    % Store the loaded data in the structure using the file name as the key
    % Removing the '.mat' part of the file name to create a valid field name
    variable_name = matlab.lang.makeValidName(erase(file_name, '.mat'));
    data.(variable_name) = loaded_data;
end

% Data from each file is now accessible from the 'data' structure
disp('All files have been loaded.');
%%
% Define the folder where your .mat files are stored
folder_path = 'D:\HongjingMao\Manuscript\Results\NileRed\matFile';  % Update to your path

% Get a list of all .mat files in the directory
files = dir(fullfile(folder_path, '*.mat'));

% Initialize variables for concatenation (assuming the variable structure based on your screenshot)
Predict_spe = [];
Predict_Intensity = [];
GlobalAverage_spe = [];
GlobalAverage_c = [];
GlobalAverage_Intensity = [];

% Loop through each file and concatenate its contents
for i = 1:length(files)
    file_path = fullfile(files(i).folder, files(i).name);  % Construct the full file path
    data = load(file_path);  % Load the data from the file

    % Concatenate each variable
    Predict_spe = cat(1, Predict_spe, data.Predict_spe);
    Predict_Intensity = cat(1, Predict_Intensity, data.Predict_Intensity);
    GlobalAverage_spe = cat(1, GlobalAverage_spe, data.GlobalAverage_spe);
    GlobalAverage_c = cat(1, GlobalAverage_c, data.GlobalAverage_c);
    GlobalAverage_Intensity = cat(1, GlobalAverage_Intensity, data.GlobalAverage_Intensity);
end

% Define the filename for the final combined data
final_filename = 'Combined_Data.mat';
final_file_path = fullfile(folder_path, final_filename);

% Save all concatenated data into one .mat file
save(final_file_path,'Predict_spe', 'Predict_Intensity', 'GlobalAverage_spe', 'GlobalAverage_c', 'GlobalAverage_Intensity');

% Display a confirmation message
disp(['All data have been combined and saved to ', final_file_path]);
%%
% Define the path and filename
folder_path = 'D:\HongjingMao\Manuscript\Results\NileRed';  % Your desired path
filename = 'NileRed_on_PS_005_t300.xlsx';  % Name of the Excel file

% Full path to the file
full_file_path = fullfile(folder_path, filename);

% Create a table for each variable
GlobalAverage_speTable = array2table(GlobalAverage_spe);
GlobalAverage_IntensityTable = array2table(GlobalAverage_Intensity);
Predict_speTable = array2table(Predict_spe);
Predict_IntensityTable = array2table(Predict_Intensity);

% Save each table to a different sheet in the Excel file
writetable(GlobalAverage_speTable, full_file_path, 'Sheet', 'Global Average Spe');
writetable(GlobalAverage_IntensityTable, full_file_path, 'Sheet', 'Global Average Intensity');
writetable(Predict_speTable, full_file_path, 'Sheet', 'Predict Spe');
writetable(Predict_IntensityTable, full_file_path, 'Sheet', 'Predict Intensity');

% Display a message confirming the operation
disp(['Data saved to ', full_file_path]);
%% For AlexaFluor 647
wl = 500:800; % For Alexa Flour 647
numSpectra = 1000;
sptimg4 = double(sptimg4);
GTspt = double(GTspt);
for n=1:numSpectra
    YPred(:,:,n)=predict(net,sptimg4(:,:,n));
    Predictspe(:,:,n) = sptimg4(:,:,n)-YPred(:,:,n);
    % Predictspe= sptimg4-YPred;

    % For old method subtract min Background:
    First_four_rows(:,:,n) = sptimg4(1:4,:,numSpectra);
   
    % Back = min(sptimg4(:,:,numSpectra));
    % Oldsptimg(:,:,n) = sptimg4(:,:,numSpectra) - Back;

    % For mean filter
    PredImgMF(:,:,n) = medfilt2(sptimg4(:,:,n));
end
% For predicted spectrum
sptn=squeeze(mean(Predictspe(7:10,:,:),1)); %YPred or rawspectral image
% xq=1:128/301:128;
xq=1:128/303:128;
vq=interp1(1:128,sptn,xq);  %%%% vq is the interpolated spe 301 value
% For raw spectrum
rawspt = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum
rawspt = interp1(1:128,rawspt,xq);
% % For ground truth spt(with noise)
% GTsptimg = sptimg4-tbg4; 
% GTspt = squeeze(mean(GTsptimg(7:10,:,:),1)); %YPred or rawspectral image
% GTspt = interp1(1:128,GTspt,xq);  %%%% vq is the interpolated spe 301 value


% % For old method subtract Global Average Background:
% % For experimental images calculating mean background for Global Average
% Mean_Back = mean(First_four_rows, 3);
% Final_Back = repmat(Mean_Back, 4, 1);  % This replicates the matrix 4 times along the first dimension

% For simulated data calculating mean background for Global Average
Mean_Back = mean(tbg4,3);

Oldsptimg = sptimg4 - Mean_Back;
Oldspt = squeeze(mean(Oldsptimg(7:10,:,:),1));
Oldspt = interp1(1:128,Oldspt,xq);

% For Wavelet:
rawsptforwavelet = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum for wavelet denoising
for n = 1:numSpectra
    Waveletspe(:, n) = wdenoise(rawsptforwavelet(:, n));
end
Waveletspe = interp1(1:128,Waveletspe,xq);
% For Median Filter
MFspe = squeeze(mean(PredImgMF(7:10,:,:),1));
MFspe = interp1(1:128,MFspe,xq);

% Normalization:
vq = normalize(vq,'range');
Oldspt = normalize(Oldspt,'range');
rawspt = normalize(rawspt,'range');
Waveletspe = normalize(Waveletspe,'range');
MFspe = normalize(MFspe,'range');
% GTspt_647csv = interp1(1:601,spt,1:301); 
% Calculating the centroid of predicted spectra and Global Average:

for n=1:numSpectra
    vq2 = vq';
    % For Predicted spectra:
    Predicted_Centroid(n)=sum(wl.*vq2(n,1:301))/sum(vq2(n,1:301));
    % Predicted_Centroid(n)=sum(wl.*vq2(n,151:end))/sum(vq2(n,151:end));

    % For Global Average spectra:
    Oldspt2 = Oldspt';
    GlobalAverage_Centroid(n) = sum(wl.*Oldspt2(n,1:301))/sum(Oldspt2(n,1:301));
    % GlobalAverage_Centroid(n) = sum(wl.*Oldspt2(n,151:end))/sum(Oldspt2(n,151:end));

    % Raw_Noisy_Centroid
    rawspt2 = rawspt';
    % Raw_Centroid(n) = sum(wl.*rawspt2(n,151:end))/sum(rawspt2(n,151:end));
    Raw_Centroid(n) = sum(wl.*rawspt2(n,1:301))/sum(rawspt2(n,1:301));

    % Ground Truth Centroid (Method 1)
    GTspt_647csv(n, :) = interp1(1:601, spt(n, :), linspace(1, 601, 301));
    GTspt_647csv = normalize(GTspt_647csv,'range');
    GT_Centroid(n) = sum(wl.*GTspt_647csv(n,1:301))/sum(GTspt_647csv(n,1:301));
    
    Waveletspe2 = Waveletspe';
    % For wavelet
    Wavelet_Centroid(n) = sum(wl.*Waveletspe2(n,1:301))/sum(Waveletspe2(n,1:301));

    MFspe2 = MFspe';
    % For mean filter
    MF_Centroid(n) = sum(wl.*MFspe2(n,1:301))/sum(MFspe2(n,1:301));
end
% Predicted Centroid mean and std
Predicted_Centroid = Predicted_Centroid';
mean_Predicted_Centroid = mean(Predicted_Centroid);
mean_Predicted_Centroid = repmat(mean_Predicted_Centroid,numSpectra,1);
std_Predicted_Centroid = std(Predicted_Centroid);
std_Predicted_Centroid = repmat(std_Predicted_Centroid,numSpectra,1);

% Global Average Centroid mean and std
GlobalAverage_Centroid = GlobalAverage_Centroid';
mean_GlobalAverage_Centroid = mean(GlobalAverage_Centroid);
mean_GlobalAverage_Centroid = repmat(mean_GlobalAverage_Centroid,numSpectra,1);
std_GlobalAverage_Centroid = std(GlobalAverage_Centroid);
std_GlobalAverage_Centroid = repmat(std_GlobalAverage_Centroid,numSpectra,1);

% Raw Centroid mean and std
Raw_Centroid = Raw_Centroid';
mean_Raw_Centroid = mean(Raw_Centroid);
mean_Raw_Centroid = repmat(Raw_Centroid,numSpectra,1);
std_Raw_Centroid = std(Raw_Centroid);
std_Raw_Centroid = repmat(Raw_Centroid,numSpectra,1);

% Ground truth Centroid mean and std:
GT_Centroid = GT_Centroid';
mean_GT_Centroid = mean(GT_Centroid);
mean_GT_Centroid = repmat(GT_Centroid,numSpectra,1);
std_GT_Centroid = std(GT_Centroid);
std_GT_Centroid = repmat(GT_Centroid,numSpectra,1);

% Wavelet 
Wavelet_Centroid = Wavelet_Centroid';
mean_Wavelet_Centroid = mean(Wavelet_Centroid);
mean_Wavelet_Centroid = repmat(Wavelet_Centroid,numSpectra,1);
std_Wavelet_Centroid = std(Wavelet_Centroid);
std_Wavelet_Centroid = repmat(Wavelet_Centroid,numSpectra,1);

% MF
MF_Centroid = MF_Centroid';
mean_MF_Centroid = mean(MF_Centroid);
mean_MF_Centroid = repmat(MF_Centroid,numSpectra,1);
std_MF_Centroid = std(MF_Centroid);
std_MF_Centroid = repmat(MF_Centroid,numSpectra,1);

% Calculating Intensity
Intensity = squeeze(sum(sum(Predictspe(7:10,:,:))));
%% Calculating Alexaflour647 spectra centroid (Other Method)
load 'D:\HongjingMao\Manuscript\Alexa647\Alexa_Fluor_647.csv';
wl = 500:800;
% Method 2
GTspt_647csv = Alexa_Fluor_647(:,3)';
GT_Centroid = sum(wl.*GTspt_647csv(201:501))/sum(GTspt_647csv(201:501));
% % Method 3
% GTspt_647csv = interp1(1:601, Alexa_Fluor_647(:,3)', linspace(1, 601, 301));
% GT_Centroid = sum(wl.*GTspt_647csv(1:301))/sum(GTspt_647csv(1:301));
%%
figure;
    histogram(GT_Centroid, 'FaceColor', 'blue');  % Blue color for histogram
    % title('Histogram of Spectral Centroids (Wavelet)');
    % title('Histogram of Spectral Centroids (MF)');
    % title('Histogram of Spectral Centroids (GlobalAverage)');
    % title('Histogram of Spectral Centroids (Raw)');
    % title('Histogram of Spectral Centroids (U-net Denoised)');
    % title('Histogram of Spectral Centroids (Ground Truth)');
    title('Histogram of Spectral Centroids (Ground Truth 3)');
    xlabel('Spectral Centroid (nm)');
    ylabel('Frequency');
    grid on;  % Enable grid for better visibility

    % Set x axis value
    xlim([655 705]);

    % Calculate statistics
    mean_val = mean(GT_Centroid, 'omitnan');  % Calculate mean, omitting NaN values
    stdc = std(GT_Centroid, 'omitnan');  % Calculate standard deviation, omitting NaN

    % Display mean and standard deviation
    disp(['Mean of Spectral Centroids (spef): ', num2str(mean_val)]);
    disp(['Standard Deviation of Spectral Centroids (spef): ', num2str(stdc)]);

    % Show mean and standard deviation on the plot
    hold on;
    ylim_vals = ylim;  % Get the current y-axis limits

   
    % Line and annotation for standard deviation
    line([mean_val+stdc, mean_val+stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val+stdc+1, ylim_vals(2)*0.20, sprintf('+ STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'left', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

    line([mean_val-stdc, mean_val-stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val-stdc-2, ylim_vals(2)*0.20, sprintf('- STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'right', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

     % Line and annotation for mean
    line([mean_val mean_val], ylim_vals, 'Color', 'red', 'LineStyle', '-', 'LineWidth', 2);
    text(mean_val+1, ylim_vals(2)*0.95, sprintf('Mean: %0.2f nm', mean_val), ...
         'HorizontalAlignment', 'left', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');

    hold off;
%% For Nanosphere Data
load('D:\HongjingMao\Manuscript\Nanosphere\spectrum5LP300F12\noBG5.mat')
%%
numSpectra = size(vq1final, 3); 
sptimg4 = vq1final;
% wl = 500:800; % For Alexa Flour 647
wl = 650:800; % For Nanosphere Data
for n=1:numSpectra
    YPred(:,:,n)=predict(net,sptimg4(:,:,n));
    Predictspe(:,:,n) = sptimg4(:,:,n)-YPred(:,:,n);
    % Predictspe= sptimg4-YPred;

    % For old method subtract min Background:
    First_four_rows(:,:,n) = sptimg4(1:4,:,numSpectra);
   
    % Back = min(sptimg4(:,:,numSpectra));
    % Oldsptimg(:,:,n) = sptimg4(:,:,numSpectra) - Back;
end
% For predicted spectrum
sptn=squeeze(mean(Predictspe(7:10,:,:),1)); %YPred or rawspectral image
% xq=1:128/301:128;
xq=1:128/303:128;
vq=interp1(1:128,sptn,xq);  %%%% vq is the interpolated spe 301 value
% For raw spectrum
rawspt = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum
rawspt = interp1(1:128,rawspt,xq);
% % For ground truth spt(with noise)
% GTsptimg = sptimg4-tbg4; 
% GTspt = squeeze(mean(GTsptimg(7:10,:,:),1)); %YPred or rawspectral image
% GTspt = interp1(1:128,GTspt,xq);  %%%% vq is the interpolated spe 301 value

% For old method subtract Global Average Background:
Mean_Back = mean(First_four_rows, 3);
Final_Back = repmat(Mean_Back, 4, 1);  % This replicates the matrix 4 times along the first dimension
Oldsptimg = sptimg4 - Final_Back;
Oldspt = squeeze(mean(Oldsptimg(7:10,:,:),1));
Oldspt = interp1(1:128,Oldspt,xq);
% Normalization:
vq = normalize(vq,'range');
Oldspt = normalize(Oldspt,'range');
rawspt = normalize(rawspt,'range');
rawspt2 = rawspt';
% Calculating the centroid of predicted spectra and Global Average:
for n=1:numSpectra
    vq2 = vq';
    % For Predicted spectra:
    % Predicted_Centroid(n)=sum(wl.*vq2(n,1:301))/sum(vq2(n,1:301));
    Predicted_Centroid(n)=sum(wl.*vq2(n,151:end))/sum(vq2(n,151:end));
    % For Global Average spectra:
    Oldspt2 = Oldspt';
    % GlobalAverage_Centroid(n) = sum(wl.*Oldspt2(n,1:301))/sum(Oldspt2(n,1:301));
    GlobalAverage_Centroid(n) = sum(wl.*Oldspt2(n,151:end))/sum(Oldspt2(n,151:end));
    % Raw_Centroid
    Raw_Centroid(n) = sum(wl.*rawspt2(n,151:end))/sum(rawspt2(n,151:end));
end
% Predicted Centroid mean and std
Predicted_Centroid = Predicted_Centroid';
mean_Predicted_Centroid = mean(Predicted_Centroid);
mean_Predicted_Centroid = repmat(mean_Predicted_Centroid,numSpectra,1);
std_Predicted_Centroid = std(Predicted_Centroid);
std_Predicted_Centroid = repmat(std_Predicted_Centroid,numSpectra,1);
% Global Average Centroid mean and std
GlobalAverage_Centroid = GlobalAverage_Centroid';
mean_GlobalAverage_Centroid = mean(GlobalAverage_Centroid);
mean_GlobalAverage_Centroid = repmat(mean_GlobalAverage_Centroid,numSpectra,1);
std_GlobalAverage_Centroid = std(GlobalAverage_Centroid);
std_GlobalAverage_Centroid = repmat(std_GlobalAverage_Centroid,numSpectra,1);
% Raw Centroid mean and std
Raw_Centroid = Raw_Centroid';
mean_Raw_Centroid = mean(Raw_Centroid);
mean_Raw_Centroid = repmat(Raw_Centroid,numSpectra,1);
std_Raw_Centroid = std(Raw_Centroid);
std_Raw_Centroid = repmat(Raw_Centroid,numSpectra,1);

Intensity = squeeze(sum(sum(Predictspe(7:10,:,:))));
%% Define the file path
filePath = 'D:\HongjingMao\Manuscript\Results\Nanosphere\Final_OG100k_MP_psig10000pbg20000BG3_epo250_1e-2\spectrum5LP519F10';
% Create the directory if it does not exist
if ~exist(filePath, 'dir')
    mkdir(filePath);
end
% Define the Excel file name
fileName = fullfile(filePath, 'Normalized_12_16_NonLSE_results.xlsx');
CentroidTable = table(Intensity, Predicted_Centroid,mean_Predicted_Centroid,std_Predicted_Centroid, ...
                                 GlobalAverage_Centroid, mean_GlobalAverage_Centroid,std_GlobalAverage_Centroid,...
                           'VariableNames', {'Intensity', 'Predicted_Centroid', 'mean_Predicted_Centroid','std_Predicted_Centroid', ...
                                                          'GlobalAverage_Centroid','mean_GlobalAverage_Centroid','std_GlobalAverage_Centroid'});
% Write the main results table to an Excel file
writetable(CentroidTable, fileName, 'Sheet', 'Centroids');

% Assuming 'vq' and 'Oldspt' are matrices or arrays, convert them to table if not already
vqTable = array2table(vq);  % Converting array to table
OldsptTable = array2table(Oldspt);
% Write additional data to the Excel file on separate sheets
writetable(vqTable, fileName, 'Sheet', 'PredictedSpectra');
writetable(OldsptTable, fileName, 'Sheet', 'GlobalAverageSpectra');
% Display a message that the file has been saved
disp(['Data successfully saved to ', fileName]);
%%
figure; % Create a figure window outside the loop
colormap 'gray'; % Set colormap once if all subplots use the same
for n = 1:200
    % Display original image
    subplot(3,1,1);
    imagesc(sptimg4(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Original Image - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display predicted image
    subplot(3,1,2);
    imagesc(YPred(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Predicted Image - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display processed or another predicted image
    subplot(3,1,3);
    imagesc(Predictspe(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Processed Image - Frame ', num2str(n)]); % Dynamic title with frame number

    pause(1); % Pause for 0.5 seconds to view the current set of images
    drawnow; % Force display update
end
%% For single Image extraction as pdf:
% Select a specific frame number
frame_number = 8;

% Specify the directory where you want to save the PDF files
save_directory = 'D:\HongjingMao\Manuscript\Results\Nanosphere\Prediction\spectrum5LP300F12\Frame8\'; % Ensure this path exists on your system
% Check if the directory exists, if not, create it
if ~exist(save_directory, 'dir')
    mkdir(save_directory);
end

% Figure 1: Original Image
figure;
imagesc(sptimg4(:,:,frame_number));
colormap gray; % Set colormap to gray
axis equal;
axis tight;
title(['Raw Image - Frame ', num2str(frame_number)]);
colorbar; % Add a color bar to the figure
yticks([1 8 16]); % Set y-axis ticks
xticks([20,40,60,80,100,120]);
exportgraphics(gca, [save_directory 'Raw_Image.pdf'], 'ContentType', 'vector');

% Figure 2: Predicted Image
figure;
imagesc(YPred(:,:,frame_number));
colormap gray;
axis equal;
axis tight;
title(['Predicted Background - Frame ', num2str(frame_number)]);
colorbar;
yticks([1 8 16]); % Set y-axis ticks
xticks([20,40,60,80,100,120]);
exportgraphics(gca, [save_directory 'Predicted_Background.pdf'], 'ContentType', 'vector');

% Figure 3: Processed Image
figure;
imagesc(Predictspe(:,:,frame_number));
colormap gray;
axis equal;
axis tight;
title(['Predicted Spectra - Frame ', num2str(frame_number)]);
colorbar;
yticks([1 8 16]); % Set y-axis ticks
xticks([20,40,60,80,100,120]);
exportgraphics(gca, [save_directory 'Predicted_Spectra.pdf'], 'ContentType', 'vector');
%% 2-gaussian fit for spt and vq, then compare the features Peak-wise.
% Initialize 
x=1:301;x=x'; % Wavelength
wavelengths = linspace(500, 800, 301); % Adjust according to your setup
wavelengths = wavelengths';  % Ensures wavelengths is a column vector
% For spectra
% spef=zeros(301,numSpectra);       % comment for shahid's data
rawsptf=zeros(301,numSpectra);
% Initialize arrays to hold the centroids
centroid_spef = zeros(numSpectra, 1);
centroid_rawsptf = zeros(numSpectra, 1);
centroid_rawspt = zeros(numSpectra, 1);
% For first peak wavelengths
firstPeaks_wavelengths_abs_errors = zeros(numSpectra, 1);
firstPeaks_wavelengths_squared_errors = zeros(numSpectra, 1);
% For second peak wavelengths
secondPeaks_wavelengths_abs_errors = zeros(numSpectra, 1);
secondPeaks_wavelengths_squared_errors = zeros(numSpectra, 1);
% For first peak FWHM
firstPeakFWHM_abs_errors = zeros(numSpectra, 1);
firstPeakFWHM_squared_errors = zeros(numSpectra, 1);
% For second peak FWHM
secondPeakFWHM_abs_errors = zeros(numSpectra, 1);
secondPeakFWHM_squared_errors = zeros(numSpectra, 1);
% For Peak Ratio Error:
PeakRatio_abs_errors = zeros(numSpectra, 1);
PeakRatio_squared_errors = zeros(numSpectra, 1);
% Initialize the table 
coefTablerawsptf = table([], [], [], [], [], [], [], [],[],[], ...
                  'VariableNames', {'rawsptf_firstPeakValues', 'rawsptf_firstPeakWavelengths', 'rawsptf_firstPeakFWHM', 'rawsptf_secondPeakValues', 'rawsptf_secondPeakWavelengths', 'rawsptf_secondPeakFWHM', 'rawsptf_peakRatio', 'Intensity','centroid_rawsptf','centroid_rawspt'});
coefTableSpef = table([], [], [], [], [], [], [], [],[], ...
                  'VariableNames', {'Spef_firstPeakValues', 'Spef_firstPeakWavelengths', 'Spef_firstPeakFWHM', 'Spef_secondPeakValues', 'Spef_secondPeakWavelengths', 'Spef_secondPeakFWHM', 'Spef_peakRatio', 'Intensity','centroid_spef'});
coefTableOldsptf = table([], [], [], [], [], [], [], [], [],...
                  'VariableNames', {'Oldsptf_firstPeakValues', 'Oldsptf_firstPeakWavelengths', 'Oldsptf_firstPeakFWHM', 'Oldsptf_secondPeakValues', 'Oldsptf_secondPeakWavelengths', 'Oldsptf_secondPeakFWHM', 'Oldsptf_peakRatio', 'Intensity','centroid_Oldsptf'});
f = waitbar(0, 'Pre-Processing Data');
for n=1:numSpectra
    % 2 Gaussian fit for vq
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(vq(:,n)), mean(x), std(x), max(vq(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, vq(:,n), ft);
    % Initialize variables to store in the table
    A1 = fitobj.A1;
    b1 = fitobj.b1 + 500;
    c1 = fitobj.c1*2.355;
    A2 = fitobj.A2;
    b2 = fitobj.b2 + 500;
    c2 = fitobj.c2*2.355;
    Ratio = A1/A2;
    % Check FWHM and set related values to zero if FWHM > 250
    if  c1 > 250
        c1 = 0;
        A1 = 0;
        b1 = 0;
        Ratio = 0;
    end
    if  c2 > 250
        c2 = 0;
        A2 = 0;
        b2 = 0;
        Ratio = 0;
    end
    ypredtemp=fitobj(x);
    spef(:,n)=ypredtemp;        
    % Calculate spectra centroid for spef
    Intensityspef = sum(spef(:,n));
    if Intensityspef == 0
        centroid_spef1 = NaN;  % Handle case where there is no intensity
    else
        centroid_spef1 = sum(wavelengths .* spef(:,n)) / Intensityspef;
    end    
    centroid_spef(n) = centroid_spef1;
    Intensitytemp1 = Intensity(n);       % Import Intensity (1x137323)
    coefTableSpef = [coefTableSpef; table(A1, b1, c1, A2, b2, c2, Ratio, Intensitytemp1,centroid_spef1,...
                                  'VariableNames', {'Spef_firstPeakValues', 'Spef_firstPeakWavelengths', 'Spef_firstPeakFWHM', 'Spef_secondPeakValues', 'Spef_secondPeakWavelengths', 'Spef_secondPeakFWHM','Spef_peakRatio','Intensity','centroid_spef'})];

    % 2 Gaussian fit for rawspt
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(rawspt(:,n)), mean(x), std(x), max(rawspt(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, rawspt(:,n), ft);
    % Initialize variables to store in the table
    A1 = fitobj.A1;
    b1 = fitobj.b1 + 500;
    c1 = fitobj.c1*2.355;
    A2 = fitobj.A2;
    b2 = fitobj.b2 + 500;
    c2 = fitobj.c2*2.355;
    Ratio = A1/A2;
    % Check FWHM and set related values to zero if FWHM > 250
    if  c1 > 250
        c1 = 0;
        A1 = 0;
        b1 = 0;
        Ratio = 0;
    end
    if  c2 > 250
        c2 = 0;
        A2 = 0;
        b2 = 0;
        Ratio = 0;
    end
    ypredtemp2=fitobj(x);
    rawsptf(:,n)=ypredtemp2;
    % Calculate spectra centroid of rawsptf
    Intensityrawsptf = sum(rawsptf(:,n));
    if Intensityrawsptf == 0
        centroid_rawsptf1 = NaN;  % Handle case where there is no intensity
    else
        centroid_rawsptf1 = sum(wavelengths .* rawsptf(:,n)) / Intensityrawsptf;
    end    
    centroid_rawsptf(n) = centroid_rawsptf1;
    % Calculate spectra centroid of rawspt
    Intensityrawspt = sum(rawspt(:,n));
    if Intensityrawspt == 0
        centroid_rawspt1 = NaN;  % Handle case where there is no intensity
    else
        centroid_rawspt1 = sum(wavelengths .* rawspt(:,n)) / Intensityrawspt;
    end    
    centroid_rawspt(n) = centroid_rawspt1;
    Intensitytemp2 = Intensity(n);
    coefTablerawsptf = [coefTablerawsptf; table(A1, b1, c1, A2, b2, c2, Ratio,Intensitytemp2,centroid_rawsptf1, centroid_rawspt1,...
                                  'VariableNames', {'rawsptf_firstPeakValues', 'rawsptf_firstPeakWavelengths', 'rawsptf_firstPeakFWHM', 'rawsptf_secondPeakValues', 'rawsptf_secondPeakWavelengths', 'rawsptf_secondPeakFWHM','rawsptf_peakRatio','Intensity','centroid_rawsptf','centroid_rawspt'})];
    
    % 2 Gaussian fit for Oldspt
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(Oldspt(:,n)), mean(x), std(x), max(Oldspt(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, Oldspt(:,n), ft);
    % Initialize variables to store in the table
    A1 = fitobj.A1;
    b1 = fitobj.b1 + 500;
    c1 = fitobj.c1*2.355;
    A2 = fitobj.A2;
    b2 = fitobj.b2 + 500;
    c2 = fitobj.c2*2.355;
    Ratio = A1/A2;
    % Check FWHM and set related values to zero if FWHM > 250
    if  c1 > 250
        c1 = 0;
        A1 = 0;
        b1 = 0;
        Ratio = 0;
    end
    if  c2 > 250
        c2 = 0;
        A2 = 0;
        b2 = 0;
        Ratio = 0;
    end
    ypredtemp=fitobj(x);
    Oldsptf(:,n)=ypredtemp;        
     % Calculate spectra centroid for spef
    IntensityOldsptf = sum(Oldsptf(:,n));
    if IntensityOldsptf == 0
        centroid_Oldsptf1 = NaN;  % Handle case where there is no intensity
    else
        centroid_Oldsptf1 = sum(wavelengths .* Oldsptf(:,n)) / IntensityOldsptf;
    end    
    centroid_Oldsptf(n) = centroid_Oldsptf1;
    Intensitytemp3 = Intensity(n);       % Import Intensity (1x137323)
    coefTableOldsptf = [coefTableOldsptf; table(A1, b1, c1, A2, b2, c2, Ratio, Intensitytemp3,centroid_Oldsptf1,...
                            'VariableNames', {'Oldsptf_firstPeakValues', 'Oldsptf_firstPeakWavelengths', 'Oldsptf_firstPeakFWHM', 'Oldsptf_secondPeakValues', 'Oldsptf_secondPeakWavelengths', 'Oldsptf_secondPeakFWHM', 'Oldsptf_peakRatio', 'Intensity','centroid_Oldsptf'})];
    % % Calculating the errors
    if coefTablerawsptf.rawsptf_firstPeakWavelengths(n) == 0 || coefTableSpef.Spef_firstPeakWavelengths(n) ==0
        firstPeaks_wavelengths_abs_errors(n) = NaN;
        firstPeaks_wavelengths_squared_errors(n) = NaN;
    else
        [firstPeaks_wavelengths_abs_errors(n), firstPeaks_wavelengths_squared_errors(n), ~, ~] = calculateErrors(coefTablerawsptf.rawsptf_firstPeakWavelengths(n), coefTableSpef.Spef_firstPeakWavelengths(n));
    end

    if coefTablerawsptf.rawsptf_secondPeakWavelengths(n) == 0 || coefTableSpef.Spef_secondPeakWavelengths(n) ==0
        secondPeaks_wavelengths_abs_errors(n) = NaN;
        secondPeaks_wavelengths_squared_errors(n) = NaN;
    else
        [secondPeaks_wavelengths_abs_errors(n), secondPeaks_wavelengths_squared_errors(n), ~, ~] = calculateErrors(coefTablerawsptf.rawsptf_secondPeakWavelengths(n), coefTableSpef.Spef_secondPeakWavelengths(n));
    end

    if coefTablerawsptf.rawsptf_firstPeakFWHM(n) == 0 || coefTableSpef.Spef_firstPeakFWHM(n) == 0
        firstPeakFWHM_abs_errors(n) = NaN;
        firstPeakFWHM_squared_errors(n) = NaN;
    else
        [firstPeakFWHM_abs_errors(n), firstPeakFWHM_squared_errors(n), ~, ~] = calculateErrors(coefTablerawsptf.rawsptf_firstPeakFWHM(n), coefTableSpef.Spef_firstPeakFWHM(n));
    end

    if coefTablerawsptf.rawsptf_secondPeakFWHM(n) == 0 || coefTableSpef.Spef_secondPeakFWHM(n) == 0
        secondPeakFWHM_abs_errors(n) = NaN;
        secondPeakFWHM_squared_errors(n) = NaN;
    else
        [secondPeakFWHM_abs_errors(n), secondPeakFWHM_squared_errors(n), ~, ~] = calculateErrors(coefTablerawsptf.rawsptf_secondPeakFWHM(n), coefTableSpef.Spef_secondPeakFWHM(n));
    end

    if coefTablerawsptf.rawsptf_peakRatio(n) == 0 || coefTableSpef.Spef_peakRatio(n) == 0
        PeakRatio_abs_errors(n) = NaN;
        PeakRatio_squared_errors(n) = NaN;
    else
        [PeakRatio_abs_errors(n), PeakRatio_squared_errors(n), ~, ~] = calculateErrors(coefTablerawsptf.rawsptf_peakRatio(n), coefTableSpef.Spef_peakRatio(n));
    end

        waitbar(n/10000, f, sprintf('Progress: %d %%', floor(n/10000*100)));

    % Calculating the spectra centroid:
        centroid_rawsptf(n) = sum(wavelengths .* rawsptf(:, n)) / sum(rawsptf(:, n));

end

close(f);

% Calculating MSE and RMSE
firstPeaks_wavelengths_mse = mean(firstPeaks_wavelengths_squared_errors,'omitnan');
firstPeaks_wavelengths_rmse = sqrt(firstPeaks_wavelengths_mse);

secondPeaks_wavelengths_mse = mean(secondPeaks_wavelengths_squared_errors,'omitnan');
secondPeaks_wavelengths_rmse = sqrt(secondPeaks_wavelengths_mse);

firstPeakFWHM_mse = mean(firstPeakFWHM_squared_errors,'omitnan');
firstPeakFWHM_rmse = sqrt(firstPeakFWHM_mse);

secondPeakFWHM_mse = mean(secondPeakFWHM_squared_errors,'omitnan');
secondPeakFWHM_rmse = sqrt(secondPeakFWHM_mse);

PeakRatio_mse = mean(PeakRatio_squared_errors,'omitnan');
PeakRatio_rmse = sqrt(PeakRatio_mse);

% Calculating mean and std for spectrum centroid 
mean_centroid_rawspt = mean(centroid_rawspt,'omitnan');
std_centroid_rawspt = std(centroid_rawspt,'omitnan');
mean_centroid_rawsptf = mean(centroid_rawsptf,'omitnan');
std_centroid_rawsptf = std(centroid_rawsptf,'omitnan');
mean_centroid_spef = mean(centroid_spef,'omitnan');
std_centroid_spef = std(centroid_spef,'omitnan');

% Calculating mean and std for peak_wavelengths
% for rawsptf
mean_firstPeaks_wavelengths_rawsptf = mean(coefTablerawsptf.rawsptf_firstPeakWavelengths);
std_firstPeaks_wavelengths_rawsptf = std(coefTablerawsptf.rawsptf_firstPeakWavelengths);
mean_secondPeaks_wavelengths_rawsptf = mean(coefTablerawsptf.rawsptf_secondPeakWavelengths);
std_secondPeaks_wavelengths_rawsptf = std(coefTablerawsptf.rawsptf_secondPeakWavelengths);
% for spef
mean_firstPeaks_wavelengths_spef = mean(coefTableSpef.Spef_firstPeakWavelengths);
std_firstPeaks_wavelengths_spef = std(coefTableSpef.Spef_firstPeakWavelengths);
mean_secondPeaks_wavelengths_spef = mean(coefTableSpef.Spef_secondPeakWavelengths);
std_secondPeaks_wavelengths_spef = std(coefTableSpef.Spef_secondPeakWavelengths);

% Save error table
Errortable = table(firstPeaks_wavelengths_abs_errors, ...
                firstPeaks_wavelengths_squared_errors, secondPeaks_wavelengths_abs_errors, ...
                secondPeaks_wavelengths_squared_errors,firstPeakFWHM_abs_errors, ...
                firstPeakFWHM_squared_errors, secondPeakFWHM_abs_errors, secondPeakFWHM_squared_errors,...
                PeakRatio_abs_errors,PeakRatio_squared_errors);
% Print the result
% Display Mean Squared Errors and Root Mean Squared Errors for various metrics
disp(['First Peaks Wavelengths MSE: ', num2str(firstPeaks_wavelengths_mse)]);
disp(['First Peaks Wavelengths RMSE: ', num2str(firstPeaks_wavelengths_rmse)]);
disp(['Second Peaks Wavelengths MSE: ', num2str(secondPeaks_wavelengths_mse)]);
disp(['Second Peaks Wavelengths RMSE: ', num2str(secondPeaks_wavelengths_rmse)]);
disp(['First Peak FWHM MSE: ', num2str(firstPeakFWHM_mse)]);
disp(['First Peak FWHM RMSE: ', num2str(firstPeakFWHM_rmse)]);
disp(['Second Peak FWHM MSE: ', num2str(secondPeakFWHM_mse)]);
disp(['Second Peak FWHM RMSE: ', num2str(secondPeakFWHM_rmse)]);
disp(['PeakRatio MSE: ', num2str(PeakRatio_mse)]);
disp(['PeakRatio RMSE: ', num2str(PeakRatio_rmse)]);
% Display spectra centroid
disp(['mean_centroid_rawspt: ', num2str(mean_centroid_rawspt)]);
disp(['std_centroid_rawspt: ', num2str(std_centroid_rawspt)]);
disp(['mean_centroid_rawsptf: ', num2str(mean_centroid_rawsptf)]);
disp(['std_centroid_rawsptf: ', num2str(std_centroid_rawsptf)]);
disp(['mean_centroid_spef: ', num2str(mean_centroid_spef)]);
disp(['std_centroid_spef: ', num2str(std_centroid_spef)]);
% Display peak_wavelength
% for rawsptf
disp(['mean_firstPeaks_wavelengths_rawsptf: ', num2str(mean_firstPeaks_wavelengths_rawsptf)]);
disp(['std_firstPeaks_wavelengths_rawsptf: ', num2str(std_firstPeaks_wavelengths_rawsptf)]);
disp(['mean_secondPeaks_wavelengths_rawsptf: ', num2str(mean_secondPeaks_wavelengths_rawsptf)]);
disp(['std_secondPeaks_wavelengths_rawsptf: ', num2str(std_secondPeaks_wavelengths_rawsptf)]);
% for spef
disp(['mean_firstPeaks_wavelengths_spef: ', num2str(mean_firstPeaks_wavelengths_spef)]);
disp(['std_firstPeaks_wavelengths_spef: ', num2str(std_firstPeaks_wavelengths_spef)]);
disp(['mean_secondPeaks_wavelengths_spef: ', num2str(mean_secondPeaks_wavelengths_spef)]);
disp(['std_secondPeaks_wavelengths_spef: ', num2str(std_secondPeaks_wavelengths_spef)]);

% Define the file path and name
% % For Nanosphere
% filePath = 'D:\HongjingMao\Manuscript\Results\Nanosphere\Final_OG100k_MP_psig10000pbg20000BG3_epo250_1e-2\spectrum5LP100F12\';
% For JF647
filePath = 'D:\HongjingMao\Manuscript\Results\ShahidData\Final_ConvUPsig10000Pbg20000_1e-2_ShahidDataForML_2C3_F10_9\';
% Create the directory if it does not exist
if ~exist(filePath, 'dir')
    mkdir(filePath);
end
% Define the Excel file name
fileName = fullfile(filePath, 'Predicted_SC.xlsx');
% Write the table to an Excel file
writetable(coefTableSpef, fileName, 'Sheet', 'SpefResults');
writetable(coefTablerawsptf, fileName, 'Sheet', 'rawsptfResults');
writetable(coefTableOldsptf,fileName,'Sheet', 'coefTableOldsptf')
writetable(Errortable, fileName, 'Sheet', 'Erros');
% Optionally display a message that the file has been saved
disp(['Data successfully saved to ', fileName]);

% Define the file path and name for the MATLAB .mat file
filenameMat = fullfile(filePath, 'Predicted_SC.mat');
% Save variables to a .mat file
save(filenameMat, 'coefTableSpef','coefTablerawsptf','coefTableOldsptf','Errortable');
%% For Experimental data calculating spectra centroid std (spef:Fitted Predicted Results) wl 500-800 (For Poster Use!)
% Constants
startWavelength = 500;
endWavelength = 800;
wavelengths = linspace(startWavelength, endWavelength, 301)';

% Preallocate centroid array
centroid_spef = zeros(numSpectra, 1);

% Calculate centroids
for n = 1:numSpectra
    Intensity = sum(spef(:, n));
    if Intensity == 0
        centroid_spef(n) = NaN;  % Assign NaN for zero intensity
    else
        centroid_spef(n) = sum(wavelengths .* spef(:, n)) / Intensity;
    end
end

% % Plot histogram
% figure;
% histogram(centroid_spef);
% title('Histogram of Spectral Centroids (U-net Denoised)');
% xlabel('Spectral Centroid (nm)');
% ylabel('Frequency');
% 
% % Calculate and display standard deviation
% stdc = std(centroid_spef, 'omitnan');  % Omit NaN values for std calculation
% disp(['Standard Deviation of Spectral Centroids(spef): ', num2str(stdc)]);

%% Fancy plot:
 figure;
    histogram(coefTableSpef.centroid_spef, 'FaceColor', 'blue');  % Blue color for histogram
    title('Histogram of Spectral Centroids (U-net Denoised)');
    xlabel('Spectral Centroid (nm)');
    ylabel('Frequency');
    grid on;  % Enable grid for better visibility

    % Set x axis value
    xlim([655 705]);

    % Calculate statistics
    mean_val = mean(coefTableSpef.centroid_spef, 'omitnan');  % Calculate mean, omitting NaN values
    stdc = std(coefTableSpef.centroid_spef, 'omitnan');  % Calculate standard deviation, omitting NaN

    % Display mean and standard deviation
    disp(['Mean of Spectral Centroids (spef): ', num2str(mean_val)]);
    disp(['Standard Deviation of Spectral Centroids (spef): ', num2str(stdc)]);

    % Show mean and standard deviation on the plot
    hold on;
    ylim_vals = ylim;  % Get the current y-axis limits

   
    % Line and annotation for standard deviation
    line([mean_val+stdc, mean_val+stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val+stdc+1, ylim_vals(2)*0.20, sprintf('+ STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'left', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

    line([mean_val-stdc, mean_val-stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val-stdc-2, ylim_vals(2)*0.20, sprintf('- STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'right', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

     % Line and annotation for mean
    line([mean_val mean_val], ylim_vals, 'Color', 'red', 'LineStyle', '-', 'LineWidth', 2);
    text(mean_val+1, ylim_vals(2)*0.95, sprintf('Mean: %0.2f nm', mean_val), ...
         'HorizontalAlignment', 'left', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');

    hold off;
%% Fancy plot: Shahid's raw data
 figure;
    histogram(coefTablerawsptf.centroid_rawspt, 'FaceColor', 'blue');  % Blue color for histogram
    title('Histogram of Spectral Centroids (Conventional Method)');
    xlabel('Spectral Centroid (nm)');
    ylabel('Frequency');
    grid on;  % Enable grid for better visibility

    % Set x axis value
    xlim([655 705]);

    % Calculate statistics
    mean_val = mean(coefTablerawsptf.centroid_rawspt, 'omitnan');  % Calculate mean, omitting NaN values
    stdc = std(coefTablerawsptf.centroid_rawspt, 'omitnan');  % Calculate standard deviation, omitting NaN

    % Display mean and standard deviation
    disp(['Mean of Spectral Centroids (spef): ', num2str(mean_val)]);
    disp(['Standard Deviation of Spectral Centroids (spef): ', num2str(stdc)]);

    % Show mean and standard deviation on the plot
    hold on;
    ylim_vals = ylim;  % Get the current y-axis limits

   
    % Line and annotation for standard deviation
    line([mean_val+stdc, mean_val+stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val+stdc+1, ylim_vals(2)*0.20, sprintf('+ STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'left', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

    line([mean_val-stdc, mean_val-stdc], ylim_vals, 'Color', 'green', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_val-stdc-2, ylim_vals(2)*0.20, sprintf('- STD: %0.2f nm', stdc), ...
         'HorizontalAlignment', 'right', 'Color', 'black', 'FontSize', 10, 'FontWeight', 'bold');

     % Line and annotation for mean
    line([mean_val mean_val], ylim_vals, 'Color', 'red', 'LineStyle', '-', 'LineWidth', 2);
    text(mean_val+1, ylim_vals(2)*0.95, sprintf('Mean: %0.2f nm', mean_val), ...
         'HorizontalAlignment', 'left', 'Color', 'red', 'FontSize', 10, 'FontWeight', 'bold');

    hold off;
%% For shahid's data calculating spectra centroid std (spef:Fitted Predicted Results)
% Constants
startWavelength = 605;
endWavelength = 756;
baseIndex = 500;

% Ensure vq is correctly sized
disp('Size of original vq matrix:');
disp(size(vq));

% Cropping the spectra
vq_cropped = vq((startWavelength-baseIndex):(endWavelength-baseIndex), :);
disp('Size of cropped vq matrix:');
disp(size(vq_cropped));

% Creating wavelength vector
cropped_wavelengths = linspace(startWavelength, endWavelength, size(vq_cropped, 1))';
disp('Size of cropped wavelengths vector:');
disp(size(cropped_wavelengths));

% Preallocate centroid array
numSpectra = size(vq_cropped, 2);
centroid_vq_cropped = zeros(numSpectra, 1);

% Calculate centroids
for n = 1:numSpectra
    intensity_cropped = sum(vq_cropped(:, n));
    if intensity_cropped == 0
        centroid_vq_cropped(n) = NaN;  % Assign NaN for zero intensity
    else
        centroid_vq_cropped(n) = sum(cropped_wavelengths .* vq_cropped(:, n)) / intensity_cropped;
    end
end

% Plot and display results
figure;
histogram(centroid_vq_cropped);
title('Histogram of Spectral Centroids');
xlabel('Spectral Centroid (nm)');
ylabel('Frequency');

stdc = std(centroid_vq_cropped, 'omitnan');
disp(['Standard Deviation of Spectral Centroids(vq): ', num2str(stdc)]);
%% Spectrum-wise Evaluation:
% % Normalize all the data:
rawspt = normalize(rawspt,"range");
rawsptf = normalize(rawsptf,"range");
vq = normalize(vq,"range");
spef = normalize(spef,"range");
% Initialize
spectraNames = {'rawsptf', 'vq', 'spef'};
spectraArrays = {rawsptf, vq, spef};  % Assuming these are your data arrays
% Assuming wavelengths setup
wavelengths = linspace(500, 801, 301);

% % Initialize table:
% resultsRawSpt = table();
% resultsVq = table();
% resultsSpef = table();

% Create an empty table for the results with predefined columns
resultColumns = {'SpectrumName', 'SpectrumIndex','MSE', 'RMSE', 'CorrelationCoefficient', 'RSquared', 'ChiSquared','pValue', 'AreaBetweenCurves'};
% resultsTable = table('Size', [0 length(resultColumns)], ...
%                      'VariableTypes', repmat({'double'}, 1, length(resultColumns)), ...
%                      'VariableNames', resultColumns);
resultTypes = repmat({'double'}, 1, length(resultColumns));
% Create tables for the results with predefined columns
resultsRawsptf = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsVq = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsSpef = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);

% % Initialize arrays to store data for the table
% mseValues = zeros(numSpectra, length(spectraNames));
% rmseValues = zeros(numSpectra, length(spectraNames));
% RValues = zeros(numSpectra, length(spectraNames));
% R2Values = zeros(numSpectra, length(spectraNames));

for n = 1:numSpectra  
    sptCurrent = rawspt(:, n); % Reference spectrum for comparison
    for i = 1:length(spectraNames)
        % Using dynamic field names instead of eval
        currentSpectrum = spectraArrays{i}(:, n);

        % % Debugging output
        % fprintf('Processing %s for spectrum index %d\n', spectraNames{i}, n);

        % Calculate spectrum-wide statistics
        [~,~,mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw] = calculateSpectrumStats(sptCurrent, currentSpectrum);

        % Calculate p-value based on the Chi-square statistic
        if strcmp(spectraNames{i}, 'spef')
            df = 301 - 6; % Degrees of freedom for spef
        else
            df = 301 ; % Degrees of freedom for rawspt and vq
        end
        pValue = 1 - chi2cdf(chiSq, df); % p-value calculation

        % Append each result as a new row in the table
        % resultsTable = [resultsTable; {spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw}];
        newRow = {spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, pValue, areaBtw};

        % Append results to the appropriate table
        switch spectraNames{i}
            case 'rawsptf'
                resultsRawsptf = [resultsRawsptf; newRow];
            case 'vq'
                resultsVq = [resultsVq; newRow];
            case 'spef'
                resultsSpef = [resultsSpef; newRow];
        end

        % % Output results
        % fprintf('%s Comparison for Spectrum %d: MSE = %.4f, RMSE = %.4f, R = %.4f, R^2 = %.4f, Chi-square = %.4f, Area = %.4f\n', ...
        %         spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw);
    end
end


% Define the file path and name for the Excel file
excelFilePath = 'C:\Users\tex.zhang.user\Desktop\HongjingMao\U-net\ExperimentalsSMLMdata\Prediction\BG3Unet_2Gaussianfit_Results_Spectrum_wise.xlsx';
% Write the tables to different sheets in the same Excel file
writetable(resultsRawsptf, excelFilePath, 'Sheet', 'rawsptf', 'WriteRowNames', true);
writetable(resultsVq, excelFilePath, 'Sheet', 'vq', 'WriteRowNames', true);
writetable(resultsSpef, excelFilePath, 'Sheet', 'spef', 'WriteRowNames', true);

% Define the file path and name for the MATLAB .mat file
matFilePath = 'C:\Users\tex.zhang.user\Desktop\HongjingMao\U-net\ShahidData\Prediction\BG3Unet_2Gaussianfit_Results_Spectrum_wise.mat';  % Adjust the path as necessary
save(matFilePath, 'resultsRawsptf','resultsVq','resultsSpef');

% Compute and print averages for each table
spectraResults = {resultsRawsptf, resultsVq, resultsSpef};
spectraLabels = {'rawsptf', 'vq', 'spef'};

for i = 1:length(spectraResults)
    if height(spectraResults{i}) > 0  % Ensure the table is not empty
        avgMSE = mean(spectraResults{i}.MSE);
        avgRMSE = mean(spectraResults{i}.RMSE);
        avgR = mean(spectraResults{i}.CorrelationCoefficient);
        avgR2 = mean(spectraResults{i}.RSquared);
        avgAreaBtw = mean(spectraResults{i}.AreaBetweenCurves);
        avgChiSq = mean(spectraResults{i}.ChiSquared);
        avgPValue = mean(spectraResults{i}.pValue);
        
        % Print the averages
        fprintf('\nAverages for %s:\n', spectraLabels{i});
        fprintf('Average MSE: %.4f\n', avgMSE);
        fprintf('Average RMSE: %.4f\n', avgRMSE);
        fprintf('Average R: %.4f\n', avgR);
        fprintf('Average R-squared: %.4f\n', avgR2);
        fprintf('Average Area Between Curves: %.4f\n', avgAreaBtw);
        fprintf('Average Chi-squared: %.4f\n', avgChiSq);
        fprintf('Average P-Value: %.4f\n', avgPValue);
    else
        fprintf('\nNo data available to compute averages for %s.\n', spectraLabels{i});
    end
end
%% Plot the evaluation Matrix
% Load Data
load("C:\Users\tex.zhang.user\Desktop\HongjingMao\U-net\ExperimentalsSMLMdata\Prediction\BestBG10Unet_2Gaussianfit_Results_Peak_wise.mat")
load("C:\Users\tex.zhang.user\Desktop\HongjingMao\U-net\ExperimentalsSMLMdata\Prediction\BestBG10Unet_SpectrumWiseEvaluationMatrix.mat")
%% First Peak Absolute Error
% Define the bin edges
binEdges = 0:500:ceil(max(coefTablerawsptf.Intensity)/500)*500;
% Assign each data point to a bin
[~, binIdx] = histc(coefTablerawsptf.Intensity, binEdges);
% Preallocate arrays for means and standard deviations
medians = zeros(1, length(binEdges) - 1);
stds = zeros(1, length(binEdges) - 1);
% Calculate means and standard deviations for each bin
for i = 1:length(binEdges) - 1
    binData = Errortable.firstPeaks_wavelengths_abs_errors(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, Errortable.firstPeaks_wavelengths_abs_errors, 5, 'o', 'filled');
xlabel('Intensity');
ylabel('Absolute Errors of First Peaks Wavelengths');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Second Peak Absolute Error
% Calculate means and standard deviations for each bin
for i = 1:length(binEdges) - 1
    binData = Errortable.secondPeaks_wavelengths_abs_errors(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, Errortable.secondPeaks_wavelengths_abs_errors, 5, 'o', 'filled', 'green');
xlabel('Intensity');
ylabel('Absolute Errors of Second Peaks Wavelengths');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% First Peak FWHM Absolute Error
% Calculate means and standard deviations for each bin
for i = 1:length(binEdges) - 1
    binData = Errortable.firstPeakFWHM_abs_errors(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, Errortable.firstPeakFWHM_abs_errors, 5, 'o', 'filled', 'magenta');
xlabel('Intensity');
ylabel('Absolute Errors of First Peaks FWHM');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Second Peak FWHM Absolute Error
% Calculate means and standard deviations for each bin
for i = 1:length(binEdges) - 1
    binData = Errortable.secondPeakFWHM_abs_errors(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, Errortable.secondPeakFWHM_abs_errors, 5, 'o', 'filled', 'cyan');
xlabel('Intensity');
ylabel('Absolute Errors of Second Peaks FWHM');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Peak Ratio Error (The Peak ratio Error need to be calculated)
% Filter out Big Value:
Errortable.PeakRatio_abs_errors(Errortable.PeakRatio_abs_errors > 10) = NaN;
% Calculate means and standard deviations for each bin
for i = 1:length(binEdges) - 1
    binData = Errortable.PeakRatio_abs_errors(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, Errortable.PeakRatio_abs_errors, 5, 'o', 'filled', 'red');
xlabel('Intensity');
ylabel('Absolute Errors of PeakRatio_abs_errors');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for rawspt (RMSE)
for i = 1:length(binEdges) - 1
    binData = resultsRawSpt.RMSE(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsRawSpt.RMSE, 5, 'o', 'filled', 'yellow');
xlabel('Intensity');
ylabel('RMSE of RawSpt');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for predicted results (RMSE)
for i = 1:length(binEdges) - 1
    binData = resultsVq.RMSE(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsVq.RMSE, 5, 'o', 'filled', 'red');
xlabel('Intensity');
ylabel('RMSE of Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for fitted predicted results (RMSE)
for i = 1:length(binEdges) - 1
    binData = resultsSpef.RMSE(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsSpef.RMSE, 5, 'o', 'filled', 'magenta');
xlabel('Intensity');
ylabel('RMSE of Fitted Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for rawspt (Cross-Correlation)
for i = 1:length(binEdges) - 1
    binData = resultsRawSpt.CorrelationCoefficient(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsRawSpt.CorrelationCoefficient, 5, 'o', 'filled', 'green');
xlabel('Intensity');
ylabel('CorrelationCoefficient of RawSpt');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for predicted results (Cross-Correlation)
for i = 1:length(binEdges) - 1
    binData = resultsVq.CorrelationCoefficient(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsVq.CorrelationCoefficient, 5, 'o', 'filled', 'blue');
xlabel('Intensity');
ylabel('CorrelationCoefficient of Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for fitted predicted results (Cross-Correlation)
for i = 1:length(binEdges) - 1
    binData = resultsSpef.CorrelationCoefficient(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsSpef.CorrelationCoefficient, 5, 'o', 'filled', 'cyan');
xlabel('Intensity');
ylabel('CorrelationCoefficient of Fitted Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for rawspt (AreaBetweenCurves)
for i = 1:length(binEdges) - 1
    binData = resultsRawSpt.AreaBetweenCurves(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsRawSpt.AreaBetweenCurves, 5, 'o', 'filled', 'yellow');
xlabel('Intensity');
ylabel('AreaBetweenCurves of RawSpt');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for predicted results (AreaBetweenCurves)
for i = 1:length(binEdges) - 1
    binData = resultsVq.AreaBetweenCurves(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsVq.AreaBetweenCurves, 5, 'o', 'filled', 'red');
xlabel('Intensity');
ylabel('AreaBetweenCurves of Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;
%% Spectrum-wise Comparison for fitted predicted results (AreaBetweenCurves)
for i = 1:length(binEdges) - 1
    binData = resultsSpef.AreaBetweenCurves(binIdx == i);
    if ~isempty(binData)
        medians(i) = median(binData, 'omitnan');
        stds(i) = std(binData, 'omitnan');
    else
        medians(i) = NaN;  % Handle bins with no data
        stds(i) = NaN;
    end
end
% Midpoints of bins for plotting
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;
% Create the scatter plot
scatter(coefTablerawsptf.Intensity, resultsSpef.AreaBetweenCurves, 5, 'o', 'filled', 'magenta');
xlabel('Intensity');
ylabel('AreaBetweenCurves of Fitted Predicted Spectrum');
title('Intensity vs. Error Plot');
grid on;
xticks(binEdges);  % Set x-ticks at bin edges for clarity
xlim([0, binEdges(end)]);
% Overlay error bars for mean and standard deviation
hold on;  % Retain the current plot
h = errorbar(binCenters, medians, stds, 'k-o', 'LineWidth', 2, 'MarkerFaceColor', 'red');
% Annotate each bin with mean and std values
for i = 1:length(medians)
    if ~isnan(medians(i))  % Only annotate non-NaN values
        text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    end
end
hold off;

%% Plot Error curves together:(RMSE)
% Assuming binEdges and binIdx are already defined

% Calculate medians and standard deviations for each dataset
datasets = {'resultsRawSpt.RMSE', 'resultsVq.RMSE', 'resultsSpef.RMSE'};
colors = {'yellow', 'red', 'magenta'}; % Colors for each dataset
labels = {'Raw Spt', 'Predicted', 'Fitted Predicted'};

% Initialize figure
figure;
hold on;
grid on;

% Plot settings
markers = {'o', 's', '^'}; % Different markers for each dataset
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

% Loop through each dataset
for j = 1:length(datasets)
    RMSE = eval(datasets{j}); % Dynamic variable evaluation
    medians = NaN(1, length(binEdges) - 1);
    stds = NaN(1, length(binEdges) - 1);
    
    % Compute statistics for each bin
    for i = 1:length(binEdges) - 1
        binData = RMSE(binIdx == i);
        if ~isempty(binData)
            medians(i) = median(binData, 'omitnan');
            stds(i) = std(binData, 'omitnan');
        end
    end
    
    % Scatter plot of raw data points
    % scatter(coefTableSptf.Intensity, RMSE, 5, markers{j}, 'filled', 'DisplayName', labels{j}, 'MarkerFaceColor', colors{j});
    
    % Error bars plot
    errorbar(binCenters, medians, stds, 'Color', colors{j}, 'LineStyle', '-', 'Marker', markers{j}, 'LineWidth', 2, 'MarkerFaceColor', colors{j});
    
    % %Annotate each bin with median and std with text:
    % offset = max(stds) * 0.1;  % Adjust offset based on max std deviation
    % for i = 1:length(medians)
    %     if ~isnan(medians(i))
    %         text(binCenters(i), medians(i) + stds(i) + offset, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
    %             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 6, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    %     end
    % end

    % Annotate each bin
    for i = 1:length(medians)
        if ~isnan(medians(i))
            text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
        end
    end
end

% Finalize plot
xlabel('Intensity');
ylabel('RMSE');
title('Comparison of RMSE across Spectra Types');
legend('show'); % Show legend
xticks(binEdges);
xlim([0, binEdges(end)]);
hold off;

%% Plot Error curves together:(CorrelationCoefficient)
% Assuming binEdges and binIdx are already defined

% Calculate medians and standard deviations for each dataset
datasets = {'resultsRawSpt.CorrelationCoefficient', 'resultsVq.CorrelationCoefficient', 'resultsSpef.CorrelationCoefficient'};
colors = {'yellow', 'red', 'magenta'}; % Colors for each dataset
labels = {'Raw Spt', 'Predicted', 'Fitted Predicted'};

% Initialize figure
figure;
hold on;
grid on;

% Plot settings
markers = {'o', 's', '^'}; % Different markers for each dataset
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

% Loop through each dataset
for j = 1:length(datasets)
    RMSE = eval(datasets{j}); % Dynamic variable evaluation
    medians = NaN(1, length(binEdges) - 1);
    stds = NaN(1, length(binEdges) - 1);
    
    % Compute statistics for each bin
    for i = 1:length(binEdges) - 1
        binData = RMSE(binIdx == i);
        if ~isempty(binData)
            medians(i) = median(binData, 'omitnan');
            stds(i) = std(binData, 'omitnan');
        end
    end
    
    % Scatter plot of raw data points
    % scatter(coefTableSptf.Intensity, RMSE, 5, markers{j}, 'filled', 'DisplayName', labels{j}, 'MarkerFaceColor', colors{j});
    
    % Error bars plot
    errorbar(binCenters, medians, stds, 'Color', colors{j}, 'LineStyle', '-', 'Marker', markers{j}, 'LineWidth', 2, 'MarkerFaceColor', colors{j});
    
    % Annotate each bin
    for i = 1:length(medians)
        if ~isnan(medians(i))
            text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
        end
    end

    % %Annotate each bin with median and std with text:
    % offset = max(stds) * 0.1;  % Adjust offset based on max std deviation
    % for i = 1:length(medians)
    %     if ~isnan(medians(i))
    %         text(binCenters(i), medians(i) + stds(i) + offset, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
    %             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 6, 'BackgroundColor', 'white', 'EdgeColor', 'black');
    %     end
    % end
end

% Finalize plot
xlabel('Intensity');
ylabel('RMSE');
title('Comparison of RMSE across Spectra Types');
legend('show'); % Show legend
xticks(binEdges);
xlim([0, binEdges(end)]);
hold off;
%% Plot Error curves together:(AreaBetweenCurves)
% Assuming binEdges and binIdx are already defined

% Calculate medians and standard deviations for each dataset
datasets = {'resultsRawSpt.AreaBetweenCurves', 'resultsVq.AreaBetweenCurves', 'resultsSpef.AreaBetweenCurves'};
colors = {'yellow', 'red', 'magenta'}; % Colors for each dataset
labels = {'Raw Spt', 'Predicted', 'Fitted Predicted'};

% Initialize figure
figure;
hold on;
grid on;

% Plot settings
markers = {'o', 's', '^'}; % Different markers for each dataset
binCenters = (binEdges(1:end-1) + binEdges(2:end)) / 2;

% Loop through each dataset
for j = 1:length(datasets)
    RMSE = eval(datasets{j}); % Dynamic variable evaluation
    medians = NaN(1, length(binEdges) - 1);
    stds = NaN(1, length(binEdges) - 1);
    
    % Compute statistics for each bin
    for i = 1:length(binEdges) - 1
        binData = RMSE(binIdx == i);
        if ~isempty(binData)
            medians(i) = median(binData, 'omitnan');
            stds(i) = std(binData, 'omitnan');
        end
    end
    
    % Scatter plot of raw data points
    % scatter(coefTableSptf.Intensity, RMSE, 5, markers{j}, 'filled', 'DisplayName', labels{j}, 'MarkerFaceColor', colors{j});
    
    % Error bars plot
    errorbar(binCenters, medians, stds, 'Color', colors{j}, 'LineStyle', '-', 'Marker', markers{j}, 'LineWidth', 2, 'MarkerFaceColor', colors{j});
    
    % %Annotate each bin
    % for i = 1:length(medians)
    %     if ~isnan(medians(i))
    %         text(binCenters(i), medians(i) + stds(i) + 10, sprintf('Med=%.2f\nσ=%.2f', medians(i), stds(i)), ...
    %             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', 'FontSize', 8, 'BackgroundColor', 'white');
    %     end
    % end
    
end

% Finalize plot
xlabel('Intensity');
ylabel('RMSE');
title('Comparison of RMSE across Spectra Types');
legend('show'); % Show legend
xticks(binEdges);
xlim([0, binEdges(end)]);
hold off;
%% Plot single spectrum
wavelengths = linspace(500, 801, 301);  % Adjust start and end wavelengths accordingly
num = 1;
% num = [1 2 3 4];

figure;
subplot(6,1,1)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, rawspt(:, n));
    title('(a) raw spectrum');
    pause(1); % Pause to view each plot
end
hold off;
subplot(6,1,2)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, Oldspt(:, n));
    title('(b) subtract offset (old method)');
    pause(1); % Pause to view each plot
end
hold off;
subplot(6,1,3)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, sptf(:, n));
    title('(c) fitting of raw spectrum');
    pause(1); % Pause to view each plot
end
hold off;
subplot(6,1,4)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, vq(:, n));
    title('(d) U-net');
    pause(1); % Pause to view each plot
end
hold off;
subplot (6,1,5)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, spef(:, n));
    title('(d) 2-gaussian-fitted-U-net');
    pause(1); % Pause to view each plot
end
subplot(6,1,6)
hold on;
for n = num  % Example indices of different spectra
    plot(wavelengths, spt(:, n));
    title('(e) ground truth');
    pause(1); % Pause to view each plot
end
%% check if two variables are same
areSame = isequal(spt(:,1:numSpectra), sptf);  % Returns false because data types are different
areSameSize = isequal(size(spt(:,1:numSpectra)), size(sptf));  % Returns true because sizes are the same
areSameType = strcmp(class(spt(:,1:numSpectra)), class(sptf));  % Returns false because classes are different
%% Check single sptf
    n = 122;
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(spt(:,n)), mean(x), std(x), max(spt(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, spt(:,n), ft);
    ypredtemp2=fitobj(x);
    sptf(:,1)=ypredtemp2;
%% Check single spef
    n = 122;
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(vq(:,n)), mean(x), std(x), max(vq(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, vq(:,n), ft);
    ypredtemp2=fitobj(x);
    sptf(:,1)=ypredtemp2;
%% FUNCTION
function [absError, squaredError, mse, rmse, R, R_squared, chi_square, areaBetween] = calculateSpectrumStats(spect1, spect2)
    % Error Metrics
    absError = abs(spect1 - spect2);
    squaredError = (spect1 - spect2).^2;
    mse = mean(squaredError, 'all');
    rmse = sqrt(mse);
    
    % Correlation Coefficient and R-squared
    R = corrcoef(spect1, spect2);
    R = R(1,2);  % Extract the off-diagonal element which is the correlation coefficient
    R_squared = R^2;
    
    % Chi-square Goodness of Fit
    cost_func = 'NRMSE';
    % chi_square = sum(((spect1 - spect2).^2) ./ spect2, 'all');
    chi_square = goodnessOfFit(spect1, spect2, cost_func);
    
    % % Calculate P value
    % df = 301-6;
    % p_value = 1 - chi2cdf(chi_square, df);

    % Area Between Curves
    areaBetween = trapz(abs(spect1 - spect2));
end

function fwhm = calculateFWHM(x, data, peakIndex)
    peakHeight = data(peakIndex);
    halfMax = peakHeight / 2;
    % Find indices where data first and last crosses half maximum
    indicesAboveHalfMax = find(data >= halfMax);
    if ~isempty(indicesAboveHalfMax)
        leftIndex = find(indicesAboveHalfMax < peakIndex, 1, 'last');
        rightIndex = find(indicesAboveHalfMax > peakIndex, 1, 'first');
        if isempty(leftIndex) || isempty(rightIndex)
            fwhm = NaN;  % Return NaN if no proper crossing is found
        else
            fwhm = x(rightIndex) - x(leftIndex);
        end
    else
        fwhm = NaN;
    end
end
%% calculate the absolute error, squared error, mse, rmse
function [abs_errors, squared_errors, mse, rmse] = calculateErrors(y_true, y_pred)
    % Validate input sizes
    if length(y_true) ~= length(y_pred)
        error('Input vectors y_true and y_pred must be of the same length.');
    end
    % Absolute Errors
    abs_errors = abs(y_true - y_pred);
    % Squared Errors
    squared_errors = (y_true - y_pred).^2;
    % Mean Squared Error
    mse = mean(squared_errors);
    % Root Mean Squared Error
    rmse = sqrt(mse);
    % Display calculated values 
    % fprintf('Absolute_error: %f\n', abs_errors);
    % fprintf('Squared_errors: %f\n', squared_errors);
    % fprintf('MSE: %f\n', mse);
    % fprintf('RMSE: %f\n', rmse);
end
%% check if two variables are same
areSame = isequal(Predictspe(:,1:numSpectra), Predictspe(:,1:numSpectra));  % Returns false because data types are different
% areSameSize = isequal(size(spt(:,1:numSpectra)), size(sptf));  % Returns true because sizes are the same
% areSameType = strcmp(class(spt(:,1:numSpectra)), class(sptf));  % Returns false because classes are different
