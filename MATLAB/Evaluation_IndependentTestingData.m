%% Load net and data
clearvars
close all
clc
% Load Net
addpath("D:\HongjingMao\Manuscript\Code")
% load('D:\HongjingMao\Manuscript\Results\Net\OG100k_MP_psig10000pbg20000BG3_epo250_1e-2.mat')
load('D:\HongjingMao\Manuscript\Results\Net\Final_OG100k_MP_psig10000pbg20000BG3_epo250_1e-2.mat')
net2 = load('D:\HongjingMao\Manuscript\Results\Net\OG100k_psig10000pbg20000BG3_epo250_1e-2.mat');
%% ML prediction and Preprocessing data for background prediction (For 16x128 array,oldspt = subtracting mean background over all stack,For Prelim, For manuscript!)
tic
% load Data
% BG10:
% load('D:\HongjingMao\Manuscript\TestingData\Final_OG5k_psig10000pbg20000BG3.mat')
% Alexa-647
load('D:\HongjingMao\Manuscript\Alexa647\1k_Pbg10000Psig1000_constant.mat')
%% Normalization for Normalized net
% Normalize variables using MATLAB's 'normalize' function
normalized_GTspt = normalize(GTspt, 'range');
normalized_spt = normalize(spt, 'range');  % Normalizes each column separately if 'spt' is a matrix
normalized_sptimg4 = normalize(sptimg4, 'range');
%% Make sure spt = 301x100000
spt=spt'; %ground truth spectra
% Spectra number to be processed
numSpectra = 1000;
% Prediction
sptimg4 = double(sptimg4);
tbg4 = double(tbg4);
% Initialize
YPred=zeros(16,128,numSpectra);
YPred2=zeros(16,128,numSpectra);
Predictspe = zeros(16,128,numSpectra);
Predictspe2 = zeros(16,128,numSpectra);
Oldsptimg = zeros(16,128,numSpectra);
Waveletsptimg = zeros(16,128,numSpectra);
Waveletspe = zeros(128,numSpectra);
PredImgMF=zeros(16,128,numSpectra);
meanBack = mean(tbg4,3);
% noisyImages = double(noisyImages); 
for n=1:numSpectra
    % For Conventional Net
    YPred(:,:,n)=predict(net,sptimg4(:,:,n));
    Predictspe(:,:,n) = sptimg4(:,:,n)-YPred(:,:,n);
    
    % For Up Net
    YPred2(:,:,n)=predict(net2.net,sptimg4(:,:,n));
    Predictspe2(:,:,n) = sptimg4(:,:,n)-YPred2(:,:,n);

    % For wavelet
    Waveletsptimg(:, :, n) = wdenoise2(sptimg4(:, :, n));

    % For mean filter
    PredImgMF(:,:,n) = medfilt2(sptimg4(:,:,n));

    % For old method subtract average Background of all stack:
    Oldsptimg(:,:,n) = sptimg4(:,:,n) - meanBack;  % Subtract background
end
% For predicted spectrum
sptn=squeeze(mean(Predictspe(7:10,:,:),1)); %YPred or rawspectral image
xq=1:128/303:128;
vq=interp1(1:128,sptn,xq);  %%%% vq is the interpolated spe 301 value
% For raw spectrum
rawspt = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum
rawspt = interp1(1:128,rawspt,xq);
% For Wavelet:
rawsptforwavelet = squeeze(mean(sptimg4(7:10,:,:),1)); % Raw spectrum for wavelet denoising
for n = 1:numSpectra
    Waveletspe(:, n) = wdenoise(rawsptforwavelet(:, n));
end
Waveletspe = interp1(1:128,Waveletspe,xq);
Wavelet_BG = sptimg4 - Waveletsptimg;
% For Median Filter
MFspe = squeeze(mean(PredImgMF(7:10,:,:),1));
MFspe = interp1(1:128,MFspe,xq);
MF_BG = sptimg4 - PredImgMF;
% For ground truth spt(with noise)
GTsptimg = sptimg4-tbg4; 
GTspt = squeeze(mean(GTsptimg(7:10,:,:),1)); %YPred or rawspectral image
GTspt = interp1(1:128,GTspt,xq);  %%%% vq is the interpolated spe 301 value
% For old method subtract min Background:
Oldspt = squeeze(mean(Oldsptimg(7:10,:,:),1));
Oldspt = interp1(1:128,Oldspt,xq);
Old_BG = repmat(meanBack,[1,1,5000]);
toc
%
figure; % Create a figure window outside the loop
colormap 'gray'; % Set colormap once if all subplots use the same
%% Visualization
for n = 1:100
    % Display original image
    subplot(9,1,1);
    imagesc(sptimg4(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['sptimg4 - Frame ', num2str(n)]); % Dynamic title with frame number

    % Display predicted image by ConvU
    subplot(9,1,2);
    imagesc(YPred(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Conv U-net - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display predicted image by UpU
    subplot(9,1,3);
    imagesc(YPred2(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Up U-net - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,4);
    imagesc(Predictspe(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Conv U-net spe - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,5);
    imagesc(Predictspe2(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Up U-net spe - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,6);
    imagesc(Oldsptimg(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Old spe - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,7);
    imagesc(GTsptimg(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['GT spe - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,8);
    imagesc(Waveletsptimg(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Wavelet spe - Frame ', num2str(n)]); % Dynamic title with frame number

    subplot(9,1,9);
    imagesc(PredImgMF(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['MF spe - Frame ', num2str(n)]); % Dynamic title with frame number

    pause(1); % Pause for 0.5 seconds to view the current set of images
    drawnow; % Force display update
end
%%
figure; % Create a figure window outside the loop
colormap 'gray'; % Set colormap once if all subplots use the same
for n = 1:200
    % Display original image
    subplot(3,1,1);
    imagesc(sptimg4(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Raw Image - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display predicted image
    subplot(3,1,2);
    imagesc(YPred(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Predicted Background - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display processed or another predicted image
    subplot(3,1,3);
    imagesc(Predictspe(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Predicted Spectra - Frame ', num2str(n)]); % Dynamic title with frame number

    pause(1); % Pause for 0.5 seconds to view the current set of images
    drawnow; % Force display update
end
%% For single Image extraction as pdf:
% Select a specific frame number
frame_number = 8;

% Specify the directory where you want to save the PDF files
% save_directory = 'D:\HongjingMao\Manuscript\Results\IndependentTestingData\Prediction\Frame8\'; % Ensure this path exists on your system
save_directory = 'D:\HongjingMao\Manuscript\Alexa647\Prediction'; % Ensure this path exists on your system
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
%% Normalization before Image-wise Evaluation:
% % Predictede Image Normalization:
% sptimg4_snr = normalize(sptimg4,'range'); 
% Predictspe_snr = normalize(Predictspe,'range');
% Predictspe2_snr = normalize(Predictspe2,'range');
% Oldsptimg_snr = normalize(Oldsptimg,'range');
% Waveletsptimg_snr = normalize(Waveletsptimg,'range');
% MedianFilter_snr = normalize (PredImgMF,'range');
% % BG normalization:
% YPred = normalize (YPred,'range');
% YPred2 = normalize (YPred2,'range');
% Wavelet_BG = normalize (Wavelet_BG,'range');
% Old_BG = normalize (Old_BG,'range');
% MF_BG = normalize (MF_BG,'range');
%% Calculating the SNR
% Preallocate
SNR_Raw = zeros(numSpectra, 1);  
SNR_ConvU = zeros(numSpectra, 1);  
SNR_UpU = zeros(numSpectra, 1);  
SNR_Old = zeros(numSpectra, 1);
SNR_Wavelet = zeros(numSpectra, 1);
SNR_MF = zeros(numSpectra, 1);
% IF did not do normalize please uncomment this section!!!!!!!!!!
sptimg4_snr = double(sptimg4); 
Predictspe_snr = double(Predictspe);
Predictspe2_snr = double(Predictspe2);
Oldsptimg_snr = double(Oldsptimg);
Waveletsptimg_snr = Waveletsptimg;
MedianFilter_snr = PredImgMF;
%% Yang's method 2 (For manuscript!!)
for n = 1:numSpectra
    % Raw image
    SNR_Raw(n) = calculateSNR(sptimg4_snr, n);
    % For Conv_Unet
    SNR_ConvU(n) = calculateSNR(Predictspe_snr, n);
    % For Up_Unet
    SNR_UpU(n) = calculateSNR(Predictspe2_snr, n);
    % For Wavelet
    SNR_Wavelet(n) = calculateSNR(Waveletsptimg_snr,n);
    % For Median Filter
    SNR_MF(n) = calculateSNR(MedianFilter_snr,n);
    % For conventional Method
    SNR_Old(n) = calculateSNR(Oldsptimg_snr,n);
end
% Save SNR:
% Define the path and filename
path = 'D:\HongjingMao\Manuscript\Results\Evaluation\NormalizeImagewise_OG100k_MP_psig10000pbg20000BG3_epo250_1e-2_onPbg20000testing\';
mkdir(path);
filenameMat = [path 'Image_wise_SNR.mat'];
% Save all variables to a .mat file at the specified path
save(filenameMat, 'SNR_Raw', 'SNR_ConvU','SNR_UpU', 'SNR_Wavelet', 'SNR_MF', 'SNR_Old');

% Define the Excel filename
filenameExcel = [path 'Image_wise_SNR.xlsx'];
% Prepare data for Excel
headers = {'SNR_Raw', 'SNR_ConvU','SNR_UpU', 'SNR_Wavelet', 'SNR_MF', 'SNR_Old'};
data = [SNR_Raw, SNR_ConvU, SNR_UpU, SNR_Wavelet, SNR_MF, SNR_Old];
% Combine headers and data
combinedData = [headers; num2cell(data)];
% Write to Excel at the specified path
writecell(combinedData, filenameExcel);
%% Plot the results:
% Number of bins for the histogram
numBins = 30;  % Adjust this number based on your specific data distribution

% Create a figure window
figure;

% Plot histogram for Raw SNR values
subplot(2,2,1); % This arranges the plots in a 2x2 grid, this is the first plot
histogram(SNR_Raw, numBins);
title('SNR for Raw Images');
xlabel('SNR (dB)');
ylabel('Frequency');

% Plot histogram for Unet SNR values
subplot(2,2,2); % Second plot in the 2x2 grid
histogram(SNR_Unet, numBins);
title('SNR for U-Net Predictions');
xlabel('SNR (dB)');
ylabel('Frequency');

% Plot histogram for ConvUnet SNR values
subplot(2,2,3); % Third plot in the 2x2 grid
histogram(SNR_ConvU, numBins);
title('SNR for ConvU-Net Predictions');
xlabel('SNR (dB)');
ylabel('Frequency');

% Plot histogram for Old Method SNR values
subplot(2,2,4); % Fourth plot in the 2x2 grid
histogram(SNR_Old, numBins);
title('SNR for Conventional Method');
xlabel('SNR (dB)');
ylabel('Frequency');

% Improve layout
sgtitle('Comparison of SNR Distributions Across Different Methods'); % Overall title for the subplot grid
%% Image-wise evaluation RMSE: For manuscript!
% Calculating the Image-wise RMSE:
% Conventional Net
ConvU_BG_RMSE = rmse(YPred,tbg4,[1 2]);
ConvU_BG_RMSE = squeeze(ConvU_BG_RMSE);

% ConvU_SPE_RMSE = rmse(Predictspe,GTsptimg,[1 2]);
% ConvU_SPE_RMSE = squeeze(ConvU_SPE_RMSE);

% Up Net
UpU_BG_RMSE = rmse(YPred2,tbg4,[1 2]);
UpU_BG_RMSE = squeeze(UpU_BG_RMSE);

% UpU_SPE_RMSE = rmse(Predictspe2,GTsptimg,[1 2]);
% UpU_SPE_RMSE = squeeze(UpU_SPE_RMSE);

% Wavelet
Wavelet_BG_RMSE = rmse(Wavelet_BG,tbg4,[1 2]);
Wavelet_BG_RMSE = squeeze(Wavelet_BG_RMSE);

% Wavelet_SPE_RMSE = rmse(Waveletsptimg,GTsptimg,[1 2]);
% Wavelet_SPE_RMSE = squeeze(Wavelet_SPE_RMSE);

% Median Filter
MF_BG_RMSE = rmse(MF_BG,tbg4,[1 2]);
MF_BG_RMSE = squeeze(MF_BG_RMSE);

% MF_SPE_RMSE = rmse(PredImgMF,GTsptimg,[1 2]);
% MF_SPE_RMSE = squeeze(MF_SPE_RMSE);

% Conventional Method
Old_BG_RMSE = rmse(Old_BG,tbg4,[1 2]);
Old_BG_RMSE = squeeze(Old_BG_RMSE);

% Old_SPE_RMSE = rmse(Oldsptimg,GTsptimg,[1 2]);
% Old_SPE_RMSE = squeeze(Old_SPE_RMSE);

filenameMat = [path 'Image_wise_RMSE.mat'];
% Save all variables to a .mat file at the specified path
save(filenameMat, 'ConvU_BG_RMSE', 'UpU_BG_RMSE', 'Wavelet_BG_RMSE',  'Old_BG_RMSE', 'MF_BG_RMSE');

% Define the Excel filename
filenameExcel = [path 'Image_wise_RMSE.xlsx'];
% Prepare data for Excel
headers = {'ConvU_BG_RMSE', 'UpU_BG_RMSE', 'Wavelet_BG_RMSE',  'Old_BG_RMSE', 'MF_BG_RMSE'};
data = [ConvU_BG_RMSE, UpU_BG_RMSE, Wavelet_BG_RMSE, Old_BG_RMSE, MF_BG_RMSE];
% Combine headers and data
combinedData = [headers; num2cell(data)];
% Write to Excel at the specified path
writecell(combinedData, filenameExcel);
%% Image-wise evaluation SSIM: For manuscript!
% Initialize arrays to store SSIM values for each image
ConvU_BG_SSIM = zeros(1, size(YPred, 3));
UpU_BG_SSIM = zeros(1, size(YPred2, 3));
Wavelet_BG_SSIM = zeros(1, size(Wavelet_BG, 3));
Old_BG_SSIM = zeros(1, size(Old_BG, 3));
MF_BG_SSIM = zeros(1, size(MF_BG, 3));

% Loop through each image
for i = 1:size(YPred, 3)
    ConvU_BG_SSIM(i) = ssim(YPred(:,:,i), tbg4(:,:,i));
    UpU_BG_SSIM(i) = ssim(YPred2(:,:,i), tbg4(:,:,i));
    Wavelet_BG_SSIM(i) = ssim(Wavelet_BG(:,:,i), tbg4(:,:,i));
    Old_BG_SSIM(i) = ssim(Old_BG(:,:,i), tbg4(:,:,i));
    MF_BG_SSIM(i) = ssim(MF_BG(:,:,i), tbg4(:,:,i));
end

ConvU_BG_SSIM = ConvU_BG_SSIM';
UpU_BG_SSIM = UpU_BG_SSIM';
Wavelet_BG_SSIM = Wavelet_BG_SSIM';
Old_BG_SSIM = Old_BG_SSIM';
MF_BG_SSIM = MF_BG_SSIM';

filenameMat = [path 'Image_wise_SSIM.mat'];
% Save all variables to a .mat file at the specified path
save(filenameMat, 'ConvU_BG_SSIM','UpU_BG_SSIM','Wavelet_BG_SSIM',  'Old_BG_SSIM', 'MF_BG_SSIM');

% Define the Excel filename
filenameExcel = [path 'Image_wise_SSIM.xlsx'];
% Prepare data for Excel
headers = {'ConvU_BG_SSIM', 'UpU_BG_SSIM','Wavelet_BG_SSIM','Old_BG_SSIM', 'MF_BG_SSIM'};
data = [ConvU_BG_SSIM,UpU_BG_SSIM,Wavelet_BG_SSIM,  Old_BG_SSIM, MF_BG_SSIM];
% Combine headers and data
combinedData = [headers; num2cell(data)];
% Write to Excel at the specified path
writecell(combinedData, filenameExcel);
%% Normalize:
rawspt = normalize(rawspt,"range");
spt = normalize(spt,"range");
vq = normalize(vq,"range");
Oldspt = normalize(Oldspt,"range");
Waveletspe = normalize(Waveletspe,"range");
MFspe = normalize(MFspe,"range");
%% 2-gaussian fit for spt and vq, then compare the features Peak-wise.
% Initialize every thing
x=1:301;x=x'; % Wavelength
wavelengths = x;
% For spectra
spef=zeros(301,numSpectra);
sptf=zeros(301,numSpectra);
Oldsptf = zeros(301,numSpectra);
Waveletspef = zeros(301,numSpectra);
MFspef = zeros(301,numSpectra);
% Initialize arrays to hold the centroids
centroid_Spef = zeros(numSpectra, 1);
centroid_Sptf = zeros(numSpectra, 1);
centroid_Spt = zeros(numSpectra, 1);
centroid_Oldsptf = zeros(numSpectra, 1);
centroid_Waveletspef = zeros(numSpectra, 1);
centroid_MFspef = zeros(numSpectra, 1);

% For first peak wavelengths
firstPeaks_wavelengths_errors = zeros(numSpectra, 1);
firstPeaks_wavelengths_squared_errors = zeros(numSpectra, 1);
% For second peak wavelengths
secondPeaks_wavelengths_errors = zeros(numSpectra, 1);
secondPeaks_wavelengths_squared_errors = zeros(numSpectra, 1);
% For first peak FWHM
firstPeakFWHM_errors = zeros(numSpectra, 1);
firstPeakFWHM_squared_errors = zeros(numSpectra, 1);
% For second peak FWHM
secondPeakFWHM_errors = zeros(numSpectra, 1);
secondPeakFWHM_squared_errors = zeros(numSpectra, 1);
% For Peak Ratio Error:
PeakRatio_errors = zeros(numSpectra, 1);
PeakRatio_squared_errors = zeros(numSpectra, 1);

% For old errors:
% For first peak wavelengths
firstPeaks_wavelengths_errors_old = zeros(numSpectra, 1);
firstPeaks_wavelengths_squared_errors_old = zeros(numSpectra, 1);
% For second peak wavelengths
secondPeaks_wavelengths_errors_old = zeros(numSpectra, 1);
secondPeaks_wavelengths_squared_errors_old = zeros(numSpectra, 1);
% For first peak FWHM
firstPeakFWHM_errors_old = zeros(numSpectra, 1);
firstPeakFWHM_squared_errors_old = zeros(numSpectra, 1);
% For second peak FWHM
secondPeakFWHM_errors_old = zeros(numSpectra, 1);
secondPeakFWHM_squared_errors_old = zeros(numSpectra, 1);
% For Peak Ratio Error:
PeakRatio_errors_old = zeros(numSpectra, 1);
PeakRatio_squared_errors_old = zeros(numSpectra, 1);
% Initialize the table 
coefTableSptf = table([], [], [], [], [], [], [], [], [], [], ...
                  'VariableNames', {'Sptf_firstPeakValues', 'Sptf_firstPeakWavelengths', 'Sptf_firstPeakFWHM', 'Sptf_secondPeakValues', 'Sptf_secondPeakWavelengths', 'Sptf_secondPeakFWHM', 'Sptf_peakRatio', 'Intensity','centroid_Sptf','centroid_Spt'});
coefTableSpef = table([], [], [], [], [], [], [], [], [], ...
                  'VariableNames', {'Spef_firstPeakValues', 'Spef_firstPeakWavelengths', 'Spef_firstPeakFWHM', 'Spef_secondPeakValues', 'Spef_secondPeakWavelengths', 'Spef_secondPeakFWHM', 'Spef_peakRatio', 'Intensity','centroid_Spef'});
coefTableOldsptf = table([], [], [], [], [], [], [], [], [],...
                   'VariableNames', {'Oldsptf_firstPeakValues', 'Oldsptf_firstPeakWavelengths', 'Oldsptf_firstPeakFWHM', 'Oldsptf_secondPeakValues', 'Oldsptf_secondPeakWavelengths', 'Oldsptf_secondPeakFWHM', 'Oldsptf_peakRatio', 'Intensity','centroid_Oldsptf'});
coefTableWaveletspef = table([], [], [], [], [], [], [], [], [],...
                   'VariableNames', {'Waveletspef_firstPeakValues', 'Waveletspef_firstPeakWavelengths', 'Waveletspef_firstPeakFWHM', 'Waveletspef_secondPeakValues', 'Waveletspef_secondPeakWavelengths', 'Waveletspef_secondPeakFWHM', 'Waveletspef_peakRatio', 'Intensity','centroid_Waveletspef'});
coefTableMFspef = table([], [], [], [], [], [], [], [], [],...
                   'VariableNames', {'MFspef_firstPeakValues', 'MFspef_firstPeakWavelengths', 'MFspef_firstPeakFWHM', 'MFspef_secondPeakValues', 'MFspef_secondPeakWavelengths', 'MFspef_secondPeakFWHM', 'MFspef_peakRatio', 'Intensity','centroid_MFspef'});

% LSE fitting for all spectra
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
    % Extract and compare A1 and A2
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    Intensity = sum(sum(GTsptimg(:,:,n))); 
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
    ypredtemp=fitobj(x);
    spef(:,n)=ypredtemp;
    % Calculate spectra centroid for spef
    Intensityspef = sum(spef(:,n));
    if Intensityspef == 0
        centroid_spef1 = NaN;  % Handle case where there is no intensity
    else
        centroid_spef1 = sum(wavelengths .* spef(:,n)) / Intensityspef;
    end    
    centroid_Spef(n) = centroid_spef1;
    % Intensitytemp1 = Intensity(n);      
    coefTableSpef = [coefTableSpef; table(A1, b1, c1, A2, b2, c2, Ratio, Intensity, centroid_spef1,...
                                  'VariableNames', {'Spef_firstPeakValues', 'Spef_firstPeakWavelengths', 'Spef_firstPeakFWHM', 'Spef_secondPeakValues', 'Spef_secondPeakWavelengths', 'Spef_secondPeakFWHM','Spef_peakRatio','Intensity','centroid_Spef'})];
    %2 Gaussian fit for spt
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
    % Extract and compare A1 and A2
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
    ypredtemp2=fitobj(x);
    sptf(:,n)=ypredtemp2;
    % Calculate spectra centroid of rawsptf
    Intensitysptf = sum(sptf(:,n));
    if Intensitysptf == 0
        centroid_sptf1 = NaN;  % Handle case where there is no intensity
    else
        centroid_sptf1 = sum(wavelengths .* sptf(:,n)) / Intensitysptf;
    end    
    centroid_Sptf(n) = centroid_sptf1;
    % Calculate spectra centroid of rawspt
    Intensityspt = sum(spt(:,n));
    if Intensityspt == 0
        centroid_spt1 = NaN;  % Handle case where there is no intensity
    else
        centroid_spt1 = sum(wavelengths .* spt(:,n)) / Intensityspt;
    end    
    centroid_Spt(n) = centroid_spt1;
    % Intensitytemp2 = Intensity(n);
    coefTableSptf = [coefTableSptf; table(A1, b1, c1, A2, b2, c2, Ratio,Intensity,centroid_sptf1, centroid_spt1,...
                                  'VariableNames', {'Sptf_firstPeakValues', 'Sptf_firstPeakWavelengths', 'Sptf_firstPeakFWHM', 'Sptf_secondPeakValues', 'Sptf_secondPeakWavelengths', 'Sptf_secondPeakFWHM','Sptf_peakRatio','Intensity','centroid_Sptf','centroid_Spt'})];
    
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
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
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
    % Intensitytemp3 = Intensity(n);       % Import Intensity (1x137323)
    coefTableOldsptf = [coefTableOldsptf; table(A1, b1, c1, A2, b2, c2, Ratio, Intensity, centroid_Oldsptf1,...
                            'VariableNames', {'Oldsptf_firstPeakValues', 'Oldsptf_firstPeakWavelengths', 'Oldsptf_firstPeakFWHM', 'Oldsptf_secondPeakValues', 'Oldsptf_secondPeakWavelengths', 'Oldsptf_secondPeakFWHM', 'Oldsptf_peakRatio', 'Intensity','centroid_Oldsptf'})];

    % 2 Gaussian fit for Waveletspe
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(Waveletspe(:,n)), mean(x), std(x), max(Waveletspe(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, Waveletspe(:,n), ft);
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
    ypredtemp=fitobj(x);
    Waveletspef(:,n)=ypredtemp;        
    % Calculate spectra centroid for Waveletspef
    IntensityWaveletspef = sum(Waveletspef(:,n));
    if IntensityWaveletspef == 0
        centroid_Waveletspef1 = NaN;  % Handle case where there is no intensity
    else
        centroid_Waveletspef1 = sum(wavelengths .* Waveletspef(:,n)) / IntensityWaveletspef;
    end    
    centroid_Waveletspef(n) = centroid_Waveletspef1;
    % Intensitytemp3 = Intensity(n);       % Import Intensity (1x137323)
    coefTableWaveletspef = [coefTableWaveletspef; table(A1, b1, c1, A2, b2, c2, Ratio, Intensity, centroid_Waveletspef1,...
                            'VariableNames', {'Waveletspef_firstPeakValues', 'Waveletspef_firstPeakWavelengths', 'Waveletspef_firstPeakFWHM', 'Waveletspef_secondPeakValues', 'Waveletspef_secondPeakWavelengths', 'Waveletspef_secondPeakFWHM', 'Waveletspef_peakRatio', 'Intensity','centroid_Waveletspef'})];

    % 2 Gaussian fit for MFspe
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(MFspe(:,n)), mean(x), std(x), max(MFspe(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, MFspe(:,n), ft);
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
    ypredtemp=fitobj(x);
    MFspef(:,n)=ypredtemp;        
    % Calculate spectra centroid for spef
    IntensityMFspef = sum(MFspef(:,n));
    if IntensityMFspef == 0
        centroid_MFspef1 = NaN;  % Handle case where there is no intensity
    else
        centroid_MFspef1 = sum(wavelengths .* MFspef(:,n)) / IntensityMFspef;
    end    
    centroid_MFspef(n) = centroid_MFspef1;
    % Intensitytemp3 = Intensity(n);       % Import Intensity (1x137323)
    coefTableMFspef = [coefTableMFspef; table(A1, b1, c1, A2, b2, c2, Ratio, Intensity, centroid_MFspef1,...
                            'VariableNames', {'MFspef_firstPeakValues', 'MFspef_firstPeakWavelengths', 'MFspef_firstPeakFWHM', 'MFspef_secondPeakValues', 'MFspef_secondPeakWavelengths', 'MFspef_secondPeakFWHM', 'MFspef_peakRatio', 'Intensity','centroid_MFspef'})];


    % % Calculating the errors
    % First Peak wavelengths sptf vs spef
    if coefTableSptf.Sptf_firstPeakWavelengths(n) == 0 || coefTableSpef.Spef_firstPeakWavelengths(n) ==0
        firstPeaks_wavelengths_errors(n) = NaN;
        firstPeaks_wavelengths_squared_errors(n) = NaN;
    else
        [firstPeaks_wavelengths_errors(n), firstPeaks_wavelengths_squared_errors(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_firstPeakWavelengths(n), coefTableSpef.Spef_firstPeakWavelengths(n));
    end
    % First Peak wavelengths sptf vs Oldsptf
    if coefTableSptf.Sptf_firstPeakWavelengths(n) == 0 || coefTableOldsptf.Oldsptf_firstPeakWavelengths(n) ==0
        firstPeaks_wavelengths_errors_old(n) = NaN;
        firstPeaks_wavelengths_squared_errors_old(n) = NaN;
    else
        [firstPeaks_wavelengths_errors_old(n), firstPeaks_wavelengths_squared_errors_old(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_firstPeakWavelengths(n), coefTableOldsptf.Oldsptf_firstPeakWavelengths(n));
    end

    % Second Peak wavelengths sptf vs spef
    if coefTableSptf.Sptf_secondPeakWavelengths(n) == 0 || coefTableSpef.Spef_secondPeakWavelengths(n) ==0
        secondPeaks_wavelengths_errors(n) = NaN;
        secondPeaks_wavelengths_squared_errors(n) = NaN;
    else
        [secondPeaks_wavelengths_errors(n), secondPeaks_wavelengths_squared_errors(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_secondPeakWavelengths(n), coefTableSpef.Spef_secondPeakWavelengths(n));
    end
    % Second Peak wavelengths sptf vs Oldsptf
    if coefTableSptf.Sptf_secondPeakWavelengths(n) == 0 || coefTableOldsptf.Oldsptf_secondPeakWavelengths(n) ==0
        secondPeaks_wavelengths_errors_old(n) = NaN;
        secondPeaks_wavelengths_squared_errors_old(n) = NaN;
    else
        [secondPeaks_wavelengths_errors_old(n), secondPeaks_wavelengths_squared_errors_old(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_secondPeakWavelengths(n), coefTableOldsptf.Oldsptf_secondPeakWavelengths(n));
    end

    % First Peak FWHM sptf vs spef
    if coefTableSptf.Sptf_firstPeakFWHM(n) == 0 || coefTableSpef.Spef_firstPeakFWHM(n) == 0
        firstPeakFWHM_errors(n) = NaN;
        firstPeakFWHM_squared_errors(n) = NaN;
    else
        [firstPeakFWHM_errors(n), firstPeakFWHM_squared_errors(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_firstPeakFWHM(n), coefTableSpef.Spef_firstPeakFWHM(n));
    end
    % First Peak FWHM sptf vs Oldsptf
    if coefTableSptf.Sptf_firstPeakFWHM(n) == 0 || coefTableOldsptf.Oldsptf_firstPeakFWHM(n) == 0
        firstPeakFWHM_errors_old(n) = NaN;
        firstPeakFWHM_squared_errors_old(n) = NaN;
    else
        [firstPeakFWHM_errors_old(n), firstPeakFWHM_squared_errors_old(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_firstPeakFWHM(n), coefTableOldsptf.Oldsptf_firstPeakFWHM(n));
    end

    % Second Peak FWHM sptf vs spef    
    if coefTableSptf.Sptf_secondPeakFWHM(n) == 0 || coefTableSpef.Spef_secondPeakFWHM(n) == 0
        secondPeakFWHM_errors(n) = NaN;
        secondPeakFWHM_squared_errors(n) = NaN;
    else
        [secondPeakFWHM_errors(n), secondPeakFWHM_squared_errors(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_secondPeakFWHM(n), coefTableSpef.Spef_secondPeakFWHM(n));
    end
    % Second Peak FWHM sptf vs Oldsptf    
    if coefTableSptf.Sptf_secondPeakFWHM(n) == 0 || coefTableOldsptf.Oldsptf_secondPeakFWHM(n) == 0
        secondPeakFWHM_errors_old(n) = NaN;
        secondPeakFWHM_squared_errors_old(n) = NaN;
    else
        [secondPeakFWHM_errors_old(n), secondPeakFWHM_squared_errors_old(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_secondPeakFWHM(n), coefTableOldsptf.Oldsptf_secondPeakFWHM(n));
    end

    % Peak Ration sptf vs spef
    if coefTableSptf.Sptf_peakRatio(n) == 0 || coefTableSpef.Spef_peakRatio(n) == 0
        PeakRatio_errors(n) = NaN;
        PeakRatio_squared_errors(n) = NaN;
    else
        [PeakRatio_errors(n), PeakRatio_squared_errors(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_peakRatio(n), coefTableSpef.Spef_peakRatio(n));
    end
    % Peak Ration sptf vs Oldsptf
    if coefTableSptf.Sptf_peakRatio(n) == 0 || coefTableOldsptf.Oldsptf_peakRatio(n) == 0
        PeakRatio_errors_old(n) = NaN;
        PeakRatio_squared_errors_old(n) = NaN;
    else
        [PeakRatio_errors_old(n), PeakRatio_squared_errors_old(n), ~, ~] = calculateErrors(coefTableSptf.Sptf_peakRatio(n), coefTableOldsptf.Oldsptf_peakRatio(n));
    end
        waitbar(n/10000, f, sprintf('Progress: %d %%', floor(n/10000*100)));
end

close(f);
% % Filter out FWHM > 250
% coefTableSpef.Spef_firstPeakFWHM(coefTableSpef.Spef_firstPeakFWHM > 250) = 0;
% coefTableSpef.Spef_secondPeakFWHM(coefTableSpef.Spef_secondPeakFWHM > 250) = 0;

% Calculating MSE and RMSE for spt vs spef
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

% Calculating MSE and RMSE for spt vs Oldspt
firstPeaks_wavelengths_mse_old = mean(firstPeaks_wavelengths_squared_errors_old,'omitnan');
firstPeaks_wavelengths_rmse_old = sqrt(firstPeaks_wavelengths_mse_old);

secondPeaks_wavelengths_mse_old = mean(secondPeaks_wavelengths_squared_errors_old,'omitnan');
secondPeaks_wavelengths_rmse_old = sqrt(secondPeaks_wavelengths_mse_old);

firstPeakFWHM_mse_old = mean(firstPeakFWHM_squared_errors_old,'omitnan');
firstPeakFWHM_rmse_old = sqrt(firstPeakFWHM_mse_old);

secondPeakFWHM_mse_old = mean(secondPeakFWHM_squared_errors_old,'omitnan');
secondPeakFWHM_rmse_old = sqrt(secondPeakFWHM_mse_old);

PeakRatio_mse_old = mean(PeakRatio_squared_errors_old,'omitnan');
PeakRatio_rmse_old = sqrt(PeakRatio_mse_old);

% Save error table for spt vs spef
Errortable = table(firstPeaks_wavelengths_errors, ...
                firstPeaks_wavelengths_squared_errors, secondPeaks_wavelengths_errors, ...
                secondPeaks_wavelengths_squared_errors,firstPeakFWHM_errors, ...
                firstPeakFWHM_squared_errors, secondPeakFWHM_errors, secondPeakFWHM_squared_errors,...
                PeakRatio_errors,PeakRatio_squared_errors);

% Save error table for spt vs Oldsptf
Errortable_old = table(firstPeaks_wavelengths_errors_old, ...
                firstPeaks_wavelengths_squared_errors_old, secondPeaks_wavelengths_errors_old, ...
                secondPeaks_wavelengths_squared_errors_old,firstPeakFWHM_errors_old, ...
                firstPeakFWHM_squared_errors_old, secondPeakFWHM_errors_old, secondPeakFWHM_squared_errors_old,...
                PeakRatio_errors_old,PeakRatio_squared_errors_old);
% Print the result
% Display Mean Squared Errors and Root Mean Squared Errors for sptf vs spef
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

% Display Mean Squared Errors and Root Mean Squared Errors for sptf vs Oldsptf
disp(['First Peaks Wavelengths MSE_old: ', num2str(firstPeaks_wavelengths_mse_old)]);
disp(['First Peaks Wavelengths RMSE_old: ', num2str(firstPeaks_wavelengths_rmse_old)]);
disp(['Second Peaks Wavelengths MSE_old: ', num2str(secondPeaks_wavelengths_mse_old)]);
disp(['Second Peaks Wavelengths RMSE_old: ', num2str(secondPeaks_wavelengths_rmse_old)]);
disp(['First Peak FWHM MSE_old: ', num2str(firstPeakFWHM_mse_old)]);
disp(['First Peak FWHM RMSE_old: ', num2str(firstPeakFWHM_rmse_old)]);
disp(['Second Peak FWHM MSE_old: ', num2str(secondPeakFWHM_mse_old)]);
disp(['Second Peak FWHM RMSE_old: ', num2str(secondPeakFWHM_rmse_old)]);
disp(['PeakRatio MSE_old: ', num2str(PeakRatio_mse_old)]);
disp(['PeakRatio RMSE_old: ', num2str(PeakRatio_rmse_old)]);

% Define the Excel file name
fileName = fullfile(path, 'Normalize_Peak_wise.xlsx');
% Write the tables to an Excel file
writetable(coefTableSpef, fileName, 'Sheet', 'coefTableSpef');
writetable(coefTableSptf, fileName, 'Sheet', 'coefTableSptf');
writetable(coefTableOldsptf, fileName, 'Sheet', 'coefTableOldsptf');
writetable(coefTableWaveletspef, fileName, 'Sheet', 'coefTableWaveletspef');
writetable(coefTableMFspef, fileName, 'Sheet', 'coefTableMFspef');
writetable(Errortable, fileName, 'Sheet', 'Erros');
writetable(Errortable_old, fileName, 'Sheet', 'Erros_old');

% Optionally display a message that the file has been saved
disp(['Data successfully saved to ', fileName]);
% Define the file path and name for the MATLAB .mat file
filenameMat = fullfile(path, 'Normalize_Peak_wise.mat');
% Save variables to a .mat file
save(filenameMat, 'coefTableSpef', 'coefTableSptf', 'coefTableOldsptf','coefTableWaveletspef','coefTableMFspef','Errortable', 'Errortable_old');
%% Single variable fit:
MFspe = normalize(MFspe,"range");
numSpectra = 10;
coefTableFitspe = table([], [], [], [], [], [], [], [],...
                        'VariableNames', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2','peakRatio','Intensity'});
for n=1:numSpectra
    % 2 Gaussian fit for vq
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(MFspe(:,n)), mean(x), std(x), max(MFspe(:,n))/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    [fitobj, gof] = fit(x, MFspe(:,n), ft);
    % Extract and compare A1 and A2
    if fitobj.A2 > fitobj.A1
        A1 = fitobj.A2;
        b1 = fitobj.b2 + 500;
        c1 = fitobj.c2 * 2.355;
        A2 = fitobj.A1;
        b2 = fitobj.b1 + 500;
        c2 = fitobj.c1 * 2.355;
    else
        A1 = fitobj.A1;
        b1 = fitobj.b1 + 500;
        c1 = fitobj.c1 * 2.355;
        A2 = fitobj.A2;
        b2 = fitobj.b2 + 500;
        c2 = fitobj.c2 * 2.355;
    end
    Ratio = A1/A2;
    Intensity = sum(sum(GTsptimg(:,:,n))); 
    % % Check FWHM and set related values to zero if FWHM > 250
    % if  c1 > 250
    %     c1 = 0;
    %     A1 = 0;
    %     b1 = 0;
    %     Ratio = 0;
    % end
    % if  c2 > 250
    %     c2 = 0;
    %     A2 = 0;
    %     b2 = 0;
    %     Ratio = 0;
    % end
    ypredtemp=fitobj(x);
    Fitspe(:,n)=ypredtemp;
    % % Calculate spectra centroid for spef
    % Intensity2 = sum(Fitspe(:,n));
    % if Intensity2 == 0
    %     centroid1 = NaN;  % Handle case where there is no intensity
    % else
    %     centroid1 = sum(wavelengths .* Fitspe(:,n)) / Intensity2;
    % end    
    % centroid_Fitspe(n) = centroid1;
    coefTableFitspe = [coefTableFitspe; table(A1, b1, c1, A2, b2, c2, Ratio, Intensity,...
                                  'VariableNames', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2','peakRatio','Intensity'})];
end
%% Calculate the centroid error:
load('D:\HongjingMao\Manuscript\Results\Evaluation\Abs_OG100k_psig10000pbg50000BG3_mao_250epo(best).mat')
%%
[centroid_Spef_errors, ~, ~, ~] = calculateErrors(coefTableSptf.centroid_Sptf, coefTableSpef.centroid_Spef);
[centroid_Oldsptf_errors, ~, ~, ~] = calculateErrors(coefTableSptf.centroid_Sptf, coefTableOldsptf.centroid_Oldsptf);

matFilePath = [path, 'SC.mat'];
save(matFilePath, 'centroid_Spef_errors','centroid_Oldsptf_errors');

resultsTable = table(centroid_Spef_errors, centroid_Oldsptf_errors, 'VariableNames', {'Centroid_Spef_Errors', 'Centroid_Oldsptf_Errors'});
excelFilePath = [path, 'SC.xlsx'];
% Save to Excel file
writetable(resultsTable, excelFilePath);
%% Spectrum-wise Evaluation:
% % Number of spectra
% numSpectra = 1000;
% % Normalize all the data:
spef = normalize(spef,"range");
Oldsptf = normalize(Oldsptf,"range");
Waveletspef = normalize(Waveletspef,"range");
MFspef = normalize(MFspef,"range");

% Initialize
spectraNames = {'rawspt', 'vq', 'spef', 'Oldsptf', 'Oldspt','Waveletspe','Waveletspef','MFspe','MFspef'};
spectraArrays = {rawspt, vq, spef, Oldsptf, Oldspt,Waveletspe,Waveletspef,MFspe,MFspef};  % Assuming these are your data arrays
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
resultsRawSpt = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsVq = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsSpef = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsOldsptf = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsOldspt = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsWaveletspe = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsWaveletspef = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsMFspe = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);
resultsMFspef = table('Size', [0 length(resultColumns)], 'VariableTypes', resultTypes, 'VariableNames', resultColumns);

for n = 1:numSpectra  
    sptCurrent = spt(:, n); % Reference spectrum for comparison
    for i = 1:length(spectraNames)
        % Using dynamic field names instead of eval
        currentSpectrum = spectraArrays{i}(:, n);

        % % Debugging output
        % fprintf('Processing %s for spectrum index %d\n', spectraNames{i}, n);

        % Calculate spectrum-wide statistics
        [~,~,mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw] = calculateSpectrumStats(sptCurrent, currentSpectrum);

        % Calculate p-value based on the Chi-square statistic
        if ismember(spectraNames{i}, {'spef', 'Oldsptf'})
            df = 301 - 6; % Degrees of freedom for spef and Oldsptf
        else
            df = 301; % Degrees of freedom for rawspt and vq
        end
        pValue = 1 - chi2cdf(chiSq, df); % p-value calculation

        % Append each result as a new row in the table
        % resultsTable = [resultsTable; {spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw}];
        newRow = {spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, pValue, areaBtw};

        % Append results to the appropriate table
        switch spectraNames{i}
            case 'rawspt'
                resultsRawSpt = [resultsRawSpt; newRow];
            case 'vq'
                resultsVq = [resultsVq; newRow];
            case 'spef'
                resultsSpef = [resultsSpef; newRow];
            case 'Oldsptf'
                resultsOldsptf = [resultsOldsptf; newRow];
            case 'Oldspt'
                resultsOldspt = [resultsOldspt; newRow];
            case 'Waveletspe'
                resultsWaveletspe = [resultsWaveletspe; newRow];
            case 'Waveletspef'
                resultsWaveletspef = [resultsWaveletspef; newRow];
            case 'MFspe'
                resultsMFspe = [resultsMFspe; newRow];
            case 'MFspef'
                resultsMFspef = [resultsMFspef; newRow];
        end

        % % Output results
        % fprintf('%s Comparison for Spectrum %d: MSE = %.4f, RMSE = %.4f, R = %.4f, R^2 = %.4f, Chi-square = %.4f, Area = %.4f\n', ...
        %         spectraNames{i}, n, mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw);
    end
end

% Define the file path and name for the Excel file
excelFilePath = [path,'spectrum_wise.xlsx'];
% Write the tables to different sheets in the same Excel file
writetable(resultsRawSpt, excelFilePath, 'Sheet', 'rawspt', 'WriteRowNames', true);
writetable(resultsVq, excelFilePath, 'Sheet', 'vq', 'WriteRowNames', true);
writetable(resultsSpef, excelFilePath, 'Sheet', 'spef', 'WriteRowNames', true);
writetable(resultsOldsptf, excelFilePath, 'Sheet', 'resultsOldsptf', 'WriteRowNames', true);
writetable(resultsOldspt, excelFilePath, 'Sheet', 'resultsOldspt', 'WriteRowNames', true);
writetable(resultsWaveletspe, excelFilePath, 'Sheet', 'resultsWaveletspe', 'WriteRowNames', true);
writetable(resultsWaveletspef, excelFilePath, 'Sheet', 'resultsWaveletspef', 'WriteRowNames', true);
writetable(resultsMFspe, excelFilePath, 'Sheet', 'resultsMFspe', 'WriteRowNames', true);
writetable(resultsMFspef, excelFilePath, 'Sheet', 'resultsMFspef', 'WriteRowNames', true);


% Define the file path and name for the MATLAB .mat file
matFilePath = [path,'spectrum_wise.mat'];
save(matFilePath, 'resultsRawSpt','resultsVq','resultsSpef','resultsOldsptf','resultsOldspt','resultsWaveletspe','resultsWaveletspef','resultsMFspe','resultsMFspef');

% Compute and print averages for each table
spectraResults = {resultsRawSpt, resultsVq, resultsSpef,resultsOldsptf,resultsOldspt,resultsWaveletspe,resultsWaveletspef,resultsMFspe,resultsMFspef};
spectraLabels = {'rawspt', 'vq', 'spef', 'resultsOldsptf', 'resultsOldspt','resultsWaveletspe','resultsWaveletspef','resultsMFspe','resultsMFspef'};

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
%% Plot single spectrum (still)
wavelengths = linspace(500, 801, 301);  % Adjust start and end wavelengths accordingly
num = 2;  % Assuming 'num' is a scalar, adjust as necessary or use a loop for multiple indices

% Initialize figure
figure;

% Use a loop to plot in a structured manner, 12 plots as 6 rows x 2 columns
for i = 1:11
    subplot(6,2,i); % Configure for 6 rows, 2 columns, plot i-th plot
    hold on;
    switch i
        case 1
            plot(wavelengths, rawspt(:, num));
            title('(a) Raw Spectrum');
        case 2
            plot(wavelengths, Oldspt(:, num));
            title('(b) Global Average');
        case 3
            plot(wavelengths, Oldsptf(:, num));
            title('(c) Global Average + LSE');
        case 4
            plot(wavelengths, vq(:, num));
            title('(d) SpecU');
        case 5
            plot(wavelengths, spef(:, num));
            title('(e) SpecU + LSE');
        case 6
            plot(wavelengths, Waveletspe(:, num));
            title('(f) Waveletspe');
        case 7
            plot(wavelengths, Waveletspef(:, num));
            title('(g) Waveletspef');
        case 8
            plot(wavelengths, MFspe(:, num));
            title('(h) MFspe');
        case 9
            plot(wavelengths, MFspef(:, num));
            title('(i) MFspef');
        case 10
            plot(wavelengths, spt(:, num));
            title('(j) GT');
        case 11
            plot(wavelengths, sptf(:, num));
            title('(k) GT + LSE');
    end
    pause(1); % Pause to view each plot
    hold off;
end
%% live Plot 
wavelengths = linspace(500, 801, 301);  % Adjust start and end wavelengths accordingly

% Define the range of indices
num = 1:200;

% Initialize figure
figure;
plots = cell(12,1); % Cell array to store plot handles

% Set up a plotting grid of 6 rows and 2 columns and initialize plots
for i = 1:12
    subplot(6, 2, i);
    plots{i} = plot(wavelengths, zeros(size(wavelengths)), 'LineWidth', 2); % Initialize with zero
    hold on;
    xlabel('Wavelength (nm)');
    ylabel('Intensity');
end

% Titles for each subplot
titles = {'(a) Raw Spectrum', '(b) Global Average', '(c) Global Average + LSE', ...
          '(d) SpecU', '(e) SpecU + LSE', '(f) Waveletspe', '(g) Waveletspef', ...
          '(h) MFspe', '(i) MFspef', '(j) GT', '(k) GT + LSE', '(l) Additional Plot'};

% Iterate through each spectrum index
for n = num
    for i = 1:11
        subplot(6, 2, i); % Focus subplot
        % Set new Y data depending on the plot index
        switch i
            case 1, set(plots{i}, 'YData', rawspt(:, n));
            case 2, set(plots{i}, 'YData', Oldspt(:, n));
            case 3, set(plots{i}, 'YData', Oldsptf(:, n));
            case 4, set(plots{i}, 'YData', vq(:, n));
            case 5, set(plots{i}, 'YData', spef(:, n));
            case 6, set(plots{i}, 'YData', Waveletspe(:, n));
            case 7, set(plots{i}, 'YData', Waveletspef(:, n));
            case 8, set(plots{i}, 'YData', MFspe(:, n));
            case 9, set(plots{i}, 'YData', MFspef(:, n));
            case 10, set(plots{i}, 'YData', spt(:, n));
            case 11, set(plots{i}, 'YData', sptf(:, n));
        end
        title(titles{i}); % Set titles defined in the array
        drawnow; % Update plot immediately
    end
    pause(0.1); % Brief pause to allow viewing of each update
end
%% For prism plot single spectra
for n = 1
a_Raw = rawspt(:,n);
a_SpecUNet = vq(:,n);
a_SpecUNet_LSE = spef(:,n);
a_GlobalAverage = Oldspt(:,n);
a_GlobalAverage_LSE = Oldsptf(:,n);
a_GroundTruth = spt(:,n);
a_GroundTruth_LSE = sptf(:,n);
end
%% check if two variables are same
areSame = isequal(rawspt(:,1:10), Oldspt);  % Returns false because data types are different
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
    % R = corrcoef(spect1, spect2); %This is peason         % correlation coefficient/ Pearson coefficient(corrcoef) is different from cross-correlation, use cross-coefficient(xcross) here
    % R = R(1,2);  % Extract the off-diagonal element which is the correlation coefficient
    R = corr(spect1, spect2, 'Type', 'Spearman');  % This is Spearman
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

%%  
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
function [errors, squared_errors, mse, rmse] = calculateErrors(y_true, y_pred)
    % Validate input sizes
    if length(y_true) ~= length(y_pred)
        error('Input vectors y_true and y_pred must be of the same length.');
    end
    % Errors/Absolute Errors
    % errors = abs(y_true - y_pred);    % If calculating absolute errors 
    errors = y_true - y_pred;    
    % Squared Errors
    squared_errors = (y_true - y_pred).^2;
    % Mean Squared Error
    mse = mean(squared_errors);
    % Root Mean Squared Error
    rmse = sqrt(mse);
    % Display calculated values 
    % fprintf('Absolute_error: %f\n', errors);
    % fprintf('Squared_errors: %f\n', squared_errors);
    % fprintf('MSE: %f\n', mse);
    % fprintf('RMSE: %f\n', rmse);
end

%% Calculating SNR
function [SNR] = calculateSNR(image, n)
    % Extract the signal and background data from the image matrix.
    signal = image(7:10, :, n);
    BG = image(1:4, :, n);
    
    % Calculate the peak signal-to-noise ratio (SNR).
    SNR = (max(signal(:)) - mean(BG(:))) / std(BG(:));
    
    % Convert the SNR from linear scale to dB scale.
    SNR = 10 * log10(SNR);
end
