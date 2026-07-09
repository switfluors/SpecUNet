%% Evaluation of SpecUNet on independent testing data
%
% Compares five background-removal / spectrum-recovery methods on an
% independent testing set:
%   1. Conv U-Net (SpecUNet)          -> YPred  / Predictspe
%   2. Wavelet denoising              -> Waveletsptimg / Waveletspe
%   3. Median filter                  -> PredImgMF / MFspe
%   4. Conventional (mean background) -> Oldsptimg / Oldspt
%
% Evaluations (each section can be run as an individual cell):
%   - Image-wise SNR, RMSE and SSIM against the true background (tbg4)
%   - Peak-wise two-Gaussian fits (peak position, FWHM, peak ratio errors)
%   - Spectral-centroid errors
%   - Spectrum-wise statistics (MSE, RMSE, Spearman R, chi-square, area)
%
% Expected contents of the testing .mat file:
%   sptimg4 : 16 x 128 x N noisy spectral images
%   tbg4    : 16 x 128 x N ground-truth backgrounds
%   spt     : N x 301 ground-truth spectra (transposed to 301 x N below)
%
% Toolboxes: Deep Learning, Curve Fitting, Wavelet, Image Processing,
% Statistics and Machine Learning, System Identification (goodnessOfFit).

clearvars; close all; clc

%% Configuration -- edit these paths for your system
convNetFile = 'Yourpath\net.mat';     % trained Conv U-Net (variable "net")
testingFile = 'Yourpath\TestingData.mat'; % independent testing data
resultsDir  = 'Yourpath\Testfolder';    % evaluation tables are written here
figureDir   = 'Yourpath\Testfolder\figure';    % exported PDF figures are written here
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end
if ~exist(figureDir, 'dir'), mkdir(figureDir); end
%% Load networks and data
tmp  = load(convNetFile); net  = tmp.net;
load(testingFile, 'sptimg4', 'tbg4', 'spt');
spt     = spt';             % ground-truth spectra, 301 x N
sptimg4 = double(sptimg4);
tbg4    = double(tbg4);
numSpectra = size(sptimg4, 3);    % number of spectra to evaluate
%% Background prediction with all methods
YPred         = zeros(16, 128, numSpectra);  % Conv U-Net background
Predictspe    = zeros(16, 128, numSpectra);  % Conv U-Net spectrum image
Waveletsptimg = zeros(16, 128, numSpectra);  % wavelet-denoised image
PredImgMF     = zeros(16, 128, numSpectra);  % median-filtered image
Oldsptimg     = zeros(16, 128, numSpectra);  % conventional method image

meanBack = mean(tbg4, 3);  % global average background

tic
for n = 1:numSpectra
    % Conv U-Net
    YPred(:,:,n)      = predict(net, sptimg4(:,:,n));
    Predictspe(:,:,n) = sptimg4(:,:,n) - YPred(:,:,n);

    % Wavelet denoising
    Waveletsptimg(:,:,n) = wdenoise2(sptimg4(:,:,n));

    % Median filter
    PredImgMF(:,:,n) = medfilt2(sptimg4(:,:,n));

    % Conventional method: subtract the average background of the stack
    Oldsptimg(:,:,n) = sptimg4(:,:,n) - meanBack;
end
toc
%% Extract 1-D spectra (rows 7-10) and interpolate 128 -> 301 points
xq = 1:128/303:128;  % 301-point query grid 

% Conv U-Net predicted spectrum
sptn = squeeze(mean(Predictspe(7:10,:,:), 1));
vq   = interp1(1:128, sptn, xq);

% Raw spectrum
rawspt = squeeze(mean(sptimg4(7:10,:,:), 1));
rawspt = interp1(1:128, rawspt, xq);

% Wavelet: denoise the raw 1-D spectrum
rawsptforwavelet = squeeze(mean(sptimg4(7:10,:,:), 1));
Waveletspe = zeros(128, numSpectra);
for n = 1:numSpectra
    Waveletspe(:,n) = wdenoise(rawsptforwavelet(:,n));
end
Waveletspe = interp1(1:128, Waveletspe, xq);
Wavelet_BG = sptimg4 - Waveletsptimg;

% Median filter
MFspe = squeeze(mean(PredImgMF(7:10,:,:), 1));
MFspe = interp1(1:128, MFspe, xq);
MF_BG = sptimg4 - PredImgMF;

% Ground truth 
GTsptimg = sptimg4 - tbg4;
GTspt    = squeeze(mean(GTsptimg(7:10,:,:), 1));
GTspt    = interp1(1:128, GTspt, xq);

% Conventional method
Oldspt = squeeze(mean(Oldsptimg(7:10,:,:), 1));
Oldspt = interp1(1:128, Oldspt, xq);
Old_BG = repmat(meanBack, [1, 1, numSpectra]);

%% (Optional) Browse all methods frame by frame
panels = {sptimg4,       'sptimg4';
          YPred,         'Conv U-net';
          Predictspe,    'Conv U-net spe';
          Oldsptimg,     'Old spe';
          GTsptimg,      'GT spe';
          Waveletsptimg, 'Wavelet spe';
          PredImgMF,     'MF spe'};

figure; colormap gray
for n = 1:100
    for k = 1:size(panels, 1)
        subplot(size(panels, 1), 1, k)
        imagesc(panels{k,1}(:,:,n)); axis equal
        title(sprintf('%s - Frame %d', panels{k,2}, n))
    end
    drawnow
    pause(1)
end

%% (Optional) Browse raw image / predicted background / predicted spectrum
figure; colormap gray
for n = 1:200
    subplot(3,1,1); imagesc(sptimg4(:,:,n));    axis equal
    title(sprintf('Raw Image - Frame %d', n))
    subplot(3,1,2); imagesc(YPred(:,:,n));      axis equal
    title(sprintf('Predicted Background - Frame %d', n))
    subplot(3,1,3); imagesc(Predictspe(:,:,n)); axis equal
    title(sprintf('Predicted Spectra - Frame %d', n))
    drawnow
    pause(1)
end

%% (Optional)Export a single frame as vector PDFs
frameNumber = 8;
exports = {sptimg4,    'Raw Image',            'Raw_Image.pdf';
           YPred,      'Predicted Background', 'Predicted_Background.pdf';
           Predictspe, 'Predicted Spectra',    'Predicted_Spectra.pdf'};

for k = 1:size(exports, 1)
    figure
    imagesc(exports{k,1}(:,:,frameNumber))
    colormap gray; axis equal; axis tight; colorbar
    title(sprintf('%s - Frame %d', exports{k,2}, frameNumber))
    yticks([1 8 16]); xticks(20:20:120)
    exportgraphics(gca, fullfile(figureDir, exports{k,3}), 'ContentType', 'vector');
end

%% Image-wise SNR
SNR_Raw     = zeros(numSpectra, 1);
SNR_ConvU   = zeros(numSpectra, 1);
SNR_Wavelet = zeros(numSpectra, 1);
SNR_MF      = zeros(numSpectra, 1);
SNR_Old     = zeros(numSpectra, 1);

for n = 1:numSpectra
    SNR_Raw(n)     = calculateSNR(sptimg4, n);
    SNR_ConvU(n)   = calculateSNR(Predictspe, n);
    SNR_Wavelet(n) = calculateSNR(Waveletsptimg, n);
    SNR_MF(n)      = calculateSNR(PredImgMF, n);
    SNR_Old(n)     = calculateSNR(Oldsptimg, n);
end

saveMetricsTable(resultsDir, 'Image_wise_SNR', ...
    {'SNR_Raw', 'SNR_ConvU', 'SNR_Wavelet', 'SNR_MF', 'SNR_Old'}, ...
    [SNR_Raw, SNR_ConvU, SNR_Wavelet, SNR_MF, SNR_Old]);

%% Plot SNR distributions
numBins = 30;
snrPlots = {SNR_Raw,     'SNR for Raw Images';
            SNR_ConvU,   'SNR for ConvU-Net Predictions';
            SNR_Wavelet, 'SNR for Wavelet Denoising';
            SNR_MF,      'SNR for Median Filter';
            SNR_Old,     'SNR for Conventional Method'};

figure
for k = 1:size(snrPlots, 1)
    subplot(2, 3, k)
    histogram(snrPlots{k,1}, numBins)
    title(snrPlots{k,2})
    xlabel('SNR (dB)'); ylabel('Frequency')
end
sgtitle('Comparison of SNR Distributions Across Different Methods')

%% Image-wise RMSE (predicted background vs true background)
ConvU_BG_RMSE   = squeeze(rmse(YPred,      tbg4, [1 2]));
Wavelet_BG_RMSE = squeeze(rmse(Wavelet_BG, tbg4, [1 2]));
MF_BG_RMSE      = squeeze(rmse(MF_BG,      tbg4, [1 2]));
Old_BG_RMSE     = squeeze(rmse(Old_BG,     tbg4, [1 2]));

saveMetricsTable(resultsDir, 'Image_wise_RMSE', ...
    {'ConvU_BG_RMSE',  'Wavelet_BG_RMSE', 'Old_BG_RMSE', 'MF_BG_RMSE'}, ...
    [ConvU_BG_RMSE,  Wavelet_BG_RMSE, Old_BG_RMSE, MF_BG_RMSE]);

%% Image-wise SSIM (predicted background vs true background)
ConvU_BG_SSIM   = zeros(numSpectra, 1);
UpU_BG_SSIM     = zeros(numSpectra, 1);
Wavelet_BG_SSIM = zeros(numSpectra, 1);
Old_BG_SSIM     = zeros(numSpectra, 1);
MF_BG_SSIM      = zeros(numSpectra, 1);

for n = 1:numSpectra
    ConvU_BG_SSIM(n)   = ssim(YPred(:,:,n),      tbg4(:,:,n));
    Wavelet_BG_SSIM(n) = ssim(Wavelet_BG(:,:,n), tbg4(:,:,n));
    Old_BG_SSIM(n)     = ssim(Old_BG(:,:,n),     tbg4(:,:,n));
    MF_BG_SSIM(n)      = ssim(MF_BG(:,:,n),      tbg4(:,:,n));
end

saveMetricsTable(resultsDir, 'Image_wise_SSIM', ...
    {'ConvU_BG_SSIM', 'Wavelet_BG_SSIM', 'Old_BG_SSIM', 'MF_BG_SSIM'}, ...
    [ConvU_BG_SSIM, Wavelet_BG_SSIM, Old_BG_SSIM, MF_BG_SSIM]);

%% Peak-wise evaluation: two-Gaussian fits of every spectrum
x = (1:301)';        % spectral axis used for fitting
wavelengths = x;     % weights for the centroid calculation (index units)

% Fitted curves (301 x numSpectra)
spef        = zeros(301, numSpectra);  % SpecU (Conv U-Net) + LSE
sptf        = zeros(301, numSpectra);  % ground truth + LSE
Oldsptf     = zeros(301, numSpectra);  % conventional + LSE
Waveletspef = zeros(301, numSpectra);  % wavelet + LSE
MFspef      = zeros(301, numSpectra);  % median filter + LSE

% Fit coefficients per spectrum: [A1 b1 FWHM1 A2 b2 FWHM2 peakRatio]
coefSpef    = zeros(numSpectra, 7);
coefSptf    = zeros(numSpectra, 7);
coefOldsptf = zeros(numSpectra, 7);
coefWavelet = zeros(numSpectra, 7);
coefMF      = zeros(numSpectra, 7);

% Spectral centroids
centroid_Spef        = zeros(numSpectra, 1);
centroid_Sptf        = zeros(numSpectra, 1);
centroid_Spt         = zeros(numSpectra, 1);
centroid_Oldsptf     = zeros(numSpectra, 1);
centroid_Waveletspef = zeros(numSpectra, 1);
centroid_MFspef      = zeros(numSpectra, 1);

Intensity = zeros(numSpectra, 1);  % total ground-truth intensity per frame

% Feature errors, SpecU (new) and conventional (old) vs ground-truth fit
firstPeaks_wavelengths_errors          = zeros(numSpectra, 1);
firstPeaks_wavelengths_squared_errors  = zeros(numSpectra, 1);
secondPeaks_wavelengths_errors         = zeros(numSpectra, 1);
secondPeaks_wavelengths_squared_errors = zeros(numSpectra, 1);
firstPeakFWHM_errors                   = zeros(numSpectra, 1);
firstPeakFWHM_squared_errors           = zeros(numSpectra, 1);
secondPeakFWHM_errors                  = zeros(numSpectra, 1);
secondPeakFWHM_squared_errors          = zeros(numSpectra, 1);
PeakRatio_errors                       = zeros(numSpectra, 1);
PeakRatio_squared_errors               = zeros(numSpectra, 1);

firstPeaks_wavelengths_errors_old          = zeros(numSpectra, 1);
firstPeaks_wavelengths_squared_errors_old  = zeros(numSpectra, 1);
secondPeaks_wavelengths_errors_old         = zeros(numSpectra, 1);
secondPeaks_wavelengths_squared_errors_old = zeros(numSpectra, 1);
firstPeakFWHM_errors_old                   = zeros(numSpectra, 1);
firstPeakFWHM_squared_errors_old           = zeros(numSpectra, 1);
secondPeakFWHM_errors_old                  = zeros(numSpectra, 1);
secondPeakFWHM_squared_errors_old          = zeros(numSpectra, 1);
PeakRatio_errors_old                       = zeros(numSpectra, 1);
PeakRatio_squared_errors_old               = zeros(numSpectra, 1);

f = waitbar(0, 'Fitting spectra');
for n = 1:numSpectra
    Intensity(n) = sum(GTsptimg(:,:,n), 'all');

    [coefSpef(n,:),    spef(:,n)]        = fitTwoGaussians(x, vq(:,n));
    [coefSptf(n,:),    sptf(:,n)]        = fitTwoGaussians(x, spt(:,n));
    [coefOldsptf(n,:), Oldsptf(:,n)]     = fitTwoGaussians(x, Oldspt(:,n));
    [coefWavelet(n,:), Waveletspef(:,n)] = fitTwoGaussians(x, Waveletspe(:,n));
    [coefMF(n,:),      MFspef(:,n)]      = fitTwoGaussians(x, MFspe(:,n));

    centroid_Spef(n)        = spectralCentroid(wavelengths, spef(:,n));
    centroid_Sptf(n)        = spectralCentroid(wavelengths, sptf(:,n));
    centroid_Spt(n)         = spectralCentroid(wavelengths, spt(:,n));
    centroid_Oldsptf(n)     = spectralCentroid(wavelengths, Oldsptf(:,n));
    centroid_Waveletspef(n) = spectralCentroid(wavelengths, Waveletspef(:,n));
    centroid_MFspef(n)      = spectralCentroid(wavelengths, MFspef(:,n));

    % Feature errors vs the ground-truth fit (coefficient columns:
    % 2 = first-peak wavelength, 5 = second-peak wavelength,
    % 3 = first-peak FWHM, 6 = second-peak FWHM, 7 = peak ratio)
    [firstPeaks_wavelengths_errors(n),      firstPeaks_wavelengths_squared_errors(n)]      = peakError(coefSptf(n,2), coefSpef(n,2));
    [firstPeaks_wavelengths_errors_old(n),  firstPeaks_wavelengths_squared_errors_old(n)]  = peakError(coefSptf(n,2), coefOldsptf(n,2));
    [secondPeaks_wavelengths_errors(n),     secondPeaks_wavelengths_squared_errors(n)]     = peakError(coefSptf(n,5), coefSpef(n,5));
    [secondPeaks_wavelengths_errors_old(n), secondPeaks_wavelengths_squared_errors_old(n)] = peakError(coefSptf(n,5), coefOldsptf(n,5));
    [firstPeakFWHM_errors(n),               firstPeakFWHM_squared_errors(n)]               = peakError(coefSptf(n,3), coefSpef(n,3));
    [firstPeakFWHM_errors_old(n),           firstPeakFWHM_squared_errors_old(n)]           = peakError(coefSptf(n,3), coefOldsptf(n,3));
    [secondPeakFWHM_errors(n),              secondPeakFWHM_squared_errors(n)]              = peakError(coefSptf(n,6), coefSpef(n,6));
    [secondPeakFWHM_errors_old(n),          secondPeakFWHM_squared_errors_old(n)]          = peakError(coefSptf(n,6), coefOldsptf(n,6));
    [PeakRatio_errors(n),                   PeakRatio_squared_errors(n)]                   = peakError(coefSptf(n,7), coefSpef(n,7));
    [PeakRatio_errors_old(n),               PeakRatio_squared_errors_old(n)]               = peakError(coefSptf(n,7), coefOldsptf(n,7));

    waitbar(n/numSpectra, f, sprintf('Progress: %d %%', floor(n/numSpectra*100)));
end
close(f)

% Assemble coefficient tables (column names kept from the original analysis)
coefTableSpef = array2table([coefSpef, Intensity, centroid_Spef], 'VariableNames', ...
    {'Spef_firstPeakValues', 'Spef_firstPeakWavelengths', 'Spef_firstPeakFWHM', ...
     'Spef_secondPeakValues', 'Spef_secondPeakWavelengths', 'Spef_secondPeakFWHM', ...
     'Spef_peakRatio', 'Intensity', 'centroid_Spef'});
coefTableSptf = array2table([coefSptf, Intensity, centroid_Sptf, centroid_Spt], 'VariableNames', ...
    {'Sptf_firstPeakValues', 'Sptf_firstPeakWavelengths', 'Sptf_firstPeakFWHM', ...
     'Sptf_secondPeakValues', 'Sptf_secondPeakWavelengths', 'Sptf_secondPeakFWHM', ...
     'Sptf_peakRatio', 'Intensity', 'centroid_Sptf', 'centroid_Spt'});
coefTableOldsptf = array2table([coefOldsptf, Intensity, centroid_Oldsptf], 'VariableNames', ...
    {'Oldsptf_firstPeakValues', 'Oldsptf_firstPeakWavelengths', 'Oldsptf_firstPeakFWHM', ...
     'Oldsptf_secondPeakValues', 'Oldsptf_secondPeakWavelengths', 'Oldsptf_secondPeakFWHM', ...
     'Oldsptf_peakRatio', 'Intensity', 'centroid_Oldsptf'});
coefTableWaveletspef = array2table([coefWavelet, Intensity, centroid_Waveletspef], 'VariableNames', ...
    {'Waveletspef_firstPeakValues', 'Waveletspef_firstPeakWavelengths', 'Waveletspef_firstPeakFWHM', ...
     'Waveletspef_secondPeakValues', 'Waveletspef_secondPeakWavelengths', 'Waveletspef_secondPeakFWHM', ...
     'Waveletspef_peakRatio', 'Intensity', 'centroid_Waveletspef'});
coefTableMFspef = array2table([coefMF, Intensity, centroid_MFspef], 'VariableNames', ...
    {'MFspef_firstPeakValues', 'MFspef_firstPeakWavelengths', 'MFspef_firstPeakFWHM', ...
     'MFspef_secondPeakValues', 'MFspef_secondPeakWavelengths', 'MFspef_secondPeakFWHM', ...
     'MFspef_peakRatio', 'Intensity', 'centroid_MFspef'});

% MSE and RMSE, SpecU vs ground truth
firstPeaks_wavelengths_mse   = mean(firstPeaks_wavelengths_squared_errors, 'omitnan');
firstPeaks_wavelengths_rmse  = sqrt(firstPeaks_wavelengths_mse);
secondPeaks_wavelengths_mse  = mean(secondPeaks_wavelengths_squared_errors, 'omitnan');
secondPeaks_wavelengths_rmse = sqrt(secondPeaks_wavelengths_mse);
firstPeakFWHM_mse            = mean(firstPeakFWHM_squared_errors, 'omitnan');
firstPeakFWHM_rmse           = sqrt(firstPeakFWHM_mse);
secondPeakFWHM_mse           = mean(secondPeakFWHM_squared_errors, 'omitnan');
secondPeakFWHM_rmse          = sqrt(secondPeakFWHM_mse);
PeakRatio_mse                = mean(PeakRatio_squared_errors, 'omitnan');
PeakRatio_rmse               = sqrt(PeakRatio_mse);

% MSE and RMSE, conventional method vs ground truth
firstPeaks_wavelengths_mse_old   = mean(firstPeaks_wavelengths_squared_errors_old, 'omitnan');
firstPeaks_wavelengths_rmse_old  = sqrt(firstPeaks_wavelengths_mse_old);
secondPeaks_wavelengths_mse_old  = mean(secondPeaks_wavelengths_squared_errors_old, 'omitnan');
secondPeaks_wavelengths_rmse_old = sqrt(secondPeaks_wavelengths_mse_old);
firstPeakFWHM_mse_old            = mean(firstPeakFWHM_squared_errors_old, 'omitnan');
firstPeakFWHM_rmse_old           = sqrt(firstPeakFWHM_mse_old);
secondPeakFWHM_mse_old           = mean(secondPeakFWHM_squared_errors_old, 'omitnan');
secondPeakFWHM_rmse_old          = sqrt(secondPeakFWHM_mse_old);
PeakRatio_mse_old                = mean(PeakRatio_squared_errors_old, 'omitnan');
PeakRatio_rmse_old               = sqrt(PeakRatio_mse_old);

Errortable = table(firstPeaks_wavelengths_errors, ...
    firstPeaks_wavelengths_squared_errors, secondPeaks_wavelengths_errors, ...
    secondPeaks_wavelengths_squared_errors, firstPeakFWHM_errors, ...
    firstPeakFWHM_squared_errors, secondPeakFWHM_errors, secondPeakFWHM_squared_errors, ...
    PeakRatio_errors, PeakRatio_squared_errors);

Errortable_old = table(firstPeaks_wavelengths_errors_old, ...
    firstPeaks_wavelengths_squared_errors_old, secondPeaks_wavelengths_errors_old, ...
    secondPeaks_wavelengths_squared_errors_old, firstPeakFWHM_errors_old, ...
    firstPeakFWHM_squared_errors_old, secondPeakFWHM_errors_old, secondPeakFWHM_squared_errors_old, ...
    PeakRatio_errors_old, PeakRatio_squared_errors_old);

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

% Save the peak-wise results (sheet names kept for compatibility)
fileName = fullfile(resultsDir, 'Normalize_Peak_wise.xlsx');
writetable(coefTableSpef, fileName, 'Sheet', 'coefTableSpef');
writetable(coefTableSptf, fileName, 'Sheet', 'coefTableSptf');
writetable(coefTableOldsptf, fileName, 'Sheet', 'coefTableOldsptf');
writetable(coefTableWaveletspef, fileName, 'Sheet', 'coefTableWaveletspef');
writetable(coefTableMFspef, fileName, 'Sheet', 'coefTableMFspef');
writetable(Errortable, fileName, 'Sheet', 'Erros');
writetable(Errortable_old, fileName, 'Sheet', 'Erros_old');
disp(['Data successfully saved to ', fileName]);

save(fullfile(resultsDir, 'Normalize_Peak_wise.mat'), ...
    'coefTableSpef', 'coefTableSptf', 'coefTableOldsptf', ...
    'coefTableWaveletspef', 'coefTableMFspef', 'Errortable', 'Errortable_old');

%% Spectral-centroid errors
centroid_Spef_errors    = calculateErrors(coefTableSptf.centroid_Sptf, coefTableSpef.centroid_Spef);
centroid_Oldsptf_errors = calculateErrors(coefTableSptf.centroid_Sptf, coefTableOldsptf.centroid_Oldsptf);

save(fullfile(resultsDir, 'SC.mat'), 'centroid_Spef_errors', 'centroid_Oldsptf_errors');

resultsTable = table(centroid_Spef_errors, centroid_Oldsptf_errors, ...
    'VariableNames', {'Centroid_Spef_Errors', 'Centroid_Oldsptf_Errors'});
writetable(resultsTable, fullfile(resultsDir, 'SC.xlsx'));

%% Spectrum-wise evaluation
% Normalize the fitted spectra to [0, 1] before comparison
spef        = normalize(spef, 'range');
Oldsptf     = normalize(Oldsptf, 'range');
Waveletspef = normalize(Waveletspef, 'range');
MFspef      = normalize(MFspef, 'range');

spectraNames  = {'rawspt', 'vq', 'spef', 'Oldsptf', 'Oldspt', 'Waveletspe', 'Waveletspef', 'MFspe', 'MFspef'};
spectraArrays = {rawspt, vq, spef, Oldsptf, Oldspt, Waveletspe, Waveletspef, MFspe, MFspef};
tableNames    = {'resultsRawSpt', 'resultsVq', 'resultsSpef', 'resultsOldsptf', 'resultsOldspt', ...
                 'resultsWaveletspe', 'resultsWaveletspef', 'resultsMFspe', 'resultsMFspef'};
sheetNames    = {'rawspt', 'vq', 'spef', 'resultsOldsptf', 'resultsOldspt', ...
                 'resultsWaveletspe', 'resultsWaveletspef', 'resultsMFspe', 'resultsMFspef'};
resultColumns = {'SpectrumName', 'SpectrumIndex', 'MSE', 'RMSE', 'CorrelationCoefficient', ...
                 'RSquared', 'ChiSquared', 'pValue', 'AreaBetweenCurves'};

excelFilePath = fullfile(resultsDir, 'spectrum_wise.xlsx');
results = struct();

for i = 1:length(spectraNames)
    metrics = zeros(numSpectra, 7);  % [MSE RMSE R R2 chiSq pValue area]

    % Degrees of freedom: 301 - 6 fit parameters for the Gaussian-fitted
    % spectra used in the manuscript comparison, 301 otherwise
    if ismember(spectraNames{i}, {'spef', 'Oldsptf'})
        df = 301 - 6;
    else
        df = 301;
    end

    for n = 1:numSpectra
        sptCurrent = spt(:, n);  % reference spectrum
        [~, ~, mseVal, rmseVal, RVal, R2Val, chiSq, areaBtw] = ...
            calculateSpectrumStats(sptCurrent, spectraArrays{i}(:, n));
        pValue = 1 - chi2cdf(chiSq, df);
        metrics(n,:) = [mseVal, rmseVal, RVal, R2Val, chiSq, pValue, areaBtw];
    end

    T = [table(repmat(spectraNames(i), numSpectra, 1), (1:numSpectra)', ...
               'VariableNames', resultColumns(1:2)), ...
         array2table(metrics, 'VariableNames', resultColumns(3:9))];
    results.(tableNames{i}) = T;
    writetable(T, excelFilePath, 'Sheet', sheetNames{i});
end

save(fullfile(resultsDir, 'spectrum_wise.mat'), '-struct', 'results');

% Print averages for each method
for i = 1:length(tableNames)
    T = results.(tableNames{i});
    fprintf('\nAverages for %s:\n', spectraNames{i});
    fprintf('Average MSE: %.4f\n', mean(T.MSE));
    fprintf('Average RMSE: %.4f\n', mean(T.RMSE));
    fprintf('Average R: %.4f\n', mean(T.CorrelationCoefficient));
    fprintf('Average R-squared: %.4f\n', mean(T.RSquared));
    fprintf('Average Area Between Curves: %.4f\n', mean(T.AreaBetweenCurves));
    fprintf('Average Chi-squared: %.4f\n', mean(T.ChiSquared));
    fprintf('Average P-Value: %.4f\n', mean(T.pValue));
end

%% (Optional) Plot a single spectrum, all methods
wavelengthsNm = linspace(500, 801, 301);
num = 2;  % spectrum index to plot

spectrumPlots = {rawspt,      '(a) Raw Spectrum';
                 Oldspt,      '(b) Global Average';
                 Oldsptf,     '(c) Global Average + LSE';
                 vq,          '(d) SpecU';
                 spef,        '(e) SpecU + LSE';
                 Waveletspe,  '(f) Waveletspe';
                 Waveletspef, '(g) Waveletspef';
                 MFspe,       '(h) MFspe';
                 MFspef,      '(i) MFspef';
                 spt,         '(j) GT';
                 sptf,        '(k) GT + LSE'};

figure
for i = 1:size(spectrumPlots, 1)
    subplot(6, 2, i)
    plot(wavelengthsNm, spectrumPlots{i,1}(:, num))
    title(spectrumPlots{i,2})
end

%% (Optional) Live plot: browse spectra across all methods
figure
nPlots = size(spectrumPlots, 1);
plots = cell(nPlots, 1);
for i = 1:nPlots
    subplot(6, 2, i)
    plots{i} = plot(wavelengthsNm, zeros(size(wavelengthsNm)), 'LineWidth', 2);
    title(spectrumPlots{i,2})
    xlabel('Wavelength (nm)'); ylabel('Intensity')
end

for n = 1:200
    for i = 1:nPlots
        set(plots{i}, 'YData', spectrumPlots{i,1}(:, n));
    end
    drawnow
    pause(0.1)
end

%% ------------------------- Local functions -------------------------

function [coefs, yFit] = fitTwoGaussians(x, y)
% Two-Gaussian least-squares fit of a spectrum.
% Returns COEFS = [A1 b1 FWHM1 A2 b2 FWHM2 peakRatio] with the peaks sorted
% by amplitude (largest first), centers converted from spectral index to
% wavelength (+500 nm) and sigmas converted to FWHM (*2.355), and YFIT the
% fitted curve evaluated at x.
    ft = fittype('A1*exp(-((x-b1)/c1)^2) + A2*exp(-((x-b2)/c2)^2)', ...
        'independent', 'x', ...
        'coefficients', {'A1', 'b1', 'c1', 'A2', 'b2', 'c2'});
    opts = fitoptions('Method', 'NonlinearLeastSquares', ...
                      'StartPoint', [max(y), mean(x), std(x), max(y)/2, mean(x) + 1, std(x)], ...
                      'Lower', [0, min(x), 0, 0, min(x), 0], ...
                      'Upper', [Inf, max(x), Inf, Inf, max(x), Inf], ...
                      'MaxFunEvals', 600, ...
                      'MaxIter', 400);
    ft = setoptions(ft, opts);
    fitobj = fit(x, y, ft);

    if fitobj.A2 > fitobj.A1
        coefs = [fitobj.A2, fitobj.b2 + 500, fitobj.c2 * 2.355, ...
                 fitobj.A1, fitobj.b1 + 500, fitobj.c1 * 2.355];
    else
        coefs = [fitobj.A1, fitobj.b1 + 500, fitobj.c1 * 2.355, ...
                 fitobj.A2, fitobj.b2 + 500, fitobj.c2 * 2.355];
    end
    coefs(7) = coefs(1) / coefs(4);  % peak ratio A1/A2
    yFit = fitobj(x);
end

function c = spectralCentroid(wavelengths, spectrum)
% Intensity-weighted centroid of a spectrum; NaN if the spectrum sums to zero.
    total = sum(spectrum);
    if total == 0
        c = NaN;
    else
        c = sum(wavelengths .* spectrum) / total;
    end
end

function [err, sqErr] = peakError(trueVal, predVal)
% Signed and squared error between two fitted peak features.
% Returns NaN when either value is exactly zero (failed / suppressed fit).
    if trueVal == 0 || predVal == 0
        err = NaN;
        sqErr = NaN;
    else
        [err, sqErr] = calculateErrors(trueVal, predVal);
    end
end

function [absError, squaredError, mse, rmse, R, R_squared, chi_square, areaBetween] = calculateSpectrumStats(spect1, spect2)
% Spectrum-wise comparison statistics between a reference and a test spectrum.
    absError = abs(spect1 - spect2);
    squaredError = (spect1 - spect2).^2;
    mse = mean(squaredError, 'all');
    rmse = sqrt(mse);

    R = corr(spect1, spect2, 'Type', 'Spearman');
    R_squared = R^2;

    % NRMSE goodness of fit (used in place of a classical chi-square)
    chi_square = goodnessOfFit(spect1, spect2, 'NRMSE');

    areaBetween = trapz(abs(spect1 - spect2));
end

function [errors, squared_errors, mse, rmse] = calculateErrors(y_true, y_pred)
% Signed errors, squared errors, MSE and RMSE between two vectors.
    if length(y_true) ~= length(y_pred)
        error('Input vectors y_true and y_pred must be of the same length.');
    end
    errors = y_true - y_pred;
    squared_errors = (y_true - y_pred).^2;
    mse = mean(squared_errors);
    rmse = sqrt(mse);
end

function SNR = calculateSNR(image, n)
% Peak SNR (dB) of frame n: signal rows 7-10 vs background rows 1-4.
    signal = image(7:10, :, n);
    BG = image(1:4, :, n);
    SNR = (max(signal(:)) - mean(BG(:))) / std(BG(:));
    SNR = 10 * log10(SNR);
end

function saveMetricsTable(outDir, baseName, names, data)
% Save the columns of DATA both as named variables in <baseName>.mat and as
% a spreadsheet <baseName>.xlsx with NAMES as the header row.
    S = cell2struct(num2cell(data, 1), names, 2);
    save(fullfile(outDir, [baseName '.mat']), '-struct', 'S');
    writecell([names; num2cell(data)], fullfile(outDir, [baseName '.xlsx']));
end
