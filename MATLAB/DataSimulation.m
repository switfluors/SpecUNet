clear; clc; clf;

% Define parameters for the Point Spread Function (PSF)
lambda = 670;                  % Emission peak wavelength of Alexa 647 dye (nm)
NA = 1.49;                     % Numerical aperture of the imaging system
px = 160 / 1.5;                % Effective pixel size of the EMCCD camera (nm/pixel)
w = 0.61 * lambda / NA / 2.355; % Standard deviation of PSF calculated from diffraction-limited FWHM (nm)

numSpectra = 100;              % Number of spectra to simulate (adjust as needed)
rng('shuffle');                % Initialize random seed based on current time
w = w * (rand(1, numSpectra) * 0.5 + 1); % Simulate varying focus conditions by adjusting PSF widths
fwhm = w * 2.355;              % Corresponding full-width at half-maximum (FWHM) for each PSF

% Load and configure Gaussian spectral profiles (modify parameters as needed):
wl = 500:800; % Wavelength range (nm)
% Parameters for primary Gaussian peaks:
m1 = randi(250, 1, numSpectra) + 500; % Peak wavelengths (nm) of primary peaks
s1 = randi(50, 1, numSpectra) + 5;    % Standard deviations (nm) of primary peaks
% Parameters for secondary Gaussian peaks:
m2 = m1 + randi(80, 1, numSpectra);   % Peak wavelengths (nm) of secondary peaks, offset from primary peaks
s2 = randi(50, 1, numSpectra) + 5;    % Standard deviations (nm) of secondary peaks
p1 = randi(100, 1, numSpectra) / 100; % Relative intensity ratios of secondary peaks compared to primary peaks

% Preallocate spectral data matrix
spt = zeros(numSpectra, length(wl)); 
spt1 = zeros(numSpectra, length(wl)); 
spt2 = zeros(numSpectra, length(wl)); 

% Simulate spectra composed of two Gaussian profiles:
for n = 1:numSpectra
    spttemp1 = gaussmf(wl, [s1(n), m1(n)]);           % Primary Gaussian profile
    spttemp2 = gaussmf(wl, [s2(n), m2(n)]) * p1(n);   % Secondary Gaussian profile scaled by peak ratio p1
    spt1(n, :) = spttemp1;                            % Store primary Gaussian spectrum
    spt2(n, :) = spttemp2;                            % Store secondary Gaussian spectrum
    spt(n, :) = spttemp1 + spttemp2;                  % Combined Gaussian spectra
end
% Determine global maximum and minimum intensities for further normalization purposes:
maxA1 = max(spt1, [], 'all'); 
minA1 = min(spt1, [], 'all'); 
maxA2 = max(spt2, [], 'all'); 
minA2 = min(spt2, [], 'all');

% Initialize variables and storage arrays for spectral data processing
f = waitbar(0, 'Initializing...');

% Preallocate arrays to store simulated spectral data
sptimg4 = uint16(zeros(16, 128, numSpectra)); % Simulated raw noisy spectral images
tbg4    = uint16(zeros(16, 128, numSpectra)); % Ground truth background-only images
GTspt   = uint16(zeros(16, 128, numSpectra)); % Ground truth spectra images (signal only)
Pbg4    = uint16(zeros(1, numSpectra));       % Photon counts for ground truth background
Psig4   = uint16(zeros(1, numSpectra));       % Photon counts for ground truth signals

% Loop through each spectrum to generate simulated data
for n = 1:numSpectra
    % Set photon numbers for background and signal (modifiable for simulation scenarios)
    Pbg  = randi(10000) + 5000;  % Background photon counts (noise level)
    Psig = randi(10000) + 500;   % Signal photon counts (spectral intensity)

    % Define constants for detector simulation
    res = 2.3438;                % Spectral dispersion (nm/pixel)
    eta = 1;                     % Quantum efficiency (electrons per photon)
    EM  = 100;                   % Electron-multiplying (EM) gain
    ADU = 1;                     % Analog-to-digital conversion unit (electrons per count)
    Isig = Psig * eta;           % Signal electrons
    Ibg  = Pbg * eta;            % Background electrons
    n1   = 16;                   % Spatial dimension of the image (pixels)
    n2   = 300;                  % Number of spectral pixels after interpolation
    c1   = floor(n1 / 2);        % Convolution padding size
    Nf   = 1;                    % Number of frames per spectrum (iterations)

    % Generate spectral point-spread function (PSF) and simulate spectral image
    psf    = fspecial('gaussian', 16, w(n)/px);
    sptimg = conv2(psf, spt(n, :));
    sptimg(:, end-(c1-2):end) = [];
    sptimg(:, 1:c1) = [];
    wl2    = 500:res:800;
    sptimg2 = interp1(wl, sptimg', wl2)';
    sptimg2 = sptimg2 / sum(sptimg2(:)) * Isig;  % Normalize intensity
    sptimg2 = repmat(sptimg2, [1, 1, Nf]);       % Replicate for multiple frames if needed

    % Generate a background image uniformly distributed across pixels
    bgimg = repmat(ones(n1, length(wl2)), [1, 1, Nf]) * (Ibg / n1 / length(wl2));

    % Add Poisson (shot) noise to signal and background, plus Gaussian read noise
    Ns  = random('Poisson', sptimg2);            % Signal-dependent photon shot noise
    Nbg = random('Poisson', bgimg);              % Background photon shot noise
    Nr  = random('norm', 100, 3, n1, length(wl2), Nf); % Gaussian read noise (mean=100, sd=3)

    % Combine signal, background, and noise to produce the final simulated image
    sptimg3 = uint16(Ns + Nbg + Nr);
    tbg1    = uint16(Nbg + Nr);

    % Store simulated images and photon numbers
    sptimg4(:, :, n) = sptimg3;                        % Noisy spectral image
    tbg4(:, :, n)    = tbg1;                           % Ground truth background
    GTspt(:, :, n)   = sptimg4(:, :, n) - tbg4(:, :, n); % Ground truth signal
    Pbg4(n)          = Pbg;                            % Stored background photon count
    Psig4(n)         = Psig;                           % Stored signal photon count

    % Update progress bar
    waitbar(n / numSpectra, f, sprintf('Processing spectra: %d%% complete', floor(n / numSpectra * 100)));
end
close(f); % Close the progress bar

%% Save processed data:
filepath = 'your-file-path\simulation.mat';
save(filepath, 'sptimg4', 'tbg4', 'spt', 'GTspt', 'Pbg4', 'Psig4','w','-v7.3');

%% Visualization:
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
    imagesc(GTspt(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Ground Truth Image - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display a processed or another predicted image
    subplot(3,1,3);
    imagesc(tbg4(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Ground Truth Background - Frame ', num2str(n)]); % Dynamic title with frame number

    pause(1); % Pause for 0.5 seconds to view the current set of images
    drawnow;  % Force display update
end
