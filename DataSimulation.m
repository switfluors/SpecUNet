clear; clc; clf;

% Add path for additional code dependencies
addpath('D:\HongjingMao\U-net\Code');

% Parameters for point spread function (PSF)
lambda = 670; % Emission maxima of Alexa 647
NA = 1.49; % Numerical aperture
px = 160 / 1.5; % Pixel size of EMCCD
w = 0.61 * lambda / NA / 2.355; % Convert FWHM to standard deviation
numSpectra = 100;
% w = w * rand(1, numSpectra) * 2 + 1; % Varying focus across spectra
rng('shuffle');
w=w*(rand(1,numSpectra)*0.5+1); %varying focus 
fwhm = w * 2.355;

% Load Gaussian spectral profiles
wl = 500:800; % Wavelength range
m1 = randi(250, 1, numSpectra) + 500; % Mean of first Gaussian
s1 = randi(50, 1, numSpectra) + 5; % Standard deviation of first Gaussian
m2 = m1 + randi(80, 1, numSpectra); % Mean of second Gaussian
s2 = randi(50, 1, numSpectra) + 5; % Standard deviation of second Gaussian
p1 = randi(100, 1, numSpectra) / 100; % Peak ratio of second Gaussian

spt = zeros(numSpectra, length(wl)); % Preallocate spectral data matrix
spt1 = zeros(numSpectra, length(wl)); % Preallocate spectral data matrix
spt2 = zeros(numSpectra, length(wl)); % Preallocate spectral data matrix
for n = 1:numSpectra
    spttemp1=gaussmf(wl,[s1(n) m1(n)]);
    spttemp2=gaussmf(wl,[s2(n) m2(n)])*p1(n);
    spt1(n,:)= spttemp1;
    spt2(n,:)= spttemp2;
    spt(n,:)=spttemp1+spttemp2;
end
maxA1 = max(max(spt1)) ;
minA1 = min(min(spt1)) ;
maxA2 = max(max(spt2)) ;
minA2 = min(min(spt2)) ;

% Initialization for data processing
f = waitbar(0, 'Starting');
sptimg4 = uint16(zeros(16, 128, numSpectra));
tbg4 = uint16(zeros(16, 128, numSpectra));
GTspt = uint16(zeros(16, 128, numSpectra));
reverse_sptimg4 = uint16(zeros(8, 128, numSpectra));
reverse_tbg4 = uint16(zeros(8, 128, numSpectra));
reverse_GTspt = uint16(zeros(8, 128, numSpectra));

Pbg4 = uint16(zeros(1, numSpectra));
Psig4 = uint16(zeros(1, numSpectra));
% Pbg4 = zeros(1, numSpectra);
% Psig4 = zeros(1, numSpectra);

for n = 1:numSpectra
    Pbg=randi(10000)+5000;      % Background Noise                   % 5000/10000
    Psig = randi(10000)+500;    % signal (photons): 5000-15000       % 5000/10000 OG:10000
    res = 2.3438;               % spectral dispersion (nm/pixel) 
    eta = 1;                    % quantum efficiency
    EM = 100;                   % EM gain
    ADU = 1;                    % analog digital conversion unit
    Isig = Psig*eta;            % electrons
    Ibg = Pbg*eta;              % electrons
    n1 = 16;
    n2 = 300;
    c1 = floor(n1/2);           % for convolution
    Nf = 1;                     % # of iteration

   % generate spectral image
    psf  = fspecial('gaussian',16,w(n)/px);
    sptimg = conv2(psf,spt(n,:));
    sptimg(:,end-(c1-2):end)=[];
    sptimg(:,1:c1)=[];
    spt2 = sum(sptimg,1);

    rng('shuffle');
    wl2 = 500:res:800; % for Alexa 647 (99.8% signal)
    n3 = length(wl2);
    sptimg2 = interp1(wl,sptimg',wl2)'; 
    sptimg2 = sptimg2/sum(sptimg2(:))*Isig;
    sptimg2 = repmat(sptimg2,[1 1 Nf]);
    bgimg = repmat(ones(n1,n3),[1 1 Nf])*Ibg/n1/n3;

    % generate noise source
    Ns = random('Poisson',sptimg2);
    Nbg = random('Poisson',bgimg);
    Nr = random('norm', 100,3,n1,n3,Nf); % Original is 10   bg10/3, 3 will have better results

    sptimg3 = Ns +Nbg+ Nr;
    sptimg3=uint16(sptimg3);
    tbg1=Nbg+Nr;
    tbg1=uint16(tbg1);

    % Data storage
    sptimg4(:, :, n) = sptimg3;
    tbg4(:, :, n) = tbg1;
    GTspt(:,:,n) = sptimg4(:,:,n) - tbg4(:,:,n);
    Pbg4(:,n) = Pbg;
    Psig4(:,n) = Psig;

    reverse_sptimg4(:, :, n) = sptimg3(5:12, :);
    reverse_tbg4(:, :, n) = tbg1(5:12, :);
    reverse_GTspt(:, :, n) = GTspt(5:12, :, n);
    
    waitbar(n / numSpectra, f, sprintf('Progress: %d %%', floor(n / numSpectra * 100)));
end
close(f);
%% Save processed data
filepath = 'D:\HongjingMao\Manuscript\TestingData\OG5k_psig10000pbg10000BG3_mao.mat';
save(filepath, 'sptimg4', 'tbg4', 'spt', 'GTspt', 'reverse_sptimg4', 'reverse_tbg4', 'reverse_GTspt', 'Pbg4','Psig4','w','-v7.3');
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
    imagesc(GTspt(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Ground Truth Image - Frame ', num2str(n)]); % Dynamic title with frame number
    
    % Display processed or another predicted image
    subplot(3,1,3);
    imagesc(tbg4(:,:,n)); % Display image
    axis equal; % Set axes to be equal
    title(['Ground Truth Background - Frame ', num2str(n)]); % Dynamic title with frame number

    pause(1); % Pause for 0.5 seconds to view the current set of images
    drawnow; % Force display update
end