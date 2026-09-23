function export_gao_pms_rd_examples(varargin)
%EXPORT_GAO_PMS_RD_EXAMPLES Representative Gao-style range-Doppler evidence.
%
% Uses local public PMS raw ADC frames and text labels extracted from the
% Gao/RAMP public data format.  The RAMP conversion code treats column 2 as
% class ID and columns 3--4 as x/y object position in metres, from which range
% and angle are derived.  No radial-velocity truth is present in these labels,
% so this script exports annotation-backed range/angle evidence and RD maps,
% but does not claim range-velocity error.

ip = inputParser;
addParameter(ip,'FrameIds',30:59,@(x) isnumeric(x) && all(x>=0));
addParameter(ip,'MaxFigures',4,@(x) isnumeric(x) && isscalar(x) && x>=1);
parse(ip,varargin{:});
O = ip.Results;

clc
root = fileparts(fileparts(mfilename('fullpath')));
frameDir = fullfile(root,'tmp','public_pms');
labelDir = fullfile(root,'matlab','out','cached_grouped_revision','public_pms1000_text_labels');
outDir = fullfile(root,'matlab','out','gao_pms_evidence');
figDir = fullfile(root,'figures_ieee','gao_pms');
if ~exist(outDir,'dir'), mkdir(outDir); end
if ~exist(figDir,'dir'), mkdir(figDir); end

C = gaoRadarConfig();
rows = [];
made = 0;
for frameId = O.FrameIds(:)'
    matFile = fullfile(frameDir,sprintf('%06d.mat',frameId));
    labelFile = fullfile(labelDir,sprintf('%010d.csv',frameId));
    if ~isfile(matFile) || ~isfile(labelFile), continue; end
    S = load(matFile,'adcData');
    rd = rangeDopplerFromAdc(S.adcData,C);
    L = parsePmsLabels(labelFile);
    rows = [rows; labelsToRows(frameId,L)]; %#ok<AGROW>
    if made < O.MaxFigures
        made = made + 1;
        exportFigure(frameId,rd,L,C,figDir);
    end
end
if isempty(rows)
    T = table();
else
    T = struct2table(rows);
end
writetable(T,fullfile(outDir,'pms1000_label_inventory.csv'));
writeStatus(outDir,C);
fprintf('Exported %d representative RD figures under %s\n',made,figDir);
fprintf('Wrote label inventory: %s\n',fullfile(outDir,'pms1000_label_inventory.csv'));
end

function C = gaoRadarConfig()
% Values mirror Gao RAMP-CNN config.py for the AWR1843 public data.
C.c = 299792458;
C.fc = 77e9;
C.sampleFreq = 4e6;
C.sweepSlope = 21.0017e12;
C.nChirps = 255;
C.rangeFft = 128;
C.dopplerFft = 256;
C.rrMin = 1.0;
C.rrMax = 25.0;
C.frameRate = 30;
C.Tc = 1/(C.frameRate*C.nChirps);
C.rangeAxis = (0:C.rangeFft-1) * (C.sampleFreq/C.rangeFft) * C.c/(2*C.sweepSlope);
C.velocityAxis = ((-C.dopplerFft/2):(C.dopplerFft/2-1)) * (1/(C.dopplerFft*C.Tc)) * (C.c/C.fc)/2;
end

function rd = rangeDopplerFromAdc(adcData,C)
% adcData: [samples x chirps x rx x tx], matching the local PMS frames.
x = squeeze(mean(adcData, [3 4]));
x = x(1:C.rangeFft,1:C.nChirps);
winR = hann(size(x,1),'periodic');
winD = hann(size(x,2),'periodic')';
xw = x .* winR .* winD;
rd = fftshift(fft(fft(xw,C.rangeFft,1),C.dopplerFft,2),2);
rd = 20*log10(abs(rd) + eps);
end

function L = parsePmsLabels(labelFile)
raw = readmatrix(labelFile);
if isempty(raw)
    L = zeros(0,6);
    return
end
if size(raw,2) < 6
    error('Expected at least six label columns in %s',labelFile);
end
L = raw(:,1:6);
end

function rows = labelsToRows(frameId,L)
rows = repmat(struct('FrameId',frameId,'TrackId',nan,'ClassId',nan, ...
    'X_m',nan,'Y_m',nan,'Width_m',nan,'Length_m',nan, ...
    'Range_m',nan,'Angle_deg',nan),size(L,1),1);
for i=1:size(L,1)
    x = L(i,3);
    y = L(i,4);
    rows(i).TrackId = L(i,1);
    rows(i).ClassId = L(i,2);
    rows(i).X_m = x;
    rows(i).Y_m = y;
    rows(i).Width_m = L(i,5);
    rows(i).Length_m = L(i,6);
    rows(i).Range_m = hypot(x,y);
    rows(i).Angle_deg = atan2d(x,y);
end
end

function exportFigure(frameId,rd,L,C,figDir)
f = figure('Visible','off','Color','w','Position',[100 100 720 420]);
imagesc(C.velocityAxis,C.rangeAxis,rd);
axis xy
colormap turbo
colorbar
xlabel('Radial velocity (m/s)')
ylabel('Range (m)')
title(sprintf('Gao PMS frame %06d: range-Doppler map with parsed labels',frameId))
hold on
for i=1:size(L,1)
    x = L(i,3);
    y = L(i,4);
    rngM = hypot(x,y);
    text(0,min(max(rngM,C.rangeAxis(1)),C.rangeAxis(end)), ...
        sprintf('class %.0f, R=%.2f m',L(i,2),rngM), ...
        'Color','w','FontSize',8,'FontWeight','bold','BackgroundColor','k');
end
hold off
exportgraphics(f,fullfile(figDir,sprintf('pms_rd_frame_%06d.png',frameId)),'Resolution',300);
exportgraphics(f,fullfile(figDir,sprintf('pms_rd_frame_%06d.pdf',frameId)));
close(f)
end

function writeStatus(outDir,C)
fid = fopen(fullfile(outDir,'pms1000_metric_status.txt'),'w');
fprintf(fid,'Gao/RAMP-CNN-compatible radar settings used for representative RD export.\n');
fprintf(fid,'sample_freq=%.6g Hz\nsweep_slope=%.6g Hz/s\nn_chirps=%d\n', ...
    C.sampleFreq,C.sweepSlope,C.nChirps);
fprintf(fid,['The exported figures use RAMP conversion semantics: column 2 is class, ' ...
    'columns 3--4 are x/y position, and range is sqrt(x^2+y^2). The labels do ' ...
    'not contain radial velocity, so true range-velocity error remains disabled.\n']);
fclose(fid);
end
