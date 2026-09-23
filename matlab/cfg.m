function C = cfg()
%CFG  Single source of truth for every parameter in the DLSWD pipeline.
%     Edit ONLY this file to change the experiment. Nothing downstream
%     hard-codes a path, threshold or hyper-parameter.

%% ---------- paths ----------
C.root        = fileparts(fileparts(mfilename('fullpath')));  % IEEE Wavelet - Submit/
C.dataFile    = fullfile(C.root,'data','ped_adc.mat');   % raw ADC tensor 4x2x255x128x30
C.outDir      = fullfile(C.root,'matlab','out');         % .mat intermediates
C.figDir      = fullfile(C.root,'Ped');                  % generated figure outputs
C.texMacro    = fullfile(C.root,'results_macros.tex');   % auto-generated numbers

%% ---------- dataset ----------
C.rxUsed      = 1;          % virtual antenna channel index
C.txUsed      = 1;
C.nFrames     = 30;         % source frames (grouping unit -- see splitByFrame)
C.snrGrid     = -20:1:5;    % dB, 26 levels
C.nRealiz     = 23;         % noise realisations per (frame,SNR)
C.rngNoise    = 20260813;   % master seed for AWGN generation

%% ---------- wavelet configuration space (768 = 8*6*2*4*2) ----------
C.wavelets    = {'bior3.5','db4','db8','coif3','sym4','fk4','fk8','haar'};
C.levels      = 2:7;
C.varEst      = {'LevelIndependent','LevelDependent'};
C.denMethod   = {'Bayes','UniversalThreshold','SURE','Minimax'};
C.threshRule  = {'s','h'};          % soft / hard
C.alpha       = 0.5;                % weighting in eq. (2)

%% ---------- CBC ----------
C.tauC        = 0.05;               % occurrence-frequency threshold
C.tauCsweep   = [0.01 0.02 0.05 0.10];
C.tauI        = 0.30;               % normalised-MI redundancy threshold
C.tauIsweep   = 0.10:0.05:0.60;

%% ---------- CNN ----------
C.imSize      = [75 75 1];
C.batchSize   = 16;
C.maxEpochs   = 30;
C.initLR      = 1e-3;
C.lrDropFactor= 0.1;
C.lrDropEpoch = 20;
C.optimizer   = 'adam';
C.l2reg       = 1e-4;
C.seeds       = [1 2 3 4 5];        % multi-seed -> mean +/- std
C.splitFrac   = [0.74 0.13 0.13];   % train / val / hardware-eval, BY FRAME

%% ---------- detection metric (non-circular evaluation) ----------
C.pfaTarget   = 1e-3;               % CFAR operating point for Pd
C.cfarGuard   = 2;
C.cfarTrain   = 8;

%% ---------- hardware ----------
C.dpuArch     = 'B4096';
C.dpuClockMHz = 300;
C.platforms   = {'KR260'};

if ~exist(C.outDir,'dir'), mkdir(C.outDir); end
if ~exist(C.figDir,'dir'), mkdir(C.figDir); end
end
