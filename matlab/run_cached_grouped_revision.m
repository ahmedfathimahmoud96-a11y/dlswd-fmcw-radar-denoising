function run_cached_grouped_revision()
%RUN_CACHED_GROUPED_REVISION Reviewer-safe re-evaluation without SCS rerun.
%
% Run one command from this repository:
%   >> cd('D:\DLSWD Framework\IEEE Wavelet - Submit\matlab')
%   >> run_cached_grouped_revision

% The script reads the supplied data and legacy oracle-label CSV files only.
% It never modifies F:\Master\1-MATLAB\PED_Wavelet_Denoising and it does not
% repeat the 768-configuration brute-force search.  It:
%   - builds 30-source / 26-SNR / 23-replicate manifest;
%   - merges legacy train/test noise realisations, then uses five
%     source-frame-disjoint folds;
%   - trains five CNN seeds per fold;
%   - compares CNN, estimated-SNR lookup, fixed configuration, and the
%     explicitly named oracle upper bound;
%   - caches three configuration-specific denoising metrics so five seeds do
%     not cause repeated wavelet work;
%   - writes source-level bootstrap CIs, CBC/NMI sensitivity, figures and
%     IEEE single-column LaTex-ready output.

% Guard rails:
%   * Legacy labels remain ORACLE labels; this code never calls them CNN output.
%   * Pd/Pfa are not invented.  The supplied folders have no target annotation
%     file, so only clearly named reference-peak preservation diagnostics are
%     reported.  Implement a separately validated detection stage once target
%     annotations are available; do not substitute reference peaks for them.

clc
thisFile = mfilename('fullpath');
matlabRoot = fileparts(thisFile);
repoRoot = fileparts(matlabRoot);
addpath(genpath(matlabRoot));

P = revisionPaths(repoRoot);
ensureDependencies();
P = configureCompute(P);
if ~exist(P.outDir,'dir'), mkdir(P.outDir); end
if ~exist(P.figDir,'dir'), mkdir(P.figDir); end

diary(P.logFile); diary on
stopDiary = onCleanup(@() diary('off'));
fprintf('Cached grouped revision started: %s\n',char(datetime('now')));
fprintf('Output directory: %s\n',P.outDir);
fprintf('Supplied data root (read only): %s\n\n',P.dataRoot);
t0 = tic;

M = buildManifest(P);
M = validateManifest(M);
writetable(M,P.manifestCsv);
save(P.manifestMat,'M','P','-v7.3');

[X,estimatedSNR] = prepareFeatures(M,P);
M.Fold = groupedFolds(M.SourceId,P.nOuterFolds,P.foldSeed);
classes = unique(M.ConfigKey,'stable');
assert(numel(classes)==3,'Expected exactly 3 cached configurations, found %d.',numel(classes));
M.Label = keyToIndex(M.ConfigKey,classes);
configs = decodeConfigurations(classes);
writetable(M,P.manifestCsv);
save(P.manifestMat,'M','P','-v7.3');

% Metric cache is resumable by configuration.  This is the overnight stage.
Q = buildMetricCache(M,configs,P);

[pred,cvSeed,foldInfo] = trainGroupedCNN(X,M,classes,P);
[samples,selection,cbcSensitivity,nmiMatrix] = evaluatePolicies( ...
    M,Q,pred,estimatedSNR,classes,foldInfo,P);
[sourceMetrics,summary,bySNR,paired] = summarizeResults(samples,P);

writeOutputs(M,configs,cvSeed,selection,cbcSensitivity,nmiMatrix, ...
    samples,sourceMetrics,summary,bySNR,paired,P);
makeFigures(bySNR,cvSeed,P);
writeLatex(summary,selection,cvSeed,paired,P);

fprintf('\nCompleted in %.2f h at %s\n',toc(t0)/3600,char(datetime('now')));
fprintf('Read first:\n  %s\n  %s\n  %s\n',P.summaryCsv,P.latexSnippet,P.logFile);
end

% ========================================================================
function P = revisionPaths(repoRoot)
P.repoRoot = repoRoot;
P.dataRoot = 'F:\Master\1-MATLAB\PED_Wavelet_Denoising';
P.trainLabels = fullfile(P.dataRoot,'Iter_3_Seminar_3Classes_GPU','bestResults.csv');
P.testLabels = fullfile(P.dataRoot,'Iter_3_Seminar_3Classes_Test_GPU','bestResults.csv');
P.trainNoisyRoot = fullfile(P.dataRoot,'Dataset','Dataset_noisy','train','ped','CSV');
P.testNoisyRoot = fullfile(P.dataRoot,'Dataset_Test','Dataset_noisy','test','ped','CSV');
P.cleanRoot = fullfile(P.dataRoot,'Dataset','Dataset_original','ped','CSV');

P.outDir = fullfile(repoRoot,'matlab','out','cached_grouped_revision');
P.figDir = fullfile(repoRoot,'Ped');
P.manifestCsv = fullfile(P.outDir,'manifest.csv');
P.manifestMat = fullfile(P.outDir,'manifest.mat');
P.featureMat = fullfile(P.outDir,'features.mat');
P.predMat = fullfile(P.outDir,'cnn_predictions.mat');
P.logFile = fullfile(P.outDir,'run.log');
P.summaryCsv = fullfile(P.outDir,'grouped_summary.csv');
P.latexSnippet = fullfile(repoRoot,'revision_results_insert.tex');

P.nOuterFolds = 5;
P.foldSeed = 20260822;
P.seeds = [1 2 3 4 5];
P.validationSourceFraction = 0.125;  % 3 of 24 outer-training sources
P.imSize = [75 75 1];
P.maxEpochs = 30;
P.batchSize = 16;
P.initialLearnRate = 1e-3;
P.learnRateDropPeriod = 20;
P.l2Regularization = 1e-4;
P.bootstrapReplicates = 5000;
P.useParallelCache = true;
P.showProgressWindow = true;
P.executionEnvironment = 'cpu';       % upgraded to GPU when available
P.gpuName = 'not available';

P.tauC = 0.05;
P.tauCsweep = [0.01 0.02 0.05 0.10];
% Smallest predeclared sensitivity-grid value that retains every >=5%%
% configuration in all five outer-training folds; see cbc_sensitivity.csv.
P.tauI = 0.50;
P.tauIsweep = 0.10:0.05:0.60;

P.rangeFft = 128;
P.dopplerFft = 256;
P.c = 299792458;
P.fc = 77e9;
P.Fs = 4e6;
P.sweepSlope = 21.0017e12;
P.Tc = 120e-6;
P.rangeResolution = (P.Fs/P.rangeFft)*P.c/(2*P.sweepSlope);
P.velocityResolution = (1/(P.dopplerFft*P.Tc))*(P.c/P.fc)/2;
end

function ensureDependencies()
required = {'wdenoise2','trainNetwork','imageInputLayer','imresize'};
missing = required(cellfun(@(f) exist(f,'file')~=2,required));
assert(isempty(missing),'Missing MATLAB functions: %s',strjoin(missing,', '));
end

function P = configureCompute(P)
% DWT/wdenoise2 is CPU code; the CNN is explicitly trained on the GPU when
% MATLAB detects one.  Do not move the full image tensor to gpuArray: MATLAB
% streams mini-batches to the GPU and avoids exhausting video memory.
try
    if canUseGPU
        dev = gpuDevice;
        P.executionEnvironment = 'gpu';
        P.gpuName = dev.Name;
        fprintf('GPU enabled for CNN training: %s (%g GB available).\n', ...
            dev.Name,dev.AvailableMemory/2^30);
    else
        warning('CachedRevision:NoGPU', ...
            'No supported MATLAB GPU is available. CNN training will use CPU.');
    end
catch ME
    warning('CachedRevision:GPUProbeFailed', ...
        'GPU detection failed (%s). CNN training will use CPU.',ME.message);
end
end

% ========================================================================
function M = buildManifest(P)
if isfile(P.manifestMat)
    S = load(P.manifestMat,'M');
    if isfield(S,'M') && height(S.M)==17940
        M = S.M;
        fprintf('Using cached manifest: %s\n',P.manifestMat);
        return
    end
end
A = readLegacyResults(P.trainLabels,"legacy_train",P.trainNoisyRoot,P.cleanRoot);
B = readLegacyResults(P.testLabels,"legacy_test",P.testNoisyRoot,P.cleanRoot);
M = sortrows([A;B],{'SourceId','SNRdB','Origin','Replicate'});
end

function T = readLegacyResults(csvFile,origin,noisyRoot,cleanRoot)
raw = readcell(csvFile,'Delimiter',',');
assert(size(raw,2)==18,'Unexpected legacy CSV layout: %s',csvFile);
N = size(raw,1);
fileName = string(raw(:,3));
sourceId = zeros(N,1); fileSNR = zeros(N,1); replicate = zeros(N,1);
for i=1:N
    tok = regexp(char(fileName(i)),'^ped(\d+)_(-?\d+)_(\d+)\.csv$','tokens','once');
    assert(~isempty(tok),'Cannot parse legacy filename: %s',fileName(i));
    sourceId(i)=str2double(tok{1});
    fileSNR(i)=str2double(tok{2});
    replicate(i)=str2double(tok{3});
end
snr = numericColumn(raw(:,2));
assert(isequal(snr,fileSNR),'SNR and filename differ in %s.',csvFile);
wavelet = normalizeText(raw(:,4));
method = normalizeMethod(raw(:,5));
rule = normalizeRule(raw(:,6));
noiseEstimate = normalizeText(raw(:,7));
level = numericColumn(raw(:,8));
configKey = wavelet+"|"+method+"|"+rule+"|"+noiseEstimate+"|"+string(level);

rawPath = strings(N,1); cleanPath = strings(N,1);
for i=1:N
    rawPath(i) = fullfile(noisyRoot,sprintf('SNR%d',snr(i)),char(fileName(i)));
    cleanPath(i) = fullfile(cleanRoot,sprintf('ped%d.csv',sourceId(i)));
end
T = table(sourceId,snr,replicate,repmat(string(origin),N,1),fileName,rawPath,cleanPath, ...
    wavelet,method,rule,noiseEstimate,level,configKey, ...
    numericColumn(raw(:,9)),numericColumn(raw(:,10)),numericColumn(raw(:,11)), ...
    numericColumn(raw(:,12)),numericColumn(raw(:,13)),numericColumn(raw(:,14)), ...
    numericColumn(raw(:,15)),numericColumn(raw(:,16)), ...
    'VariableNames',{'SourceId','SNRdB','Replicate','Origin','FileName','RawPath', ...
    'CleanPath','Wavelet','DenoisingMethod','ThresholdRule','NoiseEstimate','Level', ...
    'ConfigKey','OracleCorrNoisy','OracleCorrDenoised','OracleCorrGainPct', ...
    'OracleSNRBefore','OracleSNRAfter','OracleSNRGain','OracleMSEBefore','OracleMSEAfter'});
end

function x = numericColumn(c)
x = zeros(numel(c),1);
for i=1:numel(c)
    if isnumeric(c{i}), x(i)=double(c{i});
    else, x(i)=str2double(string(c{i})); end
end
assert(all(isfinite(x)),'Non-numeric cached result field.');
end

function s = normalizeText(c), s = strtrim(string(c)); end
function s = normalizeMethod(c)
s = normalizeText(c);
s(strcmpi(s,'Sure')) = "SURE";
s(strcmpi(s,'Universal')) = "UniversalThreshold";
end
function s = normalizeRule(c)
s = normalizeText(c);
s(strcmpi(s,'s')) = "Soft";
s(strcmpi(s,'h')) = "Hard";
end

function M = validateManifest(M)
assert(height(M)==17940,'Expected 17,940 records; found %d.',height(M));
assert(isequal(unique(M.SourceId)',1:30),'Expected source IDs 1:30.');
assert(isequal(unique(M.SNRdB)',-20:5),'Expected SNR range -20:5 dB.');
assert(numel(unique(M.ConfigKey))==3,'Expected 3 cached oracle classes.');

[G,~,~] = findgroups(M.SourceId,M.SNRdB);
perSourceSNR = splitapply(@numel,M.SourceId,G);
assert(all(perSourceSNR==23),'Each source/SNR combination must have 23 noise realisations.');
assert(sum(M.Origin=="legacy_train")==15600 && sum(M.Origin=="legacy_test")==2340, ...
    'Unexpected legacy train/test counts.');
for p=[M.RawPath; M.CleanPath]'
    assert(isfile(p),'Missing input data file: %s',p);
end
end

% ========================================================================
function [X,estimatedSNR] = prepareFeatures(M,P)
% Features use real(dataNoisy), matching the legacy wdenoise2 label workflow.
if isfile(P.featureMat)
    S=load(P.featureMat,'X','estimatedSNR');
    if isfield(S,'X') && size(S.X,4)==height(M)
        X=S.X; estimatedSNR=S.estimatedSNR;
        fprintf('Using cached features: %s\n',P.featureMat);
        return
    end
end
N=height(M);
X=zeros(P.imSize(1),P.imSize(2),1,N,'single');
estimatedSNR=zeros(N,1,'single');
fprintf('Creating %d CNN features from supplied CSV files...\n',N);
h = openProgressWindow(P,'DLSWD revision: features','Creating CNN features');
closeBar = onCleanup(@() closeProgressWindow(h));
for n=1:N
    x=readComplexCsv(M.RawPath(n));
    a=real(x);
    a=(a-min(a(:)))/max(max(a(:))-min(a(:)),eps);
    X(:,:,1,n)=single(imresize(a,P.imSize(1:2),'bilinear'));
    estimatedSNR(n)=single(referenceFreeSNR(a));
    if mod(n,50)==0 || n==N
        updateProgressWindow(h,n/N,sprintf('CNN features: %d / %d',n,N));
        fprintf('  feature %d / %d (%.0f%%)\n',n,N,100*n/N);
    end
end
save(P.featureMat,'X','estimatedSNR','-v7.3');
end

function x = readComplexCsv(fileName)
x=readmatrix(char(fileName));
assert(isnumeric(x) && ~isempty(x),'Cannot read numerical data: %s',fileName);
assert(isequal(size(x),[128 255]),'Expected 128x255 matrix in %s.',fileName);
x=double(x);
end

function s = referenceFreeSNR(x)
p=mean(abs(fft(x,[],1)).^2,2);
v=sort(p(:));
pn=median(v(1:max(1,round(0.25*numel(v)))));
ps=max(mean(p)-pn,eps);
s=10*log10(ps/max(pn,eps));
end

function foldId = groupedFolds(sourceId,K,seed)
src=unique(sourceId,'sorted');
assert(mod(numel(src),K)==0,'Source count must divide evenly into folds.');
rng(seed,'twister'); src=src(randperm(numel(src)));
foldId=zeros(numel(sourceId),1);
for i=1:numel(src), foldId(sourceId==src(i))=mod(i-1,K)+1; end
end

function idx = keyToIndex(keys,classes)
idx=zeros(numel(keys),1);
for k=1:numel(classes), idx(keys==classes(k))=k; end
assert(all(idx>0),'Cached label does not map to class library.');
end

function configs = decodeConfigurations(keys)
K=numel(keys);
configs=table(strings(K,1),strings(K,1),strings(K,1),strings(K,1),zeros(K,1),keys, ...
    'VariableNames',{'Wavelet','Method','Rule','NoiseEstimate','Level','Key'});
for k=1:K
    z=split(keys(k),'|');
    assert(numel(z)==5,'Malformed configuration key: %s',keys(k));
    % Table brace assignment cannot mix four string columns and one double
    % column.  Assign each typed variable explicitly.
    configs.Wavelet(k)=z(1);
    configs.Method(k)=z(2);
    configs.Rule(k)=z(3);
    configs.NoiseEstimate(k)=z(4);
    configs.Level(k)=str2double(z(5));
end
end

% ========================================================================
function Q = buildMetricCache(M,configs,P)
% Calculate each config once for each record.  The result is re-used by every
% CNN seed.  Clean data are used only after configuration choice for metrics.
N=height(M); K=height(configs);
rawPaths=M.RawPath; cleanPaths=M.CleanPath;
Q.Corr=nan(N,K,'single'); Q.SNR=nan(N,K,'single'); Q.MSE=nan(N,K,'single');
Q.RefRangeErr_m=nan(N,K,'single'); Q.RefVelocityErr_mps=nan(N,K,'single');
Q.RDNMSE=nan(N,K,'single');
for k=1:K
    cacheFile=fullfile(P.outDir,sprintf('metrics_config_%d.mat',k));
    if isfile(cacheFile)
        S=load(cacheFile,'corrOut','snrOut','mseOut','rangeErr','velErr','rdnmse');
        if numel(S.corrOut)==N
            Q.Corr(:,k)=S.corrOut; Q.SNR(:,k)=S.snrOut; Q.MSE(:,k)=S.mseOut;
            Q.RefRangeErr_m(:,k)=S.rangeErr; Q.RefVelocityErr_mps(:,k)=S.velErr;
            Q.RDNMSE(:,k)=S.rdnmse;
            fprintf('Using metric cache: %s\n',configs.Key(k));
            continue
        end
    end
    fprintf('Metric cache %d/%d: %s\n',k,K,configs.Key(k));
    cfgRow=configs(k,:);
    corrOut=nan(N,1,'single'); snrOut=corrOut; mseOut=corrOut;
    rangeErr=corrOut; velErr=corrOut; rdnmse=corrOut;
    h = openProgressWindow(P,'DLSWD revision: wavelet metrics', ...
        sprintf('Configuration %d/%d: %s',k,K,configs.Key(k)));
    closeBar = onCleanup(@() closeProgressWindow(h));
    parallelOK = P.useParallelCache && startPoolIfPossible();
    if parallelOK
        q = progressDataQueue(h,N,sprintf('Wavelet configuration %d/%d',k,K));
        parfor n=1:N
            [corrOut(n),snrOut(n),mseOut(n),rangeErr(n),velErr(n),rdnmse(n)] = ...
                applyAndMeasure(rawPaths(n),cleanPaths(n),cfgRow,P);
            send(q,1);
        end
    else
        for n=1:N
            [corrOut(n),snrOut(n),mseOut(n),rangeErr(n),velErr(n),rdnmse(n)] = ...
                applyAndMeasure(rawPaths(n),cleanPaths(n),cfgRow,P);
            if mod(n,50)==0 || n==N
                updateProgressWindow(h,n/N,sprintf('Wavelet configuration %d/%d: %d / %d',k,K,n,N));
                fprintf('  metric %d / %d (%.0f%%)\n',n,N,100*n/N);
            end
        end
    end
    save(cacheFile,'corrOut','snrOut','mseOut','rangeErr','velErr','rdnmse','-v7.3');
    Q.Corr(:,k)=corrOut; Q.SNR(:,k)=snrOut; Q.MSE(:,k)=mseOut;
    Q.RefRangeErr_m(:,k)=rangeErr; Q.RefVelocityErr_mps(:,k)=velErr;
    Q.RDNMSE(:,k)=rdnmse;
end
end

function tf = startPoolIfPossible()
tf = false;
try
    if isempty(gcp('nocreate')), parpool('local'); end
    tf = ~isempty(gcp('nocreate'));
catch ME
    warning('CachedRevision:ParallelUnavailable', ...
        'Parallel worker unavailable (%s). Running serially.',ME.message);
end
end

function h = openProgressWindow(P,titleText,message)
h = [];
if ~P.showProgressWindow, return; end
try
    % A graphical waitbar is available in desktop MATLAB.  In batch/terminal
    % MATLAB, the textual updates from progressDataQueue remain visible.
    if usejava('desktop')
        h = waitbar(0,message,'Name',titleText,'CreateCancelBtn','');
    end
catch ME
    warning('CachedRevision:ProgressWindowUnavailable', ...
        'Cannot open graphical progress window (%s).',ME.message);
end
end

function updateProgressWindow(h,fraction,message)
persistent lastPercent
if ~isempty(h) && isgraphics(h)
    waitbar(min(max(fraction,0),1),h,message);
    drawnow limitrate
else
    % VS Code and MATLAB -batch do not show a waitbar window.  Keep a compact
    % textual progress bar in the Command Window/terminal in that case.
    percent = floor(100*min(max(fraction,0),1));
    if isempty(lastPercent) || percent~=lastPercent
        nFill = round(percent/5);
        bar = [repmat('=',1,nFill),repmat('.',1,20-nFill)];
        fprintf('\r[%s] %3d%% %s',bar,percent,message);
        if percent==100, fprintf('\n'); end
        lastPercent = percent;
    end
end
end

function closeProgressWindow(h)
if ~isempty(h) && isgraphics(h), close(h); end
end

function q = progressDataQueue(h,total,labelText)
% DataQueue sends work-completion signals from parfor workers back to the
% MATLAB client, where waitbar can be updated safely.
q = parallel.pool.DataQueue;
done = 0;
lastPrinted = -1;
afterEach(q,@tick);
    function tick(~)
        done = done+1;
        fraction = done/total;
        updateProgressWindow(h,fraction,sprintf('%s: %d / %d',labelText,done,total));
        percentage = floor(100*fraction);
        if percentage>=lastPrinted+5 || done==total
            fprintf('  %s: %d / %d (%d%%)\n',labelText,done,total,percentage);
            lastPrinted = percentage;
        end
    end
end

function [rho,snr,mse,rangeErr,velErr,rdnmse] = applyAndMeasure(rawPath,cleanPath,cfg,P)
x=real(readComplexCsv(rawPath));
r=real(readComplexCsv(cleanPath));
y=legacyDenoise(x,cfg);
rho=legacyCorrelation(r,y);
e=y-r;
mse=mean(e(:).^2);
snr=10*log10(sum(r(:).^2)/max(sum(e(:).^2),eps));
[rangeErr,velErr,rdnmse]=referencePeakMetrics(y,r,P);
end

function y = legacyDenoise(x,cfg)
% Exact wdenoise2 call used to create the stored legacy bestResults labels.
y=wdenoise2(x,cfg.Level,'DenoisingMethod',char(cfg.Method), ...
    'ThresholdRule',char(cfg.Rule),'NoiseEstimate',char(cfg.NoiseEstimate), ...
    'Wavelet',char(cfg.Wavelet));
end

function rho = legacyCorrelation(a,b)
a=a(:)-mean(a(:)); b=b(:)-mean(b(:));
rho=(a'*b)/max(sqrt((a'*a)*(b'*b)),eps);
end

function [rangeErr,velErr,rdnmse] = referencePeakMetrics(y,r,P)
% Reference-peak preservation only; not external ground-truth target metrics.
rdY=fftshift(fft2(y,P.rangeFft,P.dopplerFft));
rdR=fftshift(fft2(r,P.rangeFft,P.dopplerFft));
[~,iy]=max(abs(rdY(:)).^2); [ry,vy]=ind2sub(size(rdY),iy);
[~,ir]=max(abs(rdR(:)).^2); [rr,vr]=ind2sub(size(rdR),ir);
rangeErr=abs(ry-rr)*P.rangeResolution;
velErr=abs(vy-vr)*P.velocityResolution;
rdnmse=sum(abs(rdY(:)-rdR(:)).^2)/max(sum(abs(rdR(:)).^2),eps);
end

% ========================================================================
function [pred,cvSeed,foldInfo] = trainGroupedCNN(X,M,classes,P)
N=height(M); K=numel(classes); nSeed=numel(P.seeds);
if isfile(P.predMat)
    S=load(P.predMat,'pred','cvSeed','foldInfo');
    if size(S.pred,1)==N && size(S.pred,2)==nSeed
        pred=S.pred; cvSeed=S.cvSeed; foldInfo=S.foldInfo;
        fprintf('Using cached CNN predictions: %s\n',P.predMat);
        return
    end
end

pred=zeros(N,nSeed,'uint8');
rows={};
foldInfo=struct('Fold',{},'TrainIdx',{},'ValIdx',{},'TestIdx',{},'BestSeed',{});
nTrainingRuns = P.nOuterFolds*nSeed;
completedRuns = 0;
h = openProgressWindow(P,'DLSWD revision: CNN training', ...
    sprintf('CNN training: 0 / %d seed-fold runs',nTrainingRuns));
closeBar = onCleanup(@() closeProgressWindow(h));
for f=1:P.nOuterFolds
    te=find(M.Fold==f);
    outerTr=find(M.Fold~=f);
    [tr,va]=innerSourceSplit(M.SourceId,outerTr,P.validationSourceFraction,P.foldSeed+f);
    yTr=categorical(M.Label(tr),1:K,"C"+string(1:K));
    yVa=categorical(M.Label(va),1:K,"C"+string(1:K));
    fprintf('CNN fold %d/%d: %d train / %d validation / %d held-out records\n', ...
        f,P.nOuterFolds,numel(tr),numel(va),numel(te));
    valAcc=zeros(nSeed,1); testAcc=zeros(nSeed,1); testF2=zeros(nSeed,1);
    for s=1:nSeed
        net=trainRevisionCNN(X(:,:,:,tr),yTr,X(:,:,:,va),yVa,K,P,P.seeds(s), ...
            h,completedRuns,nTrainingRuns,f);
        ypVa=double(classify(net,X(:,:,:,va),'MiniBatchSize',P.batchSize));
        ypTe=double(classify(net,X(:,:,:,te),'MiniBatchSize',P.batchSize));
        pred(te,s)=uint8(ypTe);
        valAcc(s)=mean(ypVa==M.Label(va));
        testAcc(s)=mean(ypTe==M.Label(te));
        testF2(s)=macroFbeta(confusionCounts(M.Label(te),ypTe,K),2);
        rows(end+1,:)={f,P.seeds(s),numel(tr),numel(va),numel(te), ...
            valAcc(s),testAcc(s),testF2(s)}; %#ok<AGROW>
        fprintf('  seed %d: validation %.2f%%, held-out %.2f%%, macro-F2 %.3f\n', ...
            P.seeds(s),100*valAcc(s),100*testAcc(s),testF2(s));
        completedRuns = completedRuns+1;
        updateProgressWindow(h,completedRuns/nTrainingRuns, ...
            sprintf('CNN training: %d / %d seed-fold runs complete',completedRuns,nTrainingRuns));
    end
    [~,bestSeed]=max(valAcc);
    foldInfo(end+1)=struct('Fold',f,'TrainIdx',tr,'ValIdx',va,'TestIdx',te, ...
        'BestSeed',bestSeed); %#ok<AGROW>
end
cvSeed=cell2table(rows,'VariableNames',{'Fold','Seed','TrainN','ValN','TestN', ...
    'ValidationAccuracy','TestAccuracy','TestMacroF2'});
save(P.predMat,'pred','cvSeed','foldInfo','-v7.3');
end

function [tr,va] = innerSourceSplit(sourceId,outerTr,validationFraction,seed)
src=unique(sourceId(outerTr),'sorted');
nVa=max(1,round(numel(src)*validationFraction));
rng(seed,'twister'); src=src(randperm(numel(src)));
vaSrc=src(1:nVa);
va=outerTr(ismember(sourceId(outerTr),vaSrc));
tr=outerTr(~ismember(sourceId(outerTr),vaSrc));
assert(isempty(intersect(unique(sourceId(tr)),unique(sourceId(va)))),'Inner source leakage.');
end

function net = trainRevisionCNN(Xtr,Ytr,Xva,Yva,K,P,seed,h,completedRuns,nTrainingRuns,fold)
rng(seed,'twister');
layers=[ ...
    imageInputLayer(P.imSize,'Normalization','none','Name','input')
    convolution2dLayer(3,32,'Padding','same','WeightsInitializer','he','Name','conv1')
    batchNormalizationLayer('Name','bn1')
    reluLayer('Name','relu1')
    maxPooling2dLayer(2,'Stride',2,'Name','pool1')
    convolution2dLayer(3,64,'Padding','same','WeightsInitializer','he','Name','conv2')
    batchNormalizationLayer('Name','bn2')
    reluLayer('Name','relu2')
    maxPooling2dLayer(2,'Stride',2,'Name','pool2')
    dropoutLayer(0.25,'Name','dropout')
    fullyConnectedLayer(K,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classout')];
options=trainingOptions('adam','MaxEpochs',P.maxEpochs, ...
    'MiniBatchSize',P.batchSize,'InitialLearnRate',P.initialLearnRate, ...
    'LearnRateSchedule','piecewise','LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',P.learnRateDropPeriod,'L2Regularization',P.l2Regularization, ...
    'Shuffle','every-epoch','ValidationData',{Xva,Yva}, ...
    'ValidationFrequency',max(1,floor(numel(Ytr)/P.batchSize)), ...
    'ExecutionEnvironment',P.executionEnvironment,'Verbose',false,'Plots','none', ...
    'OutputFcn',@(info) cnnProgress(info,h,completedRuns,nTrainingRuns,P,fold,seed));
net=trainNetwork(Xtr,Ytr,layers,options);
end

function stop = cnnProgress(info,h,completedRuns,nTrainingRuns,P,fold,seed)
stop = false;
if isfield(info,'Epoch') && (strcmp(info.State,'iteration') || strcmp(info.State,'done'))
    local = min(max(info.Epoch/P.maxEpochs,0),1);
    total = (completedRuns+local)/nTrainingRuns;
    updateProgressWindow(h,total,sprintf( ...
        'CNN fold %d, seed %d: epoch %d/%d (%d/%d runs)', ...
        fold,seed,ceil(info.Epoch),P.maxEpochs,completedRuns+1,nTrainingRuns));
end
end

function C = confusionCounts(y,yp,K)
C=zeros(K,K);
for i=1:numel(y), C(y(i),yp(i))=C(y(i),yp(i))+1; end
end

function f = macroFbeta(C,beta)
tp=diag(C); fp=sum(C,1)'-tp; fn=sum(C,2)-tp;
p=tp./max(tp+fp,eps); r=tp./max(tp+fn,eps);
f=mean((1+beta^2)*p.*r./max(beta^2*p+r,eps));
end

% ========================================================================
function [samples,selection,cbcSensitivity,nmiMatrix] = evaluatePolicies(M,Q,pred,estimatedSNR,classes,foldInfo,P)
N=height(M); nSeed=size(pred,2); K=numel(classes);
deployed=zeros(N,1); lookup=zeros(N,1);
selectionRows={}; cbcRows={}; nmiRows={};
for f=1:numel(foldInfo)
    z=foldInfo(f); tr=z.TrainIdx; te=z.TestIdx;
    deployed(te)=double(pred(te,z.BestSeed));
    lookup(te)=snrLookupPredict(estimatedSNR(tr),M.SNRdB(tr),M.Label(tr),estimatedSNR(te));

    [mcs,info]=cbcSelect(M.Label(tr),cellstr(classes),P.tauC,P.tauI);
    for k=1:K
        selectionRows(end+1,:)={f,classes(k),sum(M.Label(tr)==k),mean(M.Label(tr)==k),ismember(k,mcs)}; %#ok<AGROW>
    end
    for t=P.tauCsweep
        [m2,i2]=cbcSelect(M.Label(tr),cellstr(classes),t,P.tauI);
        cbcRows(end+1,:)={f,"tauC",t,P.tauI,i2.nStep1,i2.nMCS,strjoin(string(classes(m2)),"; ")}; %#ok<AGROW>
    end
    for t=P.tauIsweep
        [m2,i2]=cbcSelect(M.Label(tr),cellstr(classes),P.tauC,t);
        cbcRows(end+1,:)={f,"tauI",P.tauC,t,i2.nStep1,i2.nMCS,strjoin(string(classes(m2)),"; ")}; %#ok<AGROW>
    end
    for a=1:numel(info.step1)
        for b=1:numel(info.step1)
            nmiRows(end+1,:)={f,classes(info.step1(a)),classes(info.step1(b)),info.NMI(a,b)}; %#ok<AGROW>
        end
    end
end
selection=cell2table(selectionRows,'VariableNames',{'Fold','Configuration','Count','Fraction','RetainedAtDefault'});
cbcSensitivity=cell2table(cbcRows,'VariableNames',{'Fold','Sweep','TauC','TauI','Step1N','MCSN','RetainedConfigurations'});
nmiMatrix=cell2table(nmiRows,'VariableNames',{'Fold','ConfigurationA','ConfigurationB','NormalisedMI'});

samples=[ ...
    policyRows("No denoising",zeros(N,1),M,Q,true,0);
    policyRows("Oracle upper bound",M.Label,M,Q,false,0);
    policyRows("Estimated-SNR lookup",lookup,M,Q,false,0);
    policyRows("DLSWD-CNN deployed",deployed,M,Q,false,0)];
for k=1:K
    fixedName = "Fixed: "+shortConfigName(classes(k));
    samples=[samples;policyRows(fixedName,repmat(k,N,1),M,Q,false,0)]; %#ok<AGROW>
end
for s=1:nSeed
    samples=[samples;policyRows("DLSWD-CNN seed "+string(P.seeds(s)),double(pred(:,s)),M,Q,false,P.seeds(s))]; %#ok<AGROW>
end

function label = shortConfigName(key)
z=split(key,'|');
method=z(2);
if method=="UniversalThreshold", method="Universal"; end
label=z(1)+"/"+method+"/L"+z(5);
end
end

function labels = snrLookupPredict(estTrain,trueSnrTrain,labelTrain,estTest)
% True input SNR calibrates the lookup only inside the training sources.
levels=unique(trueSnrTrain,'sorted');
centers=zeros(numel(levels),1); decision=zeros(numel(levels),1);
for i=1:numel(levels)
    q=trueSnrTrain==levels(i);
    centers(i)=median(estTrain(q));
    decision(i)=mode(labelTrain(q));
end
labels=zeros(numel(estTest),1);
for i=1:numel(estTest)
    [~,j]=min(abs(centers-estTest(i)));
    labels(i)=decision(j);
end
end

function T = policyRows(policy,choice,M,Q,isNoDenoise,seed)
N=height(M);
if isNoDenoise
    corr=M.OracleCorrNoisy; snr=M.OracleSNRBefore; mse=M.OracleMSEBefore;
    rangeErr=zeros(N,1); velErr=zeros(N,1); rdnmse=zeros(N,1);
else
    ii=sub2ind(size(Q.Corr),(1:N)',choice(:));
    corr=double(Q.Corr(ii)); snr=double(Q.SNR(ii)); mse=double(Q.MSE(ii));
    rangeErr=double(Q.RefRangeErr_m(ii)); velErr=double(Q.RefVelocityErr_mps(ii));
    rdnmse=double(Q.RDNMSE(ii));
end
T=table(repmat(string(policy),N,1),repmat(seed,N,1),M.Fold,M.SourceId,M.SNRdB, ...
    choice(:),corr(:),snr(:),mse(:),rangeErr(:),velErr(:),rdnmse(:), ...
    'VariableNames',{'Policy','Seed','Fold','SourceId','InputSNRdB','ConfigurationId', ...
    'Correlation','OutputSNRdB','MSE','ReferencePeakRangeErr_m', ...
    'ReferencePeakVelocityErr_mps','RDNMSE'});
end

% ========================================================================
function [sourceMetrics,summary,bySNR,paired] = summarizeResults(samples,P)
policies=unique(samples.Policy,'stable');
sourceRows={}; snrRows={}; summaryRows={};
for p=1:numel(policies)
    T=samples(samples.Policy==policies(p),:);
    seeds=unique(T.Seed);
    for sd=seeds'
        U=T(T.Seed==sd,:);
        [src,snr]=aggregateSource(U);
        src.Policy=repmat(policies(p),height(src),1); src.Seed=repmat(sd,height(src),1);
        snr.Policy=repmat(policies(p),height(snr),1); snr.Seed=repmat(sd,height(snr),1);
        sourceRows=[sourceRows;table2cell(src)]; %#ok<AGROW>
        snrRows=[snrRows;table2cell(snr)]; %#ok<AGROW>
        if sd==0
            summaryRows(end+1,:)=summaryRow(policies(p),sd,src,P); %#ok<AGROW>
        end
    end
end
sourceMetrics=cell2table(sourceRows,'VariableNames',{'SourceId','Correlation','OutputSNRdB','MSE', ...
    'ReferencePeakRangeErr_m','ReferencePeakVelocityErr_mps','RDNMSE','Policy','Seed'});
bySNR=cell2table(snrRows,'VariableNames',{'SourceId','InputSNRdB','Correlation','OutputSNRdB','MSE', ...
    'ReferencePeakRangeErr_m','ReferencePeakVelocityErr_mps','RDNMSE','Policy','Seed'});
summary=cell2table(summaryRows,'VariableNames',{'Policy','Seed','NSource','CorrelationMean','CorrelationCIlo', ...
    'CorrelationCIhi','OutputSNRdBMean','OutputSNRdBCIlo','OutputSNRdBCIhi','MSEMean','MSECIlo','MSECIhi', ...
    'ReferencePeakRangeErr_mMean','ReferencePeakVelocityErr_mpsMean','RDNMSEMean'});

a=sortrows(sourceMetrics(sourceMetrics.Policy=="DLSWD-CNN deployed" & sourceMetrics.Seed==0,:),'SourceId');
b=sortrows(sourceMetrics(sourceMetrics.Policy=="Estimated-SNR lookup" & sourceMetrics.Seed==0,:),'SourceId');
assert(isequal(a.SourceId,b.SourceId),'Source pairing failed.');
metrics=["Correlation","OutputSNRdB","MSE"]; pairedRows={};
for i=1:numel(metrics)
    d=a.(metrics(i))-b.(metrics(i));
    [lo,hi]=bootstrapCI(d,P.bootstrapReplicates,P.foldSeed+i);
    pairedRows(end+1,:)={metrics(i),mean(d),lo,hi,numel(d)}; %#ok<AGROW>
end
paired=cell2table(pairedRows,'VariableNames',{'Metric','CNNminusLookupMean','CIlo','CIhi','NSource'});
end

function [sourceAgg,snrAgg] = aggregateSource(T)
metricNames={'Correlation','OutputSNRdB','MSE','ReferencePeakRangeErr_m','ReferencePeakVelocityErr_mps','RDNMSE'};
sourceId=unique(T.SourceId,'sorted');
sourceAgg=table(sourceId,zeros(numel(sourceId),1),zeros(numel(sourceId),1),zeros(numel(sourceId),1), ...
    zeros(numel(sourceId),1),zeros(numel(sourceId),1),zeros(numel(sourceId),1), ...
    'VariableNames',{'SourceId',metricNames{:}});
for i=1:numel(sourceId)
    q=T.SourceId==sourceId(i);
    sourceAgg{i,2:end}=mean(T{q,metricNames},1,'omitnan');
end
[G,sid,snr]=findgroups(T.SourceId,T.InputSNRdB);
vals=zeros(max(G),numel(metricNames));
for g=1:max(G), vals(g,:)=mean(T{G==g,metricNames},1,'omitnan'); end
snrAgg=table(sid,snr,vals(:,1),vals(:,2),vals(:,3),vals(:,4),vals(:,5),vals(:,6), ...
    'VariableNames',{'SourceId','InputSNRdB',metricNames{:}});
end

function row = summaryRow(policy,seed,S,P)
[cLo,cHi]=bootstrapCI(S.Correlation,P.bootstrapReplicates,P.foldSeed+11);
[sLo,sHi]=bootstrapCI(S.OutputSNRdB,P.bootstrapReplicates,P.foldSeed+12);
[mLo,mHi]=bootstrapCI(S.MSE,P.bootstrapReplicates,P.foldSeed+13);
row={policy,seed,height(S),mean(S.Correlation),cLo,cHi,mean(S.OutputSNRdB),sLo,sHi, ...
    mean(S.MSE),mLo,mHi,mean(S.ReferencePeakRangeErr_m), ...
    mean(S.ReferencePeakVelocityErr_mps),mean(S.RDNMSE)};
end

function [lo,hi] = bootstrapCI(x,B,seed)
x=x(isfinite(x)); N=numel(x); assert(N>1,'Too few independent sources for a CI.');
rng(seed,'twister'); z=zeros(B,1);
for b=1:B, z(b)=mean(x(randi(N,N,1))); end
z=sort(z); lo=z(max(1,round(0.025*B))); hi=z(min(B,round(0.975*B)));
end

% ========================================================================
function writeOutputs(M,configs,cvSeed,selection,cbcSensitivity,nmiMatrix, ...
    samples,sourceMetrics,summary,bySNR,paired,P)
writetable(configs,fullfile(P.outDir,'configuration_library.csv'));
writetable(cvSeed,fullfile(P.outDir,'cnn_seed_metrics.csv'));
writetable(selection,fullfile(P.outDir,'configuration_selection_by_fold.csv'));
writetable(cbcSensitivity,fullfile(P.outDir,'cbc_sensitivity.csv'));
writetable(nmiMatrix,fullfile(P.outDir,'cbc_nmi_matrix_long.csv'));
writetable(samples,fullfile(P.outDir,'policy_metrics_per_record.csv'));
writetable(sourceMetrics,fullfile(P.outDir,'policy_metrics_per_source.csv'));
writetable(summary,P.summaryCsv);
writetable(bySNR,fullfile(P.outDir,'policy_metrics_by_source_snr.csv'));
writetable(paired,fullfile(P.outDir,'cnn_vs_lookup_paired_ci.csv'));
writeRawConfusionCounts(M,configs,P);

fid=fopen(fullfile(P.outDir,'methodological_limitations.txt'),'w');
fprintf(fid,['Cached bestResults.csv data are offline oracle-label records.  ', ...
    'Legacy train/test folders were merged as noise realisations, not used as ', ...
    'independent partitions.  All realisations of each source are in one outer fold.\n\n']);
fprintf(fid,['No external target range/velocity annotations are in the supplied data. ', ...
    'Therefore, reference-peak preservation is exported, but Pd/Pfa and true ', ...
    'target localisation metrics are intentionally not claimed.\n']);
fclose(fid);

fid=fopen(fullfile(P.outDir,'detection_metrics_unavailable.csv'),'w');
fprintf(fid,'Metric,Value,Reason\n');
fprintf(fid,'Pd,NaN,"No externally annotated target support was supplied"\n');
fprintf(fid,'Pfa,NaN,"No externally annotated non-target support was supplied"\n');
fclose(fid);
end

function writeRawConfusionCounts(M,configs,P)
% Raw counts, not row-normalised percentages: this preserves class support.
S=load(P.predMat,'pred');
K=height(configs); rows={};
for f=unique(M.Fold)'
    idx=M.Fold==f;
    for s=1:size(S.pred,2)
        C=confusionCounts(M.Label(idx),double(S.pred(idx,s)),K);
        for i=1:K
            for j=1:K
                rows(end+1,:)={f,s,configs.Key(i),configs.Key(j),C(i,j)}; %#ok<AGROW>
            end
        end
    end
end
T=cell2table(rows,'VariableNames',{'Fold','Seed','TrueConfiguration','PredictedConfiguration','Count'});
writetable(T,fullfile(P.outDir,'cnn_confusion_raw_counts.csv'));
end

function makeFigures(B,cvSeed,P)
% Fig. 10 is a single-column, Fig. 11-sized reconstruction comparison.
% The first three colours exactly follow Fig. 8's blue/orange/green
% configuration-class mapping.  The two conventional controls use neutral
% graphite and violet colours, distinct from the configuration classes and
% adaptive policies. No confidence bands are drawn so that the DLSWD-CNN and
% reference-free DLSWD-LUT traces remain visually dominant.
policy=["Fixed: fk4/Universal/L6","Fixed: db8/Bayes/L5","Fixed: db4/SURE/L7", ...
        "Estimated-SNR lookup","Oracle upper bound","DLSWD-CNN deployed"];
legendNames={'fk4Universal6','db8Bayes5','db4Sure7', ...
             'DLSWD-LUT','', 'DLSWD (Proposed)','Mean Filter','Gaussian Filter'};
colors=[0.000 0.447 0.698; 0.902 0.624 0.000; 0.000 0.620 0.451; ...
        0.720 0.150 0.490; 0.350 0.350 0.350; 0.835 0.369 0.000];
lineStyles={'-o','--s',':^','--d','--','-h'};
lineWidths=[0.85 0.85 0.85 1.75 1.00 2.40];
conventional=loadConventionalCurves(P);
conventionalColors=[0.18 0.18 0.18; 0.49 0.18 0.56];
conventionalStyles={'--v',':p'};
% Match the one-column aspect ratio used by the comparison figure.
fig=figure('Color','w','Units','inches','Position',[1 1 3.45 2.06]);
ax=axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
ax.Toolbar.Visible='off';
h=gobjects(numel(policy)+2,1);
% Draw the LUT last so its hollow magenta diamonds remain visible where
% its performance nearly coincides with the proposed DLSWD trace.
drawOrder=[1 2 3 5 6 4];
for p=drawOrder
    U=B(B.Policy==policy(p) & B.Seed==0,:);
    x=unique(U.InputSNRdB,'sorted'); mu=zeros(numel(x),1);
    for i=1:numel(x)
        z=U(U.InputSNRdB==x(i),:);
        mu(i)=mean(z.OutputSNRdB);
    end
    markerSize=2.6;
    markerFace=colors(p,:);
    if p==4
        markerSize=4.6;                % DLSWD-LUT, foreground hollow diamond
        markerFace=[1 1 1];
    end
    if p==6, markerSize=4.5; end       % DLSWD (Proposed)
    h(p)=plot(ax,x,mu,lineStyles{p},'Color',colors(p,:),'LineWidth',lineWidths(p), ...
        'MarkerSize',markerSize,'MarkerFaceColor',markerFace);
end
for c=1:2
    h(numel(policy)+c)=plot(ax,conventional.InputSNRdB,conventional.OutputSNRdB(:,c), ...
        conventionalStyles{c},'Color',conventionalColors(c,:),'LineWidth',1.10, ...
        'MarkerSize',3.0,'MarkerFaceColor','w');
end
xlabel(ax,'Input SNR before denoising (dB)');
ylabel(ax,'Post-denoising SNR (dB)');
xlim(ax,[-20 5]); ylim(ax,[-10 11]); xticks(ax,-20:5:5); yticks(ax,-10:2:10);
set(ax,'FontName','Times New Roman','FontSize',8,'LineWidth',0.8);
% The oracle stays as an unlabeled grey upper-bound trace; it is deliberately
% excluded from the legend because it is not a deployable method.
lg=legend(ax,h([1 2 3 4 7 8 6]),legendNames([1 2 3 4 7 8 6]),'Location','southeast');
set(lg,'FontName','Times New Roman','FontSize',6,'Box','on', ...
    'EdgeColor',[0.6 0.6 0.6],'Color','w','ItemTokenSize',[9 5]);
hold(ax,'off');
exportgraphics(fig,fullfile(P.figDir,'F_grouped_source_disjoint_performance.pdf'),'ContentType','vector');
exportgraphics(fig,fullfile(P.figDir,'F_grouped_source_disjoint_performance.png'),'Resolution',600);
close(fig);

fig=figure('Color','w','Units','inches','Position',[1 1 3.45 2.8]);
boxchart(categorical(string(cvSeed.Seed)),100*cvSeed.TestAccuracy,'MarkerStyle','.');
xlabel('CNN seed'); ylabel('Held-out source accuracy (%)'); grid on; box on;
set(gca,'FontName','Times New Roman','FontSize',8,'LineWidth',0.8);
exportgraphics(fig,fullfile(P.figDir,'F_grouped_cnn_seed_accuracy.pdf'),'ContentType','vector');
exportgraphics(fig,fullfile(P.figDir,'F_grouped_cnn_seed_accuracy.png'),'Resolution',600);
close(fig);
end

function C = loadConventionalCurves(P)
% Load the already measured controls; do not rerun their 17,940-record pass.
cacheFile=fullfile(P.outDir,'conventional_filter_metrics.mat');
assert(isfile(cacheFile),['Missing conventional-filter cache. Run ' ...
    'run_conventional_filter_comparison before run_cached_grouped_revision.']);
S=load(P.manifestMat,'M'); D=load(cacheFile,'boxSNR','gaussSNR');
assert(numel(D.boxSNR)==height(S.M) && numel(D.gaussSNR)==height(S.M), ...
    'Conventional-filter cache does not match the cached manifest.');
x=unique(S.M.SNRdB,'sorted'); y=zeros(numel(x),2);
for i=1:numel(x)
    idx=S.M.SNRdB==x(i);
    y(i,:)=[mean(D.boxSNR(idx),'omitnan'),mean(D.gaussSNR(idx),'omitnan')];
end
C=struct('InputSNRdB',x,'OutputSNRdB',y);
end

function writeLatex(S,selection,cvSeed,paired,P)
cnn=S(S.Policy=="DLSWD-CNN deployed",:);
lut=S(S.Policy=="Estimated-SNR lookup",:);
ora=S(S.Policy=="Oracle upper bound",:);
fixed=S(startsWith(S.Policy,"Fixed:"),:);
assert(height(cnn)==1 && height(lut)==1 && height(fixed)==3 && height(ora)==1);
fid=fopen(P.latexSnippet,'w');
fprintf(fid,'%% Auto-generated by run_cached_grouped_revision.m\n');
fprintf(fid,'%% Inspect every value before reporting it publicly.\n\n');
fprintf(fid,'%s\n','\begin{table}[t]');
fprintf(fid,'%s\n','\caption{Source-frame-disjoint five-fold reconstruction evaluation. Each input-SNR bin ($-20$ to $+5$~dB) is equally weighted within each source. Values are source-level means with 95\% bootstrap CIs; post-denoising SNR and reference MSE are evaluated against the clean signal. The oracle is an upper bound, not a deployable method.}');
fprintf(fid,'%s\n','\label{tab:grouped-evaluation}');
fprintf(fid,'%s\n','\centering\scriptsize');
fprintf(fid,'%s\n','\setlength{\tabcolsep}{1pt}\renewcommand{\arraystretch}{1.03}');
fprintf(fid,'%s\n','\begin{tabular}{@{}p{0.28\columnwidth}ccc@{}}');
fprintf(fid,'%s\n','\toprule');
fprintf(fid,'%s\n','Policy & \makecell[c]{Post-denoising\\SNR (dB)} & Correlation & \makecell[c]{Reference\\MSE} \\');
fprintf(fid,'%s\n','\midrule');
latexRow(fid,'DLSWD--CNN',cnn);
latexRow(fid,'SNR-estimate lookup',lut);
for k=1:height(fixed)
    latexRow(fid,char(strrep(fixed.Policy(k),'Fixed: ','Fixed ')),fixed(k,:));
end
latexRow(fid,'Oracle upper bound',ora);
fprintf(fid,'%s\n','\bottomrule');
fprintf(fid,'%s\n','\end{tabular}');
fprintf(fid,'%s\n','\end{table}');
fprintf(fid,'\n');

fprintf(fid,['The source-frame-disjoint five-fold protocol yields CNN configuration ', ...
    'accuracy of %.2f\\%% $\\pm$ %.2f\\%% across the 25 fold--seed evaluations. ', ...
    'All realisations from a source frame are confined to one outer fold. ', ...
    'No target-detection $P_d/P_{fa}$ claim is made without external radar annotations.\n\n'], ...
    100*mean(cvSeed.TestAccuracy),100*std(cvSeed.TestAccuracy));
for k=1:height(paired)
    fprintf(fid,'%% CNN - lookup, %s: %.5g [%.5g, %.5g], n=%d source frames\n', ...
        paired.Metric(k),paired.CNNminusLookupMean(k),paired.CIlo(k),paired.CIhi(k),paired.NSource(k));
end
fprintf(fid,'%% Fold-1 CBC occurrence frequencies:\n');
for k=1:height(selection)
    if selection.Fold(k)==1
        fprintf(fid,'%% %s: %.2f%%; retained=%d\n',char(selection.Configuration(k)), ...
            100*selection.Fraction(k),selection.RetainedAtDefault(k));
    end
end
fclose(fid);
end

function latexRow(fid,name,T)
fprintf(fid,'%s & \\makecell[c]{%.2f\\\\[-.2ex]{[%.2f, %.2f]}} & \\makecell[c]{%.3f\\\\[-.2ex]{[%.3f, %.3f]}} & \\makecell[c]{%.2f\\\\[-.2ex]{[%.2f, %.2f]}} \\\\\n', ...
    name,T.OutputSNRdBMean,T.OutputSNRdBCIlo,T.OutputSNRdBCIhi, ...
    T.CorrelationMean,T.CorrelationCIlo,T.CorrelationCIhi, ...
    T.MSEMean,T.MSECIlo,T.MSECIhi);
end
