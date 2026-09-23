function run_swista_nets_baseline(varargin)
%RUN_SWISTA_NETS_BASELINE Same-data SWISTA-Net denoising baseline.
%
% This script implements a source-frame-disjoint, identity-forward
% SWISTA-Net adaptation for the FMCW matrix denoising task.  The forward
% model is the identity operator because the available supervision is
% noisy/clean radar matrices rather than an undersampled medical-image
% inverse operator.  The network keeps the SWISTA-Net structure relevant
% to this denoising setting: seven unrolled wavelet-shrinkage iterations,
% three-level subband-adaptive Haar decomposition, trainable per-stage
% steps/thresholds/momentum, and a learned convolutional post-processing
% path.
%
% Outputs:
%   matlab/out/swista_nets/swista_nets_per_record.csv
%   matlab/out/swista_nets/swista_nets_by_source.csv
%   matlab/out/swista_nets/swista_nets_by_snr.csv
%   matlab/out/swista_nets/swista_nets_summary.csv
%   matlab/out/swista_nets/swista_nets_status.txt

ip = inputParser;
addParameter(ip,'Fold',"all",@(x) (isnumeric(x) && isscalar(x)) || ischar(x) || isstring(x));
addParameter(ip,'Epochs',4,@(x) isnumeric(x) && isscalar(x) && x>=1);
addParameter(ip,'MiniBatchSize',8,@(x) isnumeric(x) && isscalar(x) && x>=1);
addParameter(ip,'MaxTrainRecords',inf,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'MaxTestRecords',inf,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'NumIterations',7,@(x) isnumeric(x) && isscalar(x) && x>=1);
addParameter(ip,'Levels',3,@(x) isnumeric(x) && isscalar(x) && x>=1);
addParameter(ip,'LearnRate',1e-3,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'ExecutionEnvironment',"auto",@(x) ischar(x) || isstring(x));
addParameter(ip,'Seed',20260918,@(x) isnumeric(x) && isscalar(x));
parse(ip,varargin{:});
O = ip.Results;

rng(O.Seed,'twister');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'matlab')));
outDir = fullfile(root,'matlab','out','swista_nets');
if ~exist(outDir,'dir'), mkdir(outDir); end

manifestFile = fullfile(root,'matlab','out','cached_grouped_revision','manifest.mat');
assert(isfile(manifestFile),'Missing manifest: %s',manifestFile);
S = load(manifestFile,'M');
M = S.M;
assert(all(isfile(M.RawPath)) && all(isfile(M.CleanPath)), ...
    'Raw/clean CSV paths are unavailable. Check the F: data root in manifest.mat.');

useGPU = shouldUseGPU(O.ExecutionEnvironment);
folds = requestedFolds(O.Fold);
params = initializeParameters(O,useGPU);
avg = [];
avgSq = [];
iteration = 0;
allRows = table();

for f = folds
    fprintf('\nSWISTA-Net fold %d/%d\n',f,max(folds));
    trainIdx = find(M.Fold ~= f);
    testIdx = find(M.Fold == f);
    trainIdx = stratifiedLimit(M,trainIdx,O.MaxTrainRecords);
    testIdx = stratifiedLimit(M,testIdx,O.MaxTestRecords);
    fprintf('Training records: %d, test records: %d\n',numel(trainIdx),numel(testIdx));

    params = initializeParameters(O,useGPU);
    avg = [];
    avgSq = [];
    iteration = 0;
    for epoch = 1:O.Epochs
        trainIdx = trainIdx(randperm(numel(trainIdx)));
        epochLoss = 0;
        nSeen = 0;
        for b = 1:O.MiniBatchSize:numel(trainIdx)
            batchIdx = trainIdx(b:min(b+O.MiniBatchSize-1,numel(trainIdx)));
            B = loadBatch(M,batchIdx,useGPU);
            iteration = iteration + 1;
            [loss,grad] = dlfeval(@modelGradients,params,B.XNorm,B.YNorm,O);
            [params,avg,avgSq] = adamupdate(params,grad,avg,avgSq,iteration,O.LearnRate);
            batchLoss = double(gather(extractdata(loss)));
            epochLoss = epochLoss + batchLoss*numel(batchIdx);
            nSeen = nSeen + numel(batchIdx);
            if mod(iteration,25)==0 || b==1
                fprintf('  epoch %d/%d batch %d/%d loss %.5f\n', ...
                    epoch,O.Epochs,ceil(b/O.MiniBatchSize),ceil(numel(trainIdx)/O.MiniBatchSize),batchLoss);
            end
        end
        fprintf('  epoch %d mean loss %.5f\n',epoch,epochLoss/max(nSeen,1));
    end

    rows = scoreFold(M,testIdx,params,O,useGPU,f);
    allRows = [allRows; rows]; %#ok<AGROW>
    save(fullfile(outDir,sprintf('swista_net_b_fold_%d_params.mat',f)), ...
        'params','rows','O','-v7.3');
end

bySource = groupsummary(allRows,"SourceId","mean",["Correlation","OutputSNRdB","MSE"]);
bySNR = groupsummary(allRows,"SNRdB","mean",["Correlation","OutputSNRdB","MSE"]);
summary = table("SWISTA-Net",height(bySource), ...
    mean(bySource.mean_Correlation,'omitnan'), ...
    mean(bySource.mean_OutputSNRdB,'omitnan'), ...
    mean(bySource.mean_MSE,'omitnan'), ...
    'VariableNames',{'Method','NSource','CorrelationMean','OutputSNRdBMean','MSEMean'});

writetable(allRows,fullfile(outDir,'swista_nets_per_record.csv'));
writetable(bySource,fullfile(outDir,'swista_nets_by_source.csv'));
writetable(bySNR,fullfile(outDir,'swista_nets_by_snr.csv'));
writetable(summary,fullfile(outDir,'swista_nets_summary.csv'));

fid = fopen(fullfile(outDir,'swista_nets_status.txt'),'w');
fprintf(fid,'Method: SWISTA-Net adaptation for FMCW matrix denoising.\n');
fprintf(fid,'Forward operator: identity, because paired noisy/clean matrices are available.\n');
fprintf(fid,'Unrolled iterations: %d\nWavelet levels: %d\n',O.NumIterations,O.Levels);
fprintf(fid,'Trainable terms: subband steps, thresholds, FISTA momentum, and 3-layer post-processing CNN.\n');
fprintf(fid,'Folds evaluated: %s\nEpochs: %d\nMiniBatchSize: %d\n',mat2str(folds),O.Epochs,O.MiniBatchSize);
fprintf(fid,'Metrics are scored on raw matrix scale after reversing per-sample normalization.\n');
fclose(fid);

disp(summary)
fprintf('Wrote SWISTA-Net outputs under %s\n',outDir);
end

function folds = requestedFolds(x)
if ischar(x) || isstring(x)
    if strcmpi(string(x),"all")
        folds = 1:5;
    else
        folds = str2double(string(x));
    end
else
    folds = x;
end
folds = double(folds(:))';
assert(all(ismember(folds,1:5)),'Fold must be 1..5 or "all".');
end

function tf = shouldUseGPU(env)
env = string(env);
tf = false;
if strcmpi(env,"gpu")
    tf = true;
elseif strcmpi(env,"auto")
    try
        tf = gpuDeviceCount > 0;
    catch
        tf = false;
    end
end
end

function idx = stratifiedLimit(M,idx,maxN)
if ~isfinite(maxN) || numel(idx)<=maxN, return; end
G = findgroups(M.SourceId(idx),M.SNRdB(idx));
keep = [];
g = unique(G,'stable');
take = max(1,floor(maxN/numel(g)));
for i = 1:numel(g)
    q = idx(G==g(i));
    keep = [keep; q(1:min(take,numel(q)))]; %#ok<AGROW>
end
if numel(keep) < maxN
    extra = setdiff(idx,keep,'stable');
    keep = [keep; extra(1:min(numel(extra),maxN-numel(keep)))]; %#ok<AGROW>
end
idx = keep(1:min(maxN,numel(keep)));
end

function params = initializeParameters(O,useGPU)
nBands = 1 + 3*O.Levels;
params.logStep = dlarray(single(log(exp(0.85)-1))*ones(O.NumIterations,nBands,'single'));
params.logTheta = dlarray(single(log(exp(0.035)-1))*ones(O.NumIterations,nBands,'single'));
params.betaLogit = dlarray(zeros(O.NumIterations,1,'single'));
params.W1 = dlarray(heInit([3 3 1 16]));
params.B1 = dlarray(zeros(1,1,16,'single'));
params.W2 = dlarray(heInit([3 3 16 16]));
params.B2 = dlarray(zeros(1,1,16,'single'));
params.W3 = dlarray(single(0.01)*randn(3,3,16,1,'single'));
params.B3 = dlarray(zeros(1,1,1,'single'));
if useGPU
    params = dlupdate(@gpuArray,params);
end
end

function W = heInit(sz)
fanIn = prod(sz(1:3));
W = randn(sz,'single')*sqrt(2/fanIn);
end

function B = loadBatch(M,idx,useGPU)
N = numel(idx);
X = zeros(128,255,1,N,'single');
Y = zeros(128,255,1,N,'single');
lo = zeros(1,N,'single');
scale = zeros(1,N,'single');
rawClean = zeros(128,255,1,N,'single');
for i = 1:N
    n = idx(i);
    xRaw = single(readmatrix(char(M.RawPath(n))));
    yRaw = single(readmatrix(char(M.CleanPath(n))));
    [xNorm,yNorm,lo(i),scale(i)] = normalizePair(xRaw,yRaw);
    X(:,:,1,i) = xNorm;
    Y(:,:,1,i) = yNorm;
    rawClean(:,:,1,i) = real(yRaw);
end
if useGPU
    X = gpuArray(X);
    Y = gpuArray(Y);
end
B.XNorm = dlarray(X,'SSCB');
B.YNorm = dlarray(Y,'SSCB');
B.Lo = lo;
B.Scale = scale;
B.CleanRaw = rawClean;
end

function [xNorm,yNorm,lo,scale] = normalizePair(xRaw,yRaw)
xRaw = real(xRaw);
yRaw = real(yRaw);
lo = single(prctile(xRaw(:),1));
hi = single(prctile(xRaw(:),99));
scale = max(hi-lo,single(1e-6));
xNorm = min(max((xRaw-lo)/scale,0),1);
yNorm = min(max((yRaw-lo)/scale,0),1);
end

function [loss,grad] = modelGradients(params,X,Y,O)
YPred = forwardSWISTA(params,X,O);
loss = mean((YPred-Y).^2,'all');
grad = dlgradient(loss,params);
end

function Y = forwardSWISTA(params,X,O)
XPad = padForWavelet(X,O.Levels);
wy = haarDwt(XPad,O.Levels);
phi = wy;
prev = wy;
nBands = numel(wy);
for k = 1:O.NumIterations
    recon = haarIdwt(phi,O.Levels);
    res = XPad - recon;
    grad = haarDwt(res,O.Levels);
    next = cell(1,nBands);
    for s = 1:nBands
        step = softplus(params.logStep(k,s));
        theta = softplus(params.logTheta(k,s));
        r = phi{s} + step*grad{s};
        shrunk = softShrink(r,theta);
        if k > 1
            beta = sigmoid(params.betaLogit(k));
            next{s} = shrunk + beta*(shrunk-prev{s});
        else
            next{s} = shrunk;
        end
        prev{s} = shrunk;
    end
    phi = next;
end
YPad = haarIdwt(phi,O.Levels);
Y0 = cropToInput(YPad,X);
R = dlconv(Y0,params.W1,params.B1,'Padding','same','DataFormat','SSCB');
R = relu(R);
R = dlconv(R,params.W2,params.B2,'Padding','same','DataFormat','SSCB');
R = relu(R);
R = dlconv(R,params.W3,params.B3,'Padding','same','DataFormat','SSCB');
Y = Y0 + R;
end

function y = padForWavelet(x,levels)
targetH = ceil(size(x,1)/(2^levels))*(2^levels);
targetW = ceil(size(x,2)/(2^levels))*(2^levels);
y = x;
while size(y,1) < targetH
    y = cat(1,y,y(end,:,:,:));
end
while size(y,2) < targetW
    y = cat(2,y,y(:,end,:,:));
end
end

function y = cropToInput(x,ref)
y = x(1:size(ref,1),1:size(ref,2),:,:);
end

function coeff = haarDwt(x,levels)
coeff = {};
a = x;
details = cell(levels,3);
for lev = 1:levels
    evenR = 2:2:size(a,1);
    oddR = 1:2:size(a,1);
    evenC = 2:2:size(a,2);
    oddC = 1:2:size(a,2);
    x00 = a(oddR,oddC,:,:);
    x01 = a(oddR,evenC,:,:);
    x10 = a(evenR,oddC,:,:);
    x11 = a(evenR,evenC,:,:);
    ll = (x00+x01+x10+x11)*0.5;
    lh = (x00-x01+x10-x11)*0.5;
    hl = (x00+x01-x10-x11)*0.5;
    hh = (x00-x01-x10+x11)*0.5;
    a = ll;
    details{lev,1} = lh;
    details{lev,2} = hl;
    details{lev,3} = hh;
end
coeff{1} = a;
for lev = levels:-1:1
    coeff{end+1} = details{lev,1}; %#ok<AGROW>
    coeff{end+1} = details{lev,2}; %#ok<AGROW>
    coeff{end+1} = details{lev,3}; %#ok<AGROW>
end
end

function x = haarIdwt(coeff,levels)
a = coeff{1};
p = 2;
for lev = levels:-1:1
    lh = coeff{p};
    hl = coeff{p+1};
    hh = coeff{p+2};
    p = p + 3;
    H = size(a,1)*2;
    W = size(a,2)*2;
    x = zeros(H,W,size(a,3),size(a,4),'like',a);
    x(1:2:end,1:2:end,:,:) = (a+lh+hl+hh)*0.5;
    x(1:2:end,2:2:end,:,:) = (a-lh+hl-hh)*0.5;
    x(2:2:end,1:2:end,:,:) = (a+lh-hl-hh)*0.5;
    x(2:2:end,2:2:end,:,:) = (a-lh-hl+hh)*0.5;
    a = x;
end
end

function y = softShrink(x,theta)
y = x .* relu(1 - theta./(abs(x)+single(1e-6)));
end

function y = softplus(x)
y = log(1+exp(x)) + single(1e-6);
end

function y = sigmoid(x)
y = 1./(1+exp(-x));
end

function rows = scoreFold(M,idx,params,O,useGPU,fold)
rows = table();
for b = 1:O.MiniBatchSize:numel(idx)
    batchIdx = idx(b:min(b+O.MiniBatchSize-1,numel(idx)));
    B = loadBatch(M,batchIdx,useGPU);
    YNorm = forwardSWISTA(params,B.XNorm,O);
    YNorm = gather(extractdata(YNorm));
    N = numel(batchIdx);
    src = zeros(N,1);
    snr = zeros(N,1);
    corr = zeros(N,1);
    snro = zeros(N,1);
    mse = zeros(N,1);
    for i = 1:N
        n = batchIdx(i);
        y = double(YNorm(:,:,1,i))*double(B.Scale(i)) + double(B.Lo(i));
        r = double(B.CleanRaw(:,:,1,i));
        src(i) = M.SourceId(n);
        snr(i) = M.SNRdB(n);
        corr(i) = corrReal(y,r);
        e = y-r;
        mse(i) = mean(e(:).^2);
        snro(i) = 10*log10(sum(r(:).^2)/max(sum(e(:).^2),eps));
    end
    T = table(repmat(fold,N,1),batchIdx(:),src,snr,corr,snro,mse, ...
        'VariableNames',{'Fold','RecordIndex','SourceId','SNRdB','Correlation','OutputSNRdB','MSE'});
    rows = [rows; T]; %#ok<AGROW>
    fprintf('  scored %d/%d records\n',min(b+O.MiniBatchSize-1,numel(idx)),numel(idx));
end
end

function r = corrReal(x,y)
a = x(:)-mean(x(:));
b = y(:)-mean(y(:));
r = (a'*b)/max(sqrt((a'*a)*(b'*b)),eps);
end
