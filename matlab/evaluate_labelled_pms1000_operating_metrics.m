function evaluate_labelled_pms1000_operating_metrics(varargin)
%EVALUATE_LABELLED_PMS1000_OPERATING_METRICS Label-aware detector analysis.
%
% Reuses the cached held-out CNN and DLSWD-LUT configuration decisions from
% run_cached_grouped_revision. No SCS, CBC, feature extraction, CNN training,
% or original-data write is performed. The 30 pms1000 source frames are mapped
% exactly to public ROD2021 frames 30--59; their matching text_labels CSVs are
% stored under matlab/out/cached_grouped_revision/public_pms1000_text_labels.
%
% Scope. The original workflow denoises the real component of Rx3/Tx1. For
% label-aware range-angle analysis, that archived noisy Rx3/Tx1 matrix is used
% unchanged and the same measured real-channel noise ratio is generated with a
% fixed seed on the other seven raw-MIMO channels. Each selected configuration
% is then applied channelwise. This is a controlled synthetic-noise evaluation,
% not a new data-generation or training workflow.
%
% Reported metrics:
%   Pd: one labelled pedestrian is detected in a predeclared range-angle gate;
%   empirical background CFAR firing rate: CFAR-positive cells outside the
%       labelled target-range gate divided by valid background RD cells;
%   false detections per frame, range RMSE, and angle RMSE, all from the same
%       labelled target match. The background firing rate is not a target-free
%       calibration of a theoretical CFAR Pfa because all 30 frames contain a
%       pedestrian.
%
% Optional name/value inputs:
%   'MaxRecords'     inf (default); a small source/SNR-stratified smoke test.
%   'OutputTag'      '' (default); keeps optional test outputs separate.
%   'ForceRecompute' false (default); ignore resumable cache.

ip = inputParser;
addParameter(ip,'MaxRecords',inf,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'OutputTag','',@(x) ischar(x) || isstring(x));
addParameter(ip,'ForceRecompute',false,@(x) islogical(x) && isscalar(x));
parse(ip,varargin{:});
O = ip.Results;

clc
root = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(root,'matlab','out','cached_grouped_revision');
assert(isfolder(outDir),'Cached revision output is missing: %s',outDir);
tag = string(O.OutputTag);
if strlength(tag)>0
    assert(~contains(tag,[filesep '/' '\\']), 'OutputTag must be a simple name.');
    tag = "_" + tag;
end
F = outputFiles(outDir,tag);

S = load(fullfile(outDir,'manifest.mat'),'M','P');
M = S.M;
Pred = load(fullfile(outDir,'cnn_predictions.mat'),'pred','foldInfo');
Feat = load(fullfile(outDir,'features.mat'),'estimatedSNR');
assert(height(M)==17940,'Expected 17,940 cached records; found %d.',height(M));
assert(size(Pred.pred,1)==height(M) && numel(Feat.estimatedSNR)==height(M), ...
    'Cached selections do not match the manifest.');

[M,configs] = configurationLibrary(M);
[cnnChoice,lutChoice] = heldOutChoices(M,Pred.pred,Pred.foldInfo,Feat.estimatedSNR);
C = analysisConfig();
[labels,mapping] = loadVerifiedLabels(outDir);
sourceCube = loadSourceCube();
verifySourceAlignment(M,sourceCube);

testIdx = find(M.Origin=="legacy_test");
assert(numel(testIdx)==30*26*3,'Expected 2,340 legacy test records; found %d.',numel(testIdx));
recordIdx = chooseRecords(M,testIdx,O.MaxRecords);
fprintf('Label-aware pms1000 operating evaluation: %d / %d held-out test records.\n', ...
    numel(recordIdx),numel(testIdx));
fprintf('Policies: DLSWD (held-out CNN choice) and DLSWD-LUT.\n');
fprintf('No CNN training, SCS/CBC search, or source-dataset modification will run.\n');

Q = loadOrCreateCache(F.cache,height(M),O.ForceRecompute);
todo = recordIdx(~all(Q.Done(recordIdx,:),2));
for j = 1:numel(todo)
    n = todo(j);
    sid = M.SourceId(n);
    noisyCube = constructNoisyCube(sourceCube(:,:,:,sid),M,n);
    policyChoices = [cnnChoice(n),lutChoice(n)];
    for p = 1:2
        if Q.Done(n,p), continue; end
        sameChoice = find(policyChoices(1:p-1)==policyChoices(p) & Q.Done(n,1:p-1),1,'first');
        if ~isempty(sameChoice)
            Q.Hit(n,p)=Q.Hit(n,sameChoice);
            Q.BackgroundCFARFARate(n,p)=Q.BackgroundCFARFARate(n,sameChoice);
            Q.FalseDetectionsPerFrame(n,p)=Q.FalseDetectionsPerFrame(n,sameChoice);
            Q.RangeError_m(n,p)=Q.RangeError_m(n,sameChoice);
            Q.AngleError_deg(n,p)=Q.AngleError_deg(n,sameChoice);
            Q.DetectionCount(n,p)=Q.DetectionCount(n,sameChoice);
            Q.WaveletMs(n,p)=Q.WaveletMs(n,sameChoice);
            Q.Done(n,p)=true;
            continue
        end
        cfg = configs(policyChoices(p),:);
        t = tic;
        denoisedCube = denoiseCube(noisyCube,cfg);
        Q.WaveletMs(n,p) = 1000*toc(t);
        [det,cfarMask] = detectRangeAngle(denoisedCube,C);
        truth = labels(sid,:);
        outcome = scoreLabelledDetections(det,cfarMask,truth,C);
        Q.Hit(n,p) = outcome.Hit;
        Q.BackgroundCFARFARate(n,p) = outcome.BackgroundCFARFARate;
        Q.FalseDetectionsPerFrame(n,p) = outcome.FalseDetectionsPerFrame;
        Q.RangeError_m(n,p) = outcome.RangeError_m;
        Q.AngleError_deg(n,p) = outcome.AngleError_deg;
        Q.DetectionCount(n,p) = outcome.DetectionCount;
        Q.Done(n,p) = true;
    end
    if mod(j,25)==0 || j==numel(todo)
        save(F.cache,'Q','C','mapping','-v7.3');
        printProgress(j,numel(todo),'label-aware range-angle analysis');
    end
end
save(F.cache,'Q','C','mapping','-v7.3');

[summary,details,bySource,status] = summarizeMetrics(M,recordIdx,cnnChoice,lutChoice,Q,C);
writetable(mapping,F.mapping);
writetable(summary,F.summary);
writetable(details,F.details);
writetable(bySource,F.bySource);
writetable(status,F.status);
save(F.complete,'M','recordIdx','cnnChoice','lutChoice','configs','labels','mapping','Q','C', ...
    'summary','details','bySource','status','-v7.3');

disp(summary)
fprintf('\nCompleted label-aware outputs:\n  %s\n  %s\n  %s\n',F.summary,F.details,F.mapping);
end

function F = outputFiles(outDir,tag)
prefix = "labelled_pms1000_operating" + tag;
F.cache = fullfile(outDir,prefix + "_cache.mat");
F.mapping = fullfile(outDir,prefix + "_frame_label_mapping.csv");
F.summary = fullfile(outDir,prefix + "_summary.csv");
F.details = fullfile(outDir,prefix + "_details.csv");
F.bySource = fullfile(outDir,prefix + "_by_source.csv");
F.status = fullfile(outDir,prefix + "_metric_status.csv");
F.complete = fullfile(outDir,prefix + "_metrics.mat");
end

function C = analysisConfig()
C.rangeFft = 134;              % Native public AWR1843 processing grid.
C.dopplerFft = 256;
C.angleFft = 128;
C.rangeCrop = 3;
C.dopplerGuard = 4;
C.dopplerTrain = 7;
C.rangeGuard = 4;
C.rangeTrain = 8;
C.passPfa = 1e-4;              % Design setting, not the measured rate.
C.c = 299792458;
C.fc = 77e9;
C.Fs = 4e6;
C.sweepSlope = 21.0017e12;
C.Tc = 120e-6;
C.rangeResolution = (C.Fs/C.rangeFft)*C.c/(2*C.sweepSlope);
C.rangeGrid_m = (0:C.rangeFft-1)'*C.rangeResolution;
C.angleGrid_deg = asind(linspace(-1,1,C.angleFft));
C.rangeGate_m = 2*C.rangeResolution;
C.angleGate_deg = 10;
C.bootstrapReplicates = 5000;
C.seed = 20260824;
end

function [M,configs] = configurationLibrary(M)
classes = unique(M.ConfigKey,'stable');
assert(numel(classes)==3,'Expected exactly three retained configurations.');
if ~ismember('Label',M.Properties.VariableNames) || any(M.Label<1) || any(M.Label>3)
    M.Label = zeros(height(M),1);
    for k=1:numel(classes), M.Label(M.ConfigKey==classes(k))=k; end
end
configs = table(strings(3,1),strings(3,1),strings(3,1),strings(3,1),zeros(3,1),classes, ...
    'VariableNames',{'Wavelet','Method','Rule','NoiseEstimate','Level','Key'});
for k=1:3
    z=split(classes(k),'|');
    assert(numel(z)==5,'Malformed configuration key: %s',classes(k));
    configs.Wavelet(k)=z(1); configs.Method(k)=z(2); configs.Rule(k)=z(3);
    configs.NoiseEstimate(k)=z(4); configs.Level(k)=str2double(z(5));
end
end

function [cnnChoice,lutChoice] = heldOutChoices(M,pred,foldInfo,estimatedSNR)
N = height(M); cnnChoice = zeros(N,1); lutChoice = zeros(N,1);
for f=1:numel(foldInfo)
    z=foldInfo(f); tr=z.TrainIdx; te=z.TestIdx;
    cnnChoice(te)=double(pred(te,z.BestSeed));
    lutChoice(te)=snrLookupPredict(estimatedSNR(tr),M.SNRdB(tr),M.Label(tr),estimatedSNR(te));
end
assert(all(cnnChoice>=1 & cnnChoice<=3) && all(lutChoice>=1 & lutChoice<=3), ...
    'Could not rebuild held-out policy choices.');
end

function labels = snrLookupPredict(estTrain,trueSnrTrain,labelTrain,estTest)
levels=unique(trueSnrTrain,'sorted'); centers=zeros(numel(levels),1); decision=zeros(numel(levels),1);
for i=1:numel(levels)
    q=trueSnrTrain==levels(i); centers(i)=median(estTrain(q)); decision(i)=mode(labelTrain(q));
end
labels=zeros(numel(estTest),1);
for i=1:numel(estTest), [~,j]=min(abs(centers-estTest(i))); labels(i)=decision(j); end
end

function [labels,mapping] = loadVerifiedLabels(outDir)
labelDir = fullfile(outDir,'public_pms1000_text_labels');
assert(isfolder(labelDir),'Recovered public labels are missing: %s',labelDir);
labels = nan(30,6);
rows = cell(30,8);
for sid=1:30
    publicFrame=sid+29;
    file=fullfile(labelDir,sprintf('%010d.csv',publicFrame));
    assert(isfile(file),'Missing verified label: %s',file);
    x=readmatrix(file);
    assert(size(x,2)==6 && size(x,1)==1,'Expected one pedestrian label in %s.',file);
    assert(x(1,2)==0,'Expected pedestrian class 0 in %s.',file);
    labels(sid,:)=x(1,:);
    rows(sid,:)={sid,publicFrame,x(1,1),x(1,2),x(1,3),x(1,4),x(1,5),x(1,6)};
end
mapping=cell2table(rows,'VariableNames',{'SourceId','PublicFrame','UID','ClassId','px_m','py_m','Width_m','Length_m'});
end

function cube = loadSourceCube()
sourceFile='F:\Master\1-MATLAB\PED_Wavelet_Denoising\template data\pms1000_30fs.mat';
assert(isfile(sourceFile),'Missing raw pms1000 source: %s',sourceFile);
S=load(sourceFile,'data_frames');
assert(isequal(size(S.data_frames),[4,30*128*255*2]),'Unexpected pms1000 data shape.');
cube=zeros(128,255,8,30,'double');
for sid=1:30
    start=(sid-1)*128*255*2+1;
    frame=reshape(S.data_frames(:,start:start+128*255*2-1),[4,128,510]);
    for tx=1:2
        for rx=1:4
            cube(:,:,rx+(tx-1)*4,sid)=real(squeeze(frame(rx,:,tx:2:end)));
        end
    end
end
end

function verifySourceAlignment(M,cube)
n=find(M.SourceId==1,1,'first');
clean=readLegacyCsv(M.CleanPath(n));
assert(max(abs(clean-cube(:,:,3,1)),[],'all')==0, ...
    'Rx3/Tx1 source alignment failed; do not use public labels.');
end

function idx = chooseRecords(M,testIdx,maxRecords)
if isinf(maxRecords) || maxRecords>=numel(testIdx), idx=testIdx; return; end
G=findgroups(M.SourceId(testIdx),M.SNRdB(testIdx));
groups=unique(G,'stable'); pick=unique(round(linspace(1,numel(groups),min(maxRecords,numel(groups)))),'stable');
idx=zeros(numel(pick),1);
for i=1:numel(pick), idx(i)=testIdx(find(G==groups(pick(i)),1,'first')); end
end

function Q = loadOrCreateCache(cacheFile,N,force)
if ~force && isfile(cacheFile)
    S=load(cacheFile,'Q');
    if isfield(S,'Q') && isfield(S.Q,'Done') && isequal(size(S.Q.Done),[N,2])
        Q=S.Q; fprintf('Using resumable labelled-analysis cache: %s\n',cacheFile); return
    end
end
Q=struct(); Q.Done=false(N,2);
names={'Hit','BackgroundCFARFARate','FalseDetectionsPerFrame','RangeError_m','AngleError_deg','DetectionCount','WaveletMs'};
for i=1:numel(names), Q.(names{i})=nan(N,2); end
end

function cube = constructNoisyCube(cleanCube,M,n)
xClean=cleanCube(:,:,3);
xNoisy=readLegacyCsv(M.RawPath(n));
assert(isequal(size(xNoisy),[128 255]),'Unexpected cached noisy CSV shape.');
noiseRatio=sqrt(mean((xNoisy-xClean).^2,'all')/max(mean(xClean.^2,'all'),eps));
cube=zeros(128,255,8);
seed=20260824 + 100000*M.SourceId(n) + 1000*(M.SNRdB(n)+20) + M.Replicate(n);
rng(seed,'twister');
v=0;
for tx=1:2
    for rx=1:4
        v=v+1;
        channel=cleanCube(:,:,rx+(tx-1)*4);
        if tx==1 && rx==3
            cube(:,:,v)=xNoisy;
        else
            sigma=noiseRatio*sqrt(mean(channel.^2,'all'));
            cube(:,:,v)=channel+sigma*randn(size(channel));
        end
    end
end
end

function y = denoiseCube(x,cfg)
y=zeros(size(x));
for a=1:size(x,3)
    y(:,:,a)=wdenoise2(x(:,:,a),cfg.Level,'DenoisingMethod',char(cfg.Method), ...
        'ThresholdRule',char(cfg.Rule),'NoiseEstimate',char(cfg.NoiseEstimate), ...
        'Wavelet',char(cfg.Wavelet));
end
end

function [det,cfarMask] = detectRangeAngle(cube,C)
rangeWindow=hamming(size(cube,1));
RD=zeros(C.rangeFft,C.dopplerFft,8);
for a=1:8
    rangeData=fft(cube(:,:,a).*rangeWindow,C.rangeFft,1);
    RD(:,:,a)=fftshift(fft(rangeData,C.dopplerFft,2),2);
end
power=mean(abs(RD(:,:,1:4)),3).^2;
[detCells,cfarMask]=twoPassCfarPower(power,C);
det=zeros(size(detCells,1),4);
for i=1:size(detCells,1)
    r=detCells(i,1); d=detCells(i,2);
    det(i,:)=[C.rangeGrid_m(r), estimateAngle(RD,r,d,C), power(r,d), d];
end
end

function [det,mask] = twoPassCfarPower(power,C)
[nr,nd]=size(power); pass1=false(nr,nd);
for r=C.rangeCrop+1:nr-C.rangeCrop
    pass1(r,:)=caCfar1d(power(r,:),C.dopplerGuard,C.dopplerTrain,C.passPfa);
end
mask=false(nr,nd);
for d=find(any(pass1,1))
    z=caCfar1d(power(:,d),C.rangeGuard,C.rangeTrain,C.passPfa);
    z(1:C.rangeCrop)=false; z(end-C.rangeCrop+1:end)=false; mask(:,d)=z;
end
[rr,dd]=find(mask); keep=false(numel(rr),1);
for i=1:numel(rr)
    r1=max(1,rr(i)-1); r2=min(nr,rr(i)+1); d1=max(1,dd(i)-1); d2=min(nd,dd(i)+1);
    keep(i)=power(rr(i),dd(i))>=max(power(r1:r2,d1:d2),[],'all');
end
det=[rr(keep),dd(keep)];
end

function out = caCfar1d(x,guard,train,pfa)
x=double(x(:)).'; n=numel(x); out=false(1,n); nTrain=2*train;
alpha=nTrain*(pfa^(-1/nTrain)-1); first=train+guard+1; last=n-train-guard;
for i=first:last
    noise=mean([x(i-guard-train:i-guard-1),x(i+guard+1:i+guard+train)]);
    out(i)=x(i)>alpha*max(noise,eps);
end
end

function angle = estimateAngle(RD,r,d,C)
v=squeeze(RD(r,d,:));
phase=exp(-1i*pi*(d-C.dopplerFft/2-1)/C.dopplerFft);
v(5:8)=v(5:8)*phase;
spectrum=abs(fftshift(fft(v.*hamming(8),C.angleFft)));
[~,bin]=max(spectrum); angle=C.angleGrid_deg(bin);
end

function score = scoreLabelledDetections(det,cfarMask,truth,C)
gtRange=hypot(truth(3),truth(4)); gtAngle=atan2d(truth(3),truth(4));
background=true(size(cfarMask));
targetRange=abs(C.rangeGrid_m-gtRange)<=C.rangeGate_m;
background(targetRange,:)=false;
score.BackgroundCFARFARate=sum(cfarMask & background,'all')/max(sum(background,'all'),1);
score.DetectionCount=size(det,1);
if isempty(det)
    score.Hit=0; score.FalseDetectionsPerFrame=0; score.RangeError_m=nan; score.AngleError_deg=nan; return
end
rangeOK=abs(det(:,1)-gtRange)<=C.rangeGate_m;
angleOK=abs(wrapTo180Local(det(:,2)-gtAngle))<=C.angleGate_deg;
hit=find(rangeOK & angleOK);
if isempty(hit)
    score.Hit=0; score.FalseDetectionsPerFrame=size(det,1); score.RangeError_m=nan; score.AngleError_deg=nan; return
end
[~,order]=max(det(hit,3)); chosen=hit(order);
score.Hit=1; score.FalseDetectionsPerFrame=size(det,1)-1;
score.RangeError_m=det(chosen,1)-gtRange;
score.AngleError_deg=wrapTo180Local(det(chosen,2)-gtAngle);
end

function a = wrapTo180Local(a)
a=mod(a+180,360)-180;
end

function x = readLegacyCsv(fileName)
x=readmatrix(char(fileName));
assert(isnumeric(x) && isequal(size(x),[128 255]),'Expected 128x255 CSV: %s',fileName);
x=real(double(x));
end

function [summary,details,bySource,status] = summarizeMetrics(M,recordIdx,cnnChoice,lutChoice,Q,C)
policyNames=["DLSWD (Proposed)","DLSWD-LUT"];
rows={}; sourceRows={}; summaryRows={};
for p=1:2
    for ii=1:numel(recordIdx)
        n=recordIdx(ii); choices=[cnnChoice(n),lutChoice(n)]; k=choices(p);
        rows(end+1,:)={M.SourceId(n),M.SNRdB(n),M.Replicate(n),policyNames(p), ...
            Q.Hit(n,p),Q.BackgroundCFARFARate(n,p),Q.FalseDetectionsPerFrame(n,p), ...
            Q.RangeError_m(n,p),Q.AngleError_deg(n,p),Q.DetectionCount(n,p),Q.WaveletMs(n,p),k}; %#ok<AGROW>
    end
end
details=cell2table(rows,'VariableNames',{'SourceId','InputSNRdB','Replicate','Policy', ...
    'Detected','EmpiricalBackgroundCFARFARate','FalseDetectionsPerFrame','RangeError_m', ...
    'AngleError_deg','DetectionCount','MIMOWaveletComponentMs','SelectedConfiguration'});
for p=1:2
    T=details(details.Policy==policyNames(p),:); src=unique(T.SourceId,'sorted');
    Z=nan(numel(src),6);
    for i=1:numel(src)
        U=T(T.SourceId==src(i),:);
        Z(i,1)=mean(U.Detected);
        Z(i,2)=mean(U.EmpiricalBackgroundCFARFARate);
        Z(i,3)=mean(U.FalseDetectionsPerFrame);
        Z(i,4)=sqrt(mean(U.RangeError_m.^2,'omitnan'));
        Z(i,5)=sqrt(mean(U.AngleError_deg.^2,'omitnan'));
        Z(i,6)=mean(U.DetectionCount);
    end
    S=table(src,repmat(policyNames(p),numel(src),1),Z(:,1),Z(:,2),Z(:,3),Z(:,4),Z(:,5),Z(:,6), ...
        'VariableNames',{'SourceId','Policy','Pd','EmpiricalBackgroundCFARFARate', ...
        'FalseDetectionsPerFrame','RangeRMSE_m','AngleRMSE_deg','MeanDetectionCount'});
    sourceRows=[sourceRows;table2cell(S)]; %#ok<AGROW>
    [pdlo,pdhi]=bootstrapCI(S.Pd,C.bootstrapReplicates,C.seed+p);
    [falo,fahi]=bootstrapCI(S.EmpiricalBackgroundCFARFARate,C.bootstrapReplicates,C.seed+10+p);
    [fdlo,fdhi]=bootstrapCI(S.FalseDetectionsPerFrame,C.bootstrapReplicates,C.seed+20+p);
    [rlo,rhi]=bootstrapCI(S.RangeRMSE_m,C.bootstrapReplicates,C.seed+30+p);
    [alo,ahi]=bootstrapCI(S.AngleRMSE_deg,C.bootstrapReplicates,C.seed+40+p);
    summaryRows(end+1,:)={policyNames(p),height(S),height(T),mean(S.Pd),pdlo,pdhi, ...
        mean(S.EmpiricalBackgroundCFARFARate),falo,fahi,mean(S.FalseDetectionsPerFrame),fdlo,fdhi, ...
        mean(S.RangeRMSE_m,'omitnan'),rlo,rhi,mean(S.AngleRMSE_deg,'omitnan'),alo,ahi, ...
        median(T.MIMOWaveletComponentMs),prctile(T.MIMOWaveletComponentMs,95)}; %#ok<AGROW>
end
bySource=cell2table(sourceRows,'VariableNames',{'SourceId','Policy','Pd','EmpiricalBackgroundCFARFARate', ...
    'FalseDetectionsPerFrame','RangeRMSE_m','AngleRMSE_deg','MeanDetectionCount'});
summary=cell2table(summaryRows,'VariableNames',{'Policy','NSource','NRecord','PdMean','PdCIlo','PdCIhi', ...
    'EmpiricalBackgroundCFARFARateMean','EmpiricalBackgroundCFARFARateCIlo','EmpiricalBackgroundCFARFARateCIhi', ...
    'FalseDetectionsPerFrameMean','FalseDetectionsPerFrameCIlo','FalseDetectionsPerFrameCIhi', ...
    'RangeRMSE_mMean','RangeRMSE_mCIlo','RangeRMSE_mCIhi','AngleRMSE_degMean','AngleRMSE_degCIlo', ...
    'AngleRMSE_degCIhi','MIMOWaveletMedianMs','MIMOWaveletP95Ms'});
status=table(["Object-level Pd";"Range RMSE";"Angle RMSE";"Empirical background CFAR firing rate";"Target-free calibrated Pfa"], ...
    ["Computed";"Computed";"Computed";"Computed";"Not computed"], ...
    ["One annotated pedestrian per verified source frame; 2-range-bin and 10-degree gate."; ...
     "Computed only on detected target matches."; ...
     "Computed only on detected target matches using the 8-element virtual array."; ...
     "CFAR-positive RD cells outside the annotated target-range gate divided by valid background cells."; ...
     "All available verified frames contain a pedestrian; no target-free sequence is available for a calibrated Pfa."], ...
    'VariableNames',{'Metric','Status','Definition'});
end

function [lo,hi] = bootstrapCI(x,B,seed)
x=double(x(:)); x=x(isfinite(x)); n=numel(x);
if n<2, lo=nan; hi=nan; return; end
rng(seed,'twister'); z=zeros(B,1);
for b=1:B, z(b)=mean(x(randi(n,n,1))); end
z=sort(z); lo=z(max(1,round(0.025*B))); hi=z(min(B,round(0.975*B)));
end

function printProgress(i,n,label)
fprintf('  %s: %d / %d (%.0f%%)\n',label,i,n,100*i/n);
end
