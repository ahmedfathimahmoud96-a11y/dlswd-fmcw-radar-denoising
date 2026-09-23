function evaluate_cached_radar_operating_metrics(varargin)
%EVALUATE_CACHED_RADAR_OPERATING_METRICS Operating comparison from caches.
%
%   evaluate_cached_radar_operating_metrics
%
% Reuses the source-frame-disjoint CNN decisions and reference-free DLSWD-LUT
% decisions already stored by run_cached_grouped_revision.  It does NOT rerun
% the SCS, CBC, feature generation, or CNN training, and it never writes to
% the supplied F:\Master\... dataset.  It evaluates the selected denoising
% configurations on the same noisy 128x255 ADC matrices, using:
%   * clean-reference RD peak displacement and RD-NMSE;
%   * clean-reference two-pass CA-CFAR agreement (recall, precision, F1 and
%     unmatched detections per 1,000 valid RD cells); and
%   * host wavelet and RD/CFAR component timing.
%
% IMPORTANT TERMINOLOGY
% The clean ADC matrix supplies a processing reference, not an externally
% annotated object list.  Therefore, the CFAR measures are explicitly named
% "clean-reference CFAR agreement".  They are NOT target Pd or Pfa and are
% never exported under those names.  Object-level Pd/Pfa, range RMSE and
% angle RMSE require raw MIMO frames and matching text_labels CSV files.
%
% Optional name/value controls (useful for a quick smoke test):
%   'MaxRecords'     Inf by default; use 12 for a tiny test only.
%   'OutputTag'      '' by default; e.g. 'smoke' keeps test output separate.
%   'ForceRecompute' false by default.
%
% Full evaluation output (all under matlab/out/cached_grouped_revision):
%   radar_operating_summary.csv
%   radar_operating_by_source.csv
%   radar_operating_by_source_snr.csv
%   radar_operating_policy_agreement.csv
%   radar_operating_ground_truth_status.csv
%   radar_operating_reference_cache.mat and radar_operating_config_*.mat

ip = inputParser;
addParameter(ip,'MaxRecords',inf,@(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(ip,'OutputTag','',@(x) ischar(x) || isstring(x));
addParameter(ip,'ForceRecompute',false,@(x) islogical(x) && isscalar(x));
parse(ip,varargin{:});
O = ip.Results;

clc
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(fullfile(root,'matlab')));
outDir = fullfile(root,'matlab','out','cached_grouped_revision');
assert(isfolder(outDir),'Cached revision output is missing: %s',outDir);

tag = string(O.OutputTag);
if strlength(tag)>0
    assert(~contains(tag,[filesep '/' '\\']), 'OutputTag must be a simple name.');
    tag = "_" + tag;
end
files = outputFiles(outDir,tag);

manifestFile = fullfile(outDir,'manifest.mat');
predictionFile = fullfile(outDir,'cnn_predictions.mat');
featureFile = fullfile(outDir,'features.mat');
assert(isfile(manifestFile),'Missing manifest cache: %s',manifestFile);
assert(isfile(predictionFile),'Missing CNN prediction cache: %s',predictionFile);
assert(isfile(featureFile),'Missing feature cache: %s',featureFile);

S = load(manifestFile,'M','P');
M = S.M;
assert(height(M)==17940,'Expected 17,940 cached records; found %d.',height(M));
assert(all(isfile(M.RawPath)) && all(isfile(M.CleanPath)), ...
    'One or more raw/clean CSV paths in the cached manifest are unavailable.');

C = operatingConfig(S.P,outDir,files);
[M,classes,configs] = configurationLibrary(M);
Pred = load(predictionFile,'pred','foldInfo');
Feat = load(featureFile,'estimatedSNR');
assert(size(Pred.pred,1)==height(M),'CNN predictions do not match the manifest.');
assert(numel(Feat.estimatedSNR)==height(M),'Estimated-SNR cache does not match manifest.');

[cnnChoice,lutChoice] = heldOutChoices(M,Pred.pred,Pred.foldInfo,Feat.estimatedSNR);
recordIdx = chooseRecords(M,O.MaxRecords);
fprintf('Cached radar operating evaluation: %d / %d records.\n',numel(recordIdx),height(M));
fprintf('Policies: DLSWD (held-out CNN choice) and DLSWD-LUT (training-source lookup).\n');
fprintf('No SCS, CBC, feature, or CNN-training stage will run.\n');

reference = buildReferenceCache(M,recordIdx,C,files.reference,O.ForceRecompute);
metrics = evaluateSelectedConfigurations(M,recordIdx,cnnChoice,lutChoice,configs, ...
    reference,C,files,tag,O.ForceRecompute);

[summary,bySource,bySourceSnr,agreement] = summarizeOperatingMetrics( ...
    M,recordIdx,cnnChoice,lutChoice,metrics,reference,C);
writetable(summary,files.summary);
writetable(bySource,files.bySource);
writetable(bySourceSnr,files.bySourceSnr);
writetable(agreement,files.agreement);
writeGroundTruthStatus(M,recordIdx,files.groundTruth);

save(files.complete,'M','recordIdx','cnnChoice','lutChoice','configs','reference', ...
    'metrics','summary','bySource','bySourceSnr','agreement','C','-v7.3');

disp(summary)
disp(agreement)
fprintf('\nCompleted. Read these measured outputs:\n  %s\n  %s\n  %s\n', ...
    files.summary,files.bySource,files.agreement);
fprintf(['\nInterpretation guard rail: clean-reference CFAR agreement is an RD-map ' ...
    'fidelity metric. It is not object-level Pd/Pfa because matching target ' ...
    'annotations were not found in the supplied local folders.\n']);
end

function F = outputFiles(outDir,tag)
prefix = "radar_operating" + tag;
F.reference = fullfile(outDir,prefix + "_reference_cache.mat");
F.config = @(k) fullfile(outDir,sprintf('%s_config_%d.mat',prefix,k));
F.summary = fullfile(outDir,prefix + "_summary.csv");
F.bySource = fullfile(outDir,prefix + "_by_source.csv");
F.bySourceSnr = fullfile(outDir,prefix + "_by_source_snr.csv");
F.agreement = fullfile(outDir,prefix + "_policy_agreement.csv");
F.groundTruth = fullfile(outDir,prefix + "_ground_truth_status.csv");
F.complete = fullfile(outDir,prefix + "_metrics.mat");
end

function C = operatingConfig(P,outDir,files)
% The FFT and radar settings match the public AWR1843 data metadata and the
% dimensions used by the existing cached peak-preservation calculation.
C.rangeFft = 128;
C.dopplerFft = 256;
C.rangeCrop = 3;
C.dopplerGuard = 4;
C.dopplerTrain = 7;
C.rangeGuard = 4;
C.rangeTrain = 8;
C.passPfa = 1e-4;  % per-pass CA-CFAR setting; do not call this empirical Pfa.
C.c = 299792458;
C.fc = 77e9;
C.Fs = 4e6;
C.sweepSlope = 21.0017e12;
C.Tc = 120e-6;
C.rangeResolution = (C.Fs/C.rangeFft)*C.c/(2*C.sweepSlope);
C.velocityResolution = (1/(C.dopplerFft*C.Tc))*(C.c/C.fc)/2;
C.bootstrapReplicates = 5000;
C.seed = 20260824;
C.outDir = outDir;
C.referenceFile = files.reference;
if isfield(P,'dataRoot'), C.dataRoot = string(P.dataRoot); else, C.dataRoot = ""; end
end

function [M,classes,configs] = configurationLibrary(M)
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
N = height(M);
cnnChoice = zeros(N,1);
lutChoice = zeros(N,1);
for f=1:numel(foldInfo)
    z=foldInfo(f);
    tr=z.TrainIdx; te=z.TestIdx;
    cnnChoice(te)=double(pred(te,z.BestSeed));
    lutChoice(te)=snrLookupPredict(estimatedSNR(tr),M.SNRdB(tr),M.Label(tr),estimatedSNR(te));
end
assert(all(cnnChoice>=1 & cnnChoice<=3) && all(lutChoice>=1 & lutChoice<=3), ...
    'Failed to reconstruct held-out policy choices.');
end

function labels = snrLookupPredict(estTrain,trueSnrTrain,labelTrain,estTest)
levels=unique(trueSnrTrain,'sorted');
centers=zeros(numel(levels),1); decision=zeros(numel(levels),1);
for i=1:numel(levels)
    q=trueSnrTrain==levels(i);
    centers(i)=median(estTrain(q));
    decision(i)=mode(labelTrain(q));
end
labels=zeros(numel(estTest),1);
for i=1:numel(estTest), [~,j]=min(abs(centers-estTest(i))); labels(i)=decision(j); end
end

function idx = chooseRecords(M,maxRecords)
N=height(M);
if isinf(maxRecords) || maxRecords>=N, idx=(1:N)'; return; end
% A small test remains source/SNR stratified; full analysis always uses all data.
G=findgroups(M.SourceId,M.SNRdB);
g=unique(G,'stable');
if maxRecords<numel(g)
    pick=unique(round(linspace(1,numel(g),maxRecords)),'stable');
    idx=zeros(numel(pick),1);
    for i=1:numel(pick), idx(i)=find(G==g(pick(i)),1,'first'); end
    return
end
take=max(1,floor(maxRecords/numel(g)));
idx=[];
for i=1:numel(g)
    q=find(G==g(i)); idx=[idx; q(1:min(take,numel(q)))]; %#ok<AGROW>
    if numel(idx)>=maxRecords, break; end
end
idx=idx(1:min(maxRecords,numel(idx)));
end

function R = buildReferenceCache(M,recordIdx,C,cacheFile,force)
N=height(M);
if ~force && isfile(cacheFile)
    S=load(cacheFile,'R');
    if isfield(S,'R') && isfield(S.R,'Done') && numel(S.R.Done)==N && all(S.R.Done(recordIdx))
        R=S.R; fprintf('Using clean-reference RD/CFAR cache: %s\n',cacheFile); return
    end
end
R = struct();
R.Done=false(N,1);
R.DetCells=cell(N,1);
R.RefPeakRangeBin=nan(N,1); R.RefPeakVelocityBin=nan(N,1);
fprintf('Building clean-reference RD/CFAR cache (%d records)...\n',numel(recordIdx));
for j=1:numel(recordIdx)
    n=recordIdx(j);
    x=readCsv(M.CleanPath(n));
    [rd,pkR,pkV]=rangeDoppler(x,C);
    R.DetCells{n}=uint16(twoPassCfar(rd,C));
    R.RefPeakRangeBin(n)=pkR; R.RefPeakVelocityBin(n)=pkV; R.Done(n)=true;
    printProgress(j,numel(recordIdx),'clean-reference RD/CFAR');
end
save(cacheFile,'R','-v7.3');
end

function Q = evaluateSelectedConfigurations(M,recordIdx,cnnChoice,lutChoice,configs,R,C,files,tag,force)
N=height(M); K=height(configs);
fields={'Done','RefRecall','RefPrecision','RefF1','UnmatchedPerKCells', ...
    'PeakRangeError_m','PeakVelocityError_mps','RDNMSE','WaveletMs','RDCfarMs', ...
    'DetectionCount'};
for f=1:numel(fields), Q.(fields{f})=nan(N,K); end
Q.Done=false(N,K);
for k=1:K
    needed=recordIdx(cnnChoice(recordIdx)==k | lutChoice(recordIdx)==k);
    cacheFile=files.config(k);
    if ~force && isfile(cacheFile)
        S=load(cacheFile,'Qk');
        if isfield(S,'Qk') && isfield(S.Qk,'Done') && numel(S.Qk.Done)==N && all(S.Qk.Done(needed))
            Q=copyColumn(Q,S.Qk,k); fprintf('Using RD/CFAR cache for %s.\n',configs.Key(k)); continue
        end
    end
    Qk=blankColumn(N);
    if isfile(cacheFile) && ~force
        S=load(cacheFile,'Qk');
        if isfield(S,'Qk'), Qk=S.Qk; end
    end
    todo=needed(~Qk.Done(needed));
    fprintf('Configuration %d/%d (%s): %d selected records to process.\n', ...
        k,K,configs.Key(k),numel(todo));
    for j=1:numel(todo)
        n=todo(j);
        x=readCsv(M.RawPath(n));
        tw=tic; y=applyWavelet(x,configs(k,:)); Qk.WaveletMs(n)=1000*toc(tw);
        tr=tic; [rd,pkR,pkV]=rangeDoppler(y,C); det=twoPassCfar(rd,C);
        Qk.RDCfarMs(n)=1000*toc(tr);
        [rec,pre,f1,unmatched]=cfarAgreement(det,R.DetCells{n},C);
        Qk.RefRecall(n)=rec; Qk.RefPrecision(n)=pre; Qk.RefF1(n)=f1;
        Qk.UnmatchedPerKCells(n)=1000*unmatched/validCells(C);
        Qk.PeakRangeError_m(n)=abs(pkR-R.RefPeakRangeBin(n))*C.rangeResolution;
        Qk.PeakVelocityError_mps(n)=abs(pkV-R.RefPeakVelocityBin(n))*C.velocityResolution;
        ref=readCsv(M.CleanPath(n));
        rdRef=fftshift(fft2(ref,C.rangeFft,C.dopplerFft));
        Qk.RDNMSE(n)=sum(abs(rd(:)-rdRef(:)).^2)/max(sum(abs(rdRef(:)).^2),eps);
        Qk.DetectionCount(n)=size(det,1); Qk.Done(n)=true;
        if mod(j,100)==0 || j==numel(todo)
            save(cacheFile,'Qk','-v7.3');
        end
        printProgress(j,numel(todo),sprintf('configuration %d RD/CFAR',k));
    end
    save(cacheFile,'Qk','-v7.3');
    Q=copyColumn(Q,Qk,k);
end
end

function Qk = blankColumn(N)
Qk=struct(); Qk.Done=false(N,1);
names={'RefRecall','RefPrecision','RefF1','UnmatchedPerKCells','PeakRangeError_m', ...
    'PeakVelocityError_mps','RDNMSE','WaveletMs','RDCfarMs','DetectionCount'};
for i=1:numel(names), Qk.(names{i})=nan(N,1); end
end

function Q = copyColumn(Q,Qk,k)
names=fieldnames(Qk);
for i=1:numel(names)
    if strcmp(names{i},'Done'), Q.Done(:,k)=Qk.Done; else, Q.(names{i})(:,k)=Qk.(names{i}); end
end
end

function x = readCsv(fileName)
x=readmatrix(char(fileName));
assert(isnumeric(x) && isequal(size(x),[128 255]),'Expected 128x255 CSV: %s',fileName);
% The archived legacy labels and cached reconstruction metrics were produced
% from the real component of this single virtual-antenna ADC channel.
x=real(double(x));
end

function y = applyWavelet(x,cfg)
y=wdenoise2(x,cfg.Level,'DenoisingMethod',char(cfg.Method), ...
    'ThresholdRule',char(cfg.Rule),'NoiseEstimate',char(cfg.NoiseEstimate), ...
    'Wavelet',char(cfg.Wavelet));
end

function [rd,peakRange,peakVelocity] = rangeDoppler(x,C)
rd=fftshift(fft2(x,C.rangeFft,C.dopplerFft));
[~,i]=max(abs(rd(:)).^2); [peakRange,peakVelocity]=ind2sub(size(rd),i);
end

function det = twoPassCfar(rd,C)
% Two-pass CA-CFAR follows the public AWR1843 processing sequence: first
% Doppler-wise candidates at each usable range cell, then range-wise CFAR at
% candidate Doppler cells.  The numeric setting is held fixed for reference,
% CNN and LUT outputs; it is not a calibrated object-level Pfa measurement.
p=abs(rd).^2; [nr,nd]=size(p); pass1=false(nr,nd);
for r=C.rangeCrop+1:nr-C.rangeCrop
    pass1(r,:)=caCfar1d(p(r,:),C.dopplerGuard,C.dopplerTrain,C.passPfa);
end
candidateDop=find(any(pass1,1));
out=false(nr,nd);
for d=candidateDop
    z=caCfar1d(p(:,d),C.rangeGuard,C.rangeTrain,C.passPfa);
    z(1:C.rangeCrop)=false; z(end-C.rangeCrop+1:end)=false;
    out(:,d)=z;
end
% A deterministic 3x3 non-maximum suppression prevents a broad lobe from
% inflating the agreement count.  It is applied identically to both maps.
[rr,dd]=find(out); keep=false(numel(rr),1);
for i=1:numel(rr)
    r1=max(1,rr(i)-1); r2=min(nr,rr(i)+1); d1=max(1,dd(i)-1); d2=min(nd,dd(i)+1);
    keep(i)=p(rr(i),dd(i))>=max(p(r1:r2,d1:d2),[],'all');
end
det=[rr(keep),dd(keep)];
end

function d = caCfar1d(x,guard,train,pfa)
x=double(x(:)).'; n=numel(x); d=false(1,n);
nTrain=2*train; alpha=nTrain*(pfa^(-1/nTrain)-1);
first=train+guard+1; last=n-train-guard;
for i=first:last
    noise=mean([x(i-guard-train:i-guard-1),x(i+guard+1:i+guard+train)]);
    d(i)=x(i)>alpha*max(noise,eps);
end
end

function [rec,pre,f1,unmatched] = cfarAgreement(det,ref,C)
% Greedy one-to-one matching within one range and one Doppler bin.  This is
% detector agreement with the clean RD map, never an annotated-target score.
if isempty(ref), rec=nan; else, rec=0; end
if isempty(det), pre=nan; else, pre=0; end
if isempty(ref) && isempty(det), f1=nan; unmatched=0; return; end
used=false(size(ref,1),1); match=0;
for i=1:size(det,1)
    q=find(~used & abs(double(ref(:,1))-det(i,1))<=1 & abs(double(ref(:,2))-det(i,2))<=1,1,'first');
    if ~isempty(q), used(q)=true; match=match+1; end
end
if ~isempty(ref), rec=match/size(ref,1); end
if ~isempty(det), pre=match/size(det,1); end
if isnan(rec) || isnan(pre) || rec+pre==0, f1=0; else, f1=2*rec*pre/(rec+pre); end
unmatched=size(det,1)-match;
end

function n=validCells(C)
n=(C.rangeFft-2*C.rangeCrop)*(C.dopplerFft-2*(C.dopplerGuard+C.dopplerTrain));
end

function [summary,bySource,bySourceSnr,agreement] = summarizeOperatingMetrics(M,recordIdx,cnnChoice,lutChoice,Q,R,C)
policyNames=["DLSWD (Proposed)","DLSWD-LUT"];
choices=[cnnChoice,lutChoice];
metricNames={'RefRecall','RefPrecision','RefF1','UnmatchedPerKCells','PeakRangeError_m', ...
    'PeakVelocityError_mps','RDNMSE','WaveletMs','RDCfarMs','DetectionCount'};
long={}; sourceRows={}; snrRows={}; summaryRows={};
for p=1:2
    for ii=1:numel(recordIdx)
        n=recordIdx(ii); k=choices(n,p);
        long(end+1,:)={M.SourceId(n),M.SNRdB(n),policyNames(p), ...
            Q.RefRecall(n,k),Q.RefPrecision(n,k),Q.RefF1(n,k),Q.UnmatchedPerKCells(n,k), ...
            Q.PeakRangeError_m(n,k),Q.PeakVelocityError_mps(n,k),Q.RDNMSE(n,k), ...
            Q.WaveletMs(n,k),Q.RDCfarMs(n,k),Q.DetectionCount(n,k)}; %#ok<AGROW>
    end
end
T=cell2table(long,'VariableNames',{'SourceId','InputSNRdB','Policy',metricNames{:}});
for p=1:2
    U=T(T.Policy==policyNames(p),:);
    src=unique(U.SourceId,'sorted');
    Z=nan(numel(src),numel(metricNames));
    for i=1:numel(src), Z(i,:)=mean(U{U.SourceId==src(i),metricNames},1,'omitnan'); end
    S=table(src,repmat(policyNames(p),numel(src),1),Z(:,1),Z(:,2),Z(:,3),Z(:,4),Z(:,5), ...
        Z(:,6),Z(:,7),Z(:,8),Z(:,9),Z(:,10), ...
        'VariableNames',{'SourceId','Policy',metricNames{:}});
    sourceRows=[sourceRows; table2cell(S)]; %#ok<AGROW>
    [G,sid,snr]=findgroups(U.SourceId,U.InputSNRdB);
    Zsnr=nan(max(G),numel(metricNames));
    for g=1:max(G), Zsnr(g,:)=mean(U{G==g,metricNames},1,'omitnan'); end
    V=table(sid,snr,repmat(policyNames(p),max(G),1),Zsnr(:,1),Zsnr(:,2),Zsnr(:,3),Zsnr(:,4), ...
        Zsnr(:,5),Zsnr(:,6),Zsnr(:,7),Zsnr(:,8),Zsnr(:,9),Zsnr(:,10), ...
        'VariableNames',{'SourceId','InputSNRdB','Policy',metricNames{:}});
    snrRows=[snrRows; table2cell(V)]; %#ok<AGROW>
    [f1lo,f1hi]=bootstrapCI(S.RefF1,C.bootstrapReplicates,C.seed+p);
    [rlo,rhi]=bootstrapCI(S.RefRecall,C.bootstrapReplicates,C.seed+10+p);
    [plo,phi]=bootstrapCI(S.RefPrecision,C.bootstrapReplicates,C.seed+20+p);
    [ulo,uhi]=bootstrapCI(S.UnmatchedPerKCells,C.bootstrapReplicates,C.seed+30+p);
    summaryRows(end+1,:)={policyNames(p),height(S),mean(S.RefRecall,'omitnan'),rlo,rhi, ...
        mean(S.RefPrecision,'omitnan'),plo,phi,mean(S.RefF1,'omitnan'),f1lo,f1hi, ...
        mean(S.UnmatchedPerKCells,'omitnan'),ulo,uhi,mean(S.PeakRangeError_m,'omitnan'), ...
        mean(S.PeakVelocityError_mps,'omitnan'),mean(S.RDNMSE,'omitnan'), ...
        median(U.WaveletMs,'omitnan'),prctile(U.WaveletMs,95),median(U.RDCfarMs,'omitnan'), ...
        prctile(U.RDCfarMs,95)}; %#ok<AGROW>
end
bySource=cell2table(sourceRows,'VariableNames',{'SourceId','Policy',metricNames{:}});
bySourceSnr=cell2table(snrRows,'VariableNames',{'SourceId','InputSNRdB','Policy',metricNames{:}});
summary=cell2table(summaryRows,'VariableNames',{'Policy','NSource','ReferenceCFARRecallMean', ...
    'ReferenceCFARRecallCIlo','ReferenceCFARRecallCIhi','ReferenceCFARPrecisionMean', ...
    'ReferenceCFARPrecisionCIlo','ReferenceCFARPrecisionCIhi','ReferenceCFARF1Mean', ...
    'ReferenceCFARF1CIlo','ReferenceCFARF1CIhi','UnmatchedDetectionsPerKCellsMean', ...
    'UnmatchedDetectionsPerKCellsCIlo','UnmatchedDetectionsPerKCellsCIhi', ...
    'ReferencePeakRangeError_mMean','ReferencePeakVelocityError_mpsMean','RDNMSEMean', ...
    'WaveletMedianMs','WaveletP95Ms','RDCfarMedianMs','RDCfarP95Ms'});
same=cnnChoice(recordIdx)==lutChoice(recordIdx);
src=unique(M.SourceId(recordIdx),'sorted'); sameBySource=nan(numel(src),1);
for i=1:numel(src), q=recordIdx(M.SourceId(recordIdx)==src(i)); sameBySource(i)=mean(cnnChoice(q)==lutChoice(q)); end
[alo,ahi]=bootstrapCI(sameBySource,C.bootstrapReplicates,C.seed+99);
agreement=table(numel(recordIdx),numel(src),mean(same),mean(sameBySource),alo,ahi, ...
    'VariableNames',{'NRecord','NSource','RecordAgreementFraction','SourceMeanAgreementFraction', ...
    'SourceMeanAgreementCIlo','SourceMeanAgreementCIhi'});
% R is passed deliberately to keep provenance obvious at the call site.
assert(all(R.Done(recordIdx)),'Reference cache is incomplete.');
end

function [lo,hi]=bootstrapCI(x,B,seed)
x=double(x(:)); x=x(isfinite(x)); n=numel(x);
if n<2, lo=nan; hi=nan; return; end
rng(seed,'twister'); z=zeros(B,1);
for b=1:B, z(b)=mean(x(randi(n,n,1))); end
z=sort(z); lo=z(max(1,round(0.025*B))); hi=z(min(B,round(0.975*B)));
end

function writeGroundTruthStatus(M,recordIdx,fileName)
dataRoot = fileparts(char(M.RawPath(recordIdx(1))));
root = fileparts(fileparts(fileparts(fileparts(fileparts(dataRoot)))));
labelCandidates=string({fullfile(root,'text_labels'); fullfile(root,'Dataset','text_labels'); ...
    fullfile(root,'Raw_ADC_radar_dataset_for_automotive_object_detection','text_labels')});
found="";
for i=1:numel(labelCandidates)
    if isfolder(labelCandidates(i)), found=labelCandidates(i); break; end
end
if strlength(found)==0
    status="Not computed";
    reason="No matching text_labels directory was found locally; clean-reference CFAR agreement is reported instead.";
else
    status="Requires frame mapping";
    reason="text_labels exists, but the cached 30 pedestrian CSV source IDs are not yet mapped to raw-frame filenames.";
end
T=table(["Object-level Pd";"Object-level Pfa";"Range/angle RMSE"],repmat(status,3,1), ...
    repmat(reason,3,1),'VariableNames',{'Metric','Status','Reason'});
writetable(T,fileName);
end

function printProgress(i,n,label)
if mod(i,100)==0 || i==n, fprintf('  %s: %d / %d (%.0f%%)\n',label,i,n,100*i/n); end
end
