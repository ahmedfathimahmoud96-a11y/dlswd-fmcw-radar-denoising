function run_conventional_filter_comparison()
%RUN_CONVENTIONAL_FILTER_COMPARISON Same-protocol fixed-filter controls.
%
% Evaluates two fixed, non-tuned conventional denoisers on the 17,940
% supplied noisy CSV records referenced by the cached source-frame manifest:
%   1) 3x3 box (moving-average) filter
%   2) 5x5 Gaussian low-pass filter (sigma = 1 sample)
%
% The script never changes the supplied dataset and does not rerun the SCS or
% CNN.  It writes only cache and CSV files under matlab/out/.

clc
repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(repoRoot,'matlab','out','cached_grouped_revision');
manifestFile = fullfile(outDir,'manifest.mat');
cacheFile = fullfile(outDir,'conventional_filter_metrics.mat');
summaryFile = fullfile(outDir,'conventional_filter_summary.csv');
sourceFile = fullfile(outDir,'conventional_filter_source_means.csv');

assert(isfile(manifestFile),'Missing cached manifest: %s',manifestFile);
S = load(manifestFile,'M');
M = S.M;
N = height(M);
assert(N==17940,'Expected 17,940 cached records; found %d.',N);
assert(all(isfile(M.RawPath)) && all(isfile(M.CleanPath)), ...
    'One or more CSV paths in the cached manifest are unavailable.');

if isfile(cacheFile)
    C = load(cacheFile,'boxCorr','boxSNR','boxMSE','gaussCorr','gaussSNR','gaussMSE');
    required = {'boxCorr','boxSNR','boxMSE','gaussCorr','gaussSNR','gaussMSE'};
    if all(isfield(C,required)) && numel(C.boxCorr)==N && numel(C.gaussCorr)==N
        fprintf('Using conventional-filter metric cache: %s\n',cacheFile);
    else
        C = computeControls(M,cacheFile);
    end
else
    C = computeControls(M,cacheFile);
end

summary = [summarizeControl("3x3 box (moving-average)",C.boxCorr,C.boxSNR,C.boxMSE,M,2026082401); ...
           summarizeControl("5x5 Gaussian (sigma=1)",C.gaussCorr,C.gaussSNR,C.gaussMSE,M,2026082402)];
writetable(summary,summaryFile);

sourceMeans = [sourceMeansForControl("3x3 box (moving-average)",C.boxCorr,C.boxSNR,C.boxMSE,M); ...
               sourceMeansForControl("5x5 Gaussian (sigma=1)",C.gaussCorr,C.gaussSNR,C.gaussMSE,M)];
writetable(sourceMeans,sourceFile);

disp(summary)
fprintf(['\nCompleted fixed conventional-filter controls without SCS or CNN reruns.\n' ...
         'Summary: %s\nSource means: %s\n'],summaryFile,sourceFile);
end

function C = computeControls(M,cacheFile)
N = height(M);
boxCorr = nan(N,1,'single'); boxSNR = boxCorr; boxMSE = boxCorr;
gaussCorr = boxCorr; gaussSNR = boxCorr; gaussMSE = boxCorr;

sourceIds = unique(M.SourceId,'sorted');
cleanBySource = cell(numel(sourceIds),1);
for i = 1:numel(sourceIds)
    idx = find(M.SourceId==sourceIds(i),1,'first');
    cleanBySource{i} = real(readComplexCsv(M.CleanPath(idx)));
end
sourceIndex = zeros(N,1);
for i = 1:numel(sourceIds), sourceIndex(M.SourceId==sourceIds(i)) = i; end

fprintf('Computing two fixed conventional filters for %d supplied records...\n',N);
useParallel = false;
try
    if isempty(gcp('nocreate')), parpool('local'); end
    useParallel = ~isempty(gcp('nocreate'));
catch ME
    warning('ConventionalFilters:ParallelUnavailable', ...
        'Parallel pool unavailable (%s). Running serially.',ME.message);
end

if useParallel
    q = parallel.pool.DataQueue;
    done = 0; lastPercent = -1;
    afterEach(q,@progressTick);
    parfor n = 1:N
        x = real(readComplexCsv(M.RawPath(n)));
        r = cleanBySource{sourceIndex(n)};
        [boxCorr(n),boxSNR(n),boxMSE(n)] = filterMetrics(boxFilter3(x),r);
        [gaussCorr(n),gaussSNR(n),gaussMSE(n)] = filterMetrics(gaussianFilter5(x),r);
        send(q,1);
    end
else
    for n = 1:N
        x = real(readComplexCsv(M.RawPath(n)));
        r = cleanBySource{sourceIndex(n)};
        [boxCorr(n),boxSNR(n),boxMSE(n)] = filterMetrics(boxFilter3(x),r);
        [gaussCorr(n),gaussSNR(n),gaussMSE(n)] = filterMetrics(gaussianFilter5(x),r);
        if mod(n,180)==0 || n==N
            fprintf('  %d / %d (%.0f%%)\n',n,N,100*n/N);
        end
    end
end

save(cacheFile,'boxCorr','boxSNR','boxMSE','gaussCorr','gaussSNR','gaussMSE','-v7.3');
C = struct('boxCorr',boxCorr,'boxSNR',boxSNR,'boxMSE',boxMSE, ...
    'gaussCorr',gaussCorr,'gaussSNR',gaussSNR,'gaussMSE',gaussMSE);

    function progressTick(~)
        done = done + 1;
        percent = floor(100*done/N);
        if percent >= lastPercent+5 || done==N
            fprintf('  conventional controls: %d / %d (%d%%)\n',done,N,percent);
            lastPercent = percent;
        end
    end
end

function y = boxFilter3(x)
y = replicateConvolution2D(x,ones(3)/9);
end

function y = gaussianFilter5(x)
v = -2:2;
g = exp(-(v.^2)/2); g = g/sum(g);
y = replicateConvolution2D(x,g'*g);
end

function y = replicateConvolution2D(x,h)
pad = floor(size(h)/2);
xp = [repmat(x(1,:),pad(1),1); x; repmat(x(end,:),pad(1),1)];
xp = [repmat(xp(:,1),1,pad(2)), xp, repmat(xp(:,end),1,pad(2))];
y = conv2(xp,h,'valid');
end

function [rho,snr,mse] = filterMetrics(y,r)
a = r(:)-mean(r(:)); b = y(:)-mean(y(:));
rho = (a'*b)/max(sqrt((a'*a)*(b'*b)),eps);
e = y-r;
mse = mean(e(:).^2);
snr = 10*log10(sum(r(:).^2)/max(sum(e(:).^2),eps));
end

function T = sourceMeansForControl(name,corr,snr,mse,M)
src = unique(M.SourceId,'sorted');
rows = cell(numel(src),5);
for i = 1:numel(src)
    idx = M.SourceId==src(i);
    rows(i,:) = {name,src(i),mean(corr(idx)),mean(snr(idx)),mean(mse(idx))};
end
T = cell2table(rows,'VariableNames',{'Filter','SourceId','Correlation','OutputSNRdB','MSE'});
end

function T = summarizeControl(name,corr,snr,mse,M,seed)
S = sourceMeansForControl(name,corr,snr,mse,M);
[cLo,cHi] = bootstrapCI(S.Correlation,5000,seed+1);
[sLo,sHi] = bootstrapCI(S.OutputSNRdB,5000,seed+2);
[mLo,mHi] = bootstrapCI(S.MSE,5000,seed+3);
T = table(name,height(S),mean(S.Correlation),cLo,cHi,mean(S.OutputSNRdB),sLo,sHi, ...
    mean(S.MSE),mLo,mHi,'VariableNames',{'Filter','NSource','CorrelationMean', ...
    'CorrelationCIlo','CorrelationCIhi','OutputSNRdBMean','OutputSNRdBCIlo', ...
    'OutputSNRdBCIhi','MSEMean','MSECIlo','MSECIhi'});
end

function [lo,hi] = bootstrapCI(x,B,seed)
x = double(x(:)); n = numel(x);
rng(seed,'twister');
idx = randi(n,n,B);
means = mean(x(idx),1);
q = quantile(means,[0.025 0.975]);
lo = q(1); hi = q(2);
end

function x = readComplexCsv(fileName)
x = readmatrix(char(fileName));
assert(isnumeric(x) && isequal(size(x),[128 255]), ...
    'Expected a 128x255 numerical CSV: %s',fileName);
x = double(x);
end
