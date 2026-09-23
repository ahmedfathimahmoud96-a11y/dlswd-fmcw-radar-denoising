function run_golilarz_style_baseline()
%RUN_GOLILARZ_STYLE_BASELINE Same-dataset adaptive-threshold baseline.
%
% This implements a conservative Golilarz-style adaptive wavelet threshold
% control on the cached FMCW CSV matrices. It is not a reproduction of the
% satellite-image experiment; it applies the reviewer-requested idea
% (hybrid soft-hard thresholding with input-wise noise scale) under the same
% source-frame-disjoint FMCW evaluation used by the revised paper.

clc
repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(repoRoot,'matlab','out','cached_grouped_revision');
manifestFile = fullfile(outDir,'manifest.mat');
cacheFile = fullfile(outDir,'golilarz_style_metrics.mat');
summaryFile = fullfile(outDir,'golilarz_style_summary.csv');
sourceFile = fullfile(outDir,'golilarz_style_source_means.csv');

assert(isfile(manifestFile),'Missing cached manifest: %s',manifestFile);
S = load(manifestFile,'M');
M = S.M;
N = height(M);
assert(N==17940,'Expected 17,940 cached records; found %d.',N);
assert(all(isfile(M.RawPath)) && all(isfile(M.CleanPath)), ...
    'One or more CSV paths in the cached manifest are unavailable.');

if isfile(cacheFile)
    C = load(cacheFile,'corrOut','snrOut','mseOut');
    if isfield(C,'corrOut') && numel(C.corrOut)==N
        fprintf('Using Golilarz-style baseline cache: %s\n',cacheFile);
    else
        C = computeBaseline(M,cacheFile);
    end
else
    C = computeBaseline(M,cacheFile);
end

summary = summarizeControl("Golilarz-style hybrid wavelet",C.corrOut,C.snrOut,C.mseOut,M,2026091601);
writetable(summary,summaryFile);

sourceMeans = sourceMeansForControl("Golilarz-style hybrid wavelet",C.corrOut,C.snrOut,C.mseOut,M);
writetable(sourceMeans,sourceFile);

disp(summary)
fprintf('Summary: %s\nSource means: %s\n',summaryFile,sourceFile);
end

function C = computeBaseline(M,cacheFile)
N = height(M);
corrOut = nan(N,1,'single');
snrOut = corrOut;
mseOut = corrOut;

sourceIds = unique(M.SourceId,'sorted');
cleanBySource = cell(numel(sourceIds),1);
for i = 1:numel(sourceIds)
    idx = find(M.SourceId==sourceIds(i),1,'first');
    cleanBySource{i} = real(readComplexCsv(M.CleanPath(idx)));
end
sourceIndex = zeros(N,1);
for i = 1:numel(sourceIds)
    sourceIndex(M.SourceId==sourceIds(i)) = i;
end

fprintf('Computing Golilarz-style hybrid wavelet baseline for %d records...\n',N);
for n = 1:N
    x = real(readComplexCsv(M.RawPath(n)));
    r = cleanBySource{sourceIndex(n)};
    y = golilarzHybrid2D(x);
    [corrOut(n),snrOut(n),mseOut(n)] = filterMetrics(y,r);
    if mod(n,180)==0 || n==N
        fprintf('  %d / %d (%.0f%%)\n',n,N,100*n/N);
    end
end

save(cacheFile,'corrOut','snrOut','mseOut','-v7.3');
C = struct('corrOut',corrOut,'snrOut',snrOut,'mseOut',mseOut);
end

function y = golilarzHybrid2D(x)
% Apply one 2-D hybrid thresholding pass to the full slow-fast matrix. This
% keeps the same CSV-matrix protocol used by the conventional-filter controls.
w = 'db4';
lev = 4;
[cf,s] = wavedec2(x,lev,w);
approxCount = prod(s(1,:));
detail = cf(approxCount+1:end);
sigma = median(abs(detail))/0.6745;
lam = sigma*sqrt(2*log(numel(x)));
out = cf;
a = abs(detail);
sgn = sign(detail);
shrunk = detail;
shrunk(a<=lam) = 0;
mid = a>lam & a<=2*lam;
shrunk(mid) = sgn(mid).*(a(mid)-lam);
out(approxCount+1:end) = shrunk;
y = waverec2(out,s,w);
y = y(1:size(x,1),1:size(x,2));
end

function [rho,snr,mse] = filterMetrics(y,r)
a = r(:)-mean(r(:));
b = y(:)-mean(y(:));
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
x = double(x(:));
n = numel(x);
rng(seed,'twister');
idx = randi(n,n,B);
means = mean(x(idx),1);
q = quantile(means,[0.025 0.975]);
lo = q(1);
hi = q(2);
end

function x = readComplexCsv(fileName)
x = readmatrix(char(fileName));
assert(isnumeric(x) && isequal(size(x),[128 255]), ...
    'Expected a 128x255 numerical CSV: %s',fileName);
x = double(x);
end
