function [pd,pfa,thr] = cfarDetect(rangeProfile,truthMask,C)
%CFARDETECT  CA-CFAR detection on a range profile, giving Pd at a fixed Pfa.
%   Provides the NON-CIRCULAR evaluation metric requested in review: it does
%   not reuse the SNR/correlation objective of eq. (2).
p = abs(rangeProfile(:)).^2;
n = numel(p);
G = C.cfarGuard; T = C.cfarTrain;
noise = zeros(n,1);
for k = 1:n
    lo1 = max(1,k-G-T); hi1 = max(1,k-G-1);
    lo2 = min(n,k+G+1); hi2 = min(n,k+G+T);
    cell = [p(lo1:hi1); p(lo2:hi2)];
    cell(cell==0) = [];
    if isempty(cell), noise(k) = eps; else, noise(k) = mean(cell); end
end
alpha = T*( C.pfaTarget^(-1/T) - 1 );
thr   = alpha*noise;
det   = p > thr;
pd    = sum(det &  truthMask(:)) / max(sum( truthMask(:)),1);
pfa   = sum(det & ~truthMask(:)) / max(sum(~truthMask(:)),1);
end
