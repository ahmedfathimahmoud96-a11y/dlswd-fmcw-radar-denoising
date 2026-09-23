function shat = estimateSNR(x)
%ESTIMATESNR  Reference-free input-SNR estimator used by the LOOKUP BASELINE.
%
%   This baseline is the one the reviewer requires: if a simple noise-floor
%   estimate plus a table lookup matches the CNN, the CNN is not justified.
%
%   Method: the noise floor is taken as the median of the smallest 25 % of
%   the range-profile magnitude-squared bins (robust to the target peak),
%   and signal power as total power minus noise power.
X   = fft(x,[],2);                 % fast-time FFT -> range profile
p   = mean(abs(X).^2,1);           % average over slow time
ps  = sort(p(:));
Pn  = median(ps(1:max(1,round(0.25*numel(ps)))));
Ptot= mean(p(:));
Ps  = max(Ptot - Pn, eps);
shat= 10*log10(Ps/max(Pn,eps));
end
