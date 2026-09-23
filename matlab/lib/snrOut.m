function s = snrOut(xhat,xref)
%SNROUT  Output SNR in dB of a denoised frame against its clean reference.
%   Defined as 10*log10( ||xref||^2 / ||xhat-xref||^2 ), i.e. signal power
%   over residual-error power. This is the definition used throughout the
%   paper and in eq. (2); it is stated here so the metric is unambiguous.
e = xhat(:) - xref(:);
s = 10*log10( sum(abs(xref(:)).^2) / max(sum(abs(e).^2),eps) );
end
