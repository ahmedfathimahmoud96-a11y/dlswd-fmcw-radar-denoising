function [y,sigma] = addAWGNslow(x,snrdB,rs)
%ADDAWGNSLOW  Inject circularly-symmetric AWGN along the slow-time axis at a
%             prescribed frame-level SNR.
%   SNR is defined on total frame power: SNR = 10*log10(Ps/Pn) where Ps is
%   the mean power of the clean frame and Pn the variance of the added noise.
Ps    = mean(abs(x(:)).^2);
Pn    = Ps / 10^(snrdB/10);
sigma = sqrt(Pn/2);
n     = sigma*(randn(rs,size(x)) + 1i*randn(rs,size(x)));
y     = x + n;
end
