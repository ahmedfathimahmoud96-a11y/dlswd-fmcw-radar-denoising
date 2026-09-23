function y = denoiseFrame(x,cfgRow)
%DENOISEFRAME  Apply one wavelet configuration to one complex radar frame.
%
%   The DWT is applied along the SLOW-TIME axis (dimension 1 of x), which is
%   the axis along which AWGN was injected. Real and imaginary parts are
%   denoised independently with the same configuration; this preserves the
%   phase relationship required by downstream range-Doppler processing and
%   is the standard treatment for complex IQ in wavelet shrinkage.
%
%   Boundary handling: symmetric half-point extension ('sym'), the MATLAB
%   default, avoids the discontinuity that periodic extension introduces at
%   the frame edges of a finite chirp sequence.
%
%   x : [nSlow x nFast] complex
%   y : [nSlow x nFast] complex, denoised

dwtmode('sym','nodisp');

w   = char(cfgRow.wavelet);
lev = cfgRow.level;
rule= char(cfgRow.threshRule);          % 's' | 'h'

switch char(cfgRow.denMethod)
    case 'Bayes',              thrRule = 'Bayes';
    case 'UniversalThreshold', thrRule = 'UniversalThreshold';
    case 'SURE',               thrRule = 'SURE';
    case 'Minimax',            thrRule = 'Minimax';
end
switch char(cfgRow.varEst)
    case 'LevelIndependent',   noiseEst = 'LevelIndependent';
    case 'LevelDependent',     noiseEst = 'LevelDependent';
end
if strcmp(rule,'s'), sorh = 'soft'; else, sorh = 'hard'; end

y = complex(zeros(size(x),'like',real(x)));
for c = 1:size(x,2)
    col = x(:,c);
    re = wdenoise(real(col),lev,'Wavelet',w,'DenoisingMethod',thrRule, ...
                  'ThresholdRule',sorh,'NoiseEstimate',noiseEst);
    im = wdenoise(imag(col),lev,'Wavelet',w,'DenoisingMethod',thrRule, ...
                  'ThresholdRule',sorh,'NoiseEstimate',noiseEst);
    y(:,c) = complex(re,im);
end
end
