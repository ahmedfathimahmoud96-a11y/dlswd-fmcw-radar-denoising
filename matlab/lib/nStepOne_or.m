function n = nStepOne_or(info)
%NSTEPONE_OR  Deprecated. Use info.nStep1 directly.
%   Retained only so older saved cbc.mat files still resolve.
if isfield(info,'nStep1'), n = info.nStep1; else, n = numel(info.step1); end
end
