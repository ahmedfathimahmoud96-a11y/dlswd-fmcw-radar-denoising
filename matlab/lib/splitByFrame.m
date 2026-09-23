function [idxTr,idxVa,idxHw,frameAssign] = splitByFrame(frameId,frac,seed)
%SPLITBYFRAME  Frame-grouped (acquisition-grouped) partition.
%
%   This is the fix for the leakage identified in review: every noisy
%   realisation derived from a given source frame is assigned to EXACTLY ONE
%   partition, so no frame is seen in both training and evaluation.
%
%   frameId : [N x 1] source-frame index of every sample
%   frac    : [train val hw] fractions, applied to the NUMBER OF FRAMES
%   seed    : rng seed for the frame permutation (reported in the paper)
rs  = RandStream('mt19937ar','Seed',seed);
f   = unique(frameId(:));
nF  = numel(f);
perm= f(randperm(rs,nF));

nTr = round(frac(1)*nF);
nVa = round(frac(2)*nF);
nTr = max(nTr,1); nVa = max(nVa,1);
nHw = nF - nTr - nVa;
assert(nHw>=1,'Not enough frames (%d) for a 3-way grouped split.',nF);

fTr = perm(1:nTr);
fVa = perm(nTr+1:nTr+nVa);
fHw = perm(nTr+nVa+1:end);

idxTr = ismember(frameId(:),fTr);
idxVa = ismember(frameId(:),fVa);
idxHw = ismember(frameId(:),fHw);

frameAssign = struct('train',fTr,'val',fVa,'hw',fHw, ...
                     'nTrainFrames',nTr,'nValFrames',nVa,'nHwFrames',nHw);

assert(~any(idxTr&idxVa) && ~any(idxTr&idxHw) && ~any(idxVa&idxHw), ...
       'Partitions overlap -- grouped split failed.');
end
