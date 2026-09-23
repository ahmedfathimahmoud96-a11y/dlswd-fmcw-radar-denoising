function [mcs,info] = cbcSelect(labels,tags,tauC,tauI)
%CBCSELECT  Configuration Balance Check -- Algorithm 1 of the paper.
%
%   labels : [N x 1] index into tags, the argmax of eq. (2) per TRAINING sample
%   tags   : [M x 1] string, configuration names
%   tauC   : occurrence-frequency threshold (e.g. 0.05)
%   tauI   : normalised-MI redundancy threshold (e.g. 0.30)
%
%   mcs    : indices of the retained configurations (Minimized Config. Space)
%   info   : struct with counts, the pairwise NMI matrix over Step-1
%            survivors, and the prune decisions -- everything needed to
%            reproduce the reduction and to typeset it in the paper.
%
%   Determinism: candidates are visited in descending occurrence order and,
%   within a redundant pair, the MORE frequent configuration is kept. The
%   result therefore does not depend on the enumeration order of the FCS.

N = numel(labels);
M = numel(tags);
cnt = accumarray(labels(:),1,[M 1]);
frac = cnt / N;

step1 = find(frac > tauC);
[~,ord] = sort(cnt(step1),'descend');
step1 = step1(ord);

% one-hot selection indicators restricted to Step-1 survivors
Y = false(N,numel(step1));
for k = 1:numel(step1), Y(:,k) = (labels(:)==step1(k)); end

% pairwise normalised MI
K = numel(step1);
NMI = zeros(K);
for a = 1:K
    for b = a+1:K
        NMI(a,b) = nmiBinary(Y(:,a),Y(:,b));
        NMI(b,a) = NMI(a,b);
    end
end

keep = false(1,K);
prunedBy = zeros(1,K);
for a = 1:K
    red = false;
    for b = find(keep)
        if NMI(a,b) > tauI, red = true; prunedBy(a) = step1(b); break; end
    end
    if ~red, keep(a) = true; end
end

mcs = step1(keep);

info.tauC       = tauC;
info.tauI       = tauI;
info.counts     = cnt;
info.frac       = frac;
info.step1      = step1;
info.step1Tags  = tags(step1);
info.step1Frac  = frac(step1);
info.NMI        = NMI;
info.kept       = keep;
info.prunedBy   = prunedBy;
info.mcsTags    = tags(mcs);
info.nFCS       = M;
info.nObserved  = sum(cnt>0);
info.nStep1     = K;
info.nMCS       = numel(mcs);
end
