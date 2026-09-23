function M = prfMetrics(confCounts,beta)
%PRFMETRICS  Per-class precision / recall / F1 / Fbeta from RAW COUNTS.
%   confCounts : [K x K] integer matrix, rows = true class, cols = predicted
%   beta       : Fbeta weighting (2 in this paper)
%
%   Raw counts, not row-normalised percentages: percentages lose the class
%   prevalences and make precision unrecoverable.
if nargin<2, beta = 2; end
K   = size(confCounts,1);
tp  = diag(confCounts);
fp  = sum(confCounts,1)' - tp;
fn  = sum(confCounts,2)  - tp;

P = tp ./ max(tp+fp,eps);
R = tp ./ max(tp+fn,eps);
F1 = 2*P.*R ./ max(P+R,eps);
Fb = (1+beta^2)*P.*R ./ max(beta^2*P + R, eps);

M.precision = P;  M.recall = R;  M.f1 = F1;  M.fbeta = Fb;  M.beta = beta;
M.support   = sum(confCounts,2);
M.accuracy  = sum(tp)/sum(confCounts(:));
M.macroP    = mean(P); M.macroR = mean(R); M.macroF = mean(Fb);
w           = M.support/sum(M.support);
M.weightedF = sum(w.*Fb);
M.counts    = confCounts;
end
