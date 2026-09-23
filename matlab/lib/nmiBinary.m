function I = nmiBinary(y1,y2)
%NMIBINARY  Normalised mutual information between two binary selection
%           indicators, eq. (4).
%   y1,y2 : logical column vectors of equal length (1 = this configuration
%           was the argmax of eq. (2) for that sample).
%   I     : I(Y1;Y2) / min(H(Y1),H(Y2)) in [0,1].
%
%   The plug-in (maximum-likelihood) estimator is exact here because both
%   variables are discrete with known support; no binning or kNN density
%   estimation is involved.
y1 = logical(y1(:)); y2 = logical(y2(:));
N  = numel(y1);
p  = zeros(2,2);
p(1,1) = sum(~y1 & ~y2); p(1,2) = sum(~y1 &  y2);
p(2,1) = sum( y1 & ~y2); p(2,2) = sum( y1 &  y2);
p = p / N;
px = sum(p,2); py = sum(p,1);

MI = 0;
for a = 1:2
    for b = 1:2
        if p(a,b) > 0
            MI = MI + p(a,b)*log2( p(a,b) / (px(a)*py(b)) );
        end
    end
end
H = @(q) -sum(q(q>0).*log2(q(q>0)));
den = min(H(px),H(py));
if den <= 0, I = 0; else, I = MI/den; end
I = max(0,min(1,I));
end
