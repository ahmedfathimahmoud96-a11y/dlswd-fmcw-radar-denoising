function T = buildConfigSpace(C)
%BUILDCONFIGSPACE  Enumerate the Full Configuration Space (FCS), eq. (1).
%   Returns a table with one row per configuration and a short unique tag.
[iW,iL,iV,iD,iR] = ndgrid(1:numel(C.wavelets),1:numel(C.levels), ...
                          1:numel(C.varEst),1:numel(C.denMethod), ...
                          1:numel(C.threshRule));
n = numel(iW);
T = table('Size',[n 6], ...
    'VariableTypes',{'string','double','string','string','string','string'}, ...
    'VariableNames',{'wavelet','level','varEst','denMethod','threshRule','tag'});
for k = 1:n
    T.wavelet(k)    = C.wavelets{iW(k)};
    T.level(k)      = C.levels(iL(k));
    T.varEst(k)     = C.varEst{iV(k)};
    T.denMethod(k)  = C.denMethod{iD(k)};
    T.threshRule(k) = C.threshRule{iR(k)};
    T.tag(k) = sprintf('%s%s%d', erase(C.wavelets{iW(k)},'.'), ...
                       C.denMethod{iD(k)}, C.levels(iL(k)));
end
assert(height(T)==768, 'FCS size is %d, expected 768.', height(T));
end
