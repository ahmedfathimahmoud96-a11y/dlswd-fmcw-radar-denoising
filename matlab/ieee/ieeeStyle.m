function S = ieeeStyle()
%IEEESTYLE  Central definition of the IEEE figure look. Every figure in the
%           paper is produced through this file, so the style is consistent
%           by construction rather than by inspection.
%
% IEEE Transactions guidance implemented here:
%   * single-column graphic width  = 3.5 in (88.9 mm)
%   * double-column graphic width  = 7.16 in (181.9 mm)
%   * all text in the graphic 8 pt, axis labels 8 pt, no text below 6 pt
%   * Times New Roman to match the IEEEtran body font
%   * 600 dpi raster export + vector PDF export
%   * colour-blind-safe palette that also separates in greyscale print

S.fontName    = 'Times New Roman';
S.fontSize    = 8;      % tick labels
S.labelSize   = 8;      % axis labels
S.titleSize   = 8;
S.legendSize  = 7;
S.annotSize   = 7;

S.width1      = 3.5;    % in, single column
S.width2      = 7.16;   % in, double column
S.aspect      = 0.62;   % height = aspect * width  (near golden ratio)

S.lineWidth   = 1.1;
S.axLineWidth = 0.6;
S.markerSize  = 4;
S.gridAlpha   = 0.12;

S.dpi         = 600;

% Okabe-Ito palette: distinguishable for all common colour-vision
% deficiencies and monotone in luminance so it survives greyscale printing.
S.colors = [ 0.000 0.447 0.698    % blue
             0.902 0.624 0.000    % orange
             0.000 0.620 0.451    % green
             0.835 0.369 0.000    % vermillion
             0.800 0.475 0.655    % purple
             0.337 0.706 0.914    % sky
             0.941 0.894 0.259    % yellow
             0.350 0.350 0.350];  % grey

S.markers  = {'o','s','^','d','v','>','<','p'};
S.lineStys = {'-','--',':','-.','-','--',':','-.'};

% Sequential map for confusion matrices / heat maps (white -> IEEE blue)
n = 256; t = linspace(0,1,n)';
S.cmapSeq = [1-t*(1-0.000), 1-t*(1-0.447), 1-t*(1-0.698)];
end
