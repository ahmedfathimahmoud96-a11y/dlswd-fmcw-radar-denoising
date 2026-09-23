function [fh,ax] = ieeeFigure(cols,heightScale)
%IEEEFIGURE  Create a correctly sized, correctly styled figure.
%   [fh,ax] = ieeeFigure(1)      single-column figure
%   [fh,ax] = ieeeFigure(2)      double-column figure
%   [fh,ax] = ieeeFigure(1,0.85) override the height/width ratio
if nargin<1, cols = 1; end
S = ieeeStyle();
if nargin<2 || isempty(heightScale), heightScale = S.aspect; end

w = (cols==1)*S.width1 + (cols==2)*S.width2;
h = w*heightScale;

fh = figure('Units','inches','Position',[1 1 w h], ...
            'Color','w','PaperUnits','inches','PaperSize',[w h], ...
            'PaperPosition',[0 0 w h],'InvertHardcopy','off');
ax = axes(fh); %#ok<LAXES>
ieeeAxes(ax);
end
