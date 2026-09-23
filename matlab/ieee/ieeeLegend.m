function lg = ieeeLegend(ax,labels,loc)
%IEEELEGEND  Consistently styled legend.
if nargin<3 || isempty(loc), loc = 'best'; end
S = ieeeStyle();
lg = legend(ax,labels,'Location',loc);
set(lg,'FontName',S.fontName,'FontSize',S.legendSize, ...
       'Box','on','EdgeColor',[0.6 0.6 0.6],'Color','w','ItemTokenSize',[12 6]);
end
