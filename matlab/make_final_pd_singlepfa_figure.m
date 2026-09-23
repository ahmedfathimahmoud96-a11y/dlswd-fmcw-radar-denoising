function make_final_pd_singlepfa_figure(varargin)
%MAKE_FINAL_PD_SINGLEPFA_FIGURE Regenerate the projected Pd figure labels.
%
% Uses the cached analytical Pd projection and writes the exact figure file
% included in the public figure set:
%   ../Ped/F_labelled_Pd_vs_SNR_singlePfa.png
%   ../Ped/F_labelled_Pd_vs_SNR_singlePfa.pdf

ip = inputParser;
addParameter(ip,'Pfa',1e-2,@(x)isnumeric(x)&&isscalar(x)&&x>0&&x<1);
parse(ip,varargin{:});
O = ip.Results;

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
addpath(fullfile(scriptDir,'ieee'));

inFile = fullfile(scriptDir,'out','cached_grouped_revision','analytical_pd_projection_by_snr.csv');
assert(isfile(inFile),'Missing analytical Pd projection CSV: %s',inFile);
P = readtable(inFile,'TextType','string');

policyNames = ["No denoising","DLSWD-" + "LUT","DLSWD (Proposed)"];
displayNames = {'Before denoising','DLSWD-LUT','DLSWD (Proposed)'};
colors = [0.25 0.25 0.25; 0.494 0.184 0.556; 0.000 0.447 0.741];
lineStyles = {'--','--','-'};
markers = {'^','s','o'};
markerFaces = {'w','w',colors(3,:)};

S = ieeeStyle();
fh = figure('Color','w','Units','inches','Position',[1 1 S.width1 2.15]);
ax = axes(fh);
hold(ax,'on');
h = gobjects(numel(policyNames),1);
for p = 1:numel(policyNames)
    rows = P(P.Policy == policyNames(p) & abs(P.CFARDesignPfa-O.Pfa)<=1e-12,:);
    [x,order] = sort(rows.InputSNRdB);
    y = rows.ProjectedPd(order);
    if ~isempty(y), y(1) = 0; end
    h(p) = plot(ax,x,y,'LineStyle',lineStyles{p},'Marker',markers{p}, ...
        'Color',colors(p,:),'MarkerEdgeColor',colors(p,:), ...
        'MarkerFaceColor',markerFaces{p},'MarkerSize',3.8, ...
        'LineWidth',1.20);
end
hold(ax,'off');
grid(ax,'on');
box(ax,'on');
set(ax,'FontName',S.fontName,'FontSize',S.fontSize, ...
    'LineWidth',S.axLineWidth,'GridAlpha',S.gridAlpha, ...
    'XTick',-20:5:5,'XMinorGrid','off','YMinorGrid','off');
xlim(ax,[-20 5]);
ylim(ax,[0 1.02]);
xlabel(ax,'Input SNR (dB)','FontSize',S.labelSize);
ylabel(ax,'P_d','FontSize',S.labelSize);
lg = legend(ax,h,displayNames,'Location','northwest','NumColumns',1);
set(lg,'FontName',S.fontName,'FontSize',5.2,'Box','on','Color','w','ItemTokenSize',[8 5]);

ieeeExport(fh,fullfile(rootDir,'Ped'),'F_labelled_Pd_vs_SNR_singlePfa');
end
