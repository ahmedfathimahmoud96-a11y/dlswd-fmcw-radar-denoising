function make_sota_snr_sweep_figure()
%MAKE_SOTA_SNR_SWEEP_FIGURE Plot the SNR-resolved comparison curves.

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
addpath(fullfile(scriptDir, 'ieee'));

inPath = fullfile(scriptDir, 'out', 'cached_grouped_revision', 'sota_comparison_by_snr.csv');
figDir = fullfile(rootDir, 'figures_ieee');
T = readtable(inPath);

methods = [
    "5x5 Gaussian filter"
    "Golilarz-style hybrid wavelet"
    "SWISTA-Net"
    "Fixed fk4Universal6"
    "Fixed db8Bayes5"
    "Fixed db4Sure7"
    "DLSWD-LUT"
    "DLSWD-CNN"
];

displayNames = [
    "Gaussian"
    "Golilarz-style"
    "SWISTA-Net"
    "fk4Universal6"
    "db8Bayes5"
    "db4Sure7"
    "DLSWD-LUT"
    "DLSWD-CNN"
];

S = ieeeStyle();
plotOne(T, methods, displayNames, 'OutputSNRdBMean', 'Output SNR (dB)', [-10 11], ...
    S, figDir, 'F_sota_snr_improvement');
plotCorrelationBeforeAfter(T, methods, displayNames, S, figDir, 'F_sota_correlation_preservation');
end

function plotOne(T, methods, displayNames, metricName, yLabel, yLimits, S, figDir, fileName)
fh = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 S.width1 2.55]);
hold on;
for k = 1:numel(methods)
    mask = T.Method == methods(k);
    Tk = sortrows(T(mask, :), 'InputSNRdB');
    if isempty(Tk)
        warning('Method "%s" not found in SNR comparison table.', methods(k));
        continue;
    end
    c = S.colors(mod(k - 1, size(S.colors, 1)) + 1, :);
    marker = S.markers{mod(k - 1, numel(S.markers)) + 1};
    lineStyle = S.lineStys{mod(k - 1, numel(S.lineStys)) + 1};
    plot(Tk.InputSNRdB, Tk.(metricName), 'LineStyle', lineStyle, ...
        'Marker', marker, 'MarkerIndices', 1:5:height(Tk), ...
        'Color', c, 'LineWidth', 1.8, 'MarkerSize', 5.2);
end
hold off;
grid on;
box on;
set(gca, 'FontName', S.fontName, 'FontSize', S.fontSize, ...
    'LineWidth', S.axLineWidth, 'GridAlpha', S.gridAlpha);
xlim([-20 5]);
xticks(-20:5:5);
xlabel('Input SNR (dB)', 'FontSize', S.labelSize);
ylabel(yLabel, 'FontSize', S.labelSize);
ylim(yLimits);
legend(displayNames, 'Location', 'southeast', 'FontSize', 5.2, 'Box', 'on', 'NumColumns', 2);
ieeeExport(fh, figDir, fileName);
end

function plotCorrelationBeforeAfter(T, methods, displayNames, S, figDir, fileName)
base = sortrows(T(T.Method == "No denoising", :), 'InputSNRdB');
fh = figure('Color', 'w', 'Units', 'inches', 'Position', [1 1 S.width1 2.55]);
hold on;
for k = 1:numel(methods)
    Tk = sortrows(T(T.Method == methods(k), :), 'InputSNRdB');
    if isempty(Tk)
        warning('Method "%s" not found in SNR comparison table.', methods(k));
        continue;
    end
    [snrCommon, iBase, iMethod] = intersect(base.InputSNRdB, Tk.InputSNRdB);
    if isempty(snrCommon)
        warning('Method "%s" has no SNR overlap with no-denoising baseline.', methods(k));
        continue;
    end
    c = S.colors(mod(k - 1, size(S.colors, 1)) + 1, :);
    marker = S.markers{mod(k - 1, numel(S.markers)) + 1};
    lineStyle = S.lineStys{mod(k - 1, numel(S.lineStys)) + 1};
    plot(base.CorrelationMean(iBase), Tk.CorrelationMean(iMethod), ...
        'LineStyle', lineStyle, 'Marker', marker, ...
        'MarkerIndices', 1:5:numel(iMethod), 'Color', c, ...
        'LineWidth', 1.8, 'MarkerSize', 5.2);
end
hold off;
grid on;
box on;
set(gca, 'FontName', S.fontName, 'FontSize', S.fontSize, ...
    'LineWidth', S.axLineWidth, 'GridAlpha', S.gridAlpha);
xlim([0 0.85]);
ylim([0 1]);
xlabel('Correlation before denoising', 'FontSize', S.labelSize);
ylabel('Correlation after denoising', 'FontSize', S.labelSize);
legend(displayNames, 'Location', 'southeast', 'FontSize', 5.2, 'Box', 'on', 'NumColumns', 2);
ieeeExport(fh, figDir, fileName);
end
