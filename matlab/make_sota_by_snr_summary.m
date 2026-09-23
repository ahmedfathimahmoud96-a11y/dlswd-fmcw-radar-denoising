function make_sota_by_snr_summary()
%MAKE_SOTA_BY_SNR_SUMMARY Export SNR-resolved rows behind Fig. 10.
%   The public comparison is reported as an SNR sweep. This helper
%   writes the corresponding per-SNR means so reviewers can inspect the
%   trend behind the plotted curves.

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
outDir = fullfile(scriptDir, 'out', 'cached_grouped_revision');

manifestPath = fullfile(outDir, 'manifest.csv');
policyPath = fullfile(outDir, 'policy_metrics_by_source_snr.csv');
convPath = fullfile(outDir, 'conventional_filter_metrics.mat');
golPath = fullfile(outDir, 'golilarz_style_metrics.mat');
swistaPath = fullfile(scriptDir, 'out', 'swista_nets', 'swista_nets_per_record.csv');
outPath = fullfile(outDir, 'sota_comparison_by_snr.csv');
bandPath = fullfile(outDir, 'sota_comparison_by_snr_band.csv');

manifest = readtable(manifestPath);
policy = readtable(policyPath);
conv = load(convPath);
gol = load(golPath);

rows = emptyRows();

rows = [rows; groupedRowsBySourceSNR('3x3 mean filter', manifest.SourceId, manifest.SNRdB, conv.boxSNR, conv.boxCorr, conv.boxMSE)]; %#ok<AGROW>
rows = [rows; groupedRowsBySourceSNR('5x5 Gaussian filter', manifest.SourceId, manifest.SNRdB, conv.gaussSNR, conv.gaussCorr, conv.gaussMSE)]; %#ok<AGROW>
rows = [rows; groupedRowsBySourceSNR('Golilarz-style hybrid wavelet', manifest.SourceId, manifest.SNRdB, gol.snrOut, gol.corrOut, gol.mseOut)]; %#ok<AGROW>
if isfile(swistaPath)
    sw = readtable(swistaPath);
    rows = [rows; groupedRowsBySourceSNR('SWISTA-Net', ...
        sw.SourceId, sw.SNRdB, sw.OutputSNRdB, sw.Correlation, sw.MSE)]; %#ok<AGROW>
end

policyMap = {
    'No denoising',            'No denoising'
    'Fixed: fk4/Universal/L6', 'Fixed fk4Universal6'
    'Fixed: db8/Bayes/L5',     'Fixed db8Bayes5'
    'Fixed: db4/SURE/L7',      'Fixed db4Sure7'
    'Estimated-SNR lookup',    'DLSWD-LUT'
    'DLSWD-CNN deployed',      'DLSWD-CNN'
};

for k = 1:size(policyMap, 1)
    sourceName = policyMap{k, 1};
    displayName = policyMap{k, 2};
    mask = strcmp(policy.Policy, sourceName);
    rows = [rows; groupedRows(displayName, policy.InputSNRdB(mask), ...
        policy.OutputSNRdB(mask), policy.Correlation(mask), policy.MSE(mask))]; %#ok<AGROW>
end

rows = sortrows(rows, {'Method', 'InputSNRdB'});
writetable(rows, outPath);

bands = emptyBandRows();
bandNames = {'low (-20 to -11 dB)', 'mid (-10 to -1 dB)', 'high (0 to 5 dB)'};
bandEdges = [-20 -11; -10 -1; 0 5];
methods = unique(rows.Method, 'stable');
for m = 1:numel(methods)
    methodMask = rows.Method == methods(m);
    for b = 1:size(bandEdges, 1)
        bandMask = methodMask & rows.InputSNRdB >= bandEdges(b, 1) & rows.InputSNRdB <= bandEdges(b, 2);
        if any(bandMask)
            bands = [bands; table(methods(m), string(bandNames{b}), ... %#ok<AGROW>
                sum(rows.N(bandMask)), mean(rows.OutputSNRdBMean(bandMask), 'omitnan'), ...
                mean(rows.CorrelationMean(bandMask), 'omitnan'), mean(rows.MSEMean(bandMask), 'omitnan'), ...
                'VariableNames', {'Method', 'SNRBand', 'N', 'OutputSNRdBMean', 'CorrelationMean', 'MSEMean'})];
        end
    end
end
writetable(bands, bandPath);

fprintf('Wrote %s\n', erase(outPath, [rootDir filesep]));
fprintf('Wrote %s\n', erase(bandPath, [rootDir filesep]));
end

function rows = groupedRows(method, inputSNRdB, outputSNRdB, correlation, mse)
inputSNRdB = double(inputSNRdB(:));
outputSNRdB = double(outputSNRdB(:));
correlation = double(correlation(:));
mse = double(mse(:));

snrs = unique(inputSNRdB);
rows = emptyRows();
for i = 1:numel(snrs)
    mask = inputSNRdB == snrs(i);
    rows = [rows; table(string(method), snrs(i), sum(mask), ...
        mean(outputSNRdB(mask), 'omitnan'), mean(correlation(mask), 'omitnan'), mean(mse(mask), 'omitnan'), ...
        'VariableNames', {'Method', 'InputSNRdB', 'N', 'OutputSNRdBMean', 'CorrelationMean', 'MSEMean'})]; %#ok<AGROW>
end
end

function rows = groupedRowsBySourceSNR(method, sourceId, inputSNRdB, outputSNRdB, correlation, mse)
sourceId = double(sourceId(:));
inputSNRdB = double(inputSNRdB(:));
outputSNRdB = double(outputSNRdB(:));
correlation = double(correlation(:));
mse = double(mse(:));

pairs = unique([sourceId inputSNRdB], 'rows');
sourceRows = emptyRows();
for p = 1:size(pairs, 1)
    mask = sourceId == pairs(p, 1) & inputSNRdB == pairs(p, 2);
    sourceRows = [sourceRows; table(string(method), pairs(p, 2), 1, ...
        mean(outputSNRdB(mask), 'omitnan'), mean(correlation(mask), 'omitnan'), mean(mse(mask), 'omitnan'), ...
        'VariableNames', {'Method', 'InputSNRdB', 'N', 'OutputSNRdBMean', 'CorrelationMean', 'MSEMean'})]; %#ok<AGROW>
end

rows = groupedRows(method, sourceRows.InputSNRdB, sourceRows.OutputSNRdBMean, ...
    sourceRows.CorrelationMean, sourceRows.MSEMean);
end

function rows = emptyRows()
rows = table(string.empty(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    'VariableNames', {'Method', 'InputSNRdB', 'N', 'OutputSNRdBMean', 'CorrelationMean', 'MSEMean'});
end

function rows = emptyBandRows()
rows = table(string.empty(0, 1), string.empty(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), zeros(0, 1), ...
    'VariableNames', {'Method', 'SNRBand', 'N', 'OutputSNRdBMean', 'CorrelationMean', 'MSEMean'});
end
