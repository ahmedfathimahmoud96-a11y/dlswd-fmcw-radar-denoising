function ieeeExport(fh,figDir,name)
%IEEEEXPORT  Write one figure as 600-dpi PNG (for review) and vector PDF
%            (for the camera-ready). Both carry the same base name, so the
%            .tex can switch between them by changing one extension.
%
%   Robust to the common Windows failure mode in which an image viewer,
%   the Explorer preview pane, or a PDF reader still holds a handle on the
%   previous version of the file. In that case exportgraphics raises
%   "PNG library failed: Could not open file". This function retries, falls
%   back to print(), and if it still cannot write, says exactly which file
%   is locked instead of aborting the whole run.

S = ieeeStyle();
if ~exist(figDir,'dir'), mkdir(figDir); end
set(findall(fh,'-property','FontName'),'FontName',S.fontName);
drawnow;

png = fullfile(figDir,[name '.png']);
pdf = fullfile(figDir,[name '.pdf']);

okPng = writeOne(fh,png,'png',S);
okPdf = writeOne(fh,pdf,'pdf',S);

if okPng && okPdf
    fprintf('  figure -> %s (.png/.pdf)\n',name);
elseif okPng || okPdf
    fprintf(2,'  figure -> %s (PARTIAL: png=%d pdf=%d)\n',name,okPng,okPdf);
else
    fprintf(2,'  figure -> %s SKIPPED -- both files are locked.\n',name);
end
close(fh);
end

% ------------------------------------------------------------------
function ok = writeOne(fh,fname,kind,S)
ok = false;

% If a stale file exists, try to remove it first. A locked file will refuse
% here too, which is a cheaper way to detect the problem than a failed write.
if isfile(fname)
    try
        delete(fname);
    catch
        % leave it; exportgraphics may still be able to overwrite in place
    end
end

for attempt = 1:3
    try
        if exist('exportgraphics','file')                      % R2020a+
            if strcmp(kind,'png')
                exportgraphics(fh,fname,'Resolution',S.dpi,'BackgroundColor','white');
            else
                exportgraphics(fh,fname,'ContentType','vector','BackgroundColor','white');
            end
        else
            if strcmp(kind,'png')
                print(fh,fname,'-dpng',sprintf('-r%d',S.dpi));
            else
                print(fh,fname,'-dpdf','-painters');
            end
        end
        ok = true; return
    catch ME
        if attempt < 3
            pause(0.75);                                       % let the lock clear
            continue
        end
        % last resort: the older print() path, which sometimes succeeds
        % where exportgraphics does not
        try
            if strcmp(kind,'png')
                print(fh,fname,'-dpng',sprintf('-r%d',S.dpi));
            else
                print(fh,fname,'-dpdf','-painters');
            end
            ok = true; return
        catch
            fprintf(2,['    cannot write %s\n' ...
                       '    -> %s\n' ...
                       '    Close any viewer or Explorer preview showing this file ' ...
                       'and re-run.\n'],fname,ME.message);
        end
    end
end
end
