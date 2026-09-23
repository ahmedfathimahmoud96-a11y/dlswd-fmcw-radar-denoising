function r = corrCoefC(xhat,xref)
%CORRCOEFC  Magnitude of the complex correlation coefficient in [0,1].
a = xhat(:) - mean(xhat(:));
b = xref(:) - mean(xref(:));
r = abs( (a'*b) / max(sqrt((a'*a)*(b'*b)),eps) );
end
