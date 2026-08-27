% function score = max_peak_speed_detect (time, throughput, long_window, threshold, filterlength])
%
% Input arguments:
%
% time: sample time in seconds since start
% throughput: bytes transfered during previous sample interval
% long_window: period (in ms) over which fraction of time that throughput is within within threshold of peak is assessed
% threshold: see above and below
% filterlength: median filter length (set to 1 for no median filtering)
%
% Return values:
%
% score: proportion of time in the current window for which throughput is within threshold (e.g. 0.2 = 20%) of the peak value in that window
%

function score = max_peak_speed_detect (time, throughput, long_window, threshold, filterlength)

if nargin ~= 5
	error ('Usage: score = max_peak_speed_detect (time, throughput, long_window, threshold, fiterlength])');
end

% Approximately convert bin_widths_ms into steps
mean_dt = mean (time (2:end) - time(1:(end-1)));

throughput = throughput * 8 / 1000000 / mean_dt;

mean_dt = mean_dt * 1000;

%short_term_step = round (short_window / mean_dt)
long_term_step = round (long_window / mean_dt);

binned_peaks_lt = zeros (size (throughput));

binned_peaks_st = zeros (size (throughput));
score = zeros (size (throughput));

len = max (size (throughput));

k = 1;

ftp = medfilt1 (throughput, filterlength);

for idx = 1 : long_term_step : floor (len / long_term_step) * long_term_step
	binned_peaks_lt (idx + (0:(long_term_step - 1))) = max (ftp (idx + (0:(long_term_step - 1))));
	score (idx + (0:(long_term_step - 1))) = sum (throughput (idx + (0:(long_term_step - 1))) > (1 - threshold) * binned_peaks_lt (idx)) / long_term_step;
end

clf;
hold on;

yyaxis left;
plot (time, throughput, 'b-', time, ftp, 'g-', time, binned_peaks_lt, 'r-');
grid on;

ylabel ('Throughput (Mb/s)');
xlabel ('Time (seconds)');

yyaxis right;
plot (time, score, '*-k');
ylabel ('Peak speed score (0-1)');
legend ({'Raw throughput', 'Filtered throughput', 'Binned peaks', 'Peak speed score'});

exportgraphics (gca, 'peak_speed_detect_max.pdf', 'ContentType', 'vector');

hold off;

%set (gcf, 'PaperUnits', 'normalized');
%set (gcf, 'PaperPosition', [0 0 1 1]);

%print ('peak_speed_detect_max.pdf', '-dpdf');
