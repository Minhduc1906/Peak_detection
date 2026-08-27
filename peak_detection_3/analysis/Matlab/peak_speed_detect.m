% bin_widths_ms = [short_term, long_term]

function peaking = peak_speed_detect (throughput, time, bin_widths_ms)

% Approximately convert bin_widths_ms into steps
mean_dt = mean (time (2:end) - time(1:(end-1))) * 1000;

short_term_step = round (bin_widths_ms(1) / mean_dt)
long_term_step = round (bin_widths_ms(2) / mean_dt)

throughput = [zeros(long_term_step, 1); throughput];
time = [zeros(long_term_step, 1); time]; % cheating

% This is deliberately designed in an inefficient way so as to be able to process samples one at a time.

short_term_memfactor = 0.95;
upper_threshold = 2;
lower_threshold = 0.5;

moving_avg = zeros (size (throughput));
binned_avgs_st = zeros (size (throughput));
binned_peaks_st = zeros (size (throughput));
binned_troughs_st = zeros (size (throughput));
binned_avgs_lt = zeros (size (throughput));
binned_peaks_lt = zeros (size (throughput));
binned_troughs_lt = zeros (size (throughput));

len = max (size (throughput));

for idx = 2 : len
	moving_avg(idx) = short_term_memfactor * moving_avg(idx - 1) + (1 - short_term_memfactor) * throughput (idx);
end

for idx = 1 : short_term_step : len - short_term_step
	binned_avgs_st (idx + (0:(short_term_step - 1))) = mean (throughput (idx + (0:(short_term_step - 1))));
	binned_peaks_st (idx + (0:(short_term_step - 1))) = max (throughput (idx + (0:(short_term_step - 1))));
	binned_troughs_st (idx + (0:(short_term_step - 1))) = min (throughput (idx + (0:(short_term_step - 1))));
end

for idx = 1 : long_term_step : len - long_term_step
	binned_avgs_lt (idx + (0:(long_term_step - 1))) = mean (throughput (idx + (0:(long_term_step - 1))));
	binned_peaks_lt (idx + (0:(long_term_step - 1))) = max (throughput (idx + (0:(long_term_step - 1))));
	binned_troughs_lt (idx + (0:(long_term_step - 1))) = min (throughput (idx + (0:(long_term_step - 1))));
end

tpf = medfilt1 (throughput, 15);

for idx = long_term_step : len - long_term_step
	binned_avgs_lt (idx) = mean (tpf (idx - long_term_step + (1:(long_term_step))));
	binned_peaks_lt (idx) = max (tpf (idx - long_term_step + (1:(long_term_step))));
	binned_troughs_lt (idx) = min (tpf (idx - long_term_step + (1:(long_term_step))));
end

%size(time)
%size(throughput)
%size(moving_avg)
%size(binned_avgs)
%size(binned_peaks)

%close all;

%peaking = (binned_peaks_st < moving_avg * upper_threshold) & (binned_troughs_st > moving_avg * lower_threshold);

peaking = binned_peaks_st > 0.7 * binned_peaks_lt;

%plot (time, throughput, time, moving_avg, time, binned_avgs, time, binned_peaks, time, peaking * max (throughput));

%plot (time, throughput, time, moving_avg, time, binned_avgs_st, time, binned_peaks_st, time, binned_avgs_lt, time, binned_peaks_lt);

plot (time, throughput, time, binned_peaks_st, time, binned_peaks_lt, time, peaking * max (throughput));

%, time, peaking * max (throughput));
