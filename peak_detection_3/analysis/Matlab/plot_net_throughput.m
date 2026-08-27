% function [downstream, upstream] = plot_net_throughput (downstream_log, upstream_log)
%
% Input arguments:
%
% downstream_log, upstream_log: log files for the upstream and downstream interface of a given node
%
% Return values:
%
% upstream, downstream: summary of the logs for convenient plotting or analysis. Format is as follows:
%
% relative time (seconds), rx_packets, rx_bytes, tx_packets, tx_bytes, qdisc bytes, qdisc packets, qdisc drops, qdisc overlimits, qdisc backlog (queue depth)
%
% qdisc is assumed to be tbf and the only qdisc operational on this node/interface

function [downstream, upstream] = plot_net_throughput (downstream_log, upstream_log)

if nargin ~= 2
	error ('Usage: [downstream, stream] = plot_net_throughput (downstream_log, upstream_log)');
end

if exist('OCTAVE_VERSION', 'builtin') ~= 0
% Ignore the first line of each
	ds = dlmread (downstream_log, ' ', 1, 0);
	us = dlmread (upstream_log, ' ', 1, 0);
else
	ds = readmatrix (downstream_log);
	us = readmatrix (upstream_log);
end

close all;

% Relative time

t_ds = ds(2:end, 1) - ds(2, 1);
t_us = us(2:end, 1) - us(2, 1);

ds_interval = mean (t_ds(2:end) - t_ds(1:end-1));
us_interval = mean (t_us(2:end) - t_us(1:end-1));

% Packets/sec
figure (1);

plot (t_ds, ds(2:end, 4) / ds_interval, t_ds, ds(2:end, 2) / ds_interval);

xlabel ('Time (seconds)');
ylabel ('Downstream throughput (packets/s)');
grid on;
legend ({'Tx', 'Rx'});

exportgraphics (gca, 'downstream_pps_s2.pdf', 'ContentType', 'vector');

figure (2);

% Mbits/sec
plot (t_ds, 8 * ds(2:end, 5) / ds_interval / 1e6, t_ds, 8 * ds(2:end, 3) / ds_interval / 1e6);

xlabel ('Time (seconds)');
ylabel ('Downstream throughput (Mb/s)');
grid on;
legend ({'Tx', 'Rx'});

exportgraphics (gca, 'downstream_Mbps_s2.pdf', 'ContentType', 'vector');

figure (3);

% Packets/sec
plot (t_us, us(2:end, 4) / us_interval, t_us, us(2:end, 2) / us_interval);

xlabel ('Time (seconds)');
ylabel ('Upstream throughput (packets/s)');
grid on;
legend ({'Tx', 'Rx'});

exportgraphics (gca, 'upstream_pps_s2.pdf', 'ContentType', 'vector');

figure (4);

% Mbits/sec
plot (t_us, 8 * us(2:end, 5) / us_interval / 1e6, t_us, 8 * us(2:end, 3) / us_interval / 1e6);

xlabel ('Time (seconds)');
ylabel ('Upstream throughput (Mb/s)');
grid on;
legend ({'Tx', 'Rx'});

exportgraphics (gca, 'upstream_Mbps_s2.pdf', 'ContentType', 'vector');

qdisc = 11;

% Buffer occupancy
figure (5);

buffer_ds = ds(2:end, qdisc) - ds(1:end - 1, qdisc)

plot (t_ds, buffer_ds);

xlabel ('Time (seconds)');
ylabel ('Downstream tx queue buffer occupancy (packets)');
grid on;

exportgraphics (gca, 'downstream_buffer_s2.pdf', 'ContentType', 'vector');

figure(6)

buffer_us = us(2:end, qdisc) - us(1:end - 1, qdisc)

% Mbits/sec
plot (t_us, buffer_us);

xlabel ('Time (seconds)');
ylabel ('Upstream tx queue buffer occupancy (packets)');
grid on;

exportgraphics (gca, 'upstream_buffer_s2.pdf', 'ContentType', 'vector');

upstream = [t_us, us(2:end, 2:5), buffer_us];
downstream = [t_ds, ds(2:end, 2:5), buffer_ds];
