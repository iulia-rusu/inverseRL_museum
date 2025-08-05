function floorplan_trajectories_ir

pix_to_m = 0.003175;
csvfile = 'floor_overlay_labels.csv';

% Load trajectory files
csv_files = dir('cluster_*.csv');
colors = lines(length(csv_files));

figure(10); clf;
floor_rects_metric_inline(csvfile, true, pix_to_m);
hold on;

disp(length(csv_files))

%% plot all trajectories together, on top of the floorplan
for i = 1:length(csv_files)
    %for i = 8:8
    data = readmatrix(csv_files(i).name);
    x = data(:, 2) * pix_to_m;
    y = data(:, 3) * pix_to_m;
    plot(x, y, '.', 'Color', colors(i,:), 'LineWidth', 0.5, 'MarkerSize', 1);
end

title('Floor Plan and Trajectories in Meters', 'Interpreter', 'none');
xlabel('X (m)', 'Interpreter', 'none');
ylabel('Y (m)', 'Interpreter', 'none');
axis([-1000 7000 0 3500] * pix_to_m);  % ← Scaled to preserve full X range
%axis equal;
hold off;

%% plot all trajectories separately, with no floorplan

plot_separate = 1;

if plot_separate
    figure(11); clf;
    for i = 1:length(csv_files)
        subplot(4,4,i)
        data = readmatrix(csv_files(i).name);
        x = data(:, 2) * pix_to_m;
        y = data(:, 3) * pix_to_m;
        plot(x, y, '.', 'Color', colors(i,:),...
                                    'LineWidth', 0.5, 'MarkerSize', 1);
        axis([-1000 7000 0 3500] * pix_to_m);
        title_string = csv_files(i).name;
        title(sprintf('cluster %s',title_string(9:11)),...
                                      'FontSize',7,'FontWeight','normal')
    end
end

end


%% --- FLOOR PLAN FUNCTION (METERS) ---
function floor_rects_metric_inline(csvfile, show_text, pix_to_m)
T = readtable(csvfile, 'TextType', 'string');

for k = 1:height(T)
    x = T.x(k) * pix_to_m;
    y = T.y(k) * pix_to_m;
    w = T.width(k) * pix_to_m;
    h = T.height(k) * pix_to_m;

    rect_clr = .5*ones(1,3);
    rectangle('Position', [x, y, w, h], ...
        'EdgeColor', rect_clr, 'LineWidth', 1);

    if show_text
        text(x, y + h + 0.05, T.name(k), ...
            'Color', [0 0 0], 'FontSize', 8, ...
            'VerticalAlignment', 'bottom', ...
            'HorizontalAlignment', 'left', ...
            'Interpreter', 'none');
    end
end
end
