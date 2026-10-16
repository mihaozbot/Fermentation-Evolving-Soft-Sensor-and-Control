function [runs, summary] = demo_matlab(seed, hours)
    if nargin < 1
        seed = 1;
    end
    if nargin < 2
        hours = 230;
    end
    assert(hours >= 20 && hours <= 230 && abs(hours / .2 - round(hours / .2)) < 1e-8);
    root = fileparts(mfilename('fullpath'));
    addpath(fullfile(root, '..', 'vendor', 'IndPenSimMATLAB'));
    savedrng = rng;
    cleanup = onCleanup(@() rng(savedrng));
    policies = {'recipe', 'lab', 'soft'};
    runs = cell(1, 3);
    stats = [];
    for p = 1:3
        fprintf('Batch %d: %s\n', seed, policies{p});
        runs{p} = run_batch(seed, policies{p}, hours);
        r = runs{p};
        d = r.data;
        valid = isfinite(d(:, 3));
        active = d(:, 1) >= 16;
        stats(p, :) = [sqrt(mean((d(valid, 3) - d(valid, 2)).^2)), ...
                    sqrt(mean((d(valid, 4) - d(valid, 2)).^2)), d(end, 9), sum(d(:, ...
                    6)) * .2, ...
                    sum(max(d(active, 10) - d(active, 5), 0)) * .2, d(end, 12:13), ...
                    r.plans, r.invalid];
    end
    warm = runs{1}.data(:, 1) <= 16;
    for p = 2:3
        assert(max(abs(runs{1}.data(warm, [2 6 8]) - runs{p}.data(warm, [2 6 8])), ...
            [], 'all') < 1e-8);
    end
    summary = array2table(stats, 'VariableNames', {'biomass_RMSE', 'lab_hold_RMSE', ...
        'product_kg', 'feed_L', 'oxygen_deficit', 'assays', 'updates', 'plans', ...
            'invalid_forecasts'}, ...
                        'RowNames', policies);
    disp(summary);
    plot_runs(runs, root);
end

function r = run_batch(seed, policy, hours)
    dt = .2;
    N = round(hours / dt);
    plant = new_env(seed, hours, 1);
    sensor.cs = struct('id', {}, 'z', {}, 'n', {}, 'A', {}, 'b', {});
    sensor.anchor = [];
    sensor.samples = [];
    sensor.updates = 0;
    sensor.next = 1;
    initial = recipe(dt);
    actuator = [initial.Fg initial.Fs];
    feed = 0;
    offgas = [0 0];
    factor = 1;
    harvested = 0;
    z = features(0, feed, offgas, plant.y(5));
    pending = {struct('taken', 0, 'due', 4, 'z', z, 'value', sum(plant.y(12:15)))};
    predictor = new_env(0, hours, 0);
    checkpoint = predictor;
    checkpoint_k = 0;
    history = {};
    rows = zeros(N, 14);
    plans = 0;
    invalid = 0;
    for k = 1:N
        now = (k - 1) * dt;
        finish = k * dt;
        if ~isempty(pending) && pending{1}.due <= now + 1e-8
            sample = pending{1};
            pending(1) = [];
            assert(now + 1e-8 >= sample.due && sample.due - sample.taken == 4);
            sensor = receive(sensor, sample);
            if ~strcmp(policy, 'recipe')
                assert(abs(checkpoint_k * dt - sample.taken) < 1e-8);
                predictor = correct_biomass(checkpoint, sample.value, checkpoint_k);
                for j = 1:numel(history)
                    h = history{j};
                    predictor = step_env(predictor, h.k, h.action);
                    predictor = correct_online(predictor, h.measured, h.k);
                end
            end
        end
        z = features(now, feed, offgas, plant.y(5));
        estimate = estimate_biomass(sensor, z);
        nominal = recipe(finish);
        if ~strcmp(policy, 'recipe') && now >= 16 && mod(k - 1, 20) == 0
            correction = NaN;
            if strcmp(policy, 'soft')
                correction = estimate;
            end
            [factor, bad] = plan(predictor, k, actuator, correction, N);
            plans = plans + 1;
            invalid = invalid + bad;
        end
        [action, actuator] = choose(nominal, plant.y(2), plant.y(6), factor, actuator);
        if strcmp(policy, 'recipe') || now < 16
            action.Fs = nominal.Fs;
            actuator(2) = action.Fs;
        end
        plant = step_env(plant, k, action);
        y = plant.y;
        feed = feed + action.Fs * dt;
        offgas = [(10 / 7 * action.Fg) * (.204 - y(3) * (.7902 / (1 - y(3) - y(28) / ...
            100))), ...
                (55 / 28 * action.Fg) * (.0065 * y(28)) * (.7902 / (1 - .204 - y(28) ...
                    / 100) - .0330)];
        z = features(finish, feed, offgas, y(5));
        estimate = estimate_biomass(sensor, z);
        lab = NaN;
        if ~isempty(sensor.anchor)
            lab = sensor.anchor.value;
        end
        if ~strcmp(policy, 'recipe')
            measured = y([2 5 6 7 8]);
            predictor = step_env(predictor, k, action);
            predictor = correct_online(predictor, measured, k);
            history{end + 1} = struct('k', k, 'action', action, 'measured', measured);
        end
        biomass = sum(y(12:15));
        product = y(4);
        if mod(k, 60) == 0
            pending{end + 1} = struct('taken', finish, 'due', finish + 4, 'z', z, ...
                'value', biomass);
            checkpoint = predictor;
            checkpoint_k = k;
            history = {};
        end
        density = (1100 + biomass + product) / 1000;
        harvested = harvested + action.discharge / density * product * dt / 1000;
        limits = oxygen_limits(action.pressure, density * y(5));
        rows(k, :) = [finish biomass estimate lab y(2) action.Fs nominal.Fs ...
            action.Fg ...
                   harvested + product * y(5) / 1000 limits(2) limits(1) ...
                    sensor.updates sensor.updates numel(sensor.cs)];
    end
    assert(all(isfinite(rows(:, [1 2 5:14])), 'all'));
    assert(all(rows(:, 6) >= .85 * rows(:, 7) - 1e-8 & rows(:, 6) <= 1.15 * rows(:, ...
        7) + 1e-8));
    assert(all(isnan(rows(rows(:, 1) <= 4, 3))));
    assert(sensor.updates == sum((0:12:hours) + 4 < hours - 1e-8));
    r = struct('data', rows, 'sensor', sensor, 'plans', plans, 'invalid', invalid);
end

function z = features(t, feed, offgas, volume)
    z = [t / 230; feed / 20000; offgas(1) / 10; offgas(2) / 2; volume / 100000];
end

function s = receive(s, sample)
    z = sample.z;
    phi = [1; z];
    cs = s.cs;
    for i = 1:numel(cs)
        cs(i).A = .98 * cs(i).A;
        cs(i).b = .98 * cs(i).b;
    end
    dist = arrayfun(@(c) sum((z - c.z).^2), cs);
    if isempty(cs) || min(dist) > .08
        cs(end + 1) = struct('id', s.next, 'z', z, 'n', 0, 'A', zeros(6), 'b', ...
            zeros(6, 1));
        s.next = s.next + 1;
        i = numel(cs);
    else
        [~, i] = min(dist);
    end
    cs(i).n = cs(i).n + 1;
    cs(i).z = cs(i).z + (z - cs(i).z) / min(cs(i).n, 200);
    cs(i).A = cs(i).A + phi * phi';
    cs(i).b = cs(i).b + phi * sample.value / 50;
    s.samples(end + 1, :) = [cs(i).id z'];
    if numel(cs) > 30
        centers = [cs.z];
        dist = sum((reshape(centers, 5, [], 1) - reshape(centers, 5, 1, [])).^2, 1);
        dist = squeeze(dist);
        dist(1:size(dist, 1) + 1:end) = Inf;
        [~, ix] = min(dist(:));
        [i, j] = ind2sub(size(dist), ix);
        cs(i).z = (cs(i).n * cs(i).z + cs(j).n * cs(j).z) / (cs(i).n + cs(j).n);
        cs(i).n = cs(i).n + cs(j).n;
        cs(i).A = cs(i).A + cs(j).A;
        cs(i).b = cs(i).b + cs(j).b;
        cs(j) = [];
    end
    s.cs = cs;
    s.anchor = sample;
    s.updates = s.updates + 1;
end

function value = estimate_biomass(s, z)
    value = NaN;
    if isempty(s.anchor)
        return
    end
    dist = arrayfun(@(c) sum((z - c.z).^2), s.cs);
    [ds, ids] = sort(dist);
    ids = ids(1:min(6, numel(ids)));
    ds = ds(1:numel(ids));
    w = exp(-(ds - ds(1)) / max(median(ds) - ds(1), 1e-9));
    A = .01 * eye(6);
    b = zeros(6, 1);
    for j = 1:numel(ids)
        c = s.cs(ids(j));
        A = A + w(j) * c.A;
        b = b + w(j) * c.b;
    end
    value = max(0, s.anchor.value + 50 * [0; z - s.anchor.z]' * (A \ b));
end

function limits = oxygen_limits(pressure, mass)
    top = 1 + pressure;
    bottom = top + mass * .9 * 9.81e-5 / (pi * 2.1^2);
    limits = [.1 .3] * (bottom - top) / log(bottom / top) * .21 / .0251;
end

function [a, c] = choose(a, oxygen, mass, factor, c)
    limits = oxygen_limits(a.pressure, mass);
    c(1) = min(110, max(a.Fg, c(1) + 2 * min(1, max(-1, limits(2) + .5 - oxygen))));
    c(2) = min(1.15 * a.Fs, max(.85 * a.Fs, min(c(2) + .05 * a.Fs, max(c(2) - .05 * ...
        a.Fs, factor * a.Fs))));
    a.Fg = c(1);
    a.Fs = c(2);
end

function [factor, invalid] = plan(env, k, controller, estimate, N)
    candidates = [];
    invalid = 0;
    for factor = [.85 1 1.15]
        prediction = env;
        actuator = controller;
        if isfinite(estimate)
            prediction = correct_biomass(prediction, estimate, k - 1);
        end
        harvested = 0;
        deficit = 0;
        feed = 0;
        try
            for j = k:min(N, k + 39)
                [a, actuator] = choose(recipe(j * .2), prediction.y(2), ...
                    prediction.y(6), factor, actuator);
                prediction = step_env(prediction, j, a);
                y = prediction.y;
                density = (1100 + sum(y(12:15)) + y(4)) / 1000;
                limits = oxygen_limits(a.pressure, density * y(5));
                deficit = deficit + max(limits(2) - y(2), 0) * .2;
                harvested = harvested + a.discharge / density * y(4) * .2 / 1000;
                feed = feed + a.Fs * .2;
            end
            total = harvested + y(4) * y(5) / 1000;
            assert(all(isfinite([total deficit feed])));
            candidates(end + 1, :) = [deficit > 1e-8 deficit * (deficit > 1e-8) ...
                -total feed factor];
        catch
            invalid = invalid + 1;
        end
    end
    factor = 1;
    if ~isempty(candidates)
        candidates = sortrows(candidates);
        factor = candidates(1, end);
    end
end

function e = new_env(seed, hours, disturbance)
    r = zeros(17, 1);
    for j = 1:17
        rng(30 + seed + j);
        r(j) = randn;
    end
    biomass = .5 + .05 * r(1);
    x0.mux = .41 + .025 * r(1);
    x0.mup = .041 + .0025 * r(2);
    e.par = Parameter_list(x0, 85 + 10 * r(15), 150000 + 2000 * r(17), 530000 + ...
        20000 * r(16));
    e.y = [1 + .1 * r(3); 15 + .5 * r(4); .20 + .05 * r(9); 0; 58000 + 500 * r(6); ...
        62000 + 500 * r(7); ...
         10^(-(6.5 + .1 * r(10))); 297 + .5 * r(11); 0; 4; 0; biomass / 3; 2 * ...
             biomass / 3; ...
         zeros(14, 1); .038 + .001 * r(8); 0; 1400 + 50 * r(13); 1700 + 50 * r(14); ...
             0; 0];
    e.names = {'S', 'DO2', 'O2', 'P', 'V', 'Wt', 'pH', 'T', 'Q', 'Viscosity', ...
        'Culture_age', ...
             'a0', 'a1', 'a3', 'a4', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', ...
                 'n8', 'n9', ...
             'nm', 'phi0', 'CO2outgas', 'CO2_d', 'PAA', 'NH3', 'mu_P_calc', ...
                 'mu_X_calc'};
    e.X = createBatch(.2, hours);
    e = save_state(e, 1);
    e.flags = struct('SBC', 0, 'PRBS', 0, 'Faults', 0, 'Raman_spec', 0, 'T_sp', 298, ...
        'pH_sp', 6.5);
    e.hours = hours;
    e.dis = disturbance;
    rng(seed);
    noise = filter(.005, [1 -.995], randn(round(hours / .2) + 1, 1));
    e.dist = noise * [.03 .25 1500 300 .2 300000 100 .02];
end

function e = step_env(e, k, a)
    [u, e.X] = fctrl_indpensim(e.X, struct(), k, .2, e.hours, e.flags);
    if k > 65 && sum(diff(e.X.mu_X_calc.y(k - 65:k - 1)) < 0) >= 63
        e.par(2) = e.X.mu_X_calc.y(k - 1) * 5;
    end
    inp = [2 a.Fs a.Fg 100 u.Fc u.Fh u.Fb u.Fa .005 a.Fw a.pressure 4 ...
         -a.discharge a.Fpaa a.Foil 0 e.dis e.dist(k, :) 0];
    y = e.y;
    y(32:33) = 0;
    atol = ones(33, 1) * 1e-8;
    atol(7) = 1e-12;
    [t, trajectory] = ode15s(@(t, y) indpensim_ode(t, y, [], inp, e.par), ...
        linspace((k - 1) * .2, k * .2, 41), y, odeset('RelTol', 1e-5, 'AbsTol', atol));
    assert(abs(t(end) - k * .2) < 1e-8 && isreal(trajectory) && ...
        all(isfinite(trajectory), 'all'));
    e.y = trajectory(end, :)';
    % upstream floors
    e.y(1:31) = max(e.y(1:31), .001 .* (e.y(1:31) <= 0));
    if e.y(2) < 2
        e.y(2) = 1;
    end
    e = save_state(e, k);
    for name = {'Fa', 'Fb', 'Fc', 'Fh'}
        e.X.(name{1}).y(k) = u.(name{1});
    end
end

function e = save_state(e, k)
    for j = 1:33
        e.X.(e.names{j}).y(k) = e.y(j);
    end
    e.X.X.y(k) = sum(e.y(12:15));
end

function e = correct_biomass(e, value, k)
    % correct at sample time, then replay
    e.y(12:15) = e.y(12:15) * max(value, .001) / max(sum(e.y(12:15)), .001);
    e = save_state(e, max(k, 1));
end

function e = correct_online(e, measured, k)
    e.y([2 5 6 7 8]) = measured;
    e = save_state(e, k);
end

function a = recipe(t)
    names = {'Fs', 'Foil', 'Fg', 'pressure', 'Fw', 'Fpaa'};
    profiles = { [3 8; 12 15; 16 30; 20 75; 24 150; 28 30; 32 37; 36 43; 40 47; 44 ...
        51; 48 57; 52 61; 56 65; 60 72; 64 76; 68 80; 72 84; 76 90; 80 116; 160 90; ...
        230 80], ...
              [4 22; 16 30; 56 35; 60 34; 64 33; 68 32; 72 31; 76 30; 80 29; 230 ...
                  23], ...
              [8 30; 20 42; 40 55; 90 60; 200 75; 230 65], ...
              [12.4 .6; 25 .7; 30 .8; 40 .9; 100 1.1; 150 1; 200 .9; 230 .9], ...
              [50 0; 75 500; 150 100; 160 0; 170 400; 200 150; 230 250], ...
              [5 5; 40 0; 200 10; 230 4] };
    for j = 1:numel(names)
        p = profiles{j};
        index = find(p(:, 1) <= t, 1, 'last');
        if isempty(index)
            index = 1;
        end
        a.(names{j}) = p(index, 2);
    end
    a.discharge = 4000 * any(t >= [100 130 150 170 190 210] & t < [102 132 152 172 ...
        192 212]);
end

function plot_runs(runs, root)
    figure('Position', [100 100 1600 450]);
    tiledlayout(1, 3, 'Padding', 'compact');
    r = runs{3};
    d = r.data;
    t = d(:, 1);
    nexttile;
    hold on;
    plot(t, d(:, 2), 'Color', [.35 .35 .35], 'DisplayName', 'Actual biomass');
    plot(t, d(:, 3), 'Color', [0 .45 .74], 'DisplayName', 'Online estimate');
    arrived = [false; diff(d(:, 12)) > 0];
    scatter(t(arrived), d(arrived, 4), 22, [.85 .33 .1], 'filled', 'DisplayName', ...
        'Lab result (4 h delay)');
    xlabel('Time (h)');
    ylabel('Biomass (g/L)');
    legend('Location', 'best');
    grid on;
    nexttile;
    hold on;
    colors = lines(10);
    for c = r.sensor.cs
        color = colors(mod(c.id - 1, 10) + 1, :);
        samples = r.sensor.samples(r.sensor.samples(:, 1) == c.id, 4:5);
        scatter(samples(:, 1), samples(:, 2), 40, color, 'filled', 'DisplayName', ...
            sprintf('Cluster %d', c.id));
        if size(samples, 1) >= 3
            [v, lambda] = eig(cov(samples));
            if min(diag(lambda)) > 1e-12
                angle = linspace(0, 2 * pi, 100);
                ellipse = c.z(3:4) + v * diag(2 * sqrt(diag(lambda))) * [cos(angle); ...
                    sin(angle)];
                plot(ellipse(1, :), ellipse(2, :), '--', 'Color', color, ...
                    'HandleVisibility', 'off');
            end
        end
        scatter(c.z(3), c.z(4), 100, color, 'x', 'LineWidth', 2, 'HandleVisibility', ...
            'off');
    end
    scatter(NaN, NaN, 100, 'k', 'x', 'LineWidth', 2, 'DisplayName', 'Centre');
    plot(NaN, NaN, 'k--', 'DisplayName', 'Sample spread (2 SD)');
    xlabel('Normalized oxygen uptake');
    ylabel('Normalized CO2 evolution');
    legend('Location', 'best');
    grid on;
    nexttile;
    hold on;
    labels = {'Recipe', 'MPC + lab', 'MPC + online estimate'};
    colors = [.5 .5 .5; .85 .33 .1; 0 .45 .74];
    for j = 1:3
        plot(t, runs{j}.data(:, 6), 'Color', colors(j, :), 'DisplayName', labels{j});
    end
    xlabel('Time (h)');
    ylabel('Sugar feed (L/h)');
    legend('Location', 'best');
    grid on;
    exportgraphics(gcf, fullfile(root, 'demo_matlab.png'), 'Resolution', 150);
end
