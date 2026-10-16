from collections import deque
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'vendor' / 'PenSimPy'))
from pensimpy.peni_env_setup import PenSimEnv
from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data import constants as recipes
from pensimpy.constants import STEP_IN_HOURS as DT, NUM_STEPS

HORIZON, REPLAN = 8., 4.
LAB_PERIOD, LAB_DELAY, START = 12., 4., 16.


def oxygen_limits(pressure, broth_mass):
    top = 1 + pressure
    bottom = top + broth_mass * .9 * 9.81e-5 / (np.pi * 2.1**2)
    mean_pressure = (bottom - top) / np.log(bottom / top)
    saturation = mean_pressure * .21 / .0251
    return .1 * saturation, .3 * saturation


class PooledModel:
    def __init__(self, dimension=6, capacity=30):
        self.d, self.capacity, self.clusters = dimension, capacity, []
        self.updates = 0
        self.next_id, self.last_id = 1, 0

    def parameters(self, z):
        if not self.clusters:
            return np.zeros(self.d)
        distances = np.array([np.sum((z - c['z'])**2) for c in self.clusters])
        ids = np.argsort(distances)[:self.d]
        ds = distances[ids]
        weights = np.exp(-(ds - ds[0]) / max(np.median(ds) - ds[0], 1e-9))
        A, b = 0.01 * np.eye(self.d), np.zeros(self.d)
        for i, w in zip(ids, weights):
            A += w * self.clusters[i]['A']
            b += w * self.clusters[i]['b']
        return np.linalg.solve(A, b)

    def update(self, z, phi, y):
        assert np.isfinite(np.r_[z, phi, y]).all()
        cs = self.clusters

        for c in cs:
            c['A'] *= .98
            c['b'] *= .98
        distances = [np.sum((z - c['z'])**2) for c in cs]
        if not cs or min(distances) > .08:
            cs.append(dict(id=self.next_id, z=z.copy(), n=0,
                      A=np.zeros((self.d, self.d)), b=np.zeros(self.d)))
            self.next_id += 1
            i = len(cs) - 1
        else:
            i = int(np.argmin(distances))
        c = cs[i]
        self.last_id = c['id']
        c['n'] += 1
        c['z'] += (z - c['z']) / min(c['n'], 200)
        c['A'] += np.outer(phi, phi)
        c['b'] += phi * y
        self.updates += 1
        if len(cs) > self.capacity:
            centers = np.array([c['z'] for c in cs])
            distance = np.sum((centers[:, None] - centers[None, :])**2, axis=2)
            np.fill_diagonal(distance, np.inf)
            i, j = np.unravel_index(np.argmin(distance), distance.shape)
            a, b = cs[i], cs[j]
            a['z'] = (a['n'] * a['z'] + b['n'] * b['z']) / (a['n'] + b['n'])
            for key in ('n', 'A', 'b'):
                a[key] += b[key]
            cs.pop(j)


class BiomassSensor:
    def __init__(self):
        self.model = PooledModel()
        self.anchor = None
        self.latest = None
        self.received = 0
        self.samples = []

    def estimate(self, z):
        if self.anchor is None:
            return np.nan
        sampled_z, biomass = self.anchor

        theta = self.model.parameters(z)
        return max(0., float(biomass + 50 * np.r_[0., z - sampled_z] @ theta))

    # train when the lab result arrives
    def receive(self, now, sample):
        taken, due, z, biomass, volume, nominal_feed, predicted = sample
        assert now >= due and np.isclose(due - taken, LAB_DELAY)
        self.latest = (biomass, volume, nominal_feed)
        self.received += 1
        self.model.update(z, np.r_[1., z], biomass / 50)
        self.samples.append((self.model.last_id, z.copy()))
        self.anchor = (z.copy(), biomass)
        return biomass - predicted if np.isfinite(predicted) else np.nan


def features(time, feed_total, offgas, volume):
    return np.array([time / 230, feed_total / 20000, offgas[0] / 10,
                     offgas[1] / 2, volume / 100000])


class FeedController:
    def __init__(self, initial):
        self.air, self.feed = initial['Fg'], initial['Fs']

    def choose(self, nominal, oxygen, mass, factor=1.):
        action = nominal.copy()
        _, limit = oxygen_limits(nominal['pressure'], mass)
        self.air = float(np.clip(self.air + 2 * np.clip(limit + .5 - oxygen, -1, 1),
                                 nominal['Fg'], 110.))
        action['Fg'] = self.air
        desired = factor * nominal['Fs']
        limited = np.clip(desired, self.feed - .05 *
                          nominal['Fs'], self.feed + .05 * nominal['Fs'])
        action['Fs'] = float(
            np.clip(limited, .85 * nominal['Fs'], 1.15 * nominal['Fs']))

        self.feed = action['Fs']
        return action


class Predictor:
    channels = ('DO2', 'V', 'Wt', 'pH', 'T')

    def __init__(self):
        self.env = PenSimEnv(recipe(), fast=False)
        self.env.random_seed_ref = 0
        self.env.ctrl_flags.Raman_spec = 0
        self.env.ctrl_flags.Dis = 0
        _, self.state = self.env.reset()
        self.checkpoint(0)

    @staticmethod
    def biomass(env, state, k, value):
        names = ('a0', 'a1', 'a3', 'a4')
        old = sum(float(getattr(env.x0, n)) if k == 0 else read(state, n, k - 1)
                  for n in names)
        ratio = max(value, .001) / max(old, .001)
        for n in names:
            if k == 0:
                setattr(env.x0, n, float(getattr(env.x0, n)) * ratio)
            else:
                getattr(state, n).y[k - 1] *= ratio
        if k:
            state.X.y[k - 1] = max(value, .001)

    def checkpoint(self, k):
        self.saved = (k, deepcopy(self.env), deepcopy(self.state))
        self.history = []

    def advance(self, k, action, measured, remember=True):
        self.env.step(k, self.state, **action)

        for n, value in zip(self.channels, measured):
            getattr(self.state, n).y[k - 1] = value
        if remember:
            self.history.append((k, action.copy(), tuple(measured)))

    # correct the old state, replay to now
    def assay(self, taken, value):
        k, env, state = self.saved
        assert np.isclose(k * DT, taken)
        self.env, self.state = deepcopy(env), deepcopy(state)
        self.biomass(self.env, self.state, k, value)
        for step, action, measured in self.history:
            self.advance(step, action, measured, remember=False)

    def plan(self, k, controller, estimate):
        candidates, invalid = [], 0
        for factor in (.85, 1., 1.15):
            env, state, actuator = deepcopy(self.env), deepcopy(
                self.state), deepcopy(controller)
            if np.isfinite(estimate):
                self.biomass(env, state, k - 1, estimate)
            harvested, deficit, feed = 0., 0., 0.
            try:
                for j in range(k, min(NUM_STEPS + 1, k + round(HORIZON / DT))):
                    nominal = env.recipe_combo.get_values_dict_at(j * DT)
                    action = actuator.choose(nominal, read(
                        state, 'DO2', j - 2), read(state, 'Wt', j - 2), factor)
                    env.step(j, state, **action)
                    product, volume, biomass = (read(state, n, j - 1)
                                                for n in ('P', 'V', 'X'))
                    density = (1100 + product + biomass) / 1000
                    _, limit = oxygen_limits(action['pressure'], density * volume)
                    deficit += max(limit - read(state, 'DO2', j - 1), 0) * DT
                    harvested += max(-read(state, 'discharge', j - 1),
                                     0) / density * product * DT / 1000
                    feed += action['Fs'] * DT
                total = harvested + product * volume / 1000
                if not np.isfinite([total, deficit, feed]).all():
                    raise FloatingPointError('Non-finite forecast')
            except (ValueError, RuntimeError, OverflowError, FloatingPointError):
                invalid += 1
                continue

            candidates.append((deficit > 1e-8, deficit if deficit >
                              1e-8 else 0., -total, feed, factor))
        if not candidates:
            return 1., 1, invalid
        best = min(candidates)
        return best[-1], int(best[0]), invalid


def recipe():
    names = ['FS', 'FOIL', 'FG', 'PRESS', 'DISCHARGE', 'WATER', 'PAA']
    keys = ['Fs', 'Foil', 'Fg', 'pressure', 'discharge', 'Fw', 'Fpaa']
    profiles = {k: Recipe(getattr(recipes, n + '_DEFAULT_PROFILE'), k)
                for n, k in zip(names, keys)}

    harvest = [{'time': 0, 'value': 0}]
    for start in (100, 130, 150, 170, 190, 210):
        harvest += [{'time': start, 'value': 4000}, {'time': start + 2, 'value': 0}]
    profiles['discharge'] = Recipe(harvest, 'discharge')
    return RecipeCombo(profiles)


def read(x, name, i):
    return float(np.asarray(getattr(x, name).y[i]).item())


def run(seed=1, policy='recipe'):
    env = PenSimEnv(recipe(), fast=False)
    env.random_seed_ref = seed
    env.ctrl_flags.Raman_spec = 0
    assert env.ctrl_flags.Faults == 0 and env.ctrl_flags.Inhib == 2
    _, plant = env.reset()
    initial = env.recipe_combo.get_values_dict_at(DT)
    sensor, controller = BiomassSensor(), FeedController(initial)
    predictor = Predictor() if policy != 'recipe' else None
    factor, infeasible, plans, invalid_forecasts, fallbacks = 1., 0, 0, 0, 0
    oxygen, volume = float(env.x0.DO2), float(env.x0.V)
    mass, offgas, feed_total = float(env.x0.Wt), (0., 0.), 0.
    pending, rows = deque(), []

    z = features(0., 0., offgas, volume)
    pending.append((0., LAB_DELAY, z.copy(),
                    sum(float(getattr(env.x0, n))
                        for n in ('a0', 'a1', 'a3', 'a4')), volume,
                    initial['Fs'], np.nan))
    harvested, lab_error = 0., np.nan
    for k in range(1, NUM_STEPS + 1):
        now, end = (k - 1) * DT, k * DT
        while pending and pending[0][1] <= now + 1e-8:
            sample = pending.popleft()
            lab_error = sensor.receive(round(now, 8), sample)
            if predictor is not None:
                predictor.assay(sample[0], sample[3])
        z = features(now, feed_total, offgas, volume)
        estimate = sensor.estimate(z)
        nominal = env.recipe_combo.get_values_dict_at(end)
        if predictor is not None and now >= START and (k - 1) % round(REPLAN / DT) == 0:
            factor, failed, invalid = predictor.plan(
                k, controller, estimate if policy == 'soft' else np.nan)
            infeasible += failed
            invalid_forecasts += invalid
            fallbacks += int(invalid == 3)
            plans += 1
        action = controller.choose(nominal, oxygen, mass, factor)
        if policy == 'recipe' or now < START:
            action['Fs'] = controller.feed = nominal['Fs']

        _, plant, _, _ = env.step(k, plant, **action)
        assert np.isclose(read(plant, 'Fs', k - 1), action['Fs'])
        assert np.isclose(read(plant, 'Fg', k - 1), action['Fg'])
        oxygen, volume, mass = (read(plant, n, k - 1) for n in ('DO2', 'V', 'Wt'))
        offgas = tuple(read(plant, n, k - 1) for n in ('OUR', 'CER'))
        feed_total += action['Fs'] * DT
        z = features(end, feed_total, offgas, volume)

        estimate = sensor.estimate(z)
        lab_hold = sensor.latest[0] if sensor.latest is not None else np.nan
        if predictor is not None:
            predictor.advance(k, action, [read(plant, n, k - 1)
                              for n in Predictor.channels])
        truth, product = read(plant, 'X', k - 1), read(plant, 'P', k - 1)
        if k % round(LAB_PERIOD / DT) == 0:
            pending.append((end, end + LAB_DELAY, z.copy(), truth, volume,
                            nominal['Fs'], estimate))
            if predictor is not None:
                predictor.checkpoint(k)

        density = (1100 + truth + product) / 1000
        growth_limit, production_limit = oxygen_limits(
            action['pressure'], density * volume)
        harvested += max(-read(plant, 'discharge', k - 1), 0) / \
            density * product * DT / 1000
        product_kg = harvested + product * volume / 1000
        rows.append([end, truth, estimate, lab_hold, oxygen, action['Fs'],
                     nominal['Fs'], action['Fg'], product_kg, production_limit,
                     growth_limit, sensor.received, sensor.model.updates,
                     len(sensor.model.clusters), lab_error, plans,
                     infeasible, invalid_forecasts, fallbacks])
        lab_error = np.nan
    columns = ['time', 'biomass', 'estimate', 'lab_hold', 'oxygen', 'feed',
               'recipe_feed', 'air', 'product_kg', 'production_limit', 'growth_limit',
               'assays', 'updates', 'clusters', 'lab_error',
               'plans', 'infeasible', 'invalid_forecasts', 'fallbacks']
    frame = pd.DataFrame(rows, columns=columns)
    frame.attrs.update(samples=sensor.samples,
                       centers=[(c['id'], c['z'].copy())
                                for c in sensor.model.clusters])
    assert np.isfinite(frame.drop(
        columns=['estimate', 'lab_hold', 'lab_error']).to_numpy()).all()
    assert (frame.feed >= .85 * frame.recipe_feed - 1e-8).all()
    assert (frame.feed <= 1.15 * frame.recipe_feed + 1e-8).all()
    assert frame.air.between(15., 110.).all()
    assert (frame.updates == frame.assays).all()
    assert frame.loc[frame.time <= LAB_DELAY, 'estimate'].isna().all()
    return frame


def metrics(frame):
    active = frame[frame.time >= START]
    valid = frame.estimate.notna()
    return dict(biomass_RMSE=np.sqrt(np.mean(
                    (frame.estimate[valid] - frame.biomass[valid])**2)),
                lab_hold_RMSE=np.sqrt(
                    np.mean((frame.lab_hold[valid] - frame.biomass[valid])**2)),
                delayed_lab_RMSE=np.sqrt(np.nanmean(frame.lab_error**2)),
                product_kg=frame.product_kg.iloc[-1], feed_L=frame.feed.sum() * DT,
                oxygen_deficit=np.maximum(
                    active.production_limit - active.oxygen, 0).sum() * DT,
                below_10pct_h=(active.oxygen < active.growth_limit).sum() * DT,
                assays=int(frame.assays.iloc[-1]), updates=int(frame.updates.iloc[-1]),
                plans=int(frame.plans.iloc[-1]),
                infeasible=int(frame.infeasible.iloc[-1]),
                invalid_forecasts=int(frame.invalid_forecasts.iloc[-1]),
                fallbacks=int(frame.fallbacks.iloc[-1]))


def plot(frames):
    soft = frames[2]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    t = soft.time
    ax = axes[0]
    ax.plot(t, soft.biomass, color='0.35', label='Actual biomass')
    ax.plot(t, soft.estimate, color='tab:blue', label='Online estimate')
    arrived = soft.assays.diff().fillna(0) > 0
    ax.scatter(t[arrived], soft.lab_hold[arrived], color='tab:orange', s=22,
               label='Lab result (4 h delay)', zorder=3)
    ax.set(xlabel='Time (h)', ylabel='Biomass (g/L)')

    ax = axes[1]
    for cid, center in soft.attrs['centers']:
        color = plt.get_cmap('tab10')((cid - 1) % 10)
        samples = np.array([z for i, z in soft.attrs['samples'] if i == cid])
        ax.scatter(samples[:, 2], samples[:, 3], s=40,
                   color=color, label=f'Cluster {cid}')
        # sample spread, not a model boundary
        if len(samples) >= 3:
            covariance = np.cov(samples[:, 2:4], rowvar=False)
            values, vectors = np.linalg.eigh(covariance)
            if values.min() > 1e-12:
                circle = np.array([np.cos(np.linspace(0, 2 * np.pi, 100)),
                                   np.sin(np.linspace(0, 2 * np.pi, 100))])
                ellipse = center[2:4, None] + \
                    vectors @ np.diag(2 * np.sqrt(values)) @ circle
                ax.plot(*ellipse, ls='--', lw=1, color=color)
        ax.scatter(center[2], center[3], marker='X', s=130, color=color,
                   edgecolors='black', zorder=4)
    ax.scatter([], [], marker='X', s=100, color='0.5',
               edgecolors='black', label='Centre')
    ax.plot([], [], 'k--', lw=1, label='Sample spread (2 SD)')
    ax.set(xlabel='Normalized oxygen uptake', ylabel='Normalized CO2 evolution')
    ax = axes[2]
    for frame, color, label in zip(frames, ['0.5', 'tab:orange', 'tab:blue'],
                                   ['Recipe', 'MPC + lab', 'MPC + online estimate']):
        ax.plot(t, frame.feed, color=color, lw=1.2, label=label)
    ax.set(xlabel='Time (h)', ylabel='Sugar feed (L/h)')
    for ax in axes:
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(ROOT / 'demo.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    summary = []
    for seed in ((1, 2, 3) if '--validate' in sys.argv else (1,)):
        frames = []
        for policy in ('recipe', 'lab', 'soft'):
            print(f'Batch {seed}: {policy}', flush=True)
            frame = run(seed, policy)
            summary.append(dict(batch=seed, policy=policy, **metrics(frame)))
            frames.append(frame)
        warm = frames[0].time <= START
        for frame in frames[1:]:
            assert np.allclose(frames[0].loc[warm, ['biomass', 'feed', 'air']],
                               frame.loc[warm, ['biomass', 'feed', 'air']])
        difference = np.mean(abs(frames[2].feed - frames[1].feed) > 1e-8) * 100
        print(
            'Soft-sensor feed differs from lab-only feed on '
            f'{difference:.1f}% of steps', flush=True)
        if seed == 1:
            plot(frames)
    print(pd.DataFrame(summary).to_string(
        index=False, float_format=lambda x: f'{x:.3f}'))
    print('Saved demo.png. Nominal-model MPC; 8 h horizon, '
          '4 h replanning. No performance guarantee.')
