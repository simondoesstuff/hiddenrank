"""Seeded recreations of spontaneous dynamics (Fig. 1) and input geometry (Fig. 2)."""
from pathlib import Path
import json

import jax.numpy as jnp
import numpy as np

from lowrank.meanfield import GaussianRankOne, phase_boundaries, solve_fixed_points, solve_chaotic
from lowrank.network import RateNetwork, simulate


def _phase_boundaries():
    """Fig. 1 boundaries from bulk instability and Eq. 88."""
    result = phase_boundaries(np.linspace(1., 4., 81), quadrature_order=200)
    static = np.column_stack((result.g, result.static_chaos))
    chaotic = np.column_stack((result.g, result.structure_onset))
    return static, chaotic


def _simulate(m, n, chi, g, x0, drives, dt):
    net = RateNetwork(jnp.asarray(m[:, None]), jnp.asarray(n[:, None]), jnp.asarray(chi), g=g)
    return simulate(net, jnp.asarray(x0), jnp.asarray(drives), dt=dt, sample_every=max(1, round(.2 / dt)))


def _spontaneous_theory(chaotic_boundary):
    # Continue the positive branch through the stationary-to-chaotic transition.
    # Independent chaotic solves can fall onto the unstable zero-mean branch.
    gs = np.linspace(0, 3.6, 73)
    theory = np.full((len(gs), 4), np.nan)
    previous_chaotic_initial = None
    for i, g in enumerate(gs):
        p = GaussianRankOne(g=float(g), mean_m=np.sqrt(2.2), mean_n=np.sqrt(2.2))
        points = solve_fixed_points(p, seeds=[(1.4, 2.+g*g)], quadrature_order=200)
        positive = [pt for pt in points if pt.kappa[0] > 1e-3]
        pt = max(positive, key=lambda pt: pt.kappa[0]) if positive else None
        if pt is not None and pt.stable:
            theory[i] = [pt.kappa[0], pt.variance, 0., 0.]
            # A small positive temporal variance initiates chaotic continuation.
            previous_chaotic_initial = (float(pt.kappa[0]), pt.variance+.01,
                                        max(0., pt.variance-.01))
        elif g > 1:
            structured_onset = np.interp(g, chaotic_boundary[:,0], chaotic_boundary[:,1])
            if 2.2 <= structured_onset:
                # Above the chaotic pitchfork only the zero-mean branch is selected.
                initial = (0., g*g, 0.)
            else:
                initial = previous_chaotic_initial
            c = solve_chaotic(p, initial=initial, quadrature_order=200)
            if 2.2 > structured_onset and c.kappa[0] < 1e-4:
                raise RuntimeError(f"structured chaotic continuation lost its branch at g={g}")
            theory[i] = [abs(c.kappa[0]), c.long_lag_variance, c.temporal_variance, 1.]
            previous_chaotic_initial = (float(c.kappa[0]), c.variance, c.long_lag_variance)
    return gs, theory


def _input_theory():
    amplitudes = np.linspace(0, 2., 41)
    curves = np.full((3, len(amplitudes), 3), np.nan)
    for kind in range(3):
        for i, amplitude in enumerate(amplitudes):
            mean = np.sqrt(3.5) if kind == 2 else 0.
            # Input direction n's centered component; independent for kind zero.
            p = GaussianRankOne(g=.8, mean_m=mean, mean_n=mean,
                                var_input=amplitude**2,
                                cov_n_input=amplitude if kind else 0.)
            points = solve_fixed_points(p, seeds=[(-2., 4.), (0., 1.), (2., 4.)])
            for point in points:
                branch = 1 if not point.stable else 0 if point.kappa[0] < -1e-5 else 2
                curves[kind, i, branch] = point.kappa[0] if abs(point.kappa[0]) > 1e-10 else 0.
    return amplitudes, curves


def _write(output, name, metadata, **arrays):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, experiment=name, metadata=json.dumps(metadata), **arrays)
    return output


def run_dynamics_trial(name, output, *, size=512, seed=2018, trial_id=0, dt=.1, duration=40.):
    """One resumable realization with shared patterns and independent bulk disorder."""
    if size < 8 or trial_id < 0 or dt <= 0 or duration <= 0 or round(duration/dt) < 1:
        raise ValueError("size>=8, trial_id>=0, dt>0, duration>=dt required")
    steps = round(duration/dt)
    # Pattern geometry is held fixed over the ensemble, as in the original runner.
    rng = np.random.default_rng(seed)
    bulk_rng = np.random.default_rng(np.random.SeedSequence([seed, trial_id, 1818]))
    chi = bulk_rng.normal(size=(size, size)).astype(np.float32) / np.float32(np.sqrt(size))
    if name == 'spontaneous':
        from lowrank.spontaneous import simulate_spontaneous
        conditions = np.array([[.7, .8], [.7, 2.2], [2., .8], [2., 2.2]])
        base_m, base_n = rng.normal(size=(2, size))
        m = base_m[None] + np.sqrt(conditions[:, 1, None])
        n = base_n[None] + np.sqrt(conditions[:, 1, None])
        times, traces, positive = simulate_spontaneous(m, n, chi, conditions[:, 0],
            steps=steps, dt=dt, sample_every=max(1, round(.2/dt)))
        positive = np.asarray(positive)
        negative = positive * np.array([-1., 1., 1.])
        arrays = dict(conditions=conditions, times=np.asarray(times), traces=np.asarray(traces),
                      statistics=np.stack((positive, negative), axis=1))
    elif name == 'inputs':
        m, n, _ = rng.normal(size=(3, size))
        strength = (((np.arange(steps)+.5)*dt >= duration*.25)
                    & ((np.arange(steps)+.5)*dt < duration*.65))
        result = _simulate(m, n, chi, .8, np.zeros(size), strength[:, None]*n[None], dt)
        arrays = dict(times=np.asarray(result.times), states=np.asarray(result.states),
                      basis=np.stack((n, m), axis=1), onset=duration*.25, offset=duration*.65)
    else:
        raise ValueError(f"unknown experiment {name}")
    return _write(output, name, dict(size=size, seed=seed, trial_id=trial_id, dt=dt,
        duration=duration, sampling='iid Gaussian; SeedSequence([seed, trial_id, 1818])'), **arrays)


def aggregate_dynamics(inputs, output):
    """Reduce completed shards; expensive theory curves are computed only once."""
    sources = []
    for path in inputs:
        with np.load(path) as data:
            sources.append({k: np.asarray(data[k]) for k in data.files})
    if not sources:
        raise ValueError("at least one dynamics trial is required")
    sources.sort(key=lambda d: json.loads(str(d['metadata']))['trial_id'])
    first = sources[0]
    name = str(first['experiment'])
    meta = json.loads(str(first['metadata']))
    ids = []
    for source in sources:
        other = json.loads(str(source['metadata']))
        if str(source['experiment']) != name or any(other[k] != meta[k] for k in ('size', 'seed', 'dt', 'duration')):
            raise ValueError("incompatible dynamics trials")
        ids.append(other['trial_id'])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate dynamics trial IDs")
    meta.pop('trial_id')
    meta.update(trials=len(sources), trial_ids=ids, paper='Mastrogiuseppe & Ostojic (2018)')
    if name == 'spontaneous':
        static, chaotic = _phase_boundaries()
        gs, theory = _spontaneous_theory(chaotic)
        statistics = np.stack([s['statistics'] for s in sources], axis=1)
        arrays = dict(conditions=first['conditions'], times=first['times'], traces=first['traces'],
            statistics=statistics.reshape((4, -1, 3)), static_boundary=static,
            chaotic_boundary=chaotic, gs=gs, theory=theory)
    elif name == 'inputs':
        average = np.zeros_like(first['states'], dtype=np.float64)
        for source in sources:
            average += source['states']/len(sources)
        basis = first['basis']
        coordinates = average @ np.linalg.pinv(basis).T
        centered = average-average.mean(axis=0)
        zscore = centered/np.maximum(centered.std(axis=0), 1e-10)
        _, singular, vt = np.linalg.svd(zscore, full_matrices=False)
        amplitudes, curves = _input_theory()
        arrays = dict(times=first['times'], traces=average[:, :6], coordinates=coordinates,
            explained=singular[:6]**2/(singular**2).sum(),
            pc_alignment=np.abs(vt[:2] @ basis)/np.linalg.norm(basis, axis=0),
            amplitudes=amplitudes, curves=curves, onset=first['onset'], offset=first['offset'])
    else:
        raise ValueError(f"unknown experiment {name}")
    return _write(output, name, meta, **arrays)


def run_experiment(name, output, *, size=512, trials=4, seed=2018, dt=.1, duration=40.):
    """Convenience runner; Snakemake uses persistent per-trial checkpoints."""
    from tempfile import TemporaryDirectory
    if trials < 1:
        raise ValueError("trials must be positive")
    with TemporaryDirectory() as folder:
        inputs = [run_dynamics_trial(name, Path(folder)/f'{i}.npz', size=size,
                  seed=seed, trial_id=i, dt=dt, duration=duration) for i in range(trials)]
        return aggregate_dynamics(inputs, output)


def plot_experiment(source, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    with np.load(source) as data:
        name = str(data['experiment'])
        meta = json.loads(str(data['metadata']))
        plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                             'svg.fonttype': 'none'})
        if name == 'spontaneous':
            fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), layout='constrained')
            ax = axes[0, 0]
            gs, ss = np.meshgrid(np.linspace(0, 4, 301), np.linspace(0, 4, 301))
            b, c = data['static_boundary'], data['chaotic_boundary']
            onset = np.interp(ss, b[:, 1], b[:, 0], left=1, right=b[-1, 0])
            crit = np.interp(gs, c[:, 0], c[:, 1])
            phase = np.where(gs<1, np.where(ss<1, 0, 1), np.where(ss<crit, 2, np.where(gs<onset, 1, 3)))
            ax.pcolormesh(gs, ss, phase, cmap=ListedColormap(['#fafafa','#d9edf2','#f7c7ce','#ecd4bb']), shading='auto', rasterized=True, vmin=0, vmax=3)
            ax.plot(b[:,0],b[:,1],c='#555',lw=1)
            ax.plot(c[:,0],c[:,1],c='#555',lw=1)
            ax.plot([0,1,1],[1,1,0],c='#555',lw=1)
            ax.set(xlim=(0,4), ylim=(0,4), xlabel='Random strength g', ylabel='Structure strength $M_m M_n$', title='A  Spontaneous regimes (DMF)')
            for x,y,label in [(.45,.4,'Zero'),(.7,3.3,'Structured\nstationary'),(2.7,.5,'Unstructured\nchaotic'),(3.4,3.6,'Structured\nchaotic')]:
                ax.text(x,y,label,ha='center',va='center',fontsize=8)
            for i, (g, s) in enumerate(data['conditions']):
                ax.plot(g,s,'o',ms=5,color=['#333','#007f99','#dc3552','#d88526'][i])
            g = data['gs']; th=data['theory']
            ax=axes[0,1]
            for state,color,label in [(0,'#007f99','Stationary DMF'),(1,'#dc3552','Chaotic DMF')]:
                k=np.where(th[:,3]==state,th[:,0],np.nan)
                ax.plot(g,k,color=color,label=label); ax.plot(g,-k,color=color)
            ax.set(xlabel='Random strength g',ylabel=r'$\kappa$',title='B  Structured overlap ($M_mM_n=2.2$)'); ax.legend(fontsize=8)
            ax=axes[0,2]
            ax.plot(g,th[:,1],label='Static heterogeneity',color='#007f99')
            ax.plot(g,th[:,2],label='Temporal variance',color='#dc3552')
            ax.set(xlabel='Random strength g',ylabel='Activation variance',title='C  Variance components (DMF)'); ax.legend(fontsize=8)
            for i,ax in enumerate(axes[1]):
                index=[1,2,3][i]
                ax.plot(data['times'],data['traces'][index],lw=.8)
                g,s=data['conditions'][index]
                ax.set(xlabel='Time / tau',ylabel='Activation $x_i$',title=f'{"DEF"[i]}  Simulation: g={g:g}, structure={s:g}')
        else:
            fig,axes=plt.subplots(2,3,figsize=(12,7.2),layout='constrained')
            t=data['times']; onset=float(data['onset']); offset=float(data['offset'])
            ax=axes[0,0]; ax.plot(t,data['traces'],lw=1); ax.axvspan(onset,offset,color='#ddd',alpha=.4)
            ax.set(xlabel='Time / tau',ylabel='Activation $x_i$',title='A  Step input: single-unit currents')
            ax=axes[0,1]; xy=data['coordinates']; ax.plot(xy[:,0],xy[:,1],color='#007f99')
            ax.scatter(*xy[0],s=25,color='k'); ax.set(xlabel='Input coordinate',ylabel='m coordinate',title='B  Population trajectory')
            ax=axes[0,2]; ax.bar(np.arange(len(data['explained']))+1,data['explained'],color='#007f99')
            ax.set(xlabel='PC number (z-scored currents)',ylabel='Explained variance',title='C  Trial-averaged dimensionality',ylim=(0,1))
            for kind,ax in enumerate(axes[1]):
                curves=data['curves'][kind]; amplitude=data['amplitudes']
                for j in range(3):
                    ax.plot(amplitude,curves[:,j], '--' if j == 1 else '-', lw=1.4,
                            color='#81b9c5' if j == 1 else '#007f99')
                ax.set(xlabel='Input amplitude',ylabel=r'$\kappa$',title=['D  Input orthogonal to n','E  Input aligned with n','F  Overlapping m, n: bistability'][kind])
                if kind == 0:
                    ax.set_ylim(-.4, .4)
                if kind == 2:
                    ax.plot([], [], '--', color='#81b9c5', label='Unstable branch')
                    ax.legend(fontsize=8, loc='lower right')
        fig.suptitle(f'Figure {1 if name=="spontaneous" else 2} recreation | N={meta["size"]}, realizations={meta["trials"]}, seed={meta["seed"]}',fontsize=12)
        output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(output,dpi=180); plt.close(fig)
    return output
