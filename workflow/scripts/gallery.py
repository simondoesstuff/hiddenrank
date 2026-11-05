"""Build a small local gallery from Snakemake's completed figure targets."""
import html
import json
from pathlib import Path
import numpy as np

root = Path(snakemake.output[0]).parent
figures = [
    ('figure1_spontaneous', 'Spontaneous activity', 'Mean-field phase diagram, overlap and variance curves; finite-network traces.'),
    ('figure2_inputs', 'Input geometry', 'Step response, population trajectory, PCA, and mean-field response branches.'),
    ('figure3_go_nogo', 'Go / Nogo', 'Selective readout, population trajectory, and generalization to input mixtures.'),
    ('figure4_detection', 'Noisy detection', 'Bistable switching, state trajectories, and a psychometric curve.'),
    ('figure5_context', 'Context-dependent discrimination', 'The same readout changes stimulus preference with the context.'),
    ('figure6_integration', 'Context-dependent integration', 'Noisy readouts and response maps for both contextual cues.'),
]
sections = []
for stem, title, caption in figures:
    with np.load(root / 'simulations' / f'{stem}.npz') as data:
        meta = json.loads(str(data['metadata']))
    details = f"N={meta['size']}; {meta['trials']} realizations; seed={meta['seed']}; dt={meta['dt']}"
    sections.append(f'<section><h2>{html.escape(title)}</h2><p>{html.escape(caption)}</p>'
                    f'<a href="figures/{stem}.svg"><img src="figures/{stem}.png" alt="{html.escape(title)}"></a>'
                    f'<p class="meta">{details} · <a href="figures/{stem}.svg">SVG</a> · '
                    f'<a href="simulations/{stem}.npz">Data (NPZ)</a></p></section>')
root.mkdir(parents=True, exist_ok=True)
(root / 'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Low-rank RNN figure recreations</title>
<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:40px auto;padding:0 24px;color:#23313b;background:#f5f7f8}h1{font-size:30px}h2{font-size:21px}p{line-height:1.5}section{background:white;padding:20px;margin:24px 0;border-radius:10px}img{display:block;width:100%;height:auto}a{color:#14728a}.meta{font-size:13px;color:#57636b}</style>
<h1>Low-rank recurrent networks</h1><p>Key-panel recreations of Mastrogiuseppe &amp; Ostojic (2018), <i>Linking Connectivity, Dynamics, and Computations in Low-Rank Recurrent Neural Networks</i>.</p>
<p>Preview runs use smaller networks and explicitly labeled orthogonalized patterns for computational tasks. These reproduce the main qualitative panels; they are not pixel-identical copies or the published ensemble statistics. Decision thresholds use simulated resting attractors. Click a figure for its scalable version.</p>
''' + ''.join(sections) + '</html>')
