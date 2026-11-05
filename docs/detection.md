# Detection threshold and timescale (Figure 4F)

- Eq. 151: `m=w+rm*y`, `n=I+rn*y`; independent Gaussian patterns, `SI=Sw=1.2`, `rn=2`, `g=0.8`.
- **Threshold:** maximum input strength on the negative stationary branch (its saddle-node). Eq. 152 and the variance equation reduce this to one bounded scalar optimization; no network trials are needed.
- **Timescale:** positive resting branch, `tau=1/(1-lambda)` in single-neuron time units. `lambda` is the leading eigenvalue of Eq. 71's matrix `M`; the actual generator is `-Id+M` (Eq. 70).
- **Background:** upstream `4_DetectionTask/main_simulate.py` applies the independent unit-variance quenched input only during the stimulus. The fold uses variance one; resting stability uses zero. White temporal input noise is excluded from this deterministic threshold.

This distinction gives resting critical strength `rm*rn=1`, while bistability **during the stimulus** requires a larger overlap (approximately 1.784). Thus the threshold is zero over an initial interval even though resting relaxation already diverges at strength one. At `rm=2`, the predicted threshold is approximately `0.55`, close to the upstream hard-coded `0.54`. The paper's Figure 4F calculation script is absent from the public code; this reconstruction follows its published equations and the available stimulus/background protocol explicitly.

`run_detection_theory(path)` writes the curve and both critical values to NPZ; `plot_detection_theory(source, output)` writes the requested format; Snakemake requests both PNG and SVG. The plot clips the vertical timescale axis at 50 so the finite part remains legible; all computed timescales are preserved in NPZ.
