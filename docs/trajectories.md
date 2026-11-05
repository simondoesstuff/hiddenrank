# Figure 6 population trajectories

The integration figure adds the paper's state-space panels:

- **D:** projections of activations `x` onto the relevant input (`I_A` or `I_B`) and `w`, divided by N, separately for both contexts.
- **F:** scaffold coordinates of `x` during stimulation; older archives fall back to recurrent overlaps `κ = nᵀ tanh(x)/N`.
- D and F group paths by context and relevant stimulus strength. Each realization averages equally across available irrelevant strengths; thick paths average those paths across realizations. All trials, including failed Go decisions, remain included.
- D shows all sampled relevant strengths. F shows the nearest available strengths to the paper's 0.1, 0.4, 0.6, and 0.9, with actual values in its legend.
- Colors identify relevant strengths; F uses solid/dashed lines for A/B contexts. Circles and triangles mark stimulus onset and offset. Sparse archives average only the conditions available.
- The gray ring is the Gaussian mean-field scaffold of Eq. 156 with the Figure 6 parameters, before adding common readout, context, or stimulus components. It is not a fitted ring or a fixed-point prediction for the full task.

The output trace (C) uses the nearest available pair to the paper's `(0.6, 0.1)` in each context. Response maps use stored stimulus coordinates; representative condition subsets show bars instead of a fabricated grid.

Archives retain `projections[trial, condition, time, 3]` and optionally `kappa[trial, condition, time, 2]`. Existing archives without scaffold coordinates or overlaps display a regeneration notice. The gray ring can be supplied explicitly as `ring_background[point, 2]`.

New archives use `scaffold_coordinates[trial, condition, time, 2]` for panel F: activations projected onto each `m_j − 0.6w`, divided by that pattern's empirical squared norm. These state coordinates follow the paper's scaffold projection; `kappa` remains available separately and is used only for older archives. The theoretical ring uses the recorded architecture: `(r_m, r_n) = (1.38, 3)` for the supplement recipe or `(3, 1.38)` for the literal paper recipe. New figure headers name that architecture.
