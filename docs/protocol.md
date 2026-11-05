# Task protocol fidelity

Reference: Mastrogiuseppe & Ostojic (2018), Eqs. 151–161 and main-figure methods.
Supplementary code inspected at commit
[`35211877f641cd16ec239dc07330478d5b118175`](https://github.com/fmastrogiuseppe/LowRank/tree/35211877f641cd16ec239dc07330478d5b118175).

| Setting | Figure 4 source | Figure 5 source | Figure 6 paper |
| --- | --- | --- | --- |
| Discarded free transient | 10 | 10 | Not specified |
| Resting/context period | 7 | 15 | Not specified |
| Stimulus duration | 22 | 85 | Not specified |
| Sampling step | 0.1 | nominal 0.1 | Not specified |
| Stimulus noise intensity | 0.4 | 0 | 0.3 |
| Context amplitudes | None | (0.08, −0.14), reversed | (0.08, −0.18), reversed |

- Figure 4: [simulation](https://github.com/fmastrogiuseppe/LowRank/blob/35211877f641cd16ec239dc07330478d5b118175/4_DetectionTask/main_simulate.py), [integrator](https://github.com/fmastrogiuseppe/LowRank/blob/35211877f641cd16ec239dc07330478d5b118175/4_DetectionTask/fct_simulations.py).
  Initial state is `−m−n`. Independent quenched input `h` has unit variance and starts with the stimulus; it is absent at rest.
- Figure 5: [simulation](https://github.com/fmastrogiuseppe/LowRank/blob/35211877f641cd16ec239dc07330478d5b118175/5_CtxDiscriminationTask/main_simulate.py), [integrator](https://github.com/fmastrogiuseppe/LowRank/blob/35211877f641cd16ec239dc07330478d5b118175/5_CtxDiscriminationTask/fct_simulations.py).
  Initial state is `−m1−m2−n1−n2`. Context begins after the free transient and continues through the stimulus. Integration uses `odeint`; the reported sampling grid uses `linspace` with nominal step 0.1.
- The supplement contains no Figure 6 noisy rank-two script. Applying the Figure 4 noise convention or Figure 5 timing to Figure 6 is an explicit reconstruction choice.

## Noise and geometry

Figure 4 integrates scalar stimulus noise with `sqrt(dt) * sigma * Normal * I`.
Thus input coefficients supplied to an Euler drift step require noise SD `sigma / sqrt(dt)`.
The sampled stimulus amplitude SD is not `sigma` per integration step.

`protocol.task_geometry` implements paper Eqs. 151/155, using iid population covariances.
Two source discrepancies are resolved from the architecture:

- The Figure 5 script assigns context coefficient **1.6 to m and 3 to n**; the paper assigns **3 to m and 1.6 to n**. Their overlap agrees, but activity variance and cue overlap differ. Default `architecture="supplement"` reproduces the released script, and uses the analogous 1.38/3 swap for Figure 6. `architecture="paper"` retains literal paper parameters.
- Because the same `w` enters both `m` vectors, its contribution to activation variance is **bm²(k1+k2)²**. The printed Eq. 160 omits its cross term. Eq. 155 and the readout Eq. 158 require it.

For Figure 5 reproduction, the source architecture matters: the literal paper architecture loses its negative branch before either unit-strength stimulus, giving Go for both. Stationary continuation locates the preferred/irrelevant folds near 0.44/0.68 for paper parameters and 0.68/1.08 for source parameters (context A; zero irrelevant stimulus; step 0.01). The source architecture therefore preserves the Nogo response to the irrelevant unit-strength input. This diagnosis selects a figure recipe; it is not an empirical test of the paper's mathematics.

## Decision reference

The paper compares the offset endpoint to half the largest theoretical upper-state readout.
`theoretical_upper_readout` returns the largest **stable stationary raw readout**; the caller offsets it by the measured raw baseline before halving.
The default operating point is resting with the specified context and no quenched background. Go/NoGo defaults to the driven Go state because its resting output vanishes.
An explicit `stimulus=(a,b)` and `background` select a driven reference instead. Temporal noise does not enter these stationary equations.

The paper does not identify the upper-state operating point precisely enough to recover an unambiguous classifier from prose, and the released sample code contains no performance classifier. The resting reference is a documented convention, not an empirical maximum or a fitted decision threshold.
