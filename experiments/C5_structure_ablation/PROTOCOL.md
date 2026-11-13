# C5 protocol frozen before calibration and testing

Primary question: does the physical exponential-response likelihood add value beyond a direct step and a matched generic response family, and does explicitly naming separate Y/U states add observable information?

Observed Gaussian innovations are z_t=-(X_(t+1)-q X_t-c). The null mean is zero. All alternatives fit a positive amplitude A and an unknown past onset s, and maximize (positive dot(z,h))²/(variance*dot(h,h)). q,c,variance and the shared tau are frozen from C0. Only X enters the detector.

M3: for age>=0, h(age)=k-j exp(-age/tau), k=(1-q)/a, j=(exp(-dt/tau)-q)/(a-1/tau), a=-log(q)/dt. This is the exact observable kernel of the original source/transfer model.

M3-Ablation: remove explicit Y and U and directly posit the same exponential innovation mean kernel with an unknown decay constant. Its observable Gaussian likelihood, search parameters, optimized scores, alarm times and predictive distribution are exactly the same as M3. This is an intentional equivalence control, not a deliberately degraded model. A numerical superiority claim over this control would indicate a software or calibration mismatch, not a scientific contribution.

M3-Direct: h(age)=k immediately after the unknown onset, with no transfer state or time-scale parameter.

Generic GLR: h(age) is the exact AR-weighted integral of a linear ramp that reaches its plateau at R=2*kappa. Matching the response-area/mean-lag parameter kappa to M3's tau gives the same 41-point shape search grid, same onset candidates and positive analytic amplitude. It has the same effective parameter count as M3 but no exponential/first-order physical shape assumption.

Frozen M2 and frozen C4 multiscale CUSUM complete the controls. The C4 failure statistic AND its numerical thresholds remain frozen and common to all methods.

All detector thresholds are separately calibrated on new5000-trajectory normal pools at a whole25-time-unit false-alarm target5%. New normal test pools also contain5000 per noise. Two independent new event cohorts contain200 trajectories per condition each. Seeds are in config.py and are isolated from all earlier tasks.

Primary outcome: all-event coverage with warning lead at least0.5. Margins0.25 and1 are secondary. Four required M3 contrasts (M2, Direct, Ablation, matched GLR) in broad36-cell and predeclared slow-tau1/2 strength1 groups form8 comparisons: use99.375% stratified paired-cluster CIs (Bonferroni family), gain>=5percentage points, positive direction in both independent cohorts. Require comparable independent FPRs before declaring an operating-point advantage. Exact equivalence is explicitly permitted as a scientifically informative outcome; it cannot be hidden to force the strict all-control superiority criterion to pass.

Prediction readouts at12 and15 forecast2 time units ahead using only the same past X, without changing M3 after observing results. Report mean/median/tail errors and retain failures. Ordinary independent-trial confidence intervals within cells are descriptive; confirmatory decisions use innovation-cluster intervals because tau/strength conditions share innovations.

No C6 or real-system D in this task.
