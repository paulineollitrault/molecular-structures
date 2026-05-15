# Building a small reactive dataset (<1000 structures) from a single R/TS/P triplet

Reference paper: Schreiner et al., *Transition1x — a dataset for building generalizable reactive machine learning potentials*, Scientific Data 9:779 (2022), https://doi.org/10.1038/s41597-022-01870-w

This document summarizes the Transition1x methodology and adapts it for the case of a **single reaction** (reactant, product, and an existing transition state geometry) with a budget of **fewer than 1000 labeled structures**. The intended downstream use is training/fine-tuning an ML interatomic potential and/or building a reference dataset for benchmarking high-accuracy electronic structure methods.

---

## 1. What Transition1x does

The dataset generation pipeline for each of ~10k reactions:

1. **Relax endpoints** (reactant, product) until $|F| < 0.01$ eV/Å.
2. **Build initial path** in two segments: reactant → TS and TS → product, interpolated and pre-relaxed with **IDPP** (Image-Dependent Pair Potential, Smidstrup et al. JCP 2014). IDPP avoids bond crossings and gives physically reasonable starting images at negligible cost.
3. **NEB** with 10 images, spring constant $k = 0.1$ eV/Å², until $F_{\max,\perp} < 0.5$ eV/Å.
4. **CI-NEB** (Henkelman, Uberuaga, Jónsson, JCP 2000) until $F_{\max,\perp} < 0.05$ eV/Å.
5. **Diversity-based saving rule**: include the current iteration's full path in the dataset when the cumulative $\sum F_{\max}$ since the last saved iteration exceeds 0.1 eV/Å.
6. Discard reactions whose CI-NEB does not converge within 500 iterations.

**Note on CI-NEB for this work.** Schreiner runs CI-NEB because Transition1x is a general-purpose dataset where quantitative barriers are part of the deliverable. For the present use case (training an MLIP on a single known reaction with R/P/TS already in hand), CI-NEB is largely unnecessary — see Section 4 below.

DFT level: $\omega$B97x / 6-31G(d) in ORCA 5.0.2, BFGS optimizer in ASE ($\alpha = 70$, max step 0.03 Å).

The key insight: **NEB iterations themselves are the sampler**. Each BFGS step gives 10 new geometries (one per image). The path sweeps from a crude IDPP interpolation down to the MEP, covering a much broader perpendicular slice of the PES than just the converged MEP. The cumulative-$F_{\max}$ rule naturally biases the saved set toward *moving* (early) iterations and away from redundant near-converged ones.

---

## 2. Definition of $F_{\max}$

$F_{\max}$ is the **largest perpendicular force component on any image along the NEB path** at the current iteration.

In NEB, each image $i$ feels two forces:

$$\mathbf{F}_i = \mathbf{F}_i^{\perp} + \mathbf{F}_i^{\parallel,\text{spring}}$$

- $\mathbf{F}_i^{\perp} = -\nabla E(\mathbf{R}_i) + (\nabla E(\mathbf{R}_i) \cdot \hat{\boldsymbol\tau}_i)\,\hat{\boldsymbol\tau}_i$ — true PES gradient with its component along the path tangent $\hat{\boldsymbol\tau}_i$ projected out. This is what moves the image off the current path toward the MEP.
- $\mathbf{F}_i^{\parallel,\text{spring}} = k(|\mathbf{R}_{i+1} - \mathbf{R}_i| - |\mathbf{R}_i - \mathbf{R}_{i-1}|)\,\hat{\boldsymbol\tau}_i$ — spring force along the tangent; redistributes images, doesn't enter $F_{\max}$.

Convergence is tested on $\mathbf{F}^{\perp}$ only: at the true MEP this vanishes by definition, while the spring force generically does not.

Reduction to scalar (ASE convention):

$$F_{\max} = \max_{i \in \text{images}} \;\max_{a \in \text{atoms}} \; \|\mathbf{F}_{i,a}^{\perp}\|_2$$

Units in the paper: eV/Å.

Schreiner's thresholds in context:
- **0.5 eV/Å** — loose intermediate target; band qualitatively in the right valley.
- **0.05 eV/Å** — Schreiner's tight CI-NEB convergence for resolving the saddle.
- **0.01 eV/Å** — standard tight minimum-convergence for endpoint relaxations.

For the present work the relevant stopping point is **$\approx 0.1$–$0.2$ eV/Å** (loose, plain NEB only, see Section 4); the 0.05 eV/Å CI-NEB target is not needed.

Why $\sum F_{\max}$ works as a diversity criterion: BFGS steps scale like $\Delta \mathbf{R} \sim H^{-1}\mathbf{F}$, so cumulative $F_{\max}$ is a rough proxy for cumulative displacement of the band in configuration space — exactly the quantity to threshold for sampling diversity. An explicit RMSD between consecutive saved paths is more rigorous; $\sum F_{\max}$ is essentially free since the optimizer computes it anyway.

---

## 3. Adapting to one reaction, <1000 structures

Blindly copying Transition1x for a single reaction would give too few points (one converged NEB ≈ 20–60 saved iterations × 10 images = 200–600 geometries, mostly along the MEP). The budget allows richer sampling. Recommended layered strategy:

### Layer 1 — NEB trajectory snapshots (the Transition1x core)

- 10–16 images, $k = 0.05$–$0.1$ eV/Å², from IDPP-interpolated path.
- **Plain NEB only**, stop at $F_{\max,\perp} \approx 0.1$–$0.2$ eV/Å. Tighter convergence wastes BFGS steps on geometries that the diversity rule won't save anyway. CI-NEB is not needed here (see Section 4).
- Apply the cumulative-$F_{\max}$ saving rule with threshold tuned to the budget. Lower threshold → more snapshots.
- Expected yield for one reaction: ~150–400 geometries.

Gives stratified sampling along the reaction coordinate plus a perpendicular cross-section (intermediate NEB iterations are off the MEP, which is exactly what teaches the MLIP the width of the reactive channel).

### Layer 2 — Normal-mode displacements at the three stationary points

Reactant, product, and TS are underrepresented by NEB snapshots alone. Explicit local-curvature sampling matters for ML potentials.

- Compute Hessian at R, P, TS (TS Hessian probably already available from the saddle search).
- Displace along each normal mode at thermal amplitudes: $q_i \sim \mathcal{N}(0, \sigma_i^2)$ with $\sigma_i = \sqrt{k_B T / \omega_i^2}$ at $T = 300$–$1000$ K. Alternative: fixed amplitudes $\pm 0.05, \pm 0.10, \pm 0.20$ Å·amu$^{1/2}$.
- At the TS, **explicitly scan the imaginary mode** at several amplitudes — samples reaction-coordinate curvature in the most anharmonic PES region.
- Budget: ~50–100 samples per stationary point = 150–300 geometries.

This is ANI-1x-style normal-mode sampling (Smith, Nebgen, Lubbers, Isayev, Roitberg, JCP 2018).

### Layer 3 — Off-path perturbations of NEB images

Addresses the well-known failure mode where NEB-only datasets give models that work along the MEP but blow up just off it.

- Random Cartesian rattles, amplitude ~0.05–0.10 Å, applied to a subset of NEB images (e.g. every other image of selected iterations).
- Budget: ~100–200 geometries.

### Layer 4 (optional) — Short AIMD or metadynamics from R, P, TS

Anharmonic sampling that normal modes miss.

- 1–5 ps trajectories at 300–500 K with random initial velocities.
- From TS, downhill MD splits roughly 50/50 into both basins and traces the actual reactive flux corridor.
- Subsample every N steps; apply a diversity filter (CUR on SOAP, or farthest-point sampling on internal coordinates) to retain only distinct structures.
- Budget: 100–200 geometries after pruning.

### Suggested budget split (~1000 structures)

| Source | Count | Purpose |
|---|---|---|
| NEB iteration snapshots (Layer 1) | ~400 | MEP + valley structure |
| Normal-mode at R, P, TS (Layer 2) | ~300 (100 each) | Harmonic basin + TS curvature |
| TS imaginary-mode scan, large amplitude | ~50 | Anharmonic reaction-coordinate region |
| Off-path rattles of NEB images (Layer 3) | ~150 | Valley width |
| MD or metadynamics (Layer 4, optional) | ~100 | Anharmonic dynamical sampling |

After collecting candidates, apply **farthest-point sampling in a descriptor space** (SOAP, ACSF, or internal-coordinate vectors) to enforce coverage and remove redundancy, especially across layers that overlap (NEB endpoints vs. normal modes at R/P).

---

## 4. NEB vs CI-NEB — what each gives you, and why CI-NEB is skipped here

**Plain NEB.** Every image feels $\mathbf{F}_i = \mathbf{F}_i^{\perp} + \mathbf{F}_i^{\parallel,\text{spring}}$: the PES gradient projected perpendicular to the path tangent, plus a spring force along the tangent that keeps images evenly spaced. At convergence the band sits in the local minimum *perpendicular* to itself, and the highest-energy image is *near* the saddle but not on it. Barrier height from plain NEB is typically off by 0.05–0.3 eV depending on path curvature and image density.

**CI-NEB** (Henkelman, Uberuaga, Jónsson, JCP 113, 9901, 2000). For the single highest-energy image $i_{\max}$, the force law changes to

$$\mathbf{F}_{i_{\max}}^{\text{CI}} = -\nabla E(\mathbf{R}_{i_{\max}}) + 2\,(\nabla E(\mathbf{R}_{i_{\max}}) \cdot \hat{\boldsymbol\tau}_{i_{\max}})\,\hat{\boldsymbol\tau}_{i_{\max}}$$

The spring force is dropped, and the tangential gradient component is *inverted* rather than removed. Effect: the climbing image is pushed uphill along the tangent and downhill perpendicular to it — exactly the signature of a saddle-point search. At convergence it sits on the true first-order saddle.

**Why a two-stage NEB → CI-NEB protocol?** If CI-NEB is turned on from the start with a bad initial path, the climbing image climbs along the *current* tangent, which can be the wrong direction. It may drag the band onto a different saddle. Plain NEB first ensures the band is in the right valley before letting any image climb.

### Why CI-NEB is *not* needed for this MLIP dataset

CI-NEB exists to give a quantitative saddle and barrier. For training an MLIP, the goals are different:

1. **Coverage** of the relevant region of configuration space.
2. **Label accuracy** at each sampled geometry (from the high-level single-points, not the driver).
3. **Smoothness** of sampling (no holes).

CI-NEB doesn't help with any of these. It refines the position of one specific geometry — the saddle — which is mild oversampling of one point, useful for locating the TS but redundant for fitting a potential. The geometries most valuable for MLIP training are the off-MEP images from *early* NEB iterations, which sweep through perpendicular slices of the reactive valley. CI-NEB only modifies behavior in the *late* phase, where the cumulative-$F_{\max}$ rule already down-weights snapshots.

The TS is also already known from a prior saddle search, so re-locating it is not needed.

### When CI-NEB would still be worth running

Three narrow cases:

1. **TS sanity check.** If there's any doubt about the existing TS (different method, basis, or solvent than the current driver), run a short separate CI-NEB or a dedicated saddle optimizer (P-RFO, dimer) on the TS guess at the current level. This is **one calculation outside the data pipeline**, not part of the sampling sweep.
2. **Barrier metadata.** If a converged barrier height needs to be reported alongside the dataset, do one tight CI-NEB or a single-point on a refined TS — again, separate from the sampling.
3. **Downstream TS-prediction evaluation.** If the MLIP will be evaluated on its ability to locate the TS via its own NEB run, the high-level *labels* on TS-region geometries must be accurate. The labels come from the single-points; the driver still doesn't need CI-NEB. Explicit imaginary-mode sampling at the TS (Layer 2) gives better TS-region coverage than a refined CI-NEB anyway.

**Bottom line.** Run plain NEB for sampling. Stop at $F_{\max,\perp} \approx 0.1$–$0.2$ eV/Å. Don't include CI-NEB in the data-generation loop. If a quantitative barrier or a TS double-check is needed, run those separately as one-shot calculations.

---

## 5. Practical points for the multireference / strongly-correlated case

1. **Multireference character along the path.** If the TS or any NEB image has significant multireference character (high $T_1$ diagnostic, occupation numbers deviating from 0/2), single-reference DFT or CCSD(T) labels will be unreliable in the middle of the path. Run a cheap multireference diagnostic (FOD plot, $T_1$/$D_1$, or occupation numbers from a CASSCF or DMRG calculation on a small active space) on a sparse subset *before* committing the full label budget. This is exactly the MOBH35 reactions 8/9 issue.

2. **Decouple driver from reference.** Use cheap method to *drive sampling* (DFT, or an MLIP like UMA/MACE/Allegro), then run *expensive single-points* (DLPNO-CCSD(T), DLPNO-TCCSD, DMRG-NEVPT2) on the chosen geometries as labels. This is what makes the <1000-structure budget tractable: sampling stays cheap, only the labels are expensive.

3. **Sanity checks on saved geometries before labeling.**
   - Bond connectivity reasonable (no atoms < 0.7 Å apart, no fragments unexpectedly dissociated unless intentional).
   - Energy range spans the relevant chemistry — plot histogram of driver energies relative to reactant; flag outliers.
   - Spin state consistent across the path (for open-shell cases, $\langle S^2 \rangle$ within expected range).

---

## 6. Implementation pointers

- **NEB driver**: ASE (`ase.neb.NEB`, `ase.neb.IDPP`) is the path of least resistance. BFGS or FIRE optimizer. Use plain NEB only (`climb=False`); see Section 4.
- **DFT engine for the driver**: ORCA, PySCF, or Q-Chem via ASE calculator. $\omega$B97X-D3 / def2-SVP is a reasonable cheap choice; the original Grambow paper used $\omega$B97X-D3 / def2-TZVP.
- **Saving rule**: tap into the optimizer's per-step callback; accumulate $F_{\max}$ between saves; dump the current images to an HDF5 group (matching Transition1x's structure: `positions (n_images, n_atoms, 3)`, `energy (n_images,)`, `forces (n_images, n_atoms, 3)`, `atomic_numbers (n_atoms,)`) when the cumulative threshold trips.
- **Normal-mode sampling**: ASE `Vibrations` for Hessian, then displace along eigenvectors. For thermal sampling, use $\sigma_i = \sqrt{k_B T} / \omega_i$ in mass-weighted coordinates.
- **Diversity filtering**: `dscribe` for SOAP descriptors, then farthest-point sampling (e.g. `scikit-learn-extra`'s `KMedoids` or a simple greedy FPS loop). For internal coordinates, build a vector of bond lengths / angles around the reactive atoms and use the same FPS.
- **Labeling**: ORCA for DLPNO-CCSD(T); PySCF + block2 for DMRG-NEVPT2; batch as independent single-point jobs (trivially parallel).

---

## 7. Tasks for Claude Code

Implementation order, with the high-leverage pieces first:

1. **Plain-NEB pipeline** with the cumulative-$F_{\max}$ saving rule, IDPP initial path, and HDF5 output mirroring the Transition1x schema. Stop at $F_{\max,\perp} \approx 0.1$–$0.2$ eV/Å; no CI-NEB stage.
2. **Normal-mode sampler** at R, P, TS (including TS imaginary-mode scan).
3. **Off-path rattle generator** acting on saved NEB images.
4. **Diversity filter** (SOAP + FPS) to enforce ≤1000 total and remove redundancy across layers.
5. **Labeling driver** to dispatch single-points (DLPNO-CCSD(T) and/or DMRG-NEVPT2) on the final set.
6. (Optional) Short AIMD from R, P, TS with diversity-pruned subsampling.

Inputs to the pipeline: reactant `.xyz`, product `.xyz`, TS `.xyz`, charge, multiplicity, driver-level DFT settings, label-level method settings, total structure budget.

Outputs: HDF5 file with all geometries, driver energies/forces, and (once labeled) high-level energies; plus a provenance log indicating which layer each structure came from.
