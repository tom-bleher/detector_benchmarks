#!/usr/bin/env python3
"""B0 tracking benchmark: hit-seeded B0 tracks (B0TrackerCKFTracks) for single protons (gen_protons.py).

- acceptance: the proton has B0 tracker hits in >= 3 stations; its hits are grouped into stations by z, a new
  station starting after a gap of more than 100 mm, so the grouping holds for one or two layers per station;
- efficiency: accepted protons with a track of which they contributed more than half of the measurements
  (B0TrackerCKFTrackAssociations weight > 0.5), over accepted protons;
- momentum resolution: half-width of the central 68 % of (p_reco - p_true) / p_true, for the matched track
  with the highest weight;
- fake rate: tracks with no particle contributing more than half of their measurements, over all tracks.

Writes b0_tracking.json (metrics, thresholds, pass/fail per check) and b0_tracking.pdf to the output directory
and exits with status 1 if a check fails.
usage: b0_tracking.py OUTPUT_DIR DETECTOR_CONFIG RECO_FILE [RECO_FILE ...]  (file names contain protons_<P>GeV)"""
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import uproot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CROSSING_ANGLE = -0.025
STATION_GAP_MM = 100.0
# Thresholds, with margin over the realistic B0 at the optimized placement (epic_craterlake, 2026-09-30):
# efficiency 98.6 / 99.2 %, fake rate 0.3 / 2.3 %, dp/p 68 % half-width 1.48 / 2.11 % at 15 / 41 GeV
MIN_EFFICIENCY = 0.95
MAX_FAKE_RATE = 0.05
MAX_DP_OVER_P = {15: 0.020, 41: 0.028}

BRANCHES = {
    "pdg": "MCParticles.PDG", "status": "MCParticles.generatorStatus",
    "px": "MCParticles.momentum.x", "py": "MCParticles.momentum.y", "pz": "MCParticles.momentum.z",
    "hit_z": "B0TrackerHits.position.z", "hit_particle": "_B0TrackerHits_particle.index",
    "trk_px": "B0TrackerCKFTracks.momentum.x", "trk_py": "B0TrackerCKFTracks.momentum.y",
    "trk_pz": "B0TrackerCKFTracks.momentum.z",
    "assoc_weight": "B0TrackerCKFTrackAssociations.weight",
    "assoc_track": "_B0TrackerCKFTrackAssociations_rec.index",
    "assoc_particle": "_B0TrackerCKFTrackAssociations_sim.index",
}


def stations(z):
    """Number of stations among hit z positions"""
    z = np.sort(z)
    return int(len(z) > 0) + int(np.sum(np.diff(z) > STATION_GAP_MM))


def polar_angle_about_ion_axis(p):
    axis = np.array([math.sin(CROSSING_ANGLE), 0.0, math.cos(CROSSING_ANGLE)])
    return np.arccos(np.clip(p @ axis / np.linalg.norm(p), -1.0, 1.0))


def analyze(path):
    ev = uproot.open(path)["events"].arrays(list(BRANCHES.values()), library="np")
    a = {k: ev[v] for k, v in BRANCHES.items()}
    theta, accepted, matched, dp = [], [], [], []
    n_tracks = n_fake = 0
    for i in range(len(a["pdg"])):
        thrown = np.flatnonzero((a["status"][i] == 1) & (a["pdg"][i] == 2212))
        if len(thrown) != 1:
            continue
        proton = thrown[0]
        p_true = np.array([a["px"][i][proton], a["py"][i][proton], a["pz"][i][proton]])
        theta.append(polar_angle_about_ion_axis(p_true))
        accepted.append(stations(a["hit_z"][i][a["hit_particle"][i] == proton]) >= 3)
        weights = {}
        for w, t, part in zip(a["assoc_weight"][i], a["assoc_track"][i], a["assoc_particle"][i]):
            weights.setdefault(t, {})[part] = w
        n_tracks += len(a["trk_px"][i])
        n_fake += sum(1 for t in range(len(a["trk_px"][i])) if max(weights.get(t, {}).values(), default=0) <= 0.5)
        mine = [(w[proton], t) for t, w in weights.items() if w.get(proton, 0) > 0.5]
        matched.append(bool(mine))
        if mine and accepted[-1]:
            t = max(mine)[1]
            p_reco = math.sqrt(a["trk_px"][i][t] ** 2 + a["trk_py"][i][t] ** 2 + a["trk_pz"][i][t] ** 2)
            dp.append(p_reco / np.linalg.norm(p_true) - 1)
    theta, accepted, matched, dp = np.array(theta), np.array(accepted), np.array(matched), np.array(dp)
    q16, q84 = np.percentile(dp, [16, 84]) if len(dp) else (np.nan, np.nan)
    return dict(
        events=int(len(theta)), accepted=int(accepted.sum()), matched=int((accepted & matched).sum()),
        efficiency=float((accepted & matched).sum() / accepted.sum()) if accepted.sum() else float("nan"),
        tracks=n_tracks, fake_rate=n_fake / n_tracks if n_tracks else float("nan"),
        dp_over_p_68=float((q84 - q16) / 2), dp_over_p_median=float(np.median(dp)) if len(dp) else float("nan"),
    ), theta, accepted, matched, dp


def main():
    out, config, files = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax_eff, ax_dp) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    results, ok = {}, True
    for path in sorted(files, key=lambda f: int(re.search(r"protons_(\d+)GeV", f).group(1))):
        p = int(re.search(r"protons_(\d+)GeV", path).group(1))
        m, theta, accepted, matched, dp = analyze(path)
        checks = {"efficiency": m["efficiency"] >= MIN_EFFICIENCY, "fake_rate": m["fake_rate"] <= MAX_FAKE_RATE}
        if p in MAX_DP_OVER_P:
            checks["dp_over_p_68"] = m["dp_over_p_68"] <= MAX_DP_OVER_P[p]
        results[f"{p}GeV"] = dict(metrics=m, thresholds=dict(min_efficiency=MIN_EFFICIENCY, max_fake_rate=MAX_FAKE_RATE,
                                                             max_dp_over_p_68=MAX_DP_OVER_P.get(p)),
                                  checks=checks, passed=all(checks.values()))
        ok &= all(checks.values())
        print(f"{p} GeV: " + ", ".join(f"{k} {v:.4g}" for k, v in m.items()) + f"; checks {checks}")
        bins = np.linspace(4, 22, 19)
        mrad = theta * 1e3
        n_acc = np.histogram(mrad[accepted], bins)[0]
        n_eff = np.histogram(mrad[accepted & matched], bins)[0]
        color = f"C{len(results) - 1}"
        with np.errstate(invalid="ignore", divide="ignore"):
            ax_eff.stairs(n_eff / n_acc, bins, baseline=None, color=color, label=f"{p} GeV, efficiency (accepted protons)")
            ax_eff.stairs(n_acc / np.histogram(mrad, bins)[0], bins, baseline=None, color=color, ls="--",
                          label=f"{p} GeV, acceptance (>= 3 stations)")
        ax_dp.hist(100 * dp, np.linspace(-10, 10, 81), histtype="step", color=color,
                   label=f"{p} GeV: 68 % half-width {100 * m['dp_over_p_68']:.2f} %")
    ax_eff.set(xlabel="polar angle about the ion axis [mrad]", ylabel="fraction", ylim=(0, 1.05))
    ax_dp.set(xlabel="(p_reco - p_true) / p_true [%]", ylabel="matched tracks")
    for ax in (ax_eff, ax_dp):
        ax.legend(fontsize="small")
    fig.suptitle(f"ePIC simulation, {config}: single protons 4-22 mrad about the ion axis, no beam effects; "
                 "hit-seeded B0TrackerCKFTracks")
    fig.savefig(out / "b0_tracking.pdf")
    (out / "b0_tracking.json").write_text(json.dumps(dict(detector_config=config, passed=ok, results=results), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
