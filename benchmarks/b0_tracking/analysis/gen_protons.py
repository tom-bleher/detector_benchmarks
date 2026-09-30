#!/usr/bin/env python3
"""Single protons from the origin into the B0 tracker, as HepMC3 ASCII: fixed momentum, polar angle about the ion
axis (25 mrad crossing angle, rotated about y) uniform in cos(theta) from 4 to 22 mrad, uniform azimuth.
No beam effects. usage: gen_protons.py MOMENTUM_GEV N_EVENTS SEED OUTPUT"""
import math
import sys

import numpy as np

M_PROTON = 0.93827209
CROSSING_ANGLE = -0.025
THETA_MIN, THETA_MAX = 0.004, 0.022


def main():
    p, n, seed, output = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    rng = np.random.default_rng(seed)
    theta = np.arccos(rng.uniform(math.cos(THETA_MAX), math.cos(THETA_MIN), n))
    phi = rng.uniform(-math.pi, math.pi, n)
    x, y, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)
    c, s = math.cos(CROSSING_ANGLE), math.sin(CROSSING_ANGLE)
    px, py, pz = p * (c * x + s * z), p * y, p * (-s * x + c * z)
    e = math.sqrt(p**2 + M_PROTON**2)
    lines = ["HepMC::Version 3.02.06", "HepMC::Asciiv3-START_EVENT_LISTING"]
    for i in range(n):
        # a zero-momentum status-4 particle carries the production vertex
        lines += [f"E {i} 1 2", "U GEV MM", "P 1 0 22 0 0 0 0 0 4", "V -1 0 [1] @ 0 0 0 0",
                  f"P 2 -1 2212 {px[i]:.9e} {py[i]:.9e} {pz[i]:.9e} {e:.9e} {M_PROTON:.9e} 1"]
    lines.append("HepMC::Asciiv3-END_EVENT_LISTING")
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
