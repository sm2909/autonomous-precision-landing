#!/usr/bin/env python3
"""
eval_landing.py -- auxiliary batch evaluator (does NOT modify controller.py or the arena)
===========================================================================================

Runs your unmodified FlightController inside a headless (p.DIRECT) copy of the
arena physics loop and produces every number that the report leaves as [TO FILL]:

  * Section 10 table  : touch-down speed, tilt, offset, times, success rate, dropout recovery
  * Section 8 log     : one-at-a-time parameter sensitivity sweep ("tuning log", optional)
  * Dependency line   : exact Python / NumPy / OpenCV / PyBullet versions
  * Figure            : telemetry plot of a representative run (altitude, vz, error, tilt, state)

It re-uses the arena's own spawn_rover(), spawn_quadcopter() and CompetitionScorer
(imported from your arena file), and replicates its loop order, rover kinematics and
wind model exactly. Only the GUI, cv2.imshow and time.sleep are removed.

Usage (put this file next to controller.py and your arena file):
    python eval_landing.py                       # 20 nominal runs + dropout scenarios
    python eval_landing.py --runs 40 --workers 4
    python eval_landing.py --sweep --sweep-runs 8    # add the tuning sensitivity study
    python eval_landing.py --arena local_arena       # if your arena file has another name
    python eval_landing.py --renderer opengl         # GUI + OpenGL camera (slow, single worker)

Outputs go to ./eval_out/ : results.csv, summary.json, report_tables.md, run0_telemetry.png
"""
import argparse, csv, importlib, json, math, os, platform, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())

DT = 1.0 / 240.0
MAX_SIM_TIME = 60.0            # a run that has not finished by then is a TIMEOUT
CONTACT_Z = 0.16               # scorer: ROVER_SURFACE_Z + 0.01

# ---------------------------------------------------------------------------
# scenarios: name -> dict(start_offset, blank rule)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "nominal":       dict(desc="arena defaults + randomised start / rover phase / wind"),
    "far_start":     dict(desc="drone starts 6-8 m away (marker not visible: SEARCH spiral)", far=True),
    "dropout_track": dict(desc="camera blacked out for 1.0 s, 2 s after TRACK begins", blank="track"),
    "blackout_low":  dict(desc="camera permanently blacked out below 1.0 m (BLIND_DESCEND)", blank="low"),
}

# derived-gain aware overrides for the sweep (applied to the controller module in-process only)
def apply_overrides(cm, ov):
    for k, v in ov.items():
        setattr(cm, k, v)
    if "WN_XY" in ov or "ZETA_XY" in ov:
        cm.KP_POS_XY = cm.WN_XY / (2.0 * cm.ZETA_XY)
        cm.KP_VEL_XY = 2.0 * cm.ZETA_XY * cm.WN_XY
    if "WN_ATT" in ov or "ZETA_ATT" in ov:
        cm.KP_ATT = cm.WN_ATT / (2.0 * cm.ZETA_ATT)
        cm.KP_RATE = 2.0 * cm.ZETA_ATT * cm.WN_ATT
    if "TILT_MAX_LAND_DEG" in ov:
        cm.TILT_MAX_LAND = math.radians(ov["TILT_MAX_LAND_DEG"])

SWEEPS = [
    ("Z_CUT [m]",        "Z_CUT",       [0.148, 0.152, 0.156, 0.160, 0.165]),
    ("V_TOUCH [m/s]",    "V_TOUCH",     [0.15, 0.25, 0.40, 0.60]),
    ("WN_XY [rad/s]",    "WN_XY",       [2.5, 3.6, 5.0]),
    ("KI_VEL_XY",        "KI_VEL_XY",   [0.0, 1.5, 3.0]),
]


def one_run(job):
    """Executes a single episode in its own PyBullet client. Returns a result dict."""
    import pybullet as p
    import pybullet_data
    import cv2
    cv2.setNumThreads(1)

    seed, scenario, overrides, arena_name, renderer, out_dir, keep = job
    rng = random.Random(seed)
    arena = importlib.import_module(arena_name)
    cm = importlib.import_module("controller")
    saved = {}
    for k in list(overrides) + ["KP_POS_XY", "KP_VEL_XY", "KP_ATT", "KP_RATE", "TILT_MAX_LAND"]:
        if hasattr(cm, k): saved[k] = getattr(cm, k)
    apply_overrides(cm, overrides)

    class Rec(arena.CompetitionScorer):
        outcome, reason, end_err = None, "", None
        def _trigger_failure(self, reason):
            self.run_complete, self.outcome, self.reason = True, "FAIL", reason
        def _trigger_success(self, final_error):
            self.run_complete, self.outcome, self.end_err = True, "SUCCESS", final_error

    p.connect(p.GUI if renderer == "opengl" else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.loadURDF("plane.urdf")
    rover_id = arena.spawn_rover()
    drone_id = arena.spawn_quadcopter()

    # --- per-run randomisation (run 0 of 'nominal' = exact arena defaults) -------
    default_run = (scenario == "nominal" and seed == 0)
    t_off = 0.0 if default_run else rng.uniform(0.0, 60.0)
    if SCENARIOS[scenario].get("far"):
        ang = rng.uniform(0, 2 * math.pi); r = rng.uniform(6.0, 8.0)
        start = [r * math.cos(ang), r * math.sin(ang), 3.0]
    elif default_run:
        start = [0.0, 0.0, 3.0]
    else:
        start = [rng.uniform(-1, 1), rng.uniform(-1, 1), 3.0]
    p.resetBasePositionAndOrientation(drone_id, start, [0, 0, 0, 1])

    def camera():
        if renderer == "opengl":
            return arena.get_drone_camera_image(drone_id)
        pos, ori = p.getBasePositionAndOrientation(drone_id)
        R = np.array(p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        cam = np.array(pos) + R.dot([0, 0, -0.1])
        view = p.computeViewMatrix(cam, cam + R.dot([0, 0, -1]), R.dot([1, 0, 0]))
        proj = p.computeProjectionMatrixFOV(60, 320 / 240, 0.1, 20.0)
        _, _, rgb, _, _ = p.getCameraImage(320, 240, view, proj, renderer=p.ER_TINY_RENDERER)
        return cv2.cvtColor(np.reshape(rgb, (240, 320, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)

    brain = cm.FlightController()
    scorer = Rec()
    wrng = random.Random(seed * 7919 + 13)            # wind generator, reproducible per seed
    cur_w, tgt_w = [0.0] * 3, [0.0] * 3

    tel = []      # rows: 0 t | 1-3 pos | 4-6 vel | 7 tilt | 8 xy err | 9 state | 10-11 rover xy
    t = 0.0; step = 0
    t_track = t_descend = t_landed = None
    used_blind = False
    blank_until = -1.0; blank_done = False; blackout = False
    first_track_t = None

    while not scorer.run_complete and t < MAX_SIM_TIME:
        dpos, dori = p.getBasePositionAndOrientation(drone_id)
        dvel, _ = p.getBaseVelocity(drone_id)
        rpos, _ = p.getBasePositionAndOrientation(rover_id)
        tr = t + t_off
        vx = 1.2 * math.cos(tr * 0.4) + 0.5 * math.cos(tr * 1.1)
        vy = 1.2 * math.sin(tr * 0.3) + 0.5 * math.sin(tr * 0.9)
        p.resetBaseVelocity(rover_id, linearVelocity=[vx, vy, 0])
        frame = camera()

        # --- forced camera dropouts (only alters the image handed to the controller) ---
        rule = SCENARIOS[scenario].get("blank")
        if rule == "track":
            if first_track_t is None and brain.state != brain.SEARCH:
                first_track_t = t
            if (not blank_done) and first_track_t is not None and t - first_track_t >= 2.0:
                blank_until, blank_done = t + 1.0, True
        if rule == "low" and dpos[2] < 1.0:
            blackout = True
        if blackout or t < blank_until:
            frame = np.zeros_like(frame)

        scorer.update(dpos, dvel, dori, rpos, DT)
        e = math.hypot(dpos[0] - rpos[0], dpos[1] - rpos[1])
        eul = p.getEulerFromQuaternion(dori)
        tel.append((t, *dpos, *dvel, max(abs(eul[0]), abs(eul[1])), e, brain.state, *rpos[:2]))
        if scorer.run_complete:
            break

        # wind: identical to the arena
        if int(t / DT) % 120 == 0:
            tgt_w = [wrng.uniform(-2, 2), wrng.uniform(-2, 2), wrng.uniform(-0.5, 0.5)]
        for i in range(3):
            cur_w[i] += (tgt_w[i] - cur_w[i]) * 0.02
        p.applyExternalForce(drone_id, -1, cur_w, [0, 0, 0], p.LINK_FRAME)

        f1, f2, f3, f4 = brain.calculate_forces(frame, dpos, dori, dvel)
        L = 0.25
        for f, pt in ((f1, [L, L, 0]), (f2, [-L, L, 0]), (f3, [L, -L, 0]), (f4, [-L, -L, 0])):
            p.applyExternalForce(drone_id, -1, [0, 0, f], pt, p.LINK_FRAME)
        p.stepSimulation()

        if t_track is None and brain.state != brain.SEARCH: t_track = t
        if t_descend is None and brain.state == brain.DESCEND: t_descend = t
        if brain.state == brain.BLIND_DESCEND: used_blind = True
        if t_landed is None and brain.state == brain.LANDED: t_landed = t
        t += DT; step += 1

    p.disconnect()
    for k, v in saved.items(): setattr(cm, k, v)      # restore module defaults

    A = np.array(tel)
    outcome = scorer.outcome or "TIMEOUT"
    # contact sample = first step the scorer's window (z <= 0.16, err < 0.5) is entered
    idx = np.where((A[:, 3] <= CONTACT_Z) & (A[:, 8] < 0.5))[0]
    ci = int(idx[0]) if len(idx) else len(A) - 1
    st = A[:, 9].astype(int)
    desc_mask = (st == 2) | (st == 3)
    res = dict(seed=seed, scenario=scenario, overrides=json.dumps(overrides), outcome=outcome, reason=scorer.reason,
               vz_contact=float(A[ci, 6]), tilt_contact_deg=math.degrees(A[ci, 7]),
               err_contact_cm=float(A[ci, 8] * 100), err_final_cm=float(A[-1, 8] * 100),
               peak_down_speed=float(-A[desc_mask, 6].min()) if desc_mask.any() else float("nan"),
               max_tilt_deg=math.degrees(A[:, 7].max()),
               t_acquire=t_track, t_descend=t_descend, t_cutoff=t_landed, t_end=float(A[-1, 0]),
               used_blind=used_blind, start_dist=math.hypot(start[0], start[1]))
    if keep:
        np.save(os.path.join(out_dir, f"telemetry_{scenario}_{seed}.npy"), A)
    return res


# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 2
    ph = k / n; d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def summarise(rows):
    n = len(rows); k = sum(r["outcome"] == "SUCCESS" for r in rows); lo, hi = wilson(k, n)
    ok = [r for r in rows if r["outcome"] == "SUCCESS"]
    g = lambda key, src=None: [r[key] for r in (src if src is not None else rows) if r[key] is not None]
    return dict(n=n, success=k, rate=k / n if n else float("nan"), ci_lo=lo, ci_hi=hi,
                fail_reasons=sorted({r["reason"] for r in rows if r["outcome"] != "SUCCESS"}),
                vz_mean=float(np.mean(g("vz_contact"))) if n else None, vz_worst=float(min(g("vz_contact"))) if n else None,
                tilt_mean=float(np.mean(g("tilt_contact_deg"))), tilt_worst=float(max(g("tilt_contact_deg"))),
                err_mean=float(np.mean(g("err_contact_cm"))), err_worst=float(max(g("err_contact_cm"))),
                err_final_mean=float(np.mean(g("err_final_cm"))),
                t_acq_mean=float(np.mean(g("t_acquire"))) if g("t_acquire") else None,
                t_acq_worst=float(max(g("t_acquire"))) if g("t_acquire") else None,
                t_end_mean=float(np.mean(g("t_end", ok))) if ok else None,
                t_end_worst=float(max(g("t_end", ok))) if ok else None,
                blind_frac=float(np.mean([r["used_blind"] for r in rows])),
                peak_down_worst=float(max(g("peak_down_speed"))) if g("peak_down_speed") else None)

def fmt(x, f="{:.2f}"): return "n/a" if x is None else f.format(x)

def versions():
    import cv2
    try: import pybullet as pb; pbv = getattr(pb, "__version__", "installed (pybullet.getAPIVersion=%s)" % pb.getAPIVersion())
    except Exception: pbv = "n/a"
    try: aruco = "cv2.aruco OK" if hasattr(cv2, "aruco") else "cv2.aruco MISSING"
    except Exception: aruco = "?"
    return dict(python=platform.python_version(), numpy=np.__version__, opencv=cv2.__version__, aruco=aruco,
                pybullet=pbv, platform=platform.platform())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena", default="local_arena_wind", help="module name of your arena file (default local_arena_wind)")
    ap.add_argument("--runs", type=int, default=20, help="nominal runs")
    ap.add_argument("--scenario-runs", type=int, default=6, help="runs for each dropout / far-start scenario")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--sweep", action="store_true", help="also run the parameter sensitivity study (tuning log)")
    ap.add_argument("--sweep-runs", type=int, default=8)
    ap.add_argument("--renderer", choices=["tiny", "opengl"], default="tiny")
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--seed0", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.renderer == "opengl": a.workers = 1

    jobs = []
    for s in range(a.runs):
        jobs.append((a.seed0 + s, "nominal", {}, a.arena, a.renderer, a.out, s == 0))
    for sc in ("far_start", "dropout_track", "blackout_low"):
        for s in range(a.scenario_runs):
            jobs.append((a.seed0 + 1000 + s, sc, {}, a.arena, a.renderer, a.out, False))
    if a.sweep:
        for label, key, vals in SWEEPS:
            for v in vals:
                for s in range(a.sweep_runs):
                    jobs.append((a.seed0 + 2000 + s, "nominal", {key: v}, a.arena, a.renderer, a.out, False))
        for s in range(a.sweep_runs):      # baseline for the sweep with identical seeds
            jobs.append((a.seed0 + 2000 + s, "nominal", {}, a.arena, a.renderer, a.out, False))

    print(f"[eval] {len(jobs)} episodes, {a.workers} worker(s), renderer={a.renderer}")
    t0 = time.time(); results = []
    if a.workers == 1:
        for i, j in enumerate(jobs):
            results.append(one_run(j)); print(f"  {i+1}/{len(jobs)} {j[1]} seed={j[0]} -> {results[-1]['outcome']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one_run, j): j for j in jobs}
            for i, f in enumerate(as_completed(futs)):
                r = f.result(); results.append(r)
                print(f"  {i+1}/{len(jobs)} {r['scenario']} seed={r['seed']} -> {r['outcome']}", flush=True)
    print(f"[eval] done in {time.time()-t0:.0f} s")

    keys = list(results[0].keys())
    with open(os.path.join(a.out, "results.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(sorted(results, key=lambda r: (r["scenario"], r["overrides"], r["seed"])))

    sel = lambda sc, ov="{}": [r for r in results if r["scenario"] == sc and r["overrides"] == ov and (sc != "nominal" or r["seed"] < a.seed0 + 1000)]
    S = {sc: summarise(sel(sc)) for sc in SCENARIOS if sel(sc)}
    V = versions()
    nom = S["nominal"]

    md = []
    md.append("## Section 10: closed-loop results (copy into the report)\n")
    md.append(f"Nominal: {nom['n']} runs, randomised start position (+-1 m), rover phase and wind seed (run 0 = exact arena defaults). "
              f"Headless PyBullet, `{a.renderer}` camera renderer.\n")
    md.append("| Metric | Requirement | Result |\n|---|---|---|")
    md.append(f"| Touch-down vertical speed at scorer contact check [m/s] | >= -0.65 | mean {fmt(nom['vz_mean'])}, worst {fmt(nom['vz_worst'])} |")
    md.append(f"| Peak descent speed during DESCEND [m/s] | (info) | worst {fmt(nom['peak_down_worst'])} |")
    md.append(f"| Chassis tilt at touch-down [deg] | <= 30 | mean {fmt(nom['tilt_mean'],'{:.1f}')}, worst {fmt(nom['tilt_worst'],'{:.1f}')} |")
    md.append(f"| Horizontal offset from pad centre at contact [cm] | minimise (scorer gate 50 cm) | mean {fmt(nom['err_mean'],'{:.1f}')}, worst {fmt(nom['err_worst'],'{:.1f}')} |")
    md.append(f"| Time to acquire marker (leave SEARCH) [s] | n/a | mean {fmt(nom['t_acq_mean'])}, worst {fmt(nom['t_acq_worst'])} |")
    md.append(f"| Time to land (scorer success) [s] | n/a | mean {fmt(nom['t_end_mean'],'{:.1f}')}, worst {fmt(nom['t_end_worst'],'{:.1f}')} |")
    md.append(f"| Success rate, nominal | 100 % | {nom['success']}/{nom['n']} = {100*nom['rate']:.0f} % (95 % Wilson CI {100*nom['ci_lo']:.0f}-{100*nom['ci_hi']:.0f} %) |")
    if nom["fail_reasons"]: md.append(f"| Failure reasons | | {'; '.join(nom['fail_reasons'])} |")
    md.append("")
    md.append("### Robustness scenarios\n")
    md.append("| Scenario | Description | Success | Worst vz [m/s] | Worst offset [cm] | Mean time to land [s] | BLIND_DESCEND used |\n|---|---|---|---|---|---|---|")
    for sc, s in S.items():
        md.append(f"| {sc} | {SCENARIOS[sc]['desc']} | {s['success']}/{s['n']} | {fmt(s['vz_worst'])} | {fmt(s['err_worst'],'{:.1f}')} | "
                  f"{fmt(s['t_end_mean'],'{:.1f}')} | {100*s['blind_frac']:.0f} % |")
    md.append("")

    if a.sweep:
        md.append("## Section 8: parameter sensitivity (tuning log)\n")
        md.append(f"One parameter changed at a time, {a.sweep_runs} identical seeds per value (paired with the baseline).\n")
        md.append("| Parameter | Value | Success | Mean vz at contact [m/s] | Worst vz [m/s] | Mean offset [cm] | Mean land time [s] |\n|---|---|---|---|---|---|---|")
        base = summarise([r for r in results if r["overrides"] == "{}" and 2000 <= r["seed"] - a.seed0 < 3000])
        md.append(f"| baseline | (as submitted) | {base['success']}/{base['n']} | {fmt(base['vz_mean'])} | {fmt(base['vz_worst'])} | {fmt(base['err_mean'],'{:.1f}')} | {fmt(base['t_end_mean'],'{:.1f}')} |")
        for label, key, vals in SWEEPS:
            for v in vals:
                rr = [r for r in results if r["overrides"] == json.dumps({key: v})]
                if not rr: continue
                s = summarise(rr)
                md.append(f"| {label} | {v} | {s['success']}/{s['n']} | {fmt(s['vz_mean'])} | {fmt(s['vz_worst'])} | {fmt(s['err_mean'],'{:.1f}')} | {fmt(s['t_end_mean'],'{:.1f}')} |")
        md.append("")

    md.append("## Dependencies line (Appendix A)\n")
    md.append(f"Python {V['python']}, NumPy {V['numpy']}, OpenCV {V['opencv']} ({V['aruco']}), PyBullet {V['pybullet']}; tested on {V['platform']}.\n")
    open(os.path.join(a.out, "report_tables.md"), "w").write("\n".join(md))
    json.dump(dict(summary=S, versions=V), open(os.path.join(a.out, "summary.json"), "w"), indent=2, default=float)

    # --- telemetry figure for the representative run -------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        A = np.load(os.path.join(a.out, f"telemetry_nominal_{a.seed0}.npy"))
        t = A[:, 0]; names = ["SEARCH", "TRACK", "DESCEND", "BLIND", "LANDED"]
        fig, ax = plt.subplots(4, 1, figsize=(8, 8.5), sharex=True)
        ax[0].plot(t, A[:, 3], lw=1.6); ax[0].set_ylabel("altitude z [m]")
        for s_, c in zip(range(5), ["#fdf1dc", "#e3f5f2", "#e8f0fb", "#eae3f7", "#fbe6e3"]):
            m = A[:, 9] == s_
            if m.any():
                i0 = np.where(m)[0]; ax[0].axvspan(t[i0[0]], t[i0[-1]], color=c, alpha=.9, zorder=0); ax[0].text(t[i0[0]], ax[0].get_ylim()[1]*.92, names[s_], fontsize=7)
        ax[1].plot(t, A[:, 6], lw=1.2); ax[1].axhline(-0.65, color="r", ls="--"); ax[1].set_ylabel("vz [m/s]"); ax[1].text(t[0], -0.6, "limit -0.65", color="r", fontsize=8)
        ax[2].plot(t, A[:, 8] * 100, lw=1.2); ax[2].set_ylabel("XY error [cm]"); ax[2].set_yscale("log")
        ax[3].plot(t, np.degrees(A[:, 7]), lw=1.2); ax[3].axhline(30, color="r", ls="--"); ax[3].set_ylabel("max(|roll|,|pitch|) [deg]"); ax[3].set_xlabel("time [s]")
        for x in ax: x.grid(alpha=.25)
        fig.suptitle("Representative run (arena defaults)", fontsize=11); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "run0_telemetry.png"), dpi=200)
    except Exception as ex:
        print("[eval] telemetry figure skipped:", ex)

    print("\n" + "\n".join(md))
    print(f"\n[eval] files written to ./{a.out}/  (report_tables.md, results.csv, summary.json, run0_telemetry.png)")

if __name__ == "__main__":
    main()