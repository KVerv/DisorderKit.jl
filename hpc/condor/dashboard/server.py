#!/usr/bin/env python3
"""Dashboard for the Bose-Hubbard sweeps on an HTCondor pool.

Runs on the laptop and talks to the condor access point over one persistent ssh connection:

    python3 hpc/condor/dashboard/server.py      # asks for the ssh password once, opens the browser

The connection settings (host, port, login, repo folder on the access point) are read from
hpc/condor/condor.local.env (gitignored; asked for and written on the first start; template:
hpc/condor/condor.local.env.example). Command-line flags override them.

Models (MODELS below): BHCorr, the correlated Bose-Hubbard chain without symmetry on a t × μ grid
(hpc/condor/sweep_BHCorr.sh), and BHU1, the same chain with U(1)-symmetric tensors at fixed filling, swept over any two
of t, U, Δ, Ω, r (hpc/condor/sweep_BHU1.sh), and BHRand, the random-hopping chain (hoppings {t, t/3} with probability
1/2 each) swept over μ only: a preset of BHCorr (Δ = 0, Ω = 1/2, r = 0, one mean hopping t̄ = 2t/3) with its own plot of
ρ, κ and K against μ. The page picks the model, and colours the grid by job state or by an observable.

Workflow: every condor job runs the ITE and the observables of one point. *Gather*
copies finished results to this repo's data/ folder and then deletes them (and their logs) on the access point, so
nothing piles up there. The laptop is the archive: the grid shows the laptop's and the access point's results together,
*Submit* decides which points still need a run from both, *Recompute observables* uploads only the needed states and
runs observables-only jobs (a changed observables script is detected through its git hash), and *Plot* runs the
phase-diagram script locally on the laptop's data.

Options: --host, --port, --user, --remote-dir (override condor.local.env), --julia (Julia command on the access point,
default juliaup with the Manifest's version), --local-julia (Julia on the laptop, default "julia"), --http-port (8765),
--no-browser, --gather-dest (the laptop's data root, default this repo), --local DIR (testing: no ssh, DIR plays the
access point), --selftest. Python 3 standard library only.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CONFIG = os.path.join(REPO, "hpc", "condor", "condor.local.env")
JOB_ATTRS = ("ClusterId,ProcId,JobStatus,BHC_model,BHC_stage,BHC_tag,BHC_L,BHC_t,BHC_mu,BHC_x,BHC_y,BHC_points,RemoteHost,LastRemoteHost,"
             "HoldReason,EnteredCurrentStatus,QDate,CompletionDate,ExitCode,RemoveReason,"
             "RemoteUserCpu,RemoteSysCpu,JobCurrentStartDate,RequestCpus,RequestMemory,ResidentSetSize,ResidentSetSize_RAW,"
             "RemoteWallClockTime,NumJobStarts")
STALL_AFTER = 30 * 60   # s: a running job that used less than STALL_CPU cores on average after this long is flagged
STALL_CPU = 0.05
GATHER_MIN_AGE = 2  # minutes: younger result files may still be written by condor

RE_TAG = re.compile(r"^[A-Za-z0-9._+-]+$")
RE_POINT = re.compile(r"^-?\d+\.\d{4}$")
RE_AXIS = r"^(t|U|Delta|Omega|r)$"
NUM = r"^[0-9.eE+-]+$"
RE_COL = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")   # an observables CSV column
# Environment variables the actions may pass to the condor scripts, with the allowed values
PARAMS = {
    "T": NUM, "U": NUM, "R": NUM, "DELTA": NUM, "OMEGA": NUM, "FILLING": r"^\d+$", "NMAX": r"^\d+$", "DTAU": NUM,
    "MAXD": r"^\d+$", "XAXIS": RE_AXIS, "YAXIS": RE_AXIS, "OBS": r"^[A-Za-z_,]*$",
    "MAXITER": r"^\d+$", "CONVTOL": NUM, "CROSS_WIDTH": r"^\d+$", "PPN": r"^\d+$", "OBS_THREADS": r"^\d+$",
    "MEM": r"^\d+(\.\d+)? ?[KMGT]?B?$", "DISK": r"^\d+(\.\d+)? ?[KMGT]?B?$",
    "REQUIREMENTS": r"^[A-Za-z0-9_ .<>=!&|()\"']*$", "NICE": r"^(TRUE|FALSE)$",
    "N_SAMPLES": r"^\d+$", "N_EDGE": r"^\d+$", "OBS_COL": RE_COL, "LOG": r"^[01]$", "XMIN": NUM, "XMAX": NUM,
    "YMIN": NUM, "YMAX": NUM, "KAPPA_TOL": NUM, "SBSTD_TOL": NUM, "A2_TOL": NUM,
    "POINTS": r"^[0-9.\- ,]*$", "SEED": r"^\d+$", "DRYRUN": r"^[01]$", "RESUME": r"^[01]$", "CHUNK": r"^[1-9]\d*$",
}
# The models the dashboard drives. A sweep is (model, tag, L); its points are (x, y), the values of its two axes as in
# the file names (%.4f): data/<dir>/<tag>/L<L>/<X><x>/<Y><y>.jld2 (and _obs.csv), X and Y the axis names.
#   BHCorr  hpc/BoseHubbardCorrelated_NormalizedITE.jl: no symmetry, axes t and mu fixed, the tag set by the fixed
#           parameters (built in the page and by the Julia script)
#   BHU1    hpc/BoseHubbardU1_NormalizedITE.jl: U(1) tensors at fixed filling, any two of t, U, Delta, Omega, r as
#           axes; the tag "<X>-<Y>_<fixed parameters>" is built here (bhu1_tag) and passed to the scripts as TAG
#   BHRand  a preset of BHCorr (below MODELS): the same scripts, files and jobs with Δ = 0, Ω = 1/2, r = 0, i.e. random
#           hopping {1.5 t̄, 0.5 t̄} = {t, t/3}, one t̄ column swept over μ, and its own plot (ρ, κ, K against μ). Sweeps
#           and jobs of the shared folder are assigned to BHCorr or BHRand by their tag (sweep_model)
_RES = ["PPN", "MEM", "REQUIREMENTS", "NICE", "POINTS"]
MODELS = {
    "BHCorr": {
        "dir": "BoseHubbardCorrelated_NormalizedITE",
        "obs_script": "hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl",
        "plot_script": "hpc/plot_BoseHubbardCorrelated_NormalizedITE.jl",
        "scripts": {"submit": "hpc/condor/sweep_BHCorr.sh", "recompute_obs": "hpc/condor/obs_BHCorr.sh"},
        "attrs": ("BHC_t", "BHC_mu"),                   # job attributes with the point values
        "select": "",                                   # extra ClassAd constraint for its jobs
        "prefix": {"ite": "BHC_L{L}", "obs": "BHC_obs_L{L}"},
        "grid": ("TS", "MUS"),                          # the page's value lists of the two axes
        "status_args": "",                              # arguments of hpc/condor/sweep_status.jl
        "obs_marker": "t = {x}, μ = {y}",               # point marker in the observables log
        "model_params": ["U", "R", "DELTA", "OMEGA", "NMAX", "DTAU", "MAXD"],
        "extra": {"submit": [], "recompute_obs": []},
        "plot_params": ["KAPPA_TOL", "SBSTD_TOL", "A2_TOL"],
    },
    "BHU1": {
        "dir": "BoseHubbardU1_NormalizedITE",
        "obs_script": "hpc/observables_BoseHubbardU1_NormalizedITE.jl",
        "plot_script": "hpc/plot_BoseHubbardU1_NormalizedITE.jl",   # PNG of one observable over a window
        "scripts": {"submit": "hpc/condor/sweep_BHU1.sh", "recompute_obs": "hpc/condor/obs_BHU1.sh"},
        "attrs": ("BHC_x", "BHC_y"),
        "select": ' && BHC_model =?= "BHU1"',
        "prefix": {"ite": "BHU1_L{L}", "obs": "BHU1_obs_L{L}"},
        "grid": ("XS", "YS"),
        "status_args": "--model BHU1 ",
        "obs_marker": "{X} = {x}, {Y} = {y}",
        "model_params": ["T", "U", "DELTA", "OMEGA", "R", "FILLING", "NMAX", "DTAU", "MAXD"],
        "extra": {"submit": ["XAXIS", "YAXIS", "OBS", "SEED"], "recompute_obs": ["XAXIS", "YAXIS", "OBS"], "plot": ["XAXIS", "YAXIS"]},
        "plot_params": ["OBS_COL", "LOG", "XMIN", "XMAX", "YMIN", "YMAX"],
    },
}
MODELS["BHRand"] = {
    **MODELS["BHCorr"],
    "plot_script": "hpc/plot_BoseHubbardRandomHopping_NormalizedITE.jl",   # ρ, κ and K against μ
    "plot_params": [],
    "plot_suffix": "_mu",                           # PNG next to (not over) BHCorr's phase diagram of the same tag
    "preset": {"R": 0.0, "DELTA": 0.0, "OMEGA": 0.5},
}
BHU1_AXES = {"t": "T", "U": "U", "Delta": "DELTA", "Omega": "OMEGA", "r": "R"}   # axis name => parameter
RE_BHCORR_TAG = re.compile(r"^U(.+)_r(.+)_Delta(.+)_Omega(.+)_nmax\d+_dtau.+_D\d+$")


def is_preset_tag(tag, preset):
    """Whether a BHCorr tag (U<U>_r<r>_Delta<Δ>_Omega<Ω>_…) has the preset's r, Δ and Ω (compared as numbers)."""
    m = RE_BHCORR_TAG.match(tag)
    if not m:
        return False
    try:
        r, delta, omega = (float(v) for v in m.group(2, 3, 4))
    except ValueError:
        return False
    return (r, delta, omega) == (preset["R"], preset["DELTA"], preset["OMEGA"])


def sweep_model(name, tag):
    """The model a sweep of the model `name`'s folder belongs to: BHCorr tags with BHRand's preset are BHRand sweeps."""
    if name == "BHCorr" and is_preset_tag(tag, MODELS["BHRand"]["preset"]):
        return "BHRand"
    return name
COMMON = {"submit": ["MAXITER", "CONVTOL", "CROSS_WIDTH", "N_SAMPLES", "N_EDGE", "OBS_THREADS", "DISK", "DRYRUN", "RESUME"] + _RES,
          "recompute_obs": ["N_SAMPLES", "N_EDGE", "CHUNK"] + _RES, "plot": []}


def model_of(name):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}")
    return MODELS[name]


def script_params(model, action):
    m = MODELS[model]
    return m["model_params"] + m["extra"].get(action, []) + COMMON[action] + (m["plot_params"] if action == "plot" else [])


def bhu1_tag(values):
    """Tag of a BHU1 sweep: "<X>-<Y>_" and the fixed parameters (the two swept ones left out), values verbatim, e.g.
    t-Delta_U1.0_Omega0.0_r0.0_f1_nmax4_dtau0.05_D32. The page builds the same string (tagOf) to name the sweep."""
    X, Y = values["XAXIS"], values["YAXIS"]
    if X == Y:
        raise ValueError("the two sweep axes must differ")
    fixed = [f"{a}{values[BHU1_AXES[a]]}" for a in ("t", "U", "Delta", "Omega", "r") if a not in (X, Y)]
    return "_".join([f"{X}-{Y}", *fixed, f"f{values['FILLING']}", f"nmax{values['NMAX']}", f"dtau{values['DTAU']}",
                     f"D{values['MAXD']}"])


def axes_of(model, tag):
    """Names of the two axes of a sweep."""
    if model in ("BHCorr", "BHRand"):
        return "t", "mu"
    m = re.match(r"^([A-Za-z]+)-([A-Za-z]+)_", tag)
    if not m:
        raise ValueError(f"tag {tag!r} does not start with <X>-<Y>_")
    return m.group(1), m.group(2)


def point_file(model, tag, L, x, y, suffix=".jld2"):
    """Relative path of a point's state (suffix .jld2) or observables (_obs.csv) file."""
    X, Y = axes_of(model, tag)
    return f"data/{MODELS[model]['dir']}/{tag}/L{L}/{X}{x}/{Y}{y}{suffix}"


# ---- configuration (kept out of the repo)

def read_config():
    cfg = {}
    if os.path.isfile(CONFIG):
        with open(CONFIG) as f:
            for line in f:
                m = re.match(r"^\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
                if m and not line.lstrip().startswith("#"):
                    cfg[m.group(1)] = m.group(2)
    return cfg


def resolve_connection(args):
    cfg = read_config()
    keys = [("host", "CONDOR_HOST", "Access point host name"), ("port", "CONDOR_PORT", "ssh port"),
            ("user", "CONDOR_USER", "Your login on the access point"),
            ("remote_dir", "CONDOR_REMOTE_DIR", "Repo folder on the access point, relative to your home")]
    changed = False
    for attr, key, prompt in keys:
        if getattr(args, attr) is None:
            value = cfg.get(key, "")
            if not value or value.startswith("<"):
                default = "DisorderKit.jl" if key == "CONDOR_REMOTE_DIR" else ""
                value = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip() or default
                cfg[key] = value
                changed = True
            setattr(args, attr, value)
    if changed:
        with open(CONFIG, "w") as f:
            f.write("# Connection to the condor access point for hpc/condor/dashboard/server.py (gitignored, do not commit)\n")
            for _, key, _ in keys:
                f.write(f"{key}={cfg[key]}\n")
        print(f"Saved the connection settings in {CONFIG} (gitignored).")
    args.port = str(args.port)


class Remote:
    """Runs shell commands in the repo on the access point (over the ssh master connection) or locally (--local)."""

    def __init__(self, args):
        self.local = args.local
        self.host, self.port, self.user, self.dir = args.host, args.port, args.user, args.remote_dir
        self.target = f"{self.user}@{self.host}" if not self.local else ""
        self.sock = os.path.expanduser(f"~/.ssh/condor-dashboard-{self.user}@{self.host}-{self.port}.sock")
        self.julia = args.julia
        self.lost = False

    def _argv(self, cmd):
        """argv plus the script to feed on stdin. The script goes over stdin, not as an argument, because a long one
        (e.g. the rm of the logs of many gathered points) exceeds the kernel's argument length limit. It is wrapped in
        { ...; } </dev/null so bash reads it completely before running it and the commands in it do not eat it."""
        script = f"{{ cd {shlex.quote(self.local or self.dir)} && {cmd}\n}} </dev/null\n"
        if self.local:
            return ["bash", "-s"], script
        return ["ssh", "-S", self.sock, "-o", "BatchMode=yes", "-p", self.port, self.target, "bash -s"], script

    def connect(self):
        if self.local:
            return True
        check = ["ssh", "-S", self.sock, "-O", "check", "-p", self.port, self.target]
        if subprocess.run(check, capture_output=True).returncode == 0:
            return True
        print(f"Connecting to {self.target} (one password prompt; the connection is kept open for 12 h) ...")
        rc = subprocess.run(["ssh", "-M", "-S", self.sock, "-o", "ControlPersist=12h", "-o", "ServerAliveInterval=60",
                             "-p", self.port, "-fN", self.target]).returncode
        self.lost = rc != 0
        return rc == 0

    def run(self, cmd, timeout=300):
        try:
            argv, script = self._argv(cmd)
            p = subprocess.run(argv, input=script, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124, "", f"timed out after {timeout} s"
        if p.returncode == 255 and not self.local:
            self.lost = subprocess.run(["ssh", "-S", self.sock, "-O", "check", "-p", self.port, self.target],
                                       capture_output=True).returncode != 0
        return p.returncode, p.stdout, p.stderr

    def stream(self, cmd, emit):
        argv, script = self._argv(cmd)
        return stream(argv, emit, input=script)

    def path(self, rel=""):
        """rsync address of a path relative to the repo on the access point."""
        if self.local:
            return os.path.join(self.local, rel)
        return f"{self.target}:{self.dir}/{rel}"

    def rsync_ssh(self):
        return [] if self.local else ["-e", f"ssh -S {self.sock} -o BatchMode=yes -p {self.port}"]


def stream(argv, emit, input=None, **kw):
    p = subprocess.Popen(argv, stdin=subprocess.PIPE if input is not None else None, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    if input is not None:
        p.stdin.write(input)
        p.stdin.close()
    for line in p.stdout:
        emit(line.rstrip("\n"))
    return p.wait()


# ---- state of the points

QUEUE_STATE = {1: "idle", 2: "running", 4: "running", 5: "held", 6: "running", 7: "idle"}


def job_info(j, now=None):
    info = {"id": f"{j.get('ClusterId')}.{j.get('ProcId')}", "status": j.get("JobStatus"),
            "host": j.get("RemoteHost") or j.get("LastRemoteHost"), "since": j.get("EnteredCurrentStatus"),
            "hold_reason": j.get("HoldReason"), "exit_code": j.get("ExitCode"), "remove_reason": j.get("RemoveReason")}
    start = j.get("JobCurrentStartDate")
    if j.get("JobStatus") == 2 and start:
        # CPU use of the current run: average number of busy cores (condor updates RemoteUserCpu every few minutes)
        wall = max((now or time.time()) - start, 1)
        cpu = (j.get("RemoteUserCpu") or 0) + (j.get("RemoteSysCpu") or 0)
        info.update(start=start, wall=wall, cpus=j.get("RequestCpus"), cores_used=cpu / wall,
                    stalled=wall > STALL_AFTER and cpu / wall < STALL_CPU)
    if j.get("JobStatus") == 4:
        # finished job: total wall time over all its runs (RemoteWallClockTime), else that of the last run
        rt = j.get("RemoteWallClockTime")
        if not isinstance(rt, (int, float)) and j.get("CompletionDate") and start:
            rt = j["CompletionDate"] - start
        info.update(runtime=rt if isinstance(rt, (int, float)) else None, starts=j.get("NumJobStarts"))
    req = j.get("RequestMemory")
    if isinstance(req, (int, float)) and req > 0:
        # ResidentSetSize: peak of the job (KB, rounded up by condor), ResidentSetSize_RAW: current (KB). condor's
        # MemoryUsage is the expression (ResidentSetSize + 1023) / 1024, which condor_q -json does not evaluate
        peak = j.get("ResidentSetSize")
        info.update(mem_req=req, mem_peak=(peak + 1023) // 1024 if isinstance(peak, (int, float)) else None)
        if info["mem_peak"] is not None:
            info["mem_frac"] = info["mem_peak"] / req
        rss = j.get("ResidentSetSize_RAW")
        if j.get("JobStatus") == 2 and isinstance(rss, (int, float)):
            info["mem_now"] = rss / 1024
    return info


# ---- progress of running jobs, from their live stdout/stderr (condor_tail)

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RE_MARK = re.compile(r"^== (\d\d:\d\d:\d\d) (?:setup: (.*)|julia (\S+).*)$")    # run_julia.sh's phase markers
RE_INFO = re.compile(r"^\[ Info: (?:(\d\d:\d\d:\d\d) )?(.*?)\s*$")               # ITE log lines (time stamp optional)
RE_ITER = re.compile(r"^Iteration (\d+), cap = (\d+)")
RE_DE = re.compile(r"^Convergence error: \(E, dE\) = \(([^,]+), ([^)]+)\)")
ITE_STEPS = ("Partition function", "Normalization filter", "Compress", "Convergence error", "Finalize")


def parse_progress(out, err):
    """Phase of a job from its stdout (run_julia.sh markers) and its ITE step from its stderr (log lines).
    Jobs started before the markers existed: "Julia x (juliaup)" without a later julia marker means Pkg.instantiate."""
    out, err = ANSI.sub("", out or ""), ANSI.sub("", err or "")
    res = {"phase": "starting", "phase_since": None, "script": None, "iteration": None, "cap": None, "step": None,
           "step_since": None, "E": None, "dE": None, "error": None}
    for line in out.splitlines():
        m = RE_MARK.match(line)
        if m:
            res["phase_since"] = m[1]
            if m[2] is not None:
                res["phase"], res["script"] = "setup: " + m[2], None
            else:
                res["script"] = m[3]
                res["phase"] = "observables" if "observables" in m[3] else "ITE"
        elif line.startswith("Julia ") and "(juliaup)" in line:
            res["phase"], res["phase_since"] = "setup: Pkg.instantiate", None
        elif res["phase"] == "starting" and "juliaup" in line.lower():
            res["phase"] = "setup: julia install"
    for line in err.splitlines():
        if line.startswith("ERROR:"):
            res["error"] = line
        m = RE_INFO.match(line)
        if not m:
            continue
        stamp, msg = m[1], m[2]
        it = RE_ITER.match(msg)
        if it:
            res.update(iteration=int(it[1]), cap=int(it[2]), step="Iteration start", step_since=stamp)
            continue
        name = next((x for x in ITE_STEPS if msg.startswith(x)), None)
        if name is None:
            continue
        res["step"], res["step_since"] = name, stamp
        de = RE_DE.match(msg)
        if de:
            try:
                res["E"], res["dE"] = float(de[1]), float(de[2])
            except ValueError:
                pass
    if res["phase"] != "ITE" and res["phase"] != "observables":
        res.update(iteration=None, cap=None, step=None, step_since=None)
    tail = [x for x in err.splitlines() if x.strip()][-12:]
    res["tail"] = "\n".join(tail)
    return res


def status_key(p):
    """(x, y) of a point of sweep_status.jl (older versions on the access point write t and mu)."""
    return (p["x"], p["y"]) if "x" in p else (p["t"], p["mu"])


def merge(remote, local, queue, history, obs_version="", n_edge=None, attrs=("BHC_t", "BHC_mu")):
    """One record per point (x, y) from the access point's files (remote), the laptop's files (local), the queue and
    the history (jobs matched through the point attributes attrs). A state on the access point is newer than the
    laptop's copy (it is gathered later)."""
    points = {}

    def entry(key):
        return points.setdefault(key, {"x": key[0], "y": key[1], "remote": None, "local": None})

    for where, st in (("remote", remote), ("local", local)):
        for p in st.get("points", []):
            entry(status_key(p))[where] = p
    for x, y in remote.get("planned", []):
        entry((x, y))
    active = {"ite": {}, "obs": {}}
    for j in queue:
        stage = j.get("BHC_stage")
        if stage not in active or j.get("JobStatus") == 3:
            continue
        for key in job_points(j, attrs):
            entry(key)
            old = active[stage].get(key)
            if old is None or (j.get("ClusterId", 0), j.get("ProcId", 0)) > (old.get("ClusterId", 0), old.get("ProcId", 0)):
                active[stage][key] = j
    last = {"ite": {}, "obs": {}}
    for j in history:
        stage = j.get("BHC_stage")
        if stage not in last:
            continue
        rank = (j.get("CompletionDate") or j.get("EnteredCurrentStatus") or 0, j.get("ClusterId", 0))
        for key in job_points(j, attrs):
            entry(key)
            if key not in last[stage] or rank > last[stage][key][0]:
                last[stage][key] = (rank, j)

    def failure(j):
        if j.get("JobStatus") == 3:
            return "removed" + (f": {j['RemoveReason']}" if j.get("RemoveReason") else "")
        if j.get("ExitCode") not in (None, 0):
            return f"exit code {j['ExitCode']}"
        return "finished without its output files"

    out = []
    for key, p in points.items():
        r, l = p["remote"] or {}, p["local"] or {}
        f = r if r.get("state") else l            # the state that counts: the access point's (newer) or the laptop's
        o = r if r.get("obs") else l               # observables: likewise
        where = [w for w, x in (("laptop", l), ("access point", r)) if x.get("state") or x.get("obs")]
        rec = {"x": p["x"], "y": p["y"], "eps": f.get("eps"), "nsteps": f.get("nsteps"), "maxiter": f.get("maxiter"), "resumed_from": f.get("resumed_from"), "seed": f.get("seed"),
               "convtol": f.get("convtol"), "mtime": f.get("mtime"), "has_state": bool(f.get("state")),
               "n_edge": o.get("n_edge"), "obs_version": o.get("obs_version"), "where": where,
               "on_remote": bool(r.get("state") or r.get("obs")), "reason": None, "job": None, "obs_job": None,
               "values": o.get("values") or {}}
        j = active["ite"].get(key)
        if j is not None:
            rec["state"], rec["job"] = QUEUE_STATE.get(j.get("JobStatus"), "idle"), job_info(j)
            if rec["state"] == "held":
                rec["reason"] = j.get("HoldReason")
        elif f.get("state"):
            if f.get("error"):
                rec["state"], rec["reason"] = "failed", "unreadable state: " + f["error"]
            else:
                rec["state"] = "done" if f.get("converged") else "unconverged"
                if key in last["ite"] and last["ite"][key][1].get("JobStatus") == 4:
                    rec["job"] = job_info(last["ite"][key][1])   # the finished job: runtime, peak RAM
        elif key in last["ite"]:
            j = last["ite"][key][1]
            rec["state"], rec["reason"], rec["job"] = "failed", failure(j), job_info(j)
        else:
            rec["state"] = "none"
        oj = active["obs"].get(key)
        if oj is not None:
            rec["obs"], rec["obs_job"] = QUEUE_STATE.get(oj.get("JobStatus"), "idle"), job_info(oj)
        elif o.get("obs"):
            stale = (obs_version and o.get("obs_version") != obs_version) or \
                    (n_edge is not None and o.get("n_edge") != n_edge)
            rec["obs"] = "outdated" if stale else "done"
            if key in last["obs"] and last["obs"][key][1].get("JobStatus") == 4:
                rec["obs_job"] = job_info(last["obs"][key][1])
        elif key in last["obs"] and f.get("state"):
            rec["obs"], rec["obs_job"] = "failed", job_info(last["obs"][key][1])
        else:
            rec["obs"] = "none"
        out.append(rec)
    out.sort(key=lambda r: (float(r["x"]), float(r["y"])))
    return out


def job_points(j, attrs=("BHC_t", "BHC_mu")):
    """The points (x, y) of a job: BHC_points "x1_y1,x2_y2,..." (observables jobs of several points), else the point
    attributes attrs (BHC_t and BHC_mu, or BHC_x and BHC_y)."""
    pts = j.get("BHC_points")
    if isinstance(pts, str) and pts:
        return [tuple(x.split("_", 1)) for x in pts.split(",") if "_" in x]
    key = (j.get(attrs[0]), j.get(attrs[1]))
    return [] if None in key else [key]


def point_constraint(x, y, attrs=("BHC_t", "BHC_mu")):
    """ClassAd constraint matching the jobs of the point (x, y), single-point or with BHC_points."""
    a, b = attrs
    return f'(({a} =?= "{x}" && {b} =?= "{y}") || stringListMember("{x}_{y}", BHC_points) =?= true)'


def parse_progress_dump(text):
    """Output of Dashboard.progress's remote loop: @@JOB <id> <stage> <x> <y>, @@OUT <stdout>, @@ERR <stderr>."""
    jobs, cur, part = [], None, None
    for line in text.splitlines():
        if line.startswith("@@JOB ") or line == "@@END":
            if cur:
                jobs.append(cur)
            cur, part = None, None
            if line != "@@END":
                parts = line.split()
                cur = {"id": parts[1], "stage": parts[2] if len(parts) > 2 else None,
                       "x": parts[3] if len(parts) > 3 else None, "y": parts[4] if len(parts) > 4 else None,
                       "out": [], "err": []}
        elif cur is not None and line in ("@@OUT", "@@ERR"):
            part = cur["out"] if line == "@@OUT" else cur["err"]
        elif part is not None:
            part.append(line)
    if cur:
        jobs.append(cur)
    res = []
    for j in jobs:
        out, err = "\n".join(j.pop("out")), "\n".join(j.pop("err"))
        res.append({**j, **parse_progress(out, err)})
    return res


def needs_run(p, maxiter, convtol, force=False):
    """Same rule as the ITE scripts' --status, on the merged record of a point."""
    if p.get("state") in ("idle", "running", "held"):
        return None                                   # in the queue
    if force or not p.get("has_state"):
        return "run"
    eps = p.get("eps")
    eps = float("inf") if eps is None else eps
    if eps <= convtol:
        return None
    if maxiter > (p.get("maxiter") or 0):
        return "resume"                               # continues the ITE from the stored state
    if convtol < (p.get("convtol") or float("inf")):
        return "rerun"
    return None


def parse_json_section(text, default):
    text = text.strip()
    if not text:
        return default
    for line in reversed(text.splitlines()):  # sweep_status.jl prints one line; ignore warnings before it
        line = line.strip()
        if line.startswith("{") or line.startswith("["):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def sections(out):
    parts, name = {}, None
    for line in out.splitlines():
        if line.startswith("@@"):
            name = line[2:].strip()
            parts[name] = []
        elif name:
            parts[name].append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


class Dashboard:
    def __init__(self, args):
        self.args = args
        self.remote = Remote(args)
        self.lock = threading.Lock()
        self.cache = {}
        self.local_cache = {}
        self.tasks = {}
        self.current = None
        self.local_julia = shlex.split(args.local_julia)

    # ---- queries

    @staticmethod
    def check_sweep(model, tag, L):
        model_of(model)
        if not RE_TAG.match(tag):
            raise ValueError("bad tag")
        axes_of(model, tag)
        return int(L)

    def condor_constraint(self, model, tag, L):
        return f'BHC_tag =?= "{tag}" && BHC_L =?= {int(L)}' + MODELS[model]["select"]

    def obs_version(self, model):
        p = subprocess.run(["git", "-C", REPO, "hash-object", MODELS[model]["obs_script"]], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else ""

    def local_scan(self, model, tag, L):
        """sweep_status.jl on the laptop's data (cached until the folder changes)."""
        folder = os.path.join(self.args.gather_dest, "data", MODELS[model]["dir"], tag, f"L{L}")
        sig = [0, 0.0]
        for root, _, files in os.walk(folder):
            for name in files:
                sig[0] += 1
                sig[1] = max(sig[1], os.path.getmtime(os.path.join(root, name)))
        sig = tuple(sig)
        hit = self.local_cache.get((model, tag, L))
        if hit and hit[0] == sig:
            return hit[1]
        if sig[0] == 0:
            res = {"points": [], "planned": []}
        else:
            p = subprocess.run([*self.local_julia, f"--project={REPO}", "--startup-file=no",
                                os.path.join(REPO, "hpc", "condor", "sweep_status.jl"),
                                *shlex.split(MODELS[model]["status_args"]), tag, str(L)],
                               cwd=self.args.gather_dest, capture_output=True, text=True, timeout=600)
            res = parse_json_section(p.stdout, None)
            if res is None:
                raise RuntimeError("reading the laptop's data failed: " + (p.stderr.strip().splitlines() or ["?"])[-1])
        self.local_cache[(model, tag, L)] = (sig, res)
        return res

    def status(self, model, tag, L, n_edge=None, fresh=False):
        L = self.check_sweep(model, tag, L)
        m = MODELS[model]
        key = (model, tag, L, n_edge)
        with self.lock:
            hit = self.cache.get(key)
            if hit and not fresh and time.time() - hit[0] < 10:
                return hit[1]
        c = shlex.quote(self.condor_constraint(model, tag, L))
        cmd = (f"echo @@STATUS; {self.remote.julia} --project=. --startup-file=no hpc/condor/sweep_status.jl {m['status_args']}{tag} {L} 2>/dev/null; "
               f"echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} 2>/dev/null; "
               f"echo @@HISTORY; command -v condor_history >/dev/null && condor_history \"$USER\" -constraint {c} -json -attributes {JOB_ATTRS} -limit 3000 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=180)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        remote = parse_json_section(s.get("STATUS", ""), {"points": [], "planned": []})
        queue = parse_json_section(s.get("QUEUE", ""), [])
        history = parse_json_section(s.get("HISTORY", ""), [])
        local = self.local_scan(model, tag, L)
        points = merge(remote, local, queue, history, self.obs_version(model), n_edge, m["attrs"])
        counts, obs_counts = {}, {}
        for p in points:
            counts[p["state"]] = counts.get(p["state"], 0) + 1
            obs_counts[p["obs"]] = obs_counts.get(p["obs"], 0) + 1
        plots = self.plot_mtimes(model, tag, L)
        res = {"model": model, "tag": tag, "L": L, "axes": list(axes_of(model, tag)), "points": points, "counts": counts,
               "obs_counts": obs_counts,
               "clusters": sorted({j.get("ClusterId") for j in queue if j.get("ClusterId") is not None}),
               "queue_size": len(queue), "time": time.time(),
               "to_gather": sum(1 for p in points if p["on_remote"] and p["state"] not in ("idle", "running", "held")
                                and p["obs"] not in ("idle", "running", "held")),
               "on_laptop": sum(1 for p in points if "laptop" in p["where"]),
               "stalled": sum(1 for p in points if p["state"] == "running" and (p["job"] or {}).get("stalled")),
               "plot_mtime": plots.get(""), "plot_mtimes": plots}
        with self.lock:
            self.cache[key] = (time.time(), res)
        return res

    def sweeps(self):
        """Every sweep of every model: on the access point, on the laptop or in the queue."""
        lists = "".join(f"echo @@LIST_{name}; {self.remote.julia} --project=. --startup-file=no hpc/condor/sweep_status.jl "
                        f"{m['status_args']}--list 2>/dev/null; " for name, m in MODELS.items() if "preset" not in m)
        cmd = (lists + "echo @@QUEUE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint 'BHC_tag =!= undefined' "
               "-af BHC_model BHC_tag BHC_L 2>/dev/null; echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        found = {}
        for name, m in MODELS.items():
            if "preset" in m:
                continue                              # listed with the model whose folder it shares
            for x in parse_json_section(s.get(f"LIST_{name}", ""), []):
                mod = sweep_model(name, x["tag"])
                found[(mod, x["tag"], x["L"])] = {**x, "model": mod, "n_states": 0, "on_remote": x["n_states"]}
        # the laptop's sweeps (the archive)
        for model_name, m in MODELS.items():
            if "preset" in m:
                continue
            root = os.path.join(self.args.gather_dest, "data", m["dir"])
            for tag in sorted(os.listdir(root)) if os.path.isdir(root) else []:
                name = sweep_model(model_name, tag)
                try:
                    X, Y = axes_of(name, tag)
                except ValueError:
                    continue
                for Ldir in os.listdir(os.path.join(root, tag)):
                    if not re.match(r"^L\d+$", Ldir):
                        continue
                    folder = os.path.join(root, tag, Ldir)
                    states = [f for _, _, fs in os.walk(folder) for f in fs if re.match(rf"^{Y}-?[\d.]+\.jld2$", f)]
                    x = found.setdefault((name, tag, int(Ldir[1:])), {"model": name, "tag": tag, "L": int(Ldir[1:]),
                                                                       "on_remote": 0, "n_obs": 0})
                    x["n_states"] = len(states)
                    x["axes"] = [X, Y]
                    x["mtime"] = max(x.get("mtime", 0), os.path.getmtime(folder))
        queued = {}
        for parts in (line.split() for line in s.get("QUEUE", "").splitlines()):
            if len(parts) == 3 and parts[2].isdigit():
                name = parts[0] if parts[0] in MODELS else "BHCorr"   # jobs without BHC_model are BHCorr jobs
                name = sweep_model(name, parts[1])
                k = (name, parts[1], int(parts[2]))
                queued[k] = queued.get(k, 0) + 1
                if k not in found:
                    try:
                        ax = list(axes_of(name, parts[1]))
                    except ValueError:
                        continue
                    found[k] = {"model": name, "tag": parts[1], "L": int(parts[2]), "axes": ax, "n_states": 0,
                                "on_remote": 0, "n_obs": 0, "mtime": time.time()}
        for k, x in found.items():
            x["queued"] = queued.get(k, 0)
            x.setdefault("n_states", 0)
            x.setdefault("axes", list(axes_of(*k[:2])))
        return sorted(found.values(), key=lambda x: -x.get("mtime", 0))

    def log(self, model, tag, L, x, y, stage):
        L = self.check_sweep(model, tag, L)
        if not (RE_POINT.match(x) and RE_POINT.match(y)):
            raise ValueError("bad point")
        m = MODELS[model]
        X, Y = axes_of(model, tag)
        d = f"logs/{m['dir']}/{tag}/L{L}/condor" + ("/obs" if stage == "obs" else "")
        prefix = m["prefix"]["obs" if stage == "obs" else "ite"].format(L=L)
        marker = m["obs_marker"].format(X=X, Y=Y, x=x, y=y)
        # A running job's files on the access point stay empty until it ends: show its live output instead
        pc = (f'{self.condor_constraint(model, tag, L)} && BHC_stage =?= "{"obs" if stage == "obs" else "ite"}" && '
              f'{point_constraint(x, y, m["attrs"])}')
        c = shlex.quote(pc + " && JobStatus == 2")
        cmd = (f"id=$(condor_q \"$USER\" -constraint {c} -format '%d.' ClusterId -format '%d\\n' ProcId 2>/dev/null | head -1); "
               f"if [ -n \"$id\" ]; then echo \"== job $id is running: live stdout (condor_tail)\"; "
               f"timeout 30 condor_tail -maxbytes 8000 $id 2>&1 | tail -n 60; echo; echo \"== live stderr\"; "
               f"timeout 30 condor_tail -stderr -maxbytes 12000 $id 2>&1 | tail -n 60; exit 0; fi; "
               # observables jobs of several points: the log of the newest finished job of this point (by job id)
               + (f"id=$(condor_history \"$USER\" -constraint {shlex.quote(pc)} -limit 1 -format '%d.' ClusterId "
                  f"-format '%d\\n' ProcId 2>/dev/null | head -1); "
                  f"f=; [ -n \"$id\" ] && f=$(ls {d}/{prefix}_$id.err 2>/dev/null); "
                  # its setup and the section of this point (from its \"== … point k of n: <marker>\" line)
                  f"if [ -n \"$f\" ]; then o=\"${{f%.err}}.out\"; echo \"== $o (job $id, setup and this point)\"; "
                  f"awk -v m={shlex.quote(marker)} '/^== .* point [0-9]+ of [0-9]+: / {{ s = index($0, m) > 0; p = 1 }} !p || s' "
                  f"\"$o\" | tail -n 80; echo; echo \"== $f\"; tail -n 60 \"$f\"; exit 0; fi; " if stage == "obs" else "") +
               f"f=$(ls -t {d}/{prefix}_{X}{x}_{Y}{y}_*.err 2>/dev/null | head -1); "
               f"if [ -z \"$f\" ]; then echo 'No log files for this point on the access point (gathered points have no logs left).'; exit 0; fi; "
               f"o=\"${{f%.err}}.out\"; echo \"== $o\"; tail -n 60 \"$o\" 2>/dev/null; echo; echo \"== $f\"; tail -n 60 \"$f\"")
        rc, out, err = self.remote.run(cmd, timeout=60)
        return out + err

    def progress(self, model, tag, L, x=None, y=None):
        """Phase and ITE step of the running jobs of a sweep (or of one point), from their live output."""
        L = self.check_sweep(model, tag, L)
        if x is not None and not (RE_POINT.match(x) and RE_POINT.match(y)):
            raise ValueError("bad point")
        m = MODELS[model]
        c = self.condor_constraint(model, tag, L) + " && JobStatus == 2"
        if x is not None:
            c += " && " + point_constraint(x, y, m["attrs"])
        a, b = m["attrs"]
        cmd = (f"condor_q \"$USER\" -constraint {shlex.quote(c)} -af ClusterId ProcId BHC_stage {a} {b} 2>/dev/null | "
               "while read c p st x y; do echo \"@@JOB $c.$p $st $x $y\"; echo @@OUT; "
               "timeout 30 condor_tail -maxbytes 20000 $c.$p 2>&1 < /dev/null; echo; echo @@ERR; "
               "timeout 30 condor_tail -stderr -maxbytes 20000 $c.$p 2>&1 < /dev/null; echo; done; echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=300)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        jobs = parse_progress_dump(out)
        if x is not None:
            for j in jobs:   # observables jobs of several points have no point attributes
                if j.get("x") in (None, "undefined"):
                    j["x"], j["y"] = x, y
        return {"time": time.time(), "jobs": jobs}

    def plot_file(self, model, tag, L, obs=None):
        """The PNG of the plot script: BHCorr's phase diagram, BHU1's heatmap of the observable obs (None without)."""
        if not MODELS[model]["plot_script"]:
            return None
        if model == "BHU1":
            if not obs or not RE_COL.match(obs):
                return None
            return os.path.join(self.args.gather_dest, "plots", f"{MODELS[model]['dir']}_{tag}_L{int(L)}_{obs}.png")
        suffix = MODELS[model].get("plot_suffix", "")
        return os.path.join(self.args.gather_dest, "plots", f"{MODELS[model]['dir']}_{tag}_L{int(L)}{suffix}.png")

    def plot_mtimes(self, model, tag, L):
        """Modification times of the existing PNGs of a sweep: {"": t} for BHCorr, {observable: t} for BHU1."""
        if model != "BHU1":
            f = self.plot_file(model, tag, L)
            return {"": os.path.getmtime(f)} if f and os.path.isfile(f) else {}
        prefix = f"{MODELS[model]['dir']}_{tag}_L{int(L)}_"
        folder = os.path.join(self.args.gather_dest, "plots")
        out = {}
        for name in (os.listdir(folder) if os.path.isdir(folder) else []):
            obs = name[len(prefix):-4] if name.startswith(prefix) and name.endswith(".png") else ""
            if RE_COL.match(obs):
                out[obs] = os.path.getmtime(os.path.join(folder, name))
        return out

    # ---- actions (one at a time, output streamed to the page)

    def start(self, action, params):
        with self.lock:
            if self.current and not self.tasks[self.current]["done"]:
                raise RuntimeError("another action is still running")
            tid = uuid.uuid4().hex[:8]
            task = {"id": tid, "action": action, "lines": [], "done": False, "rc": None, "needs_confirm": False,
                    "started": time.time()}
            self.tasks[tid] = task
            self.current = tid
        threading.Thread(target=self._run, args=(task, action, params), daemon=True).start()
        return task

    def _run(self, task, action, params):
        emit = task["lines"].append
        try:
            rc = self._action(task, action, params, emit)
        except Exception as e:  # noqa: BLE001 - shown in the page
            emit(f"ERROR: {e}")
            rc = 1
        task["rc"], task["done"] = rc, True
        with self.lock:
            self.cache.clear()

    @staticmethod
    def _check(params, keys):
        out = {}
        for k in keys:
            v = str(params.get(k, "")).strip()
            if v == "":
                continue
            if not re.match(PARAMS[k], v):
                raise ValueError(f"invalid value for {k}: {v!r}")
            out[k] = v
        return out

    def _env(self, values):
        env = [f"{k}={shlex.quote(v)}" for k, v in values.items()]
        env.append(f'JULIA="{self.remote.julia}"')  # double quotes: $HOME expands on the access point
        return " ".join(env)

    def _sync(self, params, emit, task):
        emit("$ git status (laptop)")
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.rstrip()
        ahead = subprocess.run(["git", "-C", REPO, "rev-list", "--count", "@{u}..HEAD"],
                               capture_output=True, text=True).stdout.strip()
        problems = []
        if dirty:
            problems.append("uncommitted changes on the laptop:\n" + dirty)
        if ahead not in ("", "0"):
            problems.append(f"{ahead} commit(s) on the laptop not pushed")
        if problems and params.get("confirm") != "1":
            for p in problems:
                emit("WARNING: " + p)
            emit("These changes will NOT be used on the cluster (it runs the code on GitHub).")
            emit("Commit and push them first, or press 'Continue anyway'.")
            task["needs_confirm"] = True
            return 2
        emit("Continuing with the code on GitHub (laptop changes ignored)." if problems
             else "laptop: everything committed and pushed")
        if self.args.local:
            emit("local mode: skipping git pull on the access point")
            return 0
        emit("$ git pull --ff-only (access point)")
        rc = self.remote.stream("git pull --ff-only 2>&1", emit)
        if rc != 0:
            emit("git pull failed: resolve it on the access point (ssh, cd to the repo, git status).")
            return rc
        emit("$ Pkg.instantiate() (access point)")
        return self.remote.stream(f"{self.remote.julia} --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()' 2>&1", emit)

    @staticmethod
    def _grid(model, params):
        kx, ky = MODELS[model]["grid"]
        xs = [f"{float(v):.4f}" for v in str(params.get(kx, "")).split()]
        ys = [f"{float(v):.4f}" for v in str(params.get(ky, "")).split()]
        return [(x, y) for x in xs for y in ys]

    def _values(self, model, tag, params, action):
        """Checked environment of an action's script; for BHU1 also TAG, after checking that the parameters give the
        selected sweep's tag (the page and bhu1_tag agree)."""
        values = self._check(params, script_params(model, action))
        if model == "BHU1":
            for k in ("XAXIS", "YAXIS", "FILLING", "NMAX", "DTAU", "MAXD"):
                if k not in values:
                    raise ValueError(f"{k} is not set")
            fixed = {k: values.get(k, "") for k in MODELS["BHU1"]["model_params"]}
            want = bhu1_tag({**fixed, "XAXIS": values["XAXIS"], "YAXIS": values["YAXIS"]})
            if want != tag:
                raise ValueError(f"the parameters give the tag {want}, not the selected sweep {tag}")
            values["TAG"] = tag
        preset = MODELS[model].get("preset")
        if preset:
            for k, want in preset.items():
                if k not in values or float(values[k]) != want:
                    raise ValueError(f"{k} must be {want} for this model")
            if sweep_model("BHCorr", tag) != model:
                raise ValueError(f"the tag {tag} is not a sweep of this model")
        return values

    def _action(self, task, action, params, emit):
        model, tag, L = params.get("model", "BHCorr"), params.get("tag", ""), params.get("L", "")
        if not (model in MODELS and RE_TAG.match(tag) and str(L).isdigit()):
            raise ValueError("no sweep selected")
        L = int(L)
        m = MODELS[model]
        script = m["scripts"].get("submit" if action in ("preview", "recompute_point") else action)
        n_edge = int(params["N_EDGE"]) if str(params.get("N_EDGE", "")).isdigit() else None
        if action in ("submit", "preview", "recompute_obs") and n_edge is not None and L < 2 * n_edge + 3:
            raise ValueError(f"N_EDGE = {n_edge} leaves fewer than 2 bulk bonds for L = {L} (need L ≥ 2·N_EDGE + 3)")
        if action in ("submit", "preview"):
            values = self._values(model, tag, params, "submit")
            maxiter, convtol = int(values.get("MAXITER", 2000)), float(values.get("CONVTOL", 1e-6))
            st = self.status(model, tag, L, n_edge, fresh=True)
            known = {(p["x"], p["y"]): p for p in st["points"]}
            grid = self._grid(model, params) or list(known)
            force = params.get("FORCE") == "1"
            todo, nrun, nrerun, nresume, nskip, nqueued = [], 0, 0, 0, 0, 0
            upload = []   # states of resumed points that are only on the laptop
            for key in grid:
                p = known.get(key, {"state": "none", "has_state": False})
                why = needs_run(p, maxiter, convtol, force)
                if why is None:
                    nqueued += p.get("state") in ("idle", "running", "held")
                    nskip += p.get("state") not in ("idle", "running", "held")
                    continue
                nrun += why == "run"
                nrerun += why == "rerun"
                nresume += why == "resume"
                if why == "resume" and not p.get("on_remote"):
                    upload.append(key)
                todo.append(key)
            emit(f"{len(todo)} point(s) to submit ({nrun} new, {nresume} resumed from their state up to MAXITER, "
                 f"{nrerun} rerun from scratch for the new CONVTOL), {nskip} done, {nqueued} in the queue")
            if not todo:
                return 0
            values["POINTS"] = ",".join(f"{x} {y}" for x, y in todo)
            values["RESUME"] = "0" if force else "1"
            if action == "preview":
                values["DRYRUN"] = "1"
                if upload:
                    emit(f"(submit would first upload {len(upload)} state(s) from the laptop to resume them)")
            else:
                if upload:
                    emit(f"Uploading {len(upload)} state(s) from the laptop to the access point, to resume them ...")
                    rel = [point_file(model, tag, L, x, y) for x, y in upload]
                    rc = self._rsync(rel, self.args.gather_dest + "/", self.remote.path(), emit)
                    if rc != 0:
                        return rc
                rc = self._sync(params, emit, task)
                if rc != 0:
                    return rc
            cmd = f"env {self._env(values)} bash {script} {L} 2>&1"
            emit(f"$ bash {script} {L}  ({len(todo)} points)")
            return self.remote.stream(cmd, emit)
        if action == "recompute_point":
            # One point from scratch (also a converged one), with the SEED the page drew, so the rerun does not
            # reproduce the stored state. The stored state is overwritten.
            if model != "BHU1":
                raise ValueError("recomputing a single point is only implemented for BHU1")
            values = self._values(model, tag, params, "submit")
            if "SEED" not in values:
                raise ValueError("SEED is not set")
            pt = str(params.get("POINT", "")).split()
            if len(pt) != 2:
                raise ValueError("no point selected")
            x, y = (f"{float(v):.4f}" for v in pt)
            st = self.status(model, tag, L, n_edge, fresh=True)
            p = next((q for q in st["points"] if (q["x"], q["y"]) == (x, y)), None)
            if p is not None and p.get("state") in ("idle", "running", "held"):
                raise ValueError(f"point {x} {y} still has a job in the queue ({p['state']}); remove it first")
            values.update(POINTS=f"{x} {y}", FORCE="1", RESUME="0")
            rc = self._sync(params, emit, task)
            if rc != 0:
                return rc
            emit(f"$ bash {script} {L}  (point {x} {y} from scratch, SEED = {values['SEED']})")
            return self.remote.stream(f"env {self._env(values)} bash {script} {L} 2>&1", emit)
        if action == "recompute_obs":
            values = self._values(model, tag, params, "recompute_obs")
            st = self.status(model, tag, L, n_edge, fresh=True)
            todo = [p for p in st["points"] if p["has_state"] and p["obs"] in ("none", "outdated", "failed")
                    and p["state"] not in ("idle", "running", "held")]
            emit(f"{len(todo)} point(s) need observables (missing, outdated or failed)")
            if not todo:
                return 0
            upload = [p for p in todo if "laptop" in p["where"] and not p["on_remote"]]
            if upload:
                emit(f"Uploading {len(upload)} state(s) from the laptop to the access point ...")
                rel = [point_file(model, tag, L, p["x"], p["y"]) for p in upload]
                rc = self._rsync(rel, self.args.gather_dest + "/", self.remote.path(), emit)
                if rc != 0:
                    return rc
            rc = self._sync(params, emit, task)
            if rc != 0:
                return rc
            values["POINTS"] = ",".join(f"{p['x']} {p['y']}" for p in todo)
            emit(f"$ bash {script} {L}  ({len(todo)} points)")
            return self.remote.stream(f"env {self._env(values)} bash {script} {L} 2>&1", emit)
        if action in ("remove_held", "remove_all"):
            c = self.condor_constraint(model, tag, L) + (" && JobStatus == 5" if action == "remove_held" else "")
            cmd = f"condor_rm \"$USER\" -constraint {shlex.quote(c)} 2>&1"
            emit(f"$ condor_rm (jobs of this sweep{', held only' if action == 'remove_held' else ''})")
            return self.remote.stream(cmd, emit)
        if action == "gather":
            return self._gather(model, tag, L, emit)
        if action == "plot":
            if not m["plot_script"]:
                raise ValueError("this model has no plot script: its observables are shown as heatmaps in the grid")
            values = self._values(model, tag, params, "plot")
            if model == "BHU1" and "OBS_COL" not in values:
                raise ValueError("choose an observable in 'Colour by' first")
            env = {**os.environ, **values, "DATA_DIR": "data"}
            cmd = [*self.local_julia, f"--project={os.path.join(REPO, 'examples')}", "--startup-file=no",
                   os.path.join(REPO, m["plot_script"]), str(L)]
            emit(f"$ julia {m['plot_script']} {L}  (on the laptop, data in {self.args.gather_dest}/data)")
            return stream(cmd, emit, cwd=self.args.gather_dest, env=env)
        raise ValueError(f"unknown action {action}")

    def _rsync(self, rel, src, dst, emit, remove_source=False):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n".join(rel) + "\n")
            listfile = f.name
        try:
            cmd = ["rsync", "-a", "--itemize-changes", f"--files-from={listfile}", *self.remote.rsync_ssh()]
            if remove_source:
                cmd.append("--remove-source-files")
            return stream([*cmd, src, dst], emit)
        finally:
            os.unlink(listfile)

    def _gather(self, model, tag, L, emit):
        """Copy the finished results of this sweep to the laptop, then delete them and their logs on the access point."""
        m = MODELS[model]
        X, Y = axes_of(model, tag)
        data = f"data/{m['dir']}/{tag}/L{L}"
        logs = f"logs/{m['dir']}/{tag}/L{L}/condor"
        ite, obs = m["prefix"]["ite"].format(L=L), m["prefix"]["obs"].format(L=L)
        a, b = m["attrs"]
        c = shlex.quote(self.condor_constraint(model, tag, L))
        # Result files older than GATHER_MIN_AGE min of points without a job in the queue (any stage)
        cmd = (f"echo @@FILES; [ -d {data} ] && find {data} -type f \\( -name '{Y}*.jld2' -o -name '{Y}*_obs.csv' \\) "
               f"-mmin +{GATHER_MIN_AGE}; "
               f"echo @@ACTIVE; command -v condor_q >/dev/null && condor_q \"$USER\" -constraint {c} -af {a} {b} BHC_points 2>/dev/null; "
               f"echo @@END")
        rc, out, err = self.remote.run(cmd, timeout=120)
        if rc == 255 and not self.args.local:
            raise ConnectionError(err.strip() or "ssh connection lost")
        s = sections(out)
        active = set()
        for line in s.get("ACTIVE", "").splitlines():
            parts = line.split()
            if len(parts) == 3:
                job = {k: v for k, v in zip((a, b, "BHC_points"), parts) if v != "undefined"}
                active.update(job_points(job, m["attrs"]))
        files, points = [], set()
        for path in s.get("FILES", "").splitlines():
            mt = re.match(rf"^{re.escape(data)}/{X}(-?\d+\.\d{{4}})/{Y}(-?\d+\.\d{{4}})(_obs\.csv|\.jld2)$", path.strip())
            if mt and (mt.group(1), mt.group(2)) not in active:
                files.append(path.strip())
                points.add((mt.group(1), mt.group(2)))
        if not files:
            emit("Nothing to gather: no finished results on the access point (results younger than "
                 f"{GATHER_MIN_AGE} min or of points still in the queue wait for the next Gather).")
            return 0
        emit(f"Copying {len(files)} file(s) of {len(points)} point(s) to the laptop; each file is deleted on the access "
             "point only after it was copied completely:")
        rc = self._rsync(files, self.remote.path(), self.args.gather_dest + "/", emit, remove_source=True)
        if rc != 0:
            emit(f"rsync failed (exit {rc}): nothing more is deleted. Press Gather again to retry.")
            return rc
        missing = [f for f in files if not os.path.isfile(os.path.join(self.args.gather_dest, f))]
        if missing:
            emit(f"ERROR: {len(missing)} file(s) not found on the laptop after the copy, e.g. {missing[0]}")
            return 1
        emit(f"Copied to {os.path.join(self.args.gather_dest, data)}")
        # Logs of the gathered points, and (if no job of this sweep is left) the cluster logs, points and submit files
        rm = [f"rm -f {logs}/{ite}_{X}{x}_{Y}{y}_*.out {logs}/{ite}_{X}{x}_{Y}{y}_*.err "
              f"{logs}/obs/{obs}_{X}{x}_{Y}{y}_*.out {logs}/obs/{obs}_{X}{x}_{Y}{y}_*.err" for x, y in sorted(points)]
        if not active:
            rm.append(f"rm -f {logs}/*.log {logs}/points_*.txt {logs}/resume_*.txt {logs}/*.submit {logs}/obs/*.log "
                      f"{logs}/obs/points_*.txt {logs}/obs/*.submit")
        rm.append(f"for d in {data} logs/{m['dir']}/{tag}; do [ -d \"$d\" ] && find \"$d\" -type d -empty -delete; done; true")
        rc, _, err = self.remote.run("; ".join(rm), timeout=120)
        emit(f"Deleted the logs of {len(points)} gathered point(s) on the access point"
             + ("" if active else ", and the cluster logs and submit files (no job of this sweep left)")
             + ("" if rc == 0 else f" (with errors: {err.strip()})"))
        self.local_cache.pop((model, tag, L), None)
        return 0

    def info(self):
        rc, out, _ = self.remote.run(f"test -f Project.toml && echo repo; {self.remote.julia} --startup-file=no -e 'print(VERSION)' 2>/dev/null; echo; "
                                     "command -v condor_q >/dev/null && echo condor", timeout=60)
        lines = out.split()
        return {"host": "local test: " + self.args.local if self.args.local else self.remote.target,
                "connected": rc != 255 and not self.remote.lost,
                "repo": "repo" in lines, "condor": "condor" in lines,
                "julia": next((x for x in lines if re.match(r"^\d+\.\d+\.\d+", x)), None),
                "gather_dest": self.args.gather_dest}


def make_handler(dash):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path in ("/", "/index.html"):
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        return self._send(200, f.read(), "text/html; charset=utf-8")
                if u.path == "/api/info":
                    return self._send(200, dash.info())
                if u.path == "/api/sweeps":
                    return self._send(200, dash.sweeps())
                if u.path == "/api/status":
                    n_edge = int(q["n_edge"]) if q.get("n_edge", "").isdigit() else None
                    return self._send(200, dash.status(q.get("model", "BHCorr"), q["tag"], q["L"], n_edge, q.get("fresh") == "1"))
                if u.path == "/api/log":
                    return self._send(200, {"text": dash.log(q.get("model", "BHCorr"), q["tag"], q["L"], q["x"], q["y"],
                                                             q.get("stage", "ite"))})
                if u.path == "/api/progress":
                    return self._send(200, dash.progress(q.get("model", "BHCorr"), q["tag"], q["L"], q.get("x"), q.get("y")))
                if u.path == "/api/plot":
                    model = q.get("model", "BHCorr")
                    dash.check_sweep(model, q["tag"], q["L"])
                    path = dash.plot_file(model, q["tag"], q["L"], q.get("obs"))
                    if not path or not os.path.isfile(path):
                        return self._send(404, {"error": "no plot yet"})
                    with open(path, "rb") as f:
                        return self._send(200, f.read(), "image/png")
                if u.path == "/api/task":
                    t = dash.tasks.get(q.get("id")) or (dash.tasks.get(dash.current) if dash.current else None)
                    if t is None:
                        return self._send(200, None)
                    since = int(q.get("since", 0))
                    return self._send(200, {**{k: v for k, v in t.items() if k != "lines"}, "lines": t["lines"][since:],
                                            "n": len(t["lines"])})
                return self._send(404, {"error": "not found"})
            except ConnectionError as e:
                return self._send(503, {"error": f"connection to the access point lost ({e}); restart the dashboard"})
            except Exception as e:  # noqa: BLE001
                return self._send(500, {"error": str(e)})

        def do_POST(self):
            u = urlparse(self.path)
            try:
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if u.path == "/api/action":
                    t = dash.start(body.get("action"), body.get("params", {}))
                    return self._send(200, {"id": t["id"]})
                return self._send(404, {"error": "not found"})
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": str(e)})

    return Handler


def selftest():
    remote = {"points": [
        {"t": "0.1000", "mu": "0.2000", "state": True, "converged": True, "eps": 1e-7, "obs": True, "n_edge": 2,
         "obs_version": "v2", "error": None},
        {"t": "0.2000", "mu": "0.2000", "state": True, "converged": False, "eps": 1e-3, "obs": False, "error": None}],
        "planned": [["0.1000", "0.2000"], ["0.3000", "0.2000"], ["0.3000", "0.4000"], ["0.2000", "0.4000"]]}
    local = {"points": [
        {"t": "0.1000", "mu": "0.2000", "state": True, "converged": False, "eps": 1e-3, "obs": True, "n_edge": 2,
         "obs_version": "v1", "error": None},                                     # older copy: remote wins
        {"t": "0.1000", "mu": "0.4000", "state": True, "converged": False, "eps": 1e-3, "maxiter": 2000, "convtol": 1e-6,
         "obs": True, "n_edge": 2, "obs_version": "v1", "error": None},           # gathered, obs from an old script
        {"t": "0.5000", "mu": "0.2000", "state": True, "converged": True, "eps": 1e-8, "maxiter": 2000, "convtol": 1e-6,
         "obs": True, "n_edge": 1, "obs_version": "v2", "error": None}]}          # gathered, other N_EDGE
    queue = [
        {"ClusterId": 7, "ProcId": 0, "JobStatus": 2, "BHC_stage": "ite", "BHC_t": "0.2000", "BHC_mu": "0.2000"},
        {"ClusterId": 7, "ProcId": 1, "JobStatus": 5, "BHC_stage": "ite", "BHC_t": "0.3000", "BHC_mu": "0.2000",
         "HoldReason": "Error from slot: failed to transfer output"},
        {"ClusterId": 9, "ProcId": 0, "JobStatus": 1, "BHC_stage": "ite", "BHC_t": "0.4000", "BHC_mu": "0.2000"}]
    history = [{"ClusterId": 5, "ProcId": 3, "JobStatus": 4, "ExitCode": 1, "CompletionDate": 100, "BHC_stage": "ite",
                "BHC_t": "0.3000", "BHC_mu": "0.4000"},
               {"ClusterId": 4, "ProcId": 0, "JobStatus": 4, "ExitCode": 0, "CompletionDate": 90, "BHC_stage": "ite",
                "BHC_t": "0.1000", "BHC_mu": "0.2000", "RemoteWallClockTime": 5400.0, "NumJobStarts": 2,
                "RequestMemory": 6144, "ResidentSetSize": 3145728}]
    recs = {(p["x"], p["y"]): p for p in merge(remote, local, queue, history, "v2", 2)}
    got = {k: (p["state"], p["obs"]) for k, p in recs.items()}
    want = {("0.1000", "0.2000"): ("done", "done"), ("0.1000", "0.4000"): ("unconverged", "outdated"),
            ("0.2000", "0.2000"): ("running", "none"), ("0.3000", "0.2000"): ("held", "none"),
            ("0.3000", "0.4000"): ("failed", "none"), ("0.2000", "0.4000"): ("none", "none"),
            ("0.4000", "0.2000"): ("idle", "none"), ("0.5000", "0.2000"): ("done", "outdated")}
    assert got == want, f"\n got  {got}\n want {want}"
    assert recs[("0.1000", "0.2000")]["where"] == ["laptop", "access point"]
    j = recs[("0.1000", "0.2000")]["job"]   # finished job attached to a done point
    assert j["id"] == "4.0" and j["runtime"] == 5400.0 and j["starts"] == 2 and j["mem_peak"] == 3072, j
    assert recs[("0.5000", "0.2000")]["job"] is None   # no history job: no info
    assert recs[("0.1000", "0.4000")]["where"] == ["laptop"] and not recs[("0.1000", "0.4000")]["on_remote"]
    # submit decisions (MAXITER 2000 / 4000, CONVTOL 1e-6)
    assert needs_run(recs[("0.2000", "0.4000")], 2000, 1e-6) == "run"
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6) is None              # converged
    assert needs_run(recs[("0.1000", "0.4000")], 2000, 1e-6) is None              # unconverged, same settings
    assert needs_run(recs[("0.1000", "0.4000")], 4000, 1e-6) == "resume"          # more steps: continue the state
    assert needs_run(recs[("0.1000", "0.4000")], 2000, 1e-8) == "rerun"           # only a smaller CONVTOL
    assert needs_run(recs[("0.2000", "0.2000")], 4000, 1e-6) is None              # in the queue
    assert needs_run(recs[("0.5000", "0.2000")], 2000, 1e-6, force=True) == "run"
    assert sections("@@A\n1\n2\n@@B\n@@END\n") == {"A": "1\n2", "B": "", "END": ""}
    assert parse_json_section("warning\n[1,2]\n", None) == [1, 2]
    # observables jobs of several points (BHC_points): a running chunk applies to each of its points, a finished one
    # is attached to them; an old single-point obs job is still matched
    assert job_points({"BHC_points": "0.1000_0.2000,0.2000_0.2000"}) == [("0.1000", "0.2000"), ("0.2000", "0.2000")]
    assert job_points({"BHC_t": "0.1000", "BHC_mu": "0.2000"}) == [("0.1000", "0.2000")] and job_points({}) == []
    q2 = queue + [{"ClusterId": 11, "ProcId": 0, "JobStatus": 2, "BHC_stage": "obs",
                   "BHC_points": "0.2000_0.2000,0.1000_0.4000"}]
    h2 = history + [{"ClusterId": 10, "ProcId": 1, "JobStatus": 4, "ExitCode": 1, "CompletionDate": 95, "BHC_stage": "obs",
                     "BHC_points": "0.1000_0.2000,0.3000_0.4000", "RemoteWallClockTime": 900.0},
                    {"ClusterId": 3, "ProcId": 0, "JobStatus": 4, "ExitCode": 0, "CompletionDate": 80, "BHC_stage": "obs",
                     "BHC_t": "0.5000", "BHC_mu": "0.2000"}]
    r2 = {(p["x"], p["y"]): p for p in merge(remote, local, q2, h2, "v2", 2)}
    assert r2[("0.2000", "0.2000")]["obs"] == "running" and r2[("0.1000", "0.4000")]["obs"] == "running"
    assert r2[("0.2000", "0.2000")]["obs_job"]["id"] == "11.0"
    assert r2[("0.1000", "0.2000")]["obs"] == "done" and r2[("0.1000", "0.2000")]["obs_job"]["id"] == "10.1"   # its point ok, another failed
    assert r2[("0.3000", "0.4000")]["obs"] == "none"                     # no state: not "failed"
    assert r2[("0.5000", "0.2000")]["obs"] == "outdated" and r2[("0.5000", "0.2000")]["obs_job"]["id"] == "3.0"
    assert point_constraint("0.1", "0.2") == '((BHC_t =?= "0.1" && BHC_mu =?= "0.2") || stringListMember("0.1_0.2", BHC_points) =?= true)'
    # progress: a job of the old run_julia.sh (no markers) hanging in Pkg.instantiate
    old_out = "Host: x\nWelcome to Julia!\nNow installing Juliaup\nJulia was successfully installed\nJulia 1.12.6 (juliaup), HOME = /tmp/k"
    pr = parse_progress(old_out, "info: downloading installer\n   Configure Default Julia version set to '1.12.6'.")
    assert pr["phase"] == "setup: Pkg.instantiate" and pr["phase_since"] is None and pr["iteration"] is None, pr
    # old format in the ITE (no time stamps), colored log lines
    pr = parse_progress(old_out + "\n== 09:32:05 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 10 0.0600 1.3000\n",
                        "[ Info: Iteration 3, cap = 16\n[ Info: \x1b[36mPartition function\x1b[39m\n"
                        "[ Info: \x1b[35mCompress\x1b[39m\n")
    assert (pr["phase"], pr["phase_since"], pr["iteration"], pr["cap"], pr["step"], pr["step_since"]) == \
        ("ITE", "09:32:05", 3, 16, "Compress", None), pr
    # new format: markers and time stamps
    new_out = ("== 10:00:00 setup: waiting for the setup lock\n== 10:00:00 setup: julia install\nJulia 1.12.6 (juliaup)\n"
               "== 10:00:05 setup: Pkg.instantiate (attempt 1 of 3)\n")
    pr = parse_progress(new_out, "  Installed X\n")
    assert (pr["phase"], pr["phase_since"]) == ("setup: Pkg.instantiate (attempt 1 of 3)", "10:00:05"), pr
    ite_err = ("[ Info: 10:09:00 Iteration 1, cap = 4\n[ Info: 10:09:01 Partition function\n"
               "[ Info: 10:09:40 Convergence error: (E, dE) = (-0.28, 0.25)\n[ Info: 10:09:40 Finalize\n"
               "[ Info: 10:09:41 Iteration 2, cap = 4\n[ Info: 10:09:41 Normalization filter\n")
    pr = parse_progress(new_out + "== 10:09:00 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n", ite_err)
    assert (pr["phase"], pr["iteration"], pr["step"], pr["step_since"], pr["dE"]) == \
        ("ITE", 2, "Normalization filter", "10:09:41", 0.25), pr
    pr = parse_progress(new_out + "== 10:09:00 julia hpc/BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n"
                        "== 11:00:00 julia hpc/observables_BoseHubbardCorrelated_NormalizedITE.jl 6 0.1 0.5\n", ite_err)
    assert (pr["phase"], pr["phase_since"], pr["iteration"]) == ("observables", "11:00:00", 2), pr
    dump = ("@@JOB 7.1 ite 0.1000 0.5000\n@@OUT\n" + old_out + "\n\n@@ERR\n\n"
            "@@JOB 7.2 ite 0.2000 0.5000\n@@OUT\n" + new_out + "\n@@ERR\n" + ite_err + "\n@@END\n")
    jobs = parse_progress_dump(dump)
    assert [(j["id"], j["x"], j["phase"]) for j in jobs] == [("7.1", "0.1000", "setup: Pkg.instantiate"),
                                                              ("7.2", "0.2000", "setup: Pkg.instantiate (attempt 1 of 3)")]
    # stall flag: 27 s CPU in 19 h
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RemoteUserCpu": 20,
                   "RemoteSysCpu": 7, "RequestCpus": 8}, now=1000 + 19 * 3600)
    assert ji["stalled"] and ji["cpus"] == 8
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RemoteUserCpu": 3600 * 7,
                   "RequestCpus": 8}, now=1000 + 3600)
    assert not ji["stalled"] and abs(ji["cores_used"] - 7) < 1e-9
    # memory: an out-of-memory hold (peak above the request) and a running job
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 5, "RequestMemory": 6144, "ResidentSetSize": 7500000,
                   "ResidentSetSize_RAW": 6227456})
    assert ji["mem_frac"] > 1 and "mem_now" not in ji
    ji = job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 2, "JobCurrentStartDate": 1000, "RequestMemory": 8192,
                   "ResidentSetSize": 2500000, "ResidentSetSize_RAW": 2171020}, now=2000)
    assert ji["mem_peak"] == 2442 and abs(ji["mem_frac"] - 2442 / 8192) < 1e-12 and abs(ji["mem_now"] - 2171020 / 1024) < 1e-9
    assert "mem_req" not in job_info({"ClusterId": 1, "ProcId": 0, "JobStatus": 1})
    # ---- BHU1: free axes, tag built here, points matched through BHC_x / BHC_y
    fixed = {"T": "0.1", "U": "1.0", "DELTA": "0.0", "OMEGA": "0.0", "R": "0.0", "FILLING": "1", "NMAX": "4",
             "DTAU": "0.05", "MAXD": "32"}
    tag = bhu1_tag({**fixed, "XAXIS": "t", "YAXIS": "Delta"})
    assert tag == "t-Delta_U1.0_Omega0.0_r0.0_f1_nmax4_dtau0.05_D32", tag
    assert bhu1_tag({**fixed, "XAXIS": "Omega", "YAXIS": "r"}) == "Omega-r_t0.1_U1.0_Delta0.0_f1_nmax4_dtau0.05_D32"
    assert axes_of("BHU1", tag) == ("t", "Delta") and axes_of("BHCorr", "U1.0_r0.0") == ("t", "mu")
    assert point_file("BHU1", tag, 8, "0.1000", "0.3000") == \
        f"data/BoseHubbardU1_NormalizedITE/{tag}/L8/t0.1000/Delta0.3000.jld2"
    assert point_file("BHCorr", "U1.0_x", 6, "0.1000", "0.2000", "_obs.csv") == \
        "data/BoseHubbardCorrelated_NormalizedITE/U1.0_x/L6/t0.1000/mu0.2000_obs.csv"
    try:
        bhu1_tag({**fixed, "XAXIS": "t", "YAXIS": "t"})
        raise AssertionError("equal axes accepted")
    except ValueError:
        pass
    u_remote = {"points": [{"x": "0.1000", "y": "0.3000", "state": True, "converged": True, "eps": 1e-8, "obs": True,
                            "n_edge": 1, "obs_version": "w", "error": None, "values": {"S_bar": 0.21, "S_bstd": 0.05}}],
                "planned": [["0.1000", "0.3000"], ["0.2000", "0.3000"]]}
    u_queue = [{"ClusterId": 20, "ProcId": 0, "JobStatus": 2, "BHC_model": "BHU1", "BHC_stage": "ite",
                "BHC_x": "0.2000", "BHC_y": "0.3000"}]
    u = {(p["x"], p["y"]): p for p in merge(u_remote, {"points": []}, u_queue, [], "w", 1, MODELS["BHU1"]["attrs"])}
    assert (u[("0.1000", "0.3000")]["state"], u[("0.1000", "0.3000")]["obs"]) == ("done", "done")
    assert u[("0.1000", "0.3000")]["values"]["S_bar"] == 0.21 and u[("0.2000", "0.3000")]["state"] == "running"
    assert point_constraint("0.1", "0.2", ("BHC_x", "BHC_y")) == \
        '((BHC_x =?= "0.1" && BHC_y =?= "0.2") || stringListMember("0.1_0.2", BHC_points) =?= true)'
    assert "XAXIS" in script_params("BHU1", "submit") and "T" in script_params("BHU1", "submit")
    assert "SEED" in script_params("BHU1", "submit") and "SEED" not in script_params("BHCorr", "submit")
    assert "XAXIS" not in script_params("BHCorr", "submit") and "TS" not in script_params("BHCorr", "submit")
    # the parameters must give the selected sweep's tag
    d = Dashboard.__new__(Dashboard)
    v = d._values("BHU1", tag, {**fixed, "XAXIS": "t", "YAXIS": "Delta", "MAXITER": "100"}, "submit")
    assert v["TAG"] == tag and v["XAXIS"] == "t" and v["MAXITER"] == "100"
    try:
        d._values("BHU1", tag, {**fixed, "U": "2.0", "XAXIS": "t", "YAXIS": "Delta"}, "submit")
        raise AssertionError("tag mismatch accepted")
    except ValueError:
        pass
    # the plot action: observable and window passed, a bad column name refused
    v = d._values("BHU1", tag, {**fixed, "XAXIS": "t", "YAXIS": "Delta", "OBS_COL": "S_bar", "LOG": "0", "XMIN": "0.05",
                                "YMAX": "", "MAXITER": "100"}, "plot")
    assert v["TAG"] == tag and v["OBS_COL"] == "S_bar" and v["XMIN"] == "0.05" and "YMAX" not in v and "MAXITER" not in v
    try:
        d._values("BHU1", tag, {**fixed, "XAXIS": "t", "YAXIS": "Delta", "OBS_COL": "S_bar;rm"}, "plot")
        raise AssertionError("bad OBS_COL accepted")
    except ValueError:
        pass
    assert set(script_params("BHCorr", "plot")) == set(MODELS["BHCorr"]["model_params"] + MODELS["BHCorr"]["plot_params"])
    assert Dashboard._grid("BHU1", {"XS": "0.1 0.2", "YS": "0"}) == [("0.1000", "0.0000"), ("0.2000", "0.0000")]
    # ---- BHRand: a preset of BHCorr, sweeps assigned by tag
    rtag = "U1.0_r0.0_Delta0.0_Omega0.5_nmax4_dtau0.05_D16"
    assert sweep_model("BHCorr", rtag) == "BHRand" and sweep_model("BHCorr", "U1.0_r0.0_Delta0.2_Omega0.5_nmax4_dtau0.05_D16") == "BHCorr"
    assert sweep_model("BHCorr", "U1.0_r0_Delta0_Omega0.50_nmax4_dtau0.05_D16") == "BHRand" and sweep_model("BHU1", rtag) == "BHU1"
    assert axes_of("BHRand", rtag) == ("t", "mu")
    assert point_file("BHRand", rtag, 6, "0.1000", "0.2000") == point_file("BHCorr", rtag, 6, "0.1000", "0.2000")
    rp = {"U": "1.0", "R": "0.0", "DELTA": "0.0", "OMEGA": "0.5", "NMAX": "4", "DTAU": "0.05", "MAXD": "16"}
    assert d._values("BHRand", rtag, rp, "submit")["OMEGA"] == "0.5"
    for bad in ({**rp, "OMEGA": "0.3"}, {k: v for k, v in rp.items() if k != "DELTA"}):
        try:
            d._values("BHRand", rtag, bad, "submit")
            raise AssertionError("preset violation accepted")
        except ValueError:
            pass
    try:
        d._values("BHRand", "U1.0_r0.0_Delta0.2_Omega0.5_nmax4_dtau0.05_D16", {**rp, "DELTA": "0.0"}, "submit")
        raise AssertionError("tag of another model accepted")
    except ValueError:
        pass
    assert script_params("BHRand", "plot") == MODELS["BHCorr"]["model_params"]
    d.args = argparse.Namespace(gather_dest="/x")
    assert d.plot_file("BHRand", rtag, 6) == f"/x/plots/BoseHubbardCorrelated_NormalizedITE_{rtag}_L6_mu.png"
    assert d.plot_file("BHCorr", rtag, 6) == f"/x/plots/BoseHubbardCorrelated_NormalizedITE_{rtag}_L6.png"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host")
    ap.add_argument("--port")
    ap.add_argument("--user")
    ap.add_argument("--remote-dir")
    ap.add_argument("--julia", default=None)
    ap.add_argument("--local-julia", default="julia")
    ap.add_argument("--http-port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--local", default=None)
    ap.add_argument("--gather-dest", default=REPO)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.local:
        args.local = os.path.abspath(args.local)
        args.host, args.port, args.user, args.remote_dir = "local", "0", "local", args.local
    else:
        resolve_connection(args)
    if args.julia is None:
        with open(os.path.join(REPO, "Manifest.toml")) as f:
            version = re.search(r'^julia_version = "(.*)"', f.read(), re.M).group(1)
        args.julia = "julia" if args.local else f"$HOME/.juliaup/bin/julia +{version}"
    args.gather_dest = os.path.abspath(args.gather_dest)
    dash = Dashboard(args)
    if not dash.remote.connect():
        sys.exit("Could not connect to the access point (on the university network or VPN? right password?)")
    server = ThreadingHTTPServer(("127.0.0.1", args.http_port), make_handler(dash))
    url = f"http://localhost:{args.http_port}"
    print(f"Dashboard on {url}  (Ctrl-C to stop; jobs keep running on the cluster)")
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
