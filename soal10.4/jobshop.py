import heapq, math, sys, numpy as np

MACH0 = [3, 2, 4, 3, 1] # mesin per stasiun 1..5
ROUTE = {1: [3, 1, 2, 5], 2: [4, 1, 3], 3: [2, 5, 1, 4, 3]}
MEAN  = {1: [.50, .60, .85, .50], 2: [1.10, .80, .75], 3: [1.20, .25, .70, .90, 1.00]} # = k*ExpoMean pada ERLA(ExpoMean,2) di Arena
PROB  = [.3, .5, .2]; IAT = 0.25
WARM, END = 80.0, 800.0 # 10 hari x 8 jam ; 80 + 90 hari x 8 jam

def replicate(mach, rng, rule="FIFO"):
    """Satu replikasi. Return X = rata-rata total delay antrian per job (job yg selesai di [WARM, END])."""
    free = list(mach); queues = [[] for _ in range(6)]; ev = []; seq = 0
    def push(t, kind, data):
        nonlocal seq; seq += 1; heapq.heappush(ev, (t, seq, kind, data))
    def prio(job):
        k, ty = job['k'], job['type']
        return 0 if rule == "FIFO" else (MEAN[ty][k] if rule == "SPT" else sum(MEAN[ty][k:]))
    def start(st, job, t):
        free[st] -= 1; push(t + rng.gamma(2, MEAN[job['type']][job['k']]/2), 'D', (st, job)) # Erlang-2
    push(rng.exponential(IAT), 'A', None); delays = []
    while ev:
        t, _, kind, data = heapq.heappop(ev)
        if t > END: break
        if kind == 'A':
            push(t + rng.exponential(IAT), 'A', None)
            job = dict(type=int(rng.choice([1, 2, 3], p=PROB)), k=0, delay=0.0); kind, data = 'Q', job
        if kind == 'Q':
            job = data; st = ROUTE[job['type']][job['k']]
            if free[st-1] > 0: start(st-1, job, t)
            else: job['q'] = t; heapq.heappush(queues[st-1], (prio(job), t, id(job), job))
        elif kind == 'D':
            st, job = data; free[st] += 1
            if queues[st]:
                nj = heapq.heappop(queues[st])[3]; nj['delay'] += t - nj['q']; start(st, nj, t)
            job['k'] += 1
            if job['k'] == len(ROUTE[job['type']]):
                if t >= WARM: delays.append(job['delay'])
            else: push(t, 'Q', job)
    return float(np.mean(delays))

def h_DD(k, Pstar, n0):
    """h1 dari  P* = Integral f_t(x) * [F_t(h - x)]^(k-1) dx ,  t = Student-t dengan n0-1 derajat bebas."""
    nu = n0 - 1; x = np.linspace(-60, 60, 120001); dx = x[1] - x[0]
    lc = math.lgamma((nu+1)/2) - math.lgamma(nu/2) - 0.5*math.log(nu*math.pi)
    pdf = np.exp(lc - (nu+1)/2*np.log1p(x**2/nu)); cdf = np.cumsum(pdf)*dx - pdf*dx/2
    P = lambda h: float(np.sum(pdf*np.interp(h-x, x, cdf, left=0.0, right=1.0)**(k-1))*dx)
    lo, hi = 0.1, 20.0
    for _ in range(60):
        mid = (lo+hi)/2; lo, hi = (mid, hi) if P(mid) < Pstar else (lo, mid)
    return (lo+hi)/2

def DD(systems, n0=20, Pstar=.90, dstar=1.0, seed=2026, first_stage=None):
    k = len(systems); h = h_DD(k, Pstar, n0)
    rngs = [np.random.default_rng([seed, i]) for i in range(k)] # aliran acak independen tiap sistem
    out = []
    for i, (name, mach) in enumerate(systems.items()):
        x1 = np.array(first_stage[i], float) if first_stage else np.array([replicate(mach, rngs[i]) for _ in range(n0)])
        m1, s2 = x1.mean(), x1.var(ddof=1)
        N = max(n0+1, math.ceil(h*h*s2/dstar**2))
        x2 = np.array([replicate(mach, rngs[i]) for _ in range(N-n0)]); m2 = x2.mean()
        inner = 1 - N/n0*(1 - (N-n0)*dstar**2/(h*h*s2))
        W1 = n0/N*(1 + math.sqrt(inner)); W2 = 1 - W1
        out.append(dict(name=name, mach=mach, x1=x1, x2=x2, m1=m1, s2=s2, N=N, m2=m2, W1=W1, W2=W2, xt=W1*m1+W2*m2, inner=inner))
    return h, out

def summary(h, res, n0, Pstar, dstar):
    print(f"DD  k={len(res)}  n0={n0}  P*={Pstar}  d*={dstar}  h1={h:.4f}")
    print(f"{'Sistem':30s}{'S^2':>9s}{'N':>6s}{'Xbar1':>9s}{'Xbar2':>9s}{'W1':>8s}{'Xtilde':>9s}")
    for r in res: print(f"{r['name']:30s}{r['s2']:9.3f}{r['N']:6d}{r['m1']:9.3f}{r['m2']:9.3f}{r['W1']:8.3f}{r['xt']:9.3f}")
    rest = sorted(res, key=lambda r: r['xt']); gap = rest[1]['xt'] - rest[0]['xt']
    print(f"Pilih: {rest[0]['name']}  (selisih dgn terbaik ke-2 = {gap:.3f} jam {'>=' if gap >= dstar else '<'} d*={dstar})")
    print("Untuk rumus & perhitungan lengkap: jalankan dengan opsi --rumus")

def explain(h, res, n0, Pstar, dstar):
    k = len(res); L = "-"*78
    print(f"\n{L}\nRUMUS & PERHITUNGAN PROSEDUR DD  (k={k}, n0={n0}, P*={Pstar}, d*={dstar})\n{L}")
    print(f"[0] Konstanta h1 : solusi  P* = Integral f_t(x) * [F_t(h - x)]^(k-1) dx   (t dgn n0-1 = {n0-1} db)")
    print(f"                   -> h1 = {h:.4f}   (cek dgn Tabel 10.11 di buku)\n")
    for r in res:
        print(f"== {r['name']}  mesin per stasiun = {r['mach']}")
        print(f"[1] Tahap 1: {n0} replikasi, X_j = rata-rata total delay per job pada replikasi ke-j")
        print("    X_j =", np.round(r['x1'], 3).tolist())
        print(f"    Xbar1 = (1/n0) * sum X_j                  = {r['m1']:.4f}")
        print(f"    S^2   = sum (X_j - Xbar1)^2 / (n0-1)      = {r['s2']:.4f}")
        print(f"[2] N = max{{ n0+1 , ceil( h1^2 * S^2 / d*^2 ) }}")
        print(f"      = max{{ {n0+1} , ceil( {h:.4f}^2 * {r['s2']:.4f} / {dstar}^2 ) }} = max{{ {n0+1} , ceil({h*h*r['s2']/dstar**2:.2f}) }} = {r['N']}")
        print(f"[3] Tahap 2: tambah N-n0 = {r['N']-n0} replikasi  ->  Xbar2 = (1/(N-n0)) * sum X_j = {r['m2']:.4f}")
        print(f"[4] W1 = (n0/N) * [ 1 + sqrt( 1 - (N/n0) * (1 - (N-n0)*d*^2 / (h1^2*S^2)) ) ]")
        print(f"       = ({n0}/{r['N']}) * [ 1 + sqrt({r['inner']:.4f}) ] = {r['W1']:.4f}      W2 = 1 - W1 = {r['W2']:.4f}")
        print(f"[5] Xtilde = W1*Xbar1 + W2*Xbar2 = {r['W1']:.4f}*{r['m1']:.4f} + {r['W2']:.4f}*{r['m2']:.4f} = {r['xt']:.4f}\n")
    best = min(res, key=lambda r: r['xt']); rest = sorted(res, key=lambda r: r['xt'])
    print(f"[6] Keputusan: pilih sistem dengan Xtilde terkecil (ukuran 'makin kecil makin baik')")
    for r in rest: print(f"      {r['name']:28s} Xtilde = {r['xt']:.4f}")
    gap = rest[1]['xt'] - rest[0]['xt']
    print(f"    -> {best['name']}.  Selisih dgn terbaik ke-2 = {gap:.3f} jam {'>=' if gap >= dstar else '<'} d* = {dstar}")
    print(f"    Jaminan: P(CS) >= {Pstar} bila rata-rata terbaik memang lebih kecil >= d* dari semua yg lain.")

if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True); args = sys.argv[1:]
    stations = (1, 2, 3, 4, 5) if "--k5" in args else (1, 2, 4)
    systems = {f"Tambah mesin di stasiun {s}": [m + (j+1 == s) for j, m in enumerate(MACH0)] for s in stations}
    fs = None
    if "--arena" in args:
        files = args[args.index("--arena")+1:][:len(stations)]
        fs = [[float(l) for l in open(f).read().split()] for f in files]; assert all(len(x) == len(fs[0]) for x in fs)
    n0 = len(fs[0]) if fs else 20
    h, res = DD(systems, n0=n0, seed=7 if "--k5" in args else 2026, first_stage=fs)
    (explain if "--rumus" in args else summary)(h, res, n0, .90, 1.0)
    if "--policy" in args:
        print("\n== Tambahan: kebijakan antrian (n=30, selang 95%: mean +- t(0.975;29)*S/sqrt(30), t=2.0452) ==")
        cfgs = {"Tanpa tambahan": MACH0, **{k_: v for k_, v in systems.items()}}
        for ci, (cname, mach) in enumerate(cfgs.items()):
            for ri, rule in enumerate(("FIFO", "SPT", "SRW")):
                rng = np.random.default_rng([99, ci*10+ri]); x = np.array([replicate(mach, rng, rule) for _ in range(30)])
                print(f"{cname:30s} {rule:5s} mean={x.mean():8.3f}  +-{2.0452*x.std(ddof=1)/math.sqrt(30):6.3f}")
