#!/usr/bin/env python3
# ================================================================
# pump_review.py -- accuracy review + self-tuning REPORT for the
# ignition scanner (pump_scan.py).  READ-ONLY / propose-only:
# it NEVER modifies pump_scan.py or its scoring - it only measures
# and suggests.
#
# Input : outcomes_log.csv  (OUT_FIELDS written by pump_scan.py):
#   iso,epoch,sym,symbol,tier,score,vr,volz,creep,buyimb,ret5,nh,
#   qv,price,ret1h,ret4h,ret24h,ret72h,mfe72,mae72,win
#   (win = int(ret72h > 0.05), resolved 72h after the alert)
#
# Live scoring being audited (from pump_scan.py, do not change):
#   score = 0.50*(z(vr)+z(volz)+z(creep))/3 + 0.30*z(buyimb) + 0.20*z(nh)
#   gate  : score >= p95, ret5 < 0.20, (vr > 1.3 or volz > 1.5)
#   tiers : p95 ("top 5%") / p98 ("top tier")
#
# Sections printed:
#   [1] ACCURACY   base win rate, win rate by tier, ret72h/mfe/mae
#                  stats, win rate at +5%/+10%/+20% thresholds
#   [2] FEATURE LIFT  per feature: win rate top vs bottom tercile,
#                  AUC and Spearman rank corr vs win (dead weights
#                  become visible here)
#   [3] RETUNE     numpy IRLS logistic regression win~features on a
#                  time split (older 70% train / newest 30% test);
#                  TEST precision vs the current hand-set score,
#                  suggested normalized weights, gate tweaks,
#                  train-vs-test gap (overfit guard)
#   [4] WINNER SIGNATURE  common pattern among alerts that DID pump:
#                  per-feature winner vs loser median/IQR, AUC and
#                  standardized mean diff; STRONG-winner profile
#                  (mfe72>=0.15 or ret24h>=0.10); greedy 1-3 threshold
#                  cuts fit on the older 70% and scored on the newest
#                  30% => SUGGESTED tightened gate (exact thresholds)
#                  with test win-rate gain and alerts-kept cost
#   [4b] STOP-LOSS CALIBRATION  MAE distribution winners vs losers,
#                  winners' |mae72| p75/p90, expectancy simulation over
#                  candidate stops (-3%..-15%): realized = S if mae72<=S
#                  else ret72h; picks the mean-return-maximizing stop and
#                  reports the winners-90th-pct-MAE stop as principled
#                  alternative.  [PROVISIONAL] if resolved winners < 30.
#   [5] DIGEST     compact Discord-ready summary; also POSTed to
#                  DISCORD_WEBHOOK_PUMP if the env var is set
#
# Fewer than MIN_N resolved alerts => everything is PROVISIONAL and
# the script says "not enough data yet - keep collecting".
#
# REMINDER: this improves PRECISION / triage only.  Pump ignition
# stays a low-hit-rate signal; the output is a WATCHLIST, not a buy
# signal.
# ================================================================
import os, sys, json
import urllib.request
import numpy as np
import pandas as pd

OUTCOMES_CSV = os.environ.get("OUTCOMES_CSV", "outcomes_log.csv")
WEBHOOK      = os.environ.get("DISCORD_WEBHOOK_PUMP", "")

FEATURES = ["vr", "volz", "creep", "buyimb", "ret5", "nh"]
# per-feature weights implied by the live composite score
LIVE_W   = {"vr": 0.50/3.0, "volz": 0.50/3.0, "creep": 0.50/3.0,
            "buyimb": 0.30, "ret5": 0.0, "nh": 0.20}
WIN_TH   = 0.05      # win = ret72h > +5% (matches pump_scan.py)
MIN_N    = 150       # below this, all suggestions are provisional
EPS      = 1e-9

# ---- winner-signature settings ----------------------------------
SIG_FEATURES  = FEATURES + ["score", "qv"]   # cuttable columns from OUT_FIELDS
STRONG_MFE    = 0.15     # STRONG winner: mfe72 >= 15% ...
STRONG_R24    = 0.10     # ... or ret24h >= 10%
RULE_QS       = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MIN_KEEP_FRAC = 0.15     # a rule must keep >= 15% of train alerts ...
MIN_KEEP_ABS  = 8        # ... and >= 8 alerts (usable alert count)
MIN_GAIN      = 0.01     # each added cut must add >= +1pp train win rate
# ---- stop-loss calibration settings ------------------------------
# principle: put the stop just beyond where WINNERS rarely dip, so
# stops mostly truncate losers while letting winners run to ret72h.
STOP_GRID        = [-0.03, -0.04, -0.05, -0.06, -0.08, -0.10, -0.15]
STOP_MIN_WINNERS = 30    # fewer resolved winners => [PROVISIONAL]

# pump_scan.py p98 tier label contains this substring (jp "saijoui" = top tier)
P98_MARK = "\\u6700\\u4e0a\\u4f4d".encode("ascii").decode("unicode_escape")

# ---------------------------------------------------------------- utils
def pct(x):
    return "n/a" if (x is None or x != x) else "%.1f%%" % (100.0 * x)

def fnum(x, fmt="%+.4f"):
    return "n/a" if (x is None or x != x) else fmt % x

def load(path):
    if not os.path.exists(path):
        print("no outcomes file found: %s - nothing to review yet." % path)
        sys.exit(0)
    df = pd.read_csv(path)
    numc = (["epoch", "score"] + FEATURES +
            ["qv", "price", "ret1h", "ret4h", "ret24h", "ret72h",
             "mfe72", "mae72", "win"])
    for c in numc:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ret72h", "score"] + FEATURES).copy()
    # recompute win defensively so review always matches the definition
    df["win"] = (df["ret72h"] > WIN_TH).astype(int)
    df = df.sort_values("epoch").reset_index(drop=True)
    t = df.get("tier", pd.Series([""] * len(df))).astype(str)
    df["is98"] = t.str.contains(P98_MARK, regex=False) | t.str.contains("p98")
    return df

def auc_of(x, y):
    """rank-based AUC of feature x vs binary y (Mann-Whitney)."""
    y = np.asarray(y, int); x = np.asarray(x, float)
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(x).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))

def spearman(x, y):
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() < EPS or ry.std() < EPS:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])

def tercile_winrates(df, col):
    """win rate in bottom vs top tercile of df[col]."""
    x = df[col]
    try:
        b = pd.qcut(x, 3, labels=False, duplicates="drop")
    except ValueError:
        b = None
    if b is None or pd.Series(b).nunique() < 2:      # degenerate -> median split
        b = (x > x.median()).astype(int) * 2
    bot = df.loc[b == b.min(), "win"].mean()
    top = df.loc[b == b.max(), "win"].mean()
    return float(bot), float(top)

# ---------------------------------------------------------------- logistic (numpy IRLS)
def fit_logit(X, y, l2=1.0, iters=60):
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))
        wd = p * (1.0 - p) + 1e-6
        R = l2 * np.eye(d + 1); R[0, 0] = 0.0
        H = Xb.T @ (Xb * wd[:, None]) + R
        g = Xb.T @ (y - p) - R @ w
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w

def predict_logit(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))

def retune(df):
    """time split 70/30, IRLS logistic, compare vs live hand-set score."""
    n = len(df)
    ntr = int(round(n * 0.70))
    if ntr < 20 or n - ntr < 10:
        return None
    tr, te = df.iloc[:ntr], df.iloc[ntr:]
    if tr["win"].nunique() < 2:
        return None
    mu = tr[FEATURES].mean().to_numpy(float)
    sd = tr[FEATURES].std().to_numpy(float)
    sd = np.where(sd < EPS, 1.0, sd)
    Xtr = (tr[FEATURES].to_numpy(float) - mu) / sd
    Xte = (te[FEATURES].to_numpy(float) - mu) / sd
    w = fit_logit(Xtr, tr["win"].to_numpy(float))
    ptr = predict_logit(w, Xtr)
    pte = predict_logit(w, Xte)
    auc_tr = auc_of(ptr, tr["win"]); auc_te = auc_of(pte, te["win"])
    # precision@K on TEST: top 30% (min 5) by model prob vs by live score
    k = min(len(te), max(5, int(round(0.30 * len(te)))))
    order_m = np.argsort(-pte)[:k]
    order_l = np.argsort(-te["score"].to_numpy(float))[:k]
    yte = te["win"].to_numpy(int)
    base = float(yte.mean())
    prec_m = float(yte[order_m].mean())
    prec_l = float(yte[order_l].mean())
    coefs = w[1:]
    sw = coefs / (np.sum(np.abs(coefs)) + EPS)   # normalized weights
    return dict(n_train=ntr, n_test=n - ntr, k=k, coefs=coefs,
                sugg_w=dict(zip(FEATURES, sw)),
                auc_train=auc_tr, auc_test=auc_te,
                gap=(auc_tr - auc_te) if auc_tr == auc_tr and auc_te == auc_te else float("nan"),
                base_test=base, prec_model=prec_m, prec_live=prec_l,
                lift_model=(prec_m / base if base > 0 else float("nan")),
                lift_live=(prec_l / base if base > 0 else float("nan")))

# ---------------------------------------------------------------- winner signature
def strong_mask(df):
    """STRONG winners = alerts that clearly pumped: mfe72>=0.15 or ret24h>=0.10."""
    m = pd.Series(False, index=df.index)
    for c, th in (("mfe72", STRONG_MFE), ("ret24h", STRONG_R24)):
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            m = m | (v >= th)
    return m.fillna(False)

def smd_of(xw, xl):
    """standardized mean difference (pooled-sd Cohen's d), winners minus losers."""
    xw = np.asarray(xw, float); xl = np.asarray(xl, float)
    if len(xw) < 2 or len(xl) < 2:
        return float("nan")
    s = np.sqrt(0.5 * (np.var(xw, ddof=1) + np.var(xl, ddof=1)))
    if s < EPS:
        return float("nan")
    return float((xw.mean() - xl.mean()) / s)

def winner_profile(df):
    """per feature: winner median/IQR vs loser median/IQR, strong-winner median,
    and separation (AUC vs win + standardized mean diff)."""
    win, los = df[df["win"] == 1], df[df["win"] == 0]
    stg = df[strong_mask(df)]
    rows = []
    for f in SIG_FEATURES:
        if f not in df.columns:
            continue
        w = pd.to_numeric(win[f], errors="coerce").dropna()
        l = pd.to_numeric(los[f], errors="coerce").dropna()
        s = pd.to_numeric(stg[f], errors="coerce").dropna()
        def q(v, p):
            return float(v.quantile(p)) if len(v) else float("nan")
        rows.append(dict(feat=f,
                         wmed=q(w, 0.50), wq1=q(w, 0.25), wq3=q(w, 0.75),
                         lmed=q(l, 0.50), lq1=q(l, 0.25), lq3=q(l, 0.75),
                         smed=q(s, 0.50),
                         auc=auc_of(df[f], df["win"]),
                         smd=smd_of(w, l)))
    return rows, len(win), len(los), len(stg)

def rule_mask(d, rule):
    m = np.ones(len(d), bool)
    for f, op, thr in rule:
        x = pd.to_numeric(d[f], errors="coerce").to_numpy(float)
        m &= (x >= thr) if op == ">=" else (x <= thr)
    return m

def rule_text(rule):
    return " and ".join("%s %s %s" % (f, op, "%.4g" % thr) for f, op, thr in rule)

def derive_rule(df):
    """greedy search over 1-3 monotone threshold cuts maximizing TRAIN win rate
    while keeping a usable alert count; thresholds are fit on the older 70%
    and the rule is scored once on the newest 30% (no in-sample cherry-pick)."""
    n = len(df)
    ntr = int(round(n * 0.70))
    if ntr < 20 or n - ntr < 10:
        return None
    tr, te = df.iloc[:ntr], df.iloc[ntr:]
    ytr = tr["win"].to_numpy(int); yte = te["win"].to_numpy(int)
    if ytr.sum() == 0:
        return None
    min_keep = max(MIN_KEEP_ABS, int(round(MIN_KEEP_FRAC * ntr)))
    # candidate cuts: train quantiles, direction from train AUC (monotone)
    cands = []
    for f in SIG_FEATURES:
        if f not in tr.columns:
            continue
        a = auc_of(tr[f], tr["win"])
        if a != a:
            continue
        op = ">=" if a >= 0.5 else "<="
        xs = pd.to_numeric(tr[f], errors="coerce")
        for qq in RULE_QS:
            cands.append((f, op, float(xs.quantile(qq))))
    rule, cur = [], np.ones(ntr, bool)
    cur_wr = float(ytr.mean())
    for _ in range(3):                              # at most 3 cuts
        best = None
        for f, op, thr in cands:
            if any(f == g for g, _, _ in rule):     # one cut per feature
                continue
            x = pd.to_numeric(tr[f], errors="coerce").to_numpy(float)
            m = cur & ((x >= thr) if op == ">=" else (x <= thr))
            k = int(m.sum())
            if k < min_keep:
                continue
            wr = float(ytr[m].mean())
            if (best is None or wr > best[0] + 1e-12 or
                    (abs(wr - best[0]) <= 1e-12 and k > best[1])):
                best = (wr, k, (f, op, thr), m)
        if best is None or best[0] < cur_wr + MIN_GAIN:
            break                                   # no cut earns its keep
        cur_wr, _, cut, cur = best
        rule.append(cut)
    out = dict(rule=rule, n_train=ntr, n_test=n - ntr,
               train_base=float(ytr.mean()), train_wr=cur_wr,
               train_kept=int(cur.sum()),
               test_base=float(yte.mean()))
    if rule:
        mte = rule_mask(te, rule)
        kept = int(mte.sum())
        stg = strong_mask(te).to_numpy(bool)
        out.update(test_wr=float(yte[mte].mean()) if kept else float("nan"),
                   test_kept=kept,
                   test_kept_frac=kept / float(len(te)),
                   test_recall=(float(yte[mte].sum()) / yte.sum()
                                if yte.sum() > 0 else float("nan")),
                   test_strong_base=float(stg.mean()),
                   test_strong_wr=float(stg[mte].mean()) if kept else float("nan"))
        b = out["test_base"]
        out["test_lift"] = (out["test_wr"] / b) if (b > 0 and kept) else float("nan")
    return out

# ---------------------------------------------------------------- stop-loss calibration
def stop_calibration(df):
    """calibrate a hard stop from mae72 (max adverse excursion within 72h).
    per alert under candidate stop S (all S negative, mae72 negative):
        realized = S if mae72 <= S (stop touched) else ret72h
    reports mean realized return + fraction of winners never stopped per S,
    picks the grid S maximizing mean realized return, and also reports the
    winners' 90th-pct |mae72| stop as the principled alternative."""
    if "mae72" not in df.columns:
        return None
    d = df.dropna(subset=["mae72", "ret72h"])
    if len(d) == 0:
        return None
    mae = d["mae72"].to_numpy(float)
    ret = d["ret72h"].to_numpy(float)
    win = d["win"].to_numpy(int)
    wm, lm = mae[win == 1], mae[win == 0]
    res = dict(n=len(d), nw=int(win.sum()), nl=int(len(d) - win.sum()))
    res["wmed"] = float(np.median(wm)) if len(wm) else float("nan")
    res["lmed"] = float(np.median(lm)) if len(lm) else float("nan")
    if len(wm):
        aw = np.abs(wm)
        res["w75"] = float(np.percentile(aw, 75))
        res["w90"] = float(np.percentile(aw, 90))
    else:
        res["w75"] = res["w90"] = float("nan")
    res["nostop"] = float(ret.mean())
    rows = []
    for S in STOP_GRID:
        stopped = mae <= S
        realized = np.where(stopped, S, ret)
        rows.append(dict(S=S,
                         mean=float(realized.mean()),
                         stopped=float(stopped.mean()),
                         win_kept=(float((wm > S).mean()) if len(wm) else float("nan")),
                         winrate=float((realized > WIN_TH).mean())))
    res["rows"] = rows
    res["best"] = max(rows, key=lambda r: r["mean"])
    res["alt"] = (-res["w90"]) if res["w90"] == res["w90"] else float("nan")
    res["provisional"] = res["nw"] < STOP_MIN_WINNERS
    return res

# ---------------------------------------------------------------- suggestions
def build_suggestions(df, lifts, rt, provisional):
    """returns list of (confidence 0-1, text), best first."""
    n = len(df); out = []
    nconf = min(0.30, n / 1000.0)          # more data -> more confidence
    # 1) dead-weight features currently carrying weight in the live score
    for f in FEATURES:
        L = lifts[f]
        a, lp = L["auc"], L["lift"]
        if LIVE_W[f] > 0 and a == a and abs(a - 0.5) <= 0.03 and abs(lp) <= 0.03:
            out.append((0.45 + nconf,
                        "drop %s from the score (AUC %.2f ~ coin flip, tercile lift %+.1fpp); "
                        "re-normalize the remaining weights" % (f, a, lp * 100)))
    # 2) freshness gate: does high ret5 (already-extended) hurt even below 0.20?
    a5 = lifts["ret5"]["auc"]
    if a5 == a5 and a5 < 0.45:
        out.append((0.50 + nconf,
                    "tighten freshness gate: higher ret5 => fewer wins (AUC %.2f); "
                    "consider FRESH 0.20 -> 0.15" % a5))
    # 3) tier threshold: is p98 really better than p95?
    d98 = df[df["is98"]]; d95 = df[~df["is98"]]
    if len(d98) >= 10 and len(d95) >= 10:
        w98, w95 = d98["win"].mean(), d95["win"].mean()
        if w98 - w95 >= 0.05:
            out.append((0.55 + nconf,
                        "raise the alert gate from p95 toward p98: p98 tier wins %s vs %s "
                        "for p95-only (n=%d/%d)" % (pct(w98), pct(w95), len(d98), len(d95))))
        elif w98 <= w95:
            out.append((0.40 + nconf,
                        "p98 tier shows no edge over p95 (%s vs %s) - keep the p95 gate, "
                        "tier badge is cosmetic for now" % (pct(w98), pct(w95))))
    # 4) reweight toward the logistic fit if it beats the live score out of sample
    if rt is not None and rt["gap"] == rt["gap"]:
        beats = rt["prec_model"] - rt["prec_live"]
        overfit = rt["gap"] > 0.10
        if beats >= 0.03 and not overfit:
            wtxt = " ".join("%s %+.2f" % (f, rt["sugg_w"][f]) for f in FEATURES)
            out.append((0.60 + nconf,
                        "adopt logistic weights (test precision %s vs %s live, gap %.2f): %s"
                        % (pct(rt["prec_model"]), pct(rt["prec_live"]), rt["gap"], wtxt)))
        elif overfit:
            out.append((0.35,
                        "logistic fit looks OVERFIT (train-test AUC gap %.2f) - keep current "
                        "weights, collect more data" % rt["gap"]))
    if not out:
        out.append((0.30, "no data-supported change yet - keep current weights and gate"))
    out.sort(key=lambda t: -t[0])
    if provisional:
        out = [(c * 0.5, "[PROVISIONAL n=%d<%d] %s" % (n, MIN_N, s)) for c, s in out]
    return out

# ---------------------------------------------------------------- discord
def post_discord(text):
    if not WEBHOOK:
        print("[no webhook - digest printed only]")
        return
    for i in range(0, len(text), 1800):
        try:
            req = urllib.request.Request(
                WEBHOOK,
                data=json.dumps({"content": text[i:i + 1800]}).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "pump-review/1.0"})
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as e:
            print("discord err", e)

# ---------------------------------------------------------------- main
def main():
    df = load(OUTCOMES_CSV)
    n = len(df)
    if n == 0:
        print("outcomes_log.csv has no resolved rows yet - keep collecting.")
        return
    provisional = n < MIN_N
    P = print

    P("=" * 66)
    P("PUMP REVIEW - accuracy audit of resolved ignition alerts")
    P("file: %s | rows: %d | span: %s .. %s" %
      (OUTCOMES_CSV, n, str(df["iso"].iloc[0])[:10], str(df["iso"].iloc[-1])[:10]))
    if provisional:
        P("*** not enough data yet - keep collecting (n=%d < %d). ***" % (n, MIN_N))
        P("*** everything below is PROVISIONAL. ***")
    P("=" * 66)

    # ---- [1] accuracy -------------------------------------------------
    base = df["win"].mean()
    d98, d95 = df[df["is98"]], df[~df["is98"]]
    P("[1] ACCURACY (win = ret72h > +5%, the pump_scan.py definition)")
    P("  base win rate          : %s  (%d/%d)" % (pct(base), int(df["win"].sum()), n))
    P("  p98 tier ('top tier')  : %s  (n=%d)" %
      (pct(d98["win"].mean() if len(d98) else float("nan")), len(d98)))
    P("  p95 tier ('top 5%%')    : %s  (n=%d)" %
      (pct(d95["win"].mean() if len(d95) else float("nan")), len(d95)))
    for c in ["ret72h", "mfe72", "mae72"]:
        s = df[c].dropna()
        P("  %-6s median / mean   : %s / %s" %
          (c, fnum(s.median(), "%+.3f"), fnum(s.mean(), "%+.3f")))
    for th in [0.05, 0.10, 0.20]:
        P("  win rate at ret72h>+%2.0f%% : %s" % (th * 100, pct((df["ret72h"] > th).mean())))
    P("  win rate at ret72h> 0%%  : %s" % pct((df["ret72h"] > 0).mean()))

    # ---- [2] feature lift ---------------------------------------------
    P("")
    P("[2] FEATURE LIFT (win rate bottom vs top tercile; AUC/Spearman vs win)")
    P("  %-7s %8s %8s %8s %6s %9s  %s" %
      ("feat", "botTerc", "topTerc", "lift_pp", "AUC", "spearman", "verdict"))
    lifts = {}
    for f in FEATURES + ["score"]:
        bot, top = tercile_winrates(df, f)
        a = auc_of(df[f], df["win"]); sp = spearman(df[f], df["win"])
        lifts[f] = dict(bot=bot, top=top, lift=top - bot, auc=a, sp=sp)
        if a != a:
            verdict = "n/a"
        elif abs(a - 0.5) <= 0.03 and abs(top - bot) <= 0.03:
            verdict = "DEAD WEIGHT"
        elif a < 0.47 or (top - bot) < -0.03:
            verdict = "INVERTED (hurts)"
        elif a >= 0.58 or (top - bot) >= 0.08:
            verdict = "strong"
        else:
            verdict = "mild"
        P("  %-7s %8s %8s %+7.1f %6s %9s  %s" %
          (f, pct(bot), pct(top), (top - bot) * 100,
           fnum(a, "%.2f"), fnum(sp, "%+.2f"), verdict))

    # ---- [3] retune ----------------------------------------------------
    P("")
    P("[3] RETUNE SUGGESTION (IRLS logistic, older 70% train / newest 30% test)")
    rt = retune(df)
    if rt is None:
        P("  too few rows / single-class train split - retune skipped.")
    else:
        P("  split: train n=%d, test n=%d (time-ordered)" % (rt["n_train"], rt["n_test"]))
        P("  AUC train %.3f | AUC test %.3f | gap %+.3f %s" %
          (rt["auc_train"], rt["auc_test"], rt["gap"],
           "(OVERFIT WARNING)" if rt["gap"] > 0.10 else "(ok)"))
        P("  TEST precision@top%d: model %s vs current hand-set score %s (base %s)" %
          (rt["k"], pct(rt["prec_model"]), pct(rt["prec_live"]), pct(rt["base_test"])))
        P("  TEST lift vs base   : model %sx vs current %sx" %
          (fnum(rt["lift_model"], "%.2f"), fnum(rt["lift_live"], "%.2f")))
        P("  current live weights (pump_scan.py, unchanged): " +
          " ".join("%s %.3f" % (f, LIVE_W[f]) for f in FEATURES))
        P("  SUGGESTED normalized weights (sign = direction on z-scored feature):")
        P("    " + " ".join("%s %+.3f" % (f, rt["sugg_w"][f]) for f in FEATURES))
    suggs = build_suggestions(df, lifts, rt, provisional)
    P("  suggestions (confidence | change), propose-only:")
    for c, s in suggs:
        P("    %.2f | %s" % (c, s))
    if provisional:
        P("  NOTE: not enough data yet - keep collecting; all suggestions provisional.")

    # ---- [4] winner signature -------------------------------------------
    P("")
    P("[4] WINNER SIGNATURE (common pattern among alerts that DID pump)")
    prof, nw, nl, ns = winner_profile(df)
    sig_prov = nw < MIN_N
    P("  winners (win=1): %d | losers: %d | STRONG winners (mfe72>=%.2f or ret24h>=%.2f): %d"
      % (nw, nl, STRONG_MFE, STRONG_R24, ns))
    if sig_prov:
        P("  *** PROVISIONAL: only %d resolved winners (< %d) - thresholds will drift. ***"
          % (nw, MIN_N))
    ws = None
    if nw == 0 or nl == 0:
        P("  need both winners and losers to profile - section skipped.")
    else:
        def g3(x):
            return "n/a" if x != x else "%.3g" % x
        P("  winner ranges vs losers (median [q1,q3]); SMD = standardized mean diff:")
        P("  %-7s %24s %24s %9s %5s %6s" %
          ("feat", "WINNER med [q1,q3]", "LOSER med [q1,q3]", "strongMed", "AUC", "SMD"))
        for r in prof:
            P("  %-7s %8s [%7s,%7s] %8s [%7s,%7s] %9s %5s %6s" %
              (r["feat"], g3(r["wmed"]), g3(r["wq1"]), g3(r["wq3"]),
               g3(r["lmed"]), g3(r["lq1"]), g3(r["lq3"]), g3(r["smed"]),
               fnum(r["auc"], "%.2f"), fnum(r["smd"], "%+.2f")))
        P("")
        P("  tightened-gate search: greedy 1-3 monotone cuts, fit on older 70%,")
        P("  scored once on newest 30% (current gate = every logged alert, so the")
        P("  TEST base win rate below IS the current gate's win rate).")
        ws = derive_rule(df)
        if ws is None:
            P("  too few rows / no winners in train split - rule search skipped.")
        elif not ws["rule"]:
            P("  no threshold cut beat the current gate by >=%.0fpp on train while" % (MIN_GAIN * 100))
            P("  keeping a usable alert count - keep the current gate, collect more data.")
        else:
            P("  RULE (fit on train n=%d): %s" % (ws["n_train"], rule_text(ws["rule"])))
            P("  TRAIN win rate %s vs base %s | kept %d/%d alerts" %
              (pct(ws["train_wr"]), pct(ws["train_base"]), ws["train_kept"], ws["n_train"]))
            P("  TEST  win rate %s vs current-gate %s -> lift %sx" %
              (pct(ws["test_wr"]), pct(ws["test_base"]), fnum(ws["test_lift"], "%.2f")))
            P("  TEST  alerts kept %d/%d (%s) | winner recall %s | strong-win rate %s vs %s" %
              (ws["test_kept"], ws["n_test"], pct(ws["test_kept_frac"]),
               pct(ws["test_recall"]), pct(ws["test_strong_wr"]), pct(ws["test_strong_base"])))
            P("  SUGGESTED tightened gate (propose-only; paste into pump_scan.py 'hot'")
            P("  filter yourself if you accept the trade-off):")
            P("    keep alert only if: %s" % rule_text(ws["rule"]))
            P("    expected: win rate %s -> %s (%+.1fpp) at the cost of ~%s fewer alerts%s" %
              (pct(ws["test_base"]), pct(ws["test_wr"]),
               ((ws["test_wr"] - ws["test_base"]) * 100
                if ws["test_wr"] == ws["test_wr"] else float("nan")),
               pct(1.0 - ws["test_kept_frac"]),
               "  [PROVISIONAL]" if sig_prov else ""))
        P("  REMINDER: this raises PRECISION by dropping alerts - pump ignition stays")
        P("  a low-hit-rate signal; the output is a WATCHLIST, not a buy signal.")

    # ---- [4b] stop-loss calibration ---------------------------------------
    P("")
    P("[4b] STOP-LOSS CALIBRATION (from mae72; stop just beyond where winners rarely dip)")
    sl = stop_calibration(df)
    if sl is None or sl["nw"] == 0 or sl["nl"] == 0:
        P("  need resolved rows with mae72 and both winners and losers - section skipped.")
        sl = None
    else:
        P("  rows with mae72: %d | winners (win=1): %d | losers: %d" %
          (sl["n"], sl["nw"], sl["nl"]))
        if sl["provisional"]:
            P("  *** [PROVISIONAL]: only %d resolved winners (< %d) - stop will drift. ***"
              % (sl["nw"], STOP_MIN_WINNERS))
        P("  mae72 median        : winners %s vs losers %s" %
          (fnum(sl["wmed"], "%+.3f"), fnum(sl["lmed"], "%+.3f")))
        P("  winners |mae72|     : p75 %s | p90 %s  (winners rarely dip past these)" %
          (fnum(sl["w75"], "%.3f"), fnum(sl["w90"], "%.3f")))
        P("  simulated per alert : realized = S if mae72<=S else ret72h")
        P("  (fills at exactly S; slippage/gaps/intra-hour order not modeled)")
        P("  %8s %9s %9s %12s %12s" %
          ("stop", "meanRet", "stopped", "winnersKept", "winRate+5%"))
        for r in sl["rows"]:
            P("  %8s %9s %9s %12s %12s%s" %
              ("%.0f%%" % (r["S"] * 100), fnum(r["mean"], "%+.4f"), pct(r["stopped"]),
               pct(r["win_kept"]), pct(r["winrate"]),
               "   <-- best" if r is sl["best"] else ""))
        b = sl["best"]
        P("  no-stop mean ret72h : %s" % fnum(sl["nostop"], "%+.4f"))
        P("  DATA-DERIVED STOP   : %.0f%%  (mean realized %s vs no-stop %s, %+.1fpp)%s" %
          (b["S"] * 100, fnum(b["mean"], "%+.4f"), fnum(sl["nostop"], "%+.4f"),
           (b["mean"] - sl["nostop"]) * 100,
           "  [PROVISIONAL]" if sl["provisional"] else ""))
        P("  PRINCIPLED ALT      : %s  (winners' 90th-pct |mae72|; place the stop just beyond it)"
          % fnum(sl["alt"], "%+.3f"))
        P("  propose-only: sizing / fees / slippage not modeled - validate before use.")

    # ---- [5] digest ------------------------------------------------------
    ranked = sorted(FEATURES, key=lambda f: -(lifts[f]["lift"]
                                              if lifts[f]["lift"] == lifts[f]["lift"] else -9))
    top2, bot2 = ranked[:2], ranked[-2:][::-1]
    def dline(f):
        return "%s (%+.1fpp, AUC %s)" % (f, lifts[f]["lift"] * 100, fnum(lifts[f]["auc"], "%.2f"))
    today = str(df["iso"].iloc[-1])[:10]
    dig = []
    dig.append("**PUMP REVIEW** (thru %s, N=%d resolved)" % (today, n))
    if provisional:
        dig.append("not enough data yet - keep collecting (n<%d); all provisional" % MIN_N)
    dig.append("base win(+5%%/72h): %s | p98 tier: %s (n=%d) | p95 tier: %s (n=%d)" %
               (pct(base),
                pct(d98["win"].mean() if len(d98) else float("nan")), len(d98),
                pct(d95["win"].mean() if len(d95) else float("nan")), len(d95)))
    dig.append("top lift : " + ", ".join(dline(f) for f in top2))
    dig.append("low lift : " + ", ".join(dline(f) for f in bot2))
    dig.append("suggest  : " + suggs[0][1])
    if ws is not None and ws.get("rule"):
        dig.append("signature: %s -> TEST win %s vs gate %s, keeps %s of alerts%s" %
                   (rule_text(ws["rule"]), pct(ws["test_wr"]), pct(ws["test_base"]),
                    pct(ws["test_kept_frac"]), " [PROVISIONAL]" if sig_prov else ""))
    else:
        dig.append("signature: no tightened gate beats the current one yet - keep collecting")
    if sl is not None:
        b = sl["best"]
        dig.append("stop-loss: suggested %.0f%% (mean ret %s vs %s no-stop, %+.1fpp; winners-p90 alt %s)%s" %
                   (b["S"] * 100, fnum(b["mean"], "%+.3f"), fnum(sl["nostop"], "%+.3f"),
                    (b["mean"] - sl["nostop"]) * 100, fnum(sl["alt"], "%+.3f"),
                    " [PROVISIONAL]" if sl["provisional"] else ""))
    dig.append("_precision/triage tuning only - pumps stay low-hit; watchlist, not buy signals_")
    digest = "\n".join(dig)
    P("")
    P("[5] DISCORD DIGEST")
    P("-" * 66)
    P(digest)
    P("-" * 66)
    post_discord(digest)

if __name__ == "__main__":
    main()
