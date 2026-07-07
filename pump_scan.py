#!/usr/bin/env python3
# ================================================================
# 初動点火アラート (仕手スキャナー) V2 ── ログ&結果トラッキング付き
# data-api.binance.vision(現物・鍵不要・451回避) 検証済 lift~3.8x帯のみ通知
# 別チャンネル用Webhook: GitHub Secrets DISCORD_WEBHOOK_PUMP
# V2追加点:
#   ・全アラートを alerts_log.csv に記録(特徴量+発火時価格)
#   ・pending.json で未解決アラートを保持
#   ・毎回、72h以上経過したアラートの +1h/4h/24h/72h リターンと
#     MFE/MAE を outcomes_log.csv に追記 → ラベル付きデータが貯まる
#   ※ pump.yml は *.csv / pending.json も commit する事(git add -A 推奨)
# V2.1 additions (ADDITIVE ONLY - score/gate unchanged, collect-first):
#   * takerbuy: taker-buy ratio from RAW 12-field klines (idx9/idx5,
#     last 5 days; >0.5 = net aggressive buying). ONE raw kline fetch
#     per symbol now feeds BOTH the OHLCV features and takerbuy.
#   * obimb: order-book bid-depth share (top ~20 levels; >0.5 = more
#     bids), fetched ONLY for finalist hot candidates for speed.
#   Both are LOGGED (alerts/outcomes CSV) and shown in the alert line
#   for later validation; they do NOT gate anything yet.
#   Old CSVs (old header) are migrated in place: new columns appended,
#   old rows padded with blanks - no data loss.
# ================================================================
import os, time, json, csv, datetime as dt, numpy as np, requests, ccxt

WEBHOOK   = os.environ.get("DISCORD_WEBHOOK_PUMP","")
EVENT     = os.environ.get("GITHUB_EVENT_NAME","")
STATE     = "pump_state.json"
PENDING   = "pending.json"
ALERTS_CSV   = "alerts_log.csv"
OUTCOMES_CSV = "outcomes_log.csv"
MIN_QV   = 1_000_000
MAX_SCAN = 300
FRESH    = 0.20
TOPN     = 5
COOLDOWN = 24*3600
RESOLVE_AFTER = 72*3600        # 何秒経過したら結果確定するか(=72h)
DROP_AFTER    = 12*24*3600     # これ以上古い未解決は破棄
EPS = 1e-9
STABLE = {'USDC','FDUSD','TUSD','BUSD','DAI','EUR','USDP','AEUR','EURI','XUSD','USD1'}

EX = ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot','fetchMarkets':['spot']}})
EX.urls['api']['public'] = 'https://data-api.binance.vision/api/v3'

ALERT_FIELDS = ["iso","epoch","sym","symbol","tier","score","vr","volz","creep","buyimb","ret5","nh","qv","price","takerbuy","obimb"]
OUT_FIELDS   = ALERT_FIELDS + ["ret1h","ret4h","ret24h","ret72h","mfe72","mae72","win"]

def discord(text):
    if not WEBHOOK:
        print("[no webhook]\n"+text); return
    for i in range(0,len(text),1800):
        try: requests.post(WEBHOOK,json={"content":text[i:i+1800]},timeout=20)
        except Exception as e: print("discord err",e)

def load_json(p,default):
    try: return json.load(open(p))
    except Exception: return default

def migrate_csv(path, fields):
    # V2.1: older CSVs were written with a shorter header (no takerbuy/obimb).
    # If the existing header lacks any current field, rewrite the file once:
    # same rows, new header, blanks for the added columns. Write to a temp
    # file and os.replace() so a crash can never corrupt the original.
    if not os.path.exists(path): return
    try:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
    except Exception:
        return
    if not header or header == list(fields): return
    if all(c in header for c in fields): return   # already has every field
    try:
        with open(path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in fields})
        os.replace(tmp, path)
        print("migrated %s: header %d -> %d cols (old rows padded with blanks)"
              % (path, len(header), len(fields)))
    except Exception as e:
        print("csv migrate err", path, e)

def append_csv(path, rows, fields):
    if not rows: return
    migrate_csv(path, fields)      # V2.1: upgrade old header in place if needed
    exists = os.path.exists(path)
    with open(path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists: w.writeheader()
        for r in rows: w.writerow({k:r.get(k) for k in fields})

def slope(y):
    x=np.arange(len(y)); return float(np.polyfit(x,y,1)[0])/(np.mean(np.abs(y))+EPS)

def fo(sym,tf,lim,minlen=60):
    try: o=EX.fetch_ohlcv(sym,tf,limit=lim)
    except Exception: return None
    if not o or len(o)<minlen: return None
    a=np.array(o,float); return a[:,1],a[:,2],a[:,3],a[:,4],a[:,5]

# ---- V2.1: raw 12-field klines --------------------------------------
# ccxt fetch_ohlcv drops takerBuyBaseVolume, so hit /klines directly on
# the same data-api.binance.vision base. Raw kline row layout:
#   [0 openTime, 1 open, 2 high, 3 low, 4 close, 5 volume, 6 closeTime,
#    7 quoteVolume, 8 trades, 9 takerBuyBaseVolume, 10 takerBuyQuoteVolume, 11 ignore]
# ONE call per symbol returns BOTH the OHLCV arrays (same features as
# before) and takerbuy = sum(takerBuyBase last 5)/sum(volume last 5)
# (>0.5 = net aggressive buying). No extra API calls vs before.
RAW_KLINES = "https://data-api.binance.vision/api/v3/klines"

def fo_taker(sym,interval='1d',lim=120,minlen=60):
    try:
        resp=requests.get(RAW_KLINES,
                          params={"symbol":sym.replace('/',''),   # 'ABC/USDT' -> 'ABCUSDT'
                                  "interval":interval,"limit":lim},timeout=15)
        k=resp.json()
    except Exception:
        return None
    if not isinstance(k,list) or len(k)<minlen: return None
    try:
        a=np.array([[r[1],r[2],r[3],r[4],r[5],r[9]] for r in k],float)
    except Exception:
        return None
    time.sleep(0.03)   # polite spacing (requests bypass ccxt's rate limiter)
    O,H,L,C,V,TB=a[:,0],a[:,1],a[:,2],a[:,3],a[:,4],a[:,5]
    takerbuy=float(np.sum(TB[-5:])/(np.sum(V[-5:])+EPS))
    return O,H,L,C,V,takerbuy

def ob_imbalance(symbol):
    # V2.1: order-book depth imbalance, top ~20 levels of a 50-level book.
    # >0.5 = more resting bid size than ask size. Called ONLY for finalist
    # hot candidates (max TOPN fetches per run), never for all 300.
    try:
        ob=EX.fetch_order_book(symbol,limit=50)
        b=sum(x[1] for x in (ob.get('bids') or [])[:20])
        a=sum(x[1] for x in (ob.get('asks') or [])[:20])
        return b/(b+a) if (b+a)>0 else None
    except Exception:
        return None

def ig_raw(O,H,L,C,V):
    t=len(C)-1
    if t<50: return None
    rng=H-L+EPS
    vr=np.mean(V[t-4:t+1])/(np.mean(V[t-34:t-4])+EPS)
    volz=(V[t]-np.mean(V[t-30:t]))/(np.std(V[t-30:t])+EPS)
    creep=slope(np.log(V[t-19:t+1]+1))
    sg=slice(t-29,t+1); upv=V[sg][C[sg]>O[sg]].sum(); dnv=V[sg][C[sg]<O[sg]].sum(); buyimb=upv/(dnv+EPS)
    ret5=C[t]/C[t-5]-1
    lo=np.min(L[t-30:t+1]); hi=np.max(H[t-30:t+1]); nh=(C[t]-lo)/(hi-lo+EPS)
    return dict(vr=vr,volz=volz,creep=creep,buyimb=buyimb,ret5=ret5,nh=nh)

# ---- V2: 過去アラートの結果を確定して outcomes_log.csv に追記 ----
def resolve_pending(pending):
    nowep=time.time(); resolved=[]
    for key,a in list(pending.items()):
        age=nowep-a["epoch"]
        if age < RESOLVE_AFTER:        # まだ早い
            continue
        symbol=a.get("symbol"); p0=a.get("price")
        if not symbol or not p0:
            pending.pop(key,None); continue
        try:
            o=EX.fetch_ohlcv(symbol,'1h',since=int(a["epoch"]*1000),limit=120)
        except Exception:
            o=None
        if not o or len(o)<2:
            if age>DROP_AFTER: pending.pop(key,None)
            continue
        arr=np.array(o,float); ts=arr[:,0]; H=arr[:,2]; L=arr[:,3]; C=arr[:,4]
        since=int(a["epoch"]*1000)
        def at(h):
            tgt=since+h*3600*1000; idx=int(np.searchsorted(ts,tgt))
            idx=min(idx,len(C)-1); return float(C[idx]/p0-1)
        wmask=(ts>=since)&(ts<=since+72*3600*1000)
        mfe=float(np.max(H[wmask])/p0-1) if wmask.any() else np.nan
        mae=float(np.min(L[wmask])/p0-1) if wmask.any() else np.nan
        r72=at(72)
        row=dict(a); row.update(ret1h=at(1),ret4h=at(4),ret24h=at(24),ret72h=r72,
                                 mfe72=round(mfe,4) if mfe==mfe else "",
                                 mae72=round(mae,4) if mae==mae else "",
                                 win=int(r72>0.05))   # +5%以上を「勝ち」と暫定定義
        resolved.append(row); pending.pop(key,None)
    if resolved:
        append_csv(OUTCOMES_CSV, resolved, OUT_FIELDS)
        print("resolved %d outcomes"%len(resolved))
    return resolved

def main():
    now=dt.datetime.now(dt.timezone.utc); ep=time.time()
    state   = load_json(STATE,{})
    pending = load_json(PENDING,{})

    # 1) 過去アラートの結果確定(データ蓄積)
    resolve_pending(pending)

    EX.load_markets()
    tk=EX.fetch_tickers()
    cand=[s for s in EX.symbols if EX.markets[s].get('spot') and EX.markets[s].get('active')
          and EX.markets[s].get('quote')=='USDT' and EX.markets[s].get('base') not in STABLE]
    cand=[s for s in cand if tk.get(s) and (tk[s].get('quoteVolume') or 0)>=MIN_QV]
    cand=sorted(cand,key=lambda s:-(tk[s].get('quoteVolume') or 0))[:MAX_SCAN]
    data=[]
    for s in cand:
        # V2.1: ONE raw-kline fetch gives OHLCV features AND takerbuy
        rt=fo_taker(s,'1d',120)
        if rt is not None:
            r=rt[:5]; tb=rt[5]
        else:
            r=fo(s,'1d',120); tb=None    # fallback: ccxt path (no taker field)
        if not r: continue
        ig=ig_raw(*r)
        if ig:
            ig['takerbuy']=tb            # logged only - not in score/gate
            ig['sym']=s.split('/')[0]; ig['symbol']=s
            ig['qv']=tk[s]['quoteVolume']
            ig['price']=float(tk[s].get('last') or r[3][-1])
            data.append(ig)
    stamp=now.strftime("%m-%d %H:%M")
    if len(data)<20:
        print("data not enough"); json.dump(state,open(STATE,"w")); json.dump(pending,open(PENDING,"w")); return
    def z(k):
        a=np.array([d[k] for d in data]); return (a-a.mean())/(a.std()+EPS)
    sc=0.50*(z('vr')+z('volz')+z('creep'))/3+0.30*z('buyimb')+0.20*z('nh')
    for i,d in enumerate(data): d['score']=float(sc[i])
    p95=np.percentile(sc,95); p98=np.percentile(sc,98)
    hot=[d for d in sorted(data,key=lambda d:-d['score'])
         if d['ret5']<FRESH and (d['vr']>1.3 or d['volz']>1.5) and d['score']>=p95][:TOPN]
    # V2.1: order book only for the finalists that passed the score gate
    for d in hot:
        d['obimb']=ob_imbalance(d['symbol'])
    def tier_of(d): return "🔴最上位" if d['score']>=p98 else "🟠上位5%"
    def bflow(d):   # V2.1: buy-demand tags (TB=taker-buy ratio, OB=bid share)
        s=""
        if d.get('takerbuy') is not None: s+=" TB%.2f"%d['takerbuy']
        if d.get('obimb')    is not None: s+=" OB%.2f"%d['obimb']
        return s
    def line(d):
        return "%s `%-10s` 出来高x%.0f ・ Z%+.1f ・ 買偏%.1f ・ 5日%+.0f%% ・ $%.0fM%s"%(
            tier_of(d),d['sym'],d['vr'],d['volz'],d['buyimb'],d['ret5']*100,d['qv']/1e6,bflow(d))

    def rec(d):  # V2: アラートを記録(CSV + pending)
        row=dict(iso=now.isoformat(),epoch=round(ep,0),sym=d['sym'],symbol=d['symbol'],
                 tier=tier_of(d),score=round(d['score'],4),vr=round(d['vr'],3),volz=round(d['volz'],3),
                 creep=round(d['creep'],4),buyimb=round(d['buyimb'],3),ret5=round(d['ret5'],4),
                 nh=round(d['nh'],3),qv=round(d['qv'],0),price=d['price'],
                 takerbuy=(round(d['takerbuy'],4) if d.get('takerbuy') is not None else ""),
                 obimb=(round(d['obimb'],4) if d.get('obimb') is not None else ""))
        append_csv(ALERTS_CSV,[row],ALERT_FIELDS)
        pending["%s|%d"%(d['symbol'],int(ep))]=row

    new=[]
    for d in hot:
        k=d['sym']; prev=state.get(k)
        if prev and ep-prev<COOLDOWN: continue
        new.append(d); state[k]=ep
        rec(d)                                  # V2: 実アラートを記録

    if EVENT=="workflow_dispatch":
        body="\n".join(line(d) for d in hot) if hot else "（現在、初動の点火候補なし＝静かな相場。無理に張らない）"
        discord("✅ **初動点火スキャナー テスト** %s UTC ・ 監視 %d 銘柄\n%s\n— 監視リスト用。買いシグナルではない(上位1%%でも的中~11%%)。極小+損切り必須。"%(stamp,len(data),body))
    elif new:
        discord("🔴 **初動点火** %s UTC ・ %d件\n%s\n— 監視用。出来高継続/板を確認し極小+ハード損切り。"%(stamp,len(new),"\n".join(line(d) for d in new)))
        print("posted %d"%len(new))
    else:
        print("%s no new ignition (%d syms)"%(stamp,len(data)))

    json.dump(state,open(STATE,"w"))
    json.dump(pending,open(PENDING,"w"))

if __name__=="__main__":
    main()
