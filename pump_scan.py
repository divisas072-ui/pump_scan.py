#!/usr/bin/env python3
# ================================================================
#  初動点火アラート (仕手スキャナー) ── GitHub Actions 単発実行版
#  data-api.binance.vision(現物・鍵不要・451回避) 検証済 lift~3.8x帯のみ通知
#  別チャンネル用Webhook: GitHub Secrets DISCORD_WEBHOOK_PUMP
# ================================================================
import os, time, json, datetime as dt, numpy as np, requests, ccxt

WEBHOOK   = os.environ.get("DISCORD_WEBHOOK_PUMP","")
EVENT     = os.environ.get("GITHUB_EVENT_NAME","")
STATE     = "pump_state.json"
MIN_QV    = 1_000_000
MAX_SCAN  = 300
FRESH     = 0.20
TOPN      = 5
COOLDOWN  = 24*3600
EPS=1e-9
STABLE    = {'USDC','FDUSD','TUSD','BUSD','DAI','EUR','USDP','AEUR','EURI','XUSD','USD1'}

EX=ccxt.binance({'enableRateLimit':True,'options':{'defaultType':'spot','fetchMarkets':['spot']}})
EX.urls['api']['public']='https://data-api.binance.vision/api/v3'

def discord(text):
    if not WEBHOOK:
        print("[no webhook]\n"+text); return
    for i in range(0,len(text),1800):
        try: requests.post(WEBHOOK,json={"content":text[i:i+1800]},timeout=20)
        except Exception as e: print("discord err",e)

def slope(y): x=np.arange(len(y)); return float(np.polyfit(x,y,1)[0])/(np.mean(np.abs(y))+EPS)
def fo(sym,tf,lim,minlen=60):
    try: o=EX.fetch_ohlcv(sym,tf,limit=lim)
    except Exception: return None
    if not o or len(o)<minlen: return None
    a=np.array(o,float); return a[:,1],a[:,2],a[:,3],a[:,4],a[:,5]
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

def main():
    now=dt.datetime.now(dt.timezone.utc); ep=time.time()
    try: state=json.load(open(STATE))
    except Exception: state={}
    EX.load_markets()
    tk=EX.fetch_tickers()
    cand=[s for s in EX.symbols if EX.markets[s].get('spot') and EX.markets[s].get('active')
          and EX.markets[s].get('quote')=='USDT' and EX.markets[s].get('base') not in STABLE]
    cand=[s for s in cand if tk.get(s) and (tk[s].get('quoteVolume') or 0)>=MIN_QV]
    cand=sorted(cand,key=lambda s:-(tk[s].get('quoteVolume') or 0))[:MAX_SCAN]
    data=[]
    for s in cand:
        r=fo(s,'1d',120)
        if not r: continue
        ig=ig_raw(*r)
        if ig: ig['sym']=s.split('/')[0]; ig['qv']=tk[s]['quoteVolume']; data.append(ig)
    stamp=now.strftime("%m-%d %H:%M")
    if len(data)<20:
        print("data not enough"); return
    def z(k): a=np.array([d[k] for d in data]); return (a-a.mean())/(a.std()+EPS)
    sc=0.50*(z('vr')+z('volz')+z('creep'))/3+0.30*z('buyimb')+0.20*z('nh')
    for i,d in enumerate(data): d['score']=sc[i]
    p95=np.percentile(sc,95); p98=np.percentile(sc,98)
    hot=[d for d in sorted(data,key=lambda d:-d['score'])
         if d['ret5']<FRESH and (d['vr']>1.3 or d['volz']>1.5) and d['score']>=p95][:TOPN]
    def line(d):
        tag="🔴最上位" if d['score']>=p98 else "🟠上位5%"
        return "%s  `%-10s` 出来高x%.0f ・ Z%+.1f ・ 買偏%.1f ・ 5日%+.0f%% ・ $%.0fM"%(
            tag,d['sym'],d['vr'],d['volz'],d['buyimb'],d['ret5']*100,d['qv']/1e6)
    new=[]
    for d in hot:
        k=d['sym']; prev=state.get(k)
        if prev and ep-prev<COOLDOWN: continue
        new.append(d); state[k]=ep
    if EVENT=="workflow_dispatch":
        body="\n".join(line(d) for d in hot) if hot else "（現在、初動の点火候補なし＝静かな相場。無理に張らない）"
        discord("✅ **初動点火スキャナー テスト** %s UTC ・ 監視 %d 銘柄\n%s\n— 監視リスト用。買いシグナルではない(上位1%%でも的中~11%%)。極小+損切り必須。"%(stamp,len(data),body))
    elif new:
        discord("🔴 **初動点火** %s UTC ・ %d件\n%s\n— 監視用。出来高継続/板を確認し極小+ハード損切り。"%(stamp,len(new),"\n".join(line(d) for d in new)))
        print("posted %d"%len(new))
    else:
        print("%s no new ignition (%d syms)"%(stamp,len(data)))
    json.dump(state,open(STATE,"w"))

if __name__=="__main__":
    main()
