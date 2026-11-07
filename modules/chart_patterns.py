"""Independent daily pattern evidence, not trade admission or probability.

Own explicit initial heuristics. No score, MA50 gate, RR floor, provider call or
watcher import. All structure/volume inputs end at their recorded knowledge date.
"""
import pandas as pd

from modules.trade_watch import number
from modules.support_map import build_support_map

POLICY='chart_patterns_v30.21cm'
KINDS={'bull_flag':('Bullenflagge',1),'bear_flag':('Bärenflagge',-1),
       'head_shoulders':('Schulter-Kopf-Schulter',-1),
       'inverse_head_shoulders':('Inverse Schulter-Kopf-Schulter',1),
       'double_bottom':('Doppelboden',1),'double_top':('Doppeltop',-1)}
PHASES={'none':'Kein Beleg','forming':'Entsteht','confirmed':'Bestätigt',
        'invalid':'Verworfen','blocked':'Nicht prüfbar'}


def _empty(kind,reason='Kein regelkonformer aktueller Beleg.',phase='none'):
    name,direction=KINDS[kind]
    return dict(kind=kind,name=name,direction=direction,phase=phase,reason=reason,
                points=[],confirmed_at=None,boundary_now=None,invalidation=None,
                projection=None,projection_touched=False,volume_ratio=None,
                structure_known_at=None,break_before_known=False)


def _mean(values):
    return sum(values)/len(values)


def _volume_ratio(data,i):
    if i<20:return None
    values=data['Volume'][i-20:i+1]
    if any(number(v) is None or v<=0 for v in values):return None
    return values[-1]/_mean(values[:-1])


def _events(data,sign):
    high=data['High'] if sign==1 else [-x for x in data['Low']]
    low=data['Low'] if sign==1 else [-x for x in data['High']]
    events=[]
    for i in range(3,len(high)-3):
        h=high[i]>max(high[i-3:i]+high[i+1:i+4])
        l=low[i]<min(low[i-3:i]+low[i+1:i+4])
        if h==l:continue  # Outside bars have no identifiable intrabar ordering.
        item=(i,high[i] if h else low[i],'H' if h else 'L')
        if events and events[-1][2]==item[2]:
            if (h and item[1]>=events[-1][1]) or (l and item[1]<=events[-1][1]):events[-1]=item
        else:events.append(item)
    return events[-45:]


def _point(data,i,key,role):
    return dict(index=i,date=data['date'][i],price=data[key][i],role=role)


def _evaluate(kind,data,*,end,known,anchor,value,slope,height,invalidation,points,
              volume_required=False,volume_base=True,extra=None,expiry=None):
    out=_empty(kind,phase='forming',reason='Struktur vorhanden; Tagesbruch der Bezugslinie fehlt.')
    direction=out['direction'];n=len(data['Close'])
    buffer=max(abs(value)*.001,data['ATR'][known]*.1)
    line=lambda i:value+slope*(i-anchor)
    out.update(end_index=end,structure_known_at=data['date'][known],known_index=known,
               boundary_now=line(n-1),invalidation=invalidation,points=points,
               boundary=dict(anchor_index=anchor,anchor_date=data['date'][anchor],value=value,slope=slope),
               confirmation_buffer=buffer,volume_required=volume_required,
               evidence=extra or {})
    out['break_before_known']=any(direction*(data['Close'][i]-line(i))>buffer for i in range(end+1,known))
    confirmed=None
    for i in range(known,n):
        close=data['Close'][i]
        if direction*(close-invalidation)<0:
            out.update(phase='invalid',reason='Feste Muster-Invalidierung per Tagesschluss gebrochen.',invalidated_at=data['date'][i],invalidated_index=i);break
        if confirmed is not None and direction*(close-line(i)) < -buffer:
            out.update(phase='invalid',reason='Bestätigter Bruch per Tagesschluss zurückgenommen (Fehlausbruch).',invalidated_at=data['date'][i],invalidated_index=i);break
        if confirmed is None and expiry is not None and i>end+expiry:
            out.update(phase='invalid',reason='Flagge ohne rechtzeitige Tagesbestätigung abgelaufen.',invalidated_at=data['date'][i],invalidated_index=i);break
        if direction*(close-line(i))>buffer and confirmed is None:
            ratio=_volume_ratio(data,i)
            out['volume_ratio']=ratio
            if not volume_base or (volume_required and (ratio is None or ratio<1.2)):
                out['reason']='Kursgrenze überschritten; erforderlicher Impuls-/Flaggen- oder Ausbruchsvolumenbeleg fehlt.'
                continue
            confirmed=i
            out.update(phase='confirmed',reason='Tagesbruch der Bezugslinie bestätigt; keine Trade-Freigabe.',
                       confirmed_at=data['date'][i],volume_ratio=ratio)
            if out['break_before_known']:
                out['reason']='Tagesbruch bestätigt; erste Überschreitung lag bereits vor vollständigem Strukturbeleg. Kein rechtzeitiger Einstieg behauptet.'
    reference=confirmed if confirmed is not None else known
    target=line(reference)+direction*height
    out['projection']=target if target>0 else None
    if confirmed is not None:
        extremes=data['High'] if direction==1 else data['Low']
        # Confirmation is an end-of-day event. An earlier wick on that same
        # candle is not a subsequently observed projection touch.
        out['projection_touched']=(direction*(data['Close'][confirmed]-target)>=0 or
                                   any(direction*(v-target)>=0 for v in extremes[confirmed+1:out.get('invalidated_index',n)])) if target>0 else False
    if out['volume_ratio'] is None and not volume_required:
        out['volume_note']='Volumen nicht prüfbar; Bestätigung hier kursbasiert.'
    else:
        out['volume_note']='Mindestens 1,20× vorherige 20 Tagesvolumen erforderlich.' if volume_required else 'Volumen ergänzend, kein Pflichtfilter.'
    return out


def _reversal(data,kind):
    sign=-1 if KINDS[kind][1]==1 else 1
    events=_events(data,sign);n=len(data['Close']);hs='head_shoulders' in kind
    width=5 if hs else 3;candidates=[]
    for start in range(len(events)-width+1):
        e=events[start:start+width]
        if [x[2] for x in e] != (['H','L','H','L','H'] if hs else ['H','L','H']):continue
        first,last=e[0][0],e[-1][0];known=last+3
        if first<20 or n-1-last>35 or last-first>(140 if hs else 90) or last-first<(15 if hs else 10):continue
        if any(b[0]-a[0]<3 for a,b in zip(e,e[1:])):continue
        a,b=e[0][1],e[-1][1];scale=(abs(a)+abs(b))/2
        if abs(a-b)/scale>.03:continue
        trend=sign*(data['Close'][first]-data['Close'][first-20])/abs(data['Close'][first-20])
        if trend<.03:continue
        if hs:
            head=e[2];l1,l2=e[1],e[3]
            if head[1]-max(a,b)<max(abs(head[1])*.01,data['ATR'][known]*.4):continue
            balance=(head[0]-first)/(last-head[0])
            if not .4<=balance<=2.5:continue
            if abs(l2[1]-l1[1])/((abs(l1[1])+abs(l2[1]))/2)>.12:continue
            slope=(l2[1]-l1[1])/(l2[0]-l1[0]);anchor=l1[0];value=l1[1]
            neck=lambda i:value+slope*(i-anchor)
            if min(a-neck(first),b-neck(last))<scale*.01:continue
            height=head[1]-neck(head[0]);roles=['Linke Schulter','Nackenlinie 1','Kopf','Nackenlinie 2','Rechte Schulter']
        else:
            slope=0.;anchor=e[1][0];value=e[1][1]
            height=min(a,b)-value;roles=['Erstes Tief' if sign==-1 else 'Erstes Hoch','Nackenlinie','Zweites Tief' if sign==-1 else 'Zweites Hoch']
        if not .03<=height/scale<=.35:continue
        buffer=max(scale*.001,data['ATR'][known]*.1)
        invalid=sign*((b if hs else max(a,b))+buffer)
        points=[_point(data,i,('High' if typ=='H' else 'Low') if sign==1 else ('Low' if typ=='H' else 'High'),role)
                for (i,_,typ),role in zip(e,roles)]
        candidates.append(_evaluate(kind,data,end=last,known=known,anchor=anchor,value=sign*value,
            slope=sign*slope,height=height,invalidation=invalid,points=points,
            volume_required=sign==-1,extra=dict(prior_trend_pct=trend*100,shape_span_bars=last-first,
                extreme_difference_pct=abs(a-b)/scale*100)))
    return max(candidates,key=lambda p:p['end_index']) if candidates else _empty(kind)


def _fit(values):
    n=len(values);xmean=(n-1)/2;ymean=_mean(values)
    slope=sum((i-xmean)*(y-ymean) for i,y in enumerate(values))/sum((i-xmean)**2 for i in range(n))
    return slope,ymean-slope*xmean


def _flag(data,kind):
    sign=KINDS[kind][1];n=len(data['Close']);candidates=[]
    high=data['High'] if sign==1 else [-x for x in data['Low']]
    low=data['Low'] if sign==1 else [-x for x in data['High']]
    close=[sign*x for x in data['Close']]
    for end in range(max(35,n-12),n-1):
        for flag_days in range(3,11):
            pole_end=end-flag_days
            for pole_days in range(3,11):
                begin=pole_end-pole_days+1
                if begin<20:continue
                peak=max(high[begin:pole_end+1]);start=close[begin-1];height=peak-start
                gain=(close[pole_end]-start)/abs(start)
                if high[pole_end]!=peak or not .04<=gain<=.25 or height<2.5*data['ATR'][begin-1]:continue
                if sum(close[i]>close[i-1] for i in range(begin+1,pole_end+1))/(pole_days-1)<.6:continue
                fh=high[pole_end+1:end+1];fl=low[pole_end+1:end+1]
                depth=peak-min(fl)
                if max(fh)>peak+abs(peak)*.005 or not .05*height<=depth<=.5*height or depth>abs(peak)*.1:continue
                hs,hi=_fit(fh);ls,_=_fit(fl)
                if hs>abs(peak)*.001 or ls>abs(peak)*.001 or abs(hs-ls)>max(data['ATR'][end]*.15,height*.015):continue
                # A closed break of an already formed flag cannot be absorbed
                # into a later, wider flag and move the original breakout line.
                crossed=False
                for k in range(3,len(fh)):
                    ps,pi=_fit(fh[:k]);offset=max(h-(ps*j+pi) for j,h in enumerate(fh[:k]))
                    pad=max(abs(peak)*.001,data['ATR'][pole_end+k]*.1)
                    if close[pole_end+1+k]>ps*k+pi+offset+pad:
                        crossed=True;break
                if crossed:continue
                pole_ranges=[high[i]-low[i] for i in range(begin,pole_end+1)]
                flag_ranges=[h-l for h,l in zip(fh,fl)]
                if _mean(flag_ranges)>.8*_mean(pole_ranges):continue
                offset=max(h-(hs*i+hi) for i,h in enumerate(fh));value=hi+offset
                volumes=data['Volume'][begin-20:end+1]
                valid=all(number(v) is not None and v>0 for v in volumes)
                impulse_ratio=contraction=None
                if valid:
                    impulse=_mean(data['Volume'][begin:pole_end+1]);prior=_mean(data['Volume'][begin-20:begin])
                    impulse_ratio=impulse/prior;contraction=_mean(data['Volume'][pole_end+1:end+1])/impulse
                base_ok=valid and impulse_ratio>=1.2 and contraction<=.8
                buffer=max(abs(peak)*.001,data['ATR'][end]*.1)
                invalid=sign*(min(fl)-buffer)
                points=[_point(data,begin-1,'Close','Impulsbeginn'),
                        _point(data,pole_end,'High' if sign==1 else 'Low','Impulsende'),
                        _point(data,pole_end+1,'High' if sign==1 else 'Low','Flaggenbeginn'),
                        _point(data,end,'High' if sign==1 else 'Low','Flaggenende')]
                candidates.append(_evaluate(kind,data,end=end,known=end,anchor=pole_end+1,
                    value=sign*value,slope=sign*hs,height=height,invalidation=invalid,
                    points=points,volume_required=True,volume_base=base_ok,expiry=min(flag_days,10),
                    extra=dict(pole_days=pole_days,flag_days=flag_days,pole_height=height,
                               impulse_volume_ratio=impulse_ratio,flag_volume_ratio=contraction)))
    return max(candidates,key=lambda p:(p['end_index'],p['evidence']['flag_days'])) if candidates else _empty(kind)


def build_chart_patterns(result, *, now=None, daily_context=None):
    r=result or {};ticker=str(r.get('ticker') or r.get('Ticker') or '').upper().strip()
    context=daily_context if isinstance(daily_context,dict) and daily_context.get('ticker')==ticker else build_support_map(r,now=now)
    base=dict(policy=POLICY,ticker=ticker,available=False,patterns=[],chart=[],basis_date=context.get('basis_date',''),
              stale=bool(context.get('stale')),currency=context.get('currency',''),reason='')
    if not context.get('available'):
        return {**base,'reason':context.get('reason') or 'Tagesbasis fehlt.',
                'patterns':[_empty(k,context.get('reason') or 'Tagesbasis fehlt.','blocked') for k in KINDS]}
    source=r['df'];dates=list(source.index.date)
    cutoff=pd.Timestamp(context['basis_date']).date()
    f=source.loc[[d<=cutoff for d in dates]].tail(260).copy()
    data={k:pd.to_numeric(f[k],errors='coerce').astype(float).tolist() for k in ['Open','High','Low','Close']}
    data['Volume']=pd.to_numeric(f.Volume,errors='coerce').astype(float).tolist() if 'Volume' in f else [float('nan')]*len(f)
    data['date']=[i.date().isoformat() for i in f.index]
    tr=[max(data['High'][i]-data['Low'][i],abs(data['High'][i]-data['Close'][i-1]),abs(data['Low'][i]-data['Close'][i-1])) if i else data['High'][i]-data['Low'][i] for i in range(len(f))]
    data['ATR']=[_mean(tr[max(0,i-13):i+1]) for i in range(len(f))]
    patterns=[_flag(data,k) if 'flag' in k else _reversal(data,k) for k in KINDS]
    chart=[dict(index=i,date=data['date'][i],open=data['Open'][i],high=data['High'][i],low=data['Low'][i],close=data['Close'][i]) for i in range(max(0,len(f)-180),len(f))]
    return {**base,'available':True,'patterns':patterns,'chart':chart,'bars':len(f),
            'reason':'Eigene Tagesmuster-Heuristiken; unabhängig von Ampel und Trade-Wächter. Keine Gewinnwahrscheinlichkeit.'}


def patterns_brief(package):
    if not isinstance(package,dict) or package.get('policy')!=POLICY:return 'Neu scannen'
    if not package.get('available'):return 'Nicht prüfbar'
    patterns=package.get('patterns')
    if (not isinstance(patterns,list) or len(patterns)!=6 or
            any(not isinstance(p,dict) or p.get('kind') not in KINDS or p.get('phase') not in PHASES for p in patterns)
            or {p['kind'] for p in patterns}!=set(KINDS)):
        return 'Beleg unvollständig · neu scannen'
    rank={'confirmed':3,'forming':2,'invalid':1,'none':0,'blocked':0}
    visible=[p for p in patterns if p.get('phase') in {'confirmed','forming','invalid'}]
    visible.sort(key=lambda p:(rank[p['phase']],p.get('end_index',-1)),reverse=True)
    text=' · '.join(p['name']+': '+PHASES[p['phase']] for p in visible[:2]) or 'Kein aktueller Musterbeleg'
    return text+(' · historisch '+package.get('basis_date','') if package.get('stale') else '')


def attach_pattern_briefs(frame):
    if not isinstance(frame,pd.DataFrame):return frame
    out=frame.copy();out['Chartmuster-Kurz']=[patterns_brief(r.get('__chart_patterns')) for r in out.to_dict('records')]
    return out
