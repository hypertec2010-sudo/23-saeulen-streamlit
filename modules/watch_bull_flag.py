"""Daily bull-flag heuristic, frozen before the latest evaluation bar.

Numerical cutoffs are explicit initial parameters, not empirically optimized.
"""
import pandas as pd

ENTRY_VOLUME_RULE = {"kind":"relative_5m_volume", "lookback_groups":20, "minimum_ratio":1.3}


def bull_flag_candidate(data, atr, price):
    from modules.watch_setups import attempt,levels,atr_value
    from modules.watch_plans import bind_plan,candidate_rank
    kind = "bull_flag"
    checked=[]
    # The last bar can confirm/invalidate/chase; it cannot move flag boundaries.
    for flag_len in range(3,11):
        end = len(data)-flag_len-2
        if end < 30:
            continue
        flag = data.iloc[end+1:-1]
        for pole_len in range(3,11):
            begin = end-pole_len+1
            prior = data.iloc[begin-20:begin]
            if len(prior)!=20:
                continue
            pole = data.iloc[begin:end+1]
            start = float(data.Close.iloc[begin-1])
            peak = float(pole.High.max())
            height = peak-start
            ref_atr = atr_value(data.iloc[:begin])
            gain = (float(pole.Close.iloc[-1])/start-1)*100
            if (pole.High.idxmax()!=pole.index[-1] or not 5<=gain<=25
                    or height<3*ref_atr or float(pole.Close.iloc[-1])<peak-.25*height
                    or float((pole.Close.diff().dropna()>0).mean())<.6):
                continue
            evidence=dict(pole_start_date=pole.index[0].date().isoformat(),pole_end_date=pole.index[-1].date().isoformat(),
                pole_reference_close=start,pole_high=peak,pole_height=height,pole_days=pole_len,
                flag_start_date=flag.index[0].date().isoformat(),flag_end_date=flag.index[-1].date().isoformat(),flag_days=flag_len)
            flag_high,flag_low=float(flag.High.max()),float(flag.Low.min())
            depth = (peak-flag_low)/height
            if flag_high>peak*1.005 or not 0<depth<=.5:
                continue
            evidence.update(flag_high=flag_high,flag_low=flag_low,retracement_of_pole_pct=depth*100)
            # Flag must pause/decline, not continue climbing or widen into a new impulse.
            midpoint=(flag_len+1)//2
            if (float(flag.Close.iloc[-1])>float(flag.Close.iloc[0])*1.005
                    or flag.High.iloc[midpoint:].max()>flag.High.iloc[:midpoint].max()*1.005
                    or flag_low/peak<.92):
                continue
            pole_vol=float(pole.Volume.mean());prior_vol=float(prior.Volume.mean())
            ratio=pole_vol/prior_vol;dry=float(flag.Volume.mean())/pole_vol
            previous=data.Close.shift()
            tr=pd.concat([data.High-data.Low,(data.High-previous).abs(),(data.Low-previous).abs()],axis=1).max(axis=1)
            contraction=float(tr.loc[flag.index].mean()/tr.loc[pole.index].mean())
            start_ratio=float(pole.Volume.iloc[0])/prior_vol
            preceding_ratio=float(data.Volume.iloc[begin-1])/prior_vol
            evidence.update(pole_volume_ratio=ratio,flag_volume_ratio=dry,range_contraction_ratio=contraction,
                            pole_start_volume_ratio=start_ratio,preceding_volume_ratio=preceding_ratio)
            if ratio<1.3 or start_ratio<1.5 or preceding_ratio>=1.3:
                checked.append(attempt(kind,"flag_impulse_volume","Impuls ohne ausreichende Volumenausweitung.","forming",evidence=evidence))
                continue
            if dry>.7 or contraction>.7:
                checked.append(attempt(kind,"flag_contraction","Flaggenvolumen oder Schwankungsbreite noch nicht ausreichend kontrahiert.","forming",evidence=evidence))
                continue
            anchored_atr=atr_value(data.iloc[:-1])
            trigger=max(flag_high,peak)
            out=levels(kind,trigger,flag_low,anchored_atr,price,
                near=price>=trigger*.97,
                evidence=evidence,
                checks=["3–10 Impulstage mit 5–25 % Schlusskursanstieg und mindestens 3 vorherigen ATR",
                        "Impulsvolumen mindestens 1,3× Vorbasis; Starttag mindestens 1,5×, davor unter 1,3×",
                        "3–10 abgeschlossene Flaggentage; Rücknahme höchstens halbe Impulshöhe und 8 % vom Hoch",
                        "Flaggenvolumen und tägliche Schwankungsbreite höchstens 70 % des Impulses",
                        "Entry oberhalb von Flaggen- UND Impulshoch; Invalidierung unter der Flagge",
                        "Zusätzlich neuer 5m-Schluss mit mindestens 1,3× Volumen der 20 vorherigen vollständigen 5m-Gruppen erforderlich"])
            if out["phase"] not in {"none","blocked"} and float(data.Low.iloc[-1])<=out["invalidation"]:
                out.update(phase="invalid",reason_code="flag_broken",reason="Jüngste Tageskerze hat die feste Flaggen-Invalidierung erreicht.")
            out=bind_plan(out,data.iloc[:-1],objective=trigger+height,
                source="Einfache Flaggenmast-Projektion ab Flaggen-/Impulsausbruch",basis_date=flag.index[-1].date(),
                observed_high=float(data.High.iloc[-1]))
            out["entry_volume_rule"]=dict(ENTRY_VOLUME_RULE)
            checked.append(out)
    return max(checked,key=candidate_rank) if checked else attempt(kind,"flag_missing","Kein ausreichend klarer Impuls mit anschließender enger Bullflagge.")


def entry_volume_check(history, end, rule):
    """Use only complete groups ending BEFORE the signal group for baseline."""
    if rule!=ENTRY_VOLUME_RULE:
        return False,"Unbekannte Volumenregel; keine neue Triggerfreigabe."
    if "Volume" not in history:
        return False,"1m-Volumen fehlt; Bullflaggen-Trigger nicht bestätigt."
    from modules.trade_watch import number
    volumes=[];signal=None
    start=end-pd.Timedelta(minutes=5)
    for t,group in history.loc[history.index<end].resample('5min',origin='start_day'):
        if len(group)!=5 or group.index[0]!=t or group.index[-1]!=t+pd.Timedelta(minutes=4):
            continue
        if not group.Volume.map(lambda v:number(v) is not None and v>0).all():
            continue
        total=float(group.Volume.sum());group_end=t+pd.Timedelta(minutes=5)
        if group_end<=start:
            volumes.append(total)
        elif group_end==end:
            signal=total
    if signal is None or len(volumes)<20:
        return False,"Mindestens 20 vorherige vollständige 5m-Volumengruppen und ein vollständiger Signalblock erforderlich."
    baseline=sum(volumes[-20:])/20
    ratio=signal/baseline
    return ratio>=1.3, f"5m-Volumen {ratio:.2f}× der 20 vorherigen Gruppen; erforderlich 1,30×."
