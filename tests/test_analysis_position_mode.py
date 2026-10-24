import ast
from pathlib import Path
from datetime import datetime,timezone,date,timedelta
import re,json
import numpy as np
import pandas as pd
from modules.crv_targeting import projected_breakout_target,select_operational_tp2,structural_crv_from_plan
import pytest
from modules.live_monitor import build_decision_state_v3021az

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_engines():
    scope=dict(np=np,pd=pd,re=re,json=json,datetime=datetime,timezone=timezone,date=date,timedelta=timedelta,
     projected_breakout_target=projected_breakout_target,select_operational_tp2=select_operational_tp2,structural_crv_from_plan=structural_crv_from_plan)
    tree=ast.parse((ROOT/'analysis_core.py').read_text())
    for n in tree.body:
     if isinstance(n,ast.FunctionDef):
      n.decorator_list=[]
      exec(compile(ast.Module(body=[n],type_ignores=[]),'analysis_core.py','exec'),scope)
     elif isinstance(n,ast.Assign):
      try:ast.literal_eval(n.value)
      except (ValueError, TypeError):continue
      exec(compile(ast.Module(body=[n],type_ignores=[]),'analysis_core.py','exec'),scope)
    # Bind missing legacy helper functions from the actual app without running UI.
    app_functions={n.name:n for n in ast.parse((ROOT/'legacy_app.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    def bind(name):
     if name in scope or name not in app_functions:return
     node=app_functions[name];node.decorator_list=[]
     exec(compile(ast.Module(body=[node],type_ignores=[]),'legacy_app.py','exec'),scope)
     for child in ast.walk(node):
      if isinstance(child,ast.Name) and isinstance(child.ctx,ast.Load):bind(child.id)
    for name in ['sanitize_quality_red_flags_v1601','is_hard_red_flag_v1604','soften_growth_red_flag_item_v1604',
                 'build_stock_fomo_package_v1525','build_market_fomo_package_v1525','combine_fomo_packages_v1525']:
     bind(name)
    x=np.arange(300);close=80*np.exp(x*.0018)+.6*np.sin(x*.55)
    df=pd.DataFrame({'Close':close,'High':close+1.2,'Low':close-1.2,'Open':close-.3,'Volume':1000000+x*1000},index=pd.bdate_range('2025-08-01',periods=300))
    info=dict(longName='Synthetic Test',currency='USD',sector='Technology',industry='Software',exchange='NMS',
     profitMargins=.3,operatingMargins=.32,grossMargins=.7,returnOnEquity=.25,returnOnAssets=.18,
     revenueGrowth=.2,earningsGrowth=.3,currentRatio=2.3,quickRatio=2,debtToEquity=20,
     trailingPE=25,forwardPE=20,pegRatio=1.2,priceToSalesTrailing12Months=5,priceToBook=5,
     recommendationKey='buy',numberOfAnalystOpinions=25,recommendationMean=1.5,beta=1.1,
     shortPercentOfFloat=.02,marketCap=20000000000,totalCash=6000000000,totalDebt=1000000000,
     freeCashflow=2000000000,operatingCashflow=2500000000,totalRevenue=10000000000,dividendYield=.01)
    scope['load_data']=lambda t:(df.copy(),dict(info))
    scope['load_benchmark_data']=lambda t:df.copy()
    scope['load_extended_market_quote']=lambda t:{}
    scope['load_sector_context']=lambda *a,**kw:{}
    tree=ast.parse((ROOT/'modules/legacy_analysis_core.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_legacy_analyze_stock_impl')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'modules/legacy_analysis_core.py','exec'),scope)
    return scope,df

@pytest.mark.parametrize('engine',['analyze_stock','_legacy_analyze_stock_impl'])
@pytest.mark.parametrize('ticker',['ADBE','CRM','PLTR','TEAM'])
def test_complete_valid_setup_analysis_reaches_active_trigger(offline_engines,engine,ticker):
    # Real complete analysis function: no position_mode injection, no mocked
    # setup, score or trigger. Only market-data/provider boundaries are replaced.
    scope,_=offline_engines
    r=scope[engine](ticker,'Swing (1-4 Wochen)',20000,1,0,0,False,False)
    assert r['ticker']==ticker and r['valid_trade_setup']
    assert r['trigger_status']=='Aktiv' and r['tradeability_score']>0
    d=build_decision_state_v3021az(valid_trade_setup=r['valid_trade_setup'],
        signal_packages_complete=True,timing_score=85,conf_score=71,
        trigger_status=r['trigger_status'],crv_now=.1,crv_now_eligible=True)
    assert d['icon']=='🟢'


@pytest.mark.parametrize('engine',['analyze_stock','_legacy_analyze_stock_impl'])
def test_complete_position_analysis_still_runs(offline_engines,engine):
    scope,df=offline_engines
    r=scope[engine]('TEST','Swing (1-4 Wochen)',20000,1,0,float(df.Close.iloc[-1])*.95,False,False)
    assert r['trigger_status']=='Nicht anwendbar'
    assert r['position_action'] and r['tradeability_score']>0


@pytest.mark.parametrize('engine',['analyze_stock','_legacy_analyze_stock_impl'])
def test_complete_invalid_setup_path_still_returns_observation(offline_engines,engine):
    scope,df=offline_engines
    # A genuinely negative benchmark keeps its existing market veto.
    benchmark=df.copy()
    benchmark['Close']=np.linspace(150,50,len(df))
    scope['load_benchmark_data']=lambda t:benchmark.copy()
    r=scope[engine]('TEST','Swing (1-4 Wochen)',20000,1,0,0,False,False)
    assert not r['valid_trade_setup'] and r['trigger_status']=='Passiv'
    d=build_decision_state_v3021az(valid_trade_setup=r['valid_trade_setup'],
        signal_packages_complete=True,timing_score=85,conf_score=71,
        trigger_status=r['trigger_status'])
    assert d['icon']=='⚪'
