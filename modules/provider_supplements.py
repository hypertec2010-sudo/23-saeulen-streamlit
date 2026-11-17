"""Paced, bounded public-data caches for slow-changing Yahoo supplements."""
from copy import deepcopy
from threading import RLock
import time

INFO_TTL=15*60
SUPPLEMENT_TTL=60*60
FIELDS={'income_stmt','quarterly_income_stmt','balance_sheet','quarterly_balance_sheet',
        'cashflow','quarterly_cashflow','recommendations','calendar'}


class SupplementalCache:
    def __init__(self,provider):
        self.provider=provider;self.lock=RLock();self.values={}

    def read(self,key,ttl,loader):
        self.provider.ensure_available()
        with self.lock:
            hit=self.values.get(key)
            now=time.time()
            if hit and 0<=now-hit[0]<ttl:return hit[0],hit[1]
            self.provider._pace_request()
            try:value=loader()
            except Exception as exc:
                if self.provider.is_rate_limit_error(exc):
                    from modules.provider_manager import MarketDataRateLimitError
                    self.provider.mark_rate_limited(key[0],source='Zusatzdaten')
                    raise MarketDataRateLimitError(str(exc)) from exc
                raise
            stamp=time.time();self.values[key]=(stamp,value)
            if len(self.values)>1000:
                oldest=min(self.values,key=lambda k:self.values[k][0]);self.values.pop(oldest,None)
            return stamp,value

    def bundle(self,symbol):
        clean=self.provider.normalize_symbol(symbol)
        def fetch():
            raw=self.provider.get_ticker(clean)
            info={}
            # info and get_info are alternative interfaces, never two routine calls.
            for index,getter in enumerate((lambda:raw.get_info(),lambda:getattr(raw,'info',{}),lambda:dict(getattr(raw,'fast_info',{}) or {}))):
                try:
                    if index:self.provider._pace_request()
                    part=getter()
                    if isinstance(part,dict):info=dict(part);break
                except Exception as exc:
                    if self.provider.is_rate_limit_error(exc):raise
                    self.provider.ensure_available()
                    continue
            return raw,info
        stamp,(raw,info)=self.read((clean,'info'),INFO_TTL,fetch)
        cloned=deepcopy(info);cloned['_chsm_info_fetched_at']=stamp
        self.provider._record(clean,'info',True,f'fields={len(info)}; cache_age={max(0,time.time()-stamp):.0f}s')
        return CachedTicker(self,clean,raw,cloned),cloned


class CachedTicker:
    def __init__(self,cache,symbol,raw,info):
        self._cache=cache;self._symbol=symbol;self._raw=raw;self._info=deepcopy(info)
        self.supplemental_oldest_at=info['_chsm_info_fetched_at']

    def get_info(self):
        self._cache.provider.ensure_available()
        return deepcopy(self._info)

    def __getattr__(self,name):
        if name in FIELDS:
            stamp,value=self._cache.read((self._symbol,name),SUPPLEMENT_TTL,lambda:getattr(self._raw,name))
            self.supplemental_oldest_at=min(self.supplemental_oldest_at,stamp)
            return deepcopy(value)
        if name=='get_earnings_dates':
            def dates(*args,**kwargs):
                key=(self._symbol,name,repr(args),repr(sorted(kwargs.items())))
                stamp,value=self._cache.read(key,SUPPLEMENT_TTL,lambda:self._raw.get_earnings_dates(*args,**kwargs))
                self.supplemental_oldest_at=min(self.supplemental_oldest_at,stamp)
                return deepcopy(value)
            return dates
        return getattr(self._raw,name)
