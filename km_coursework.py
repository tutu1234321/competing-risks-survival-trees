"""Risk-specific KM trees. All predictions have shape (observations, risks, times)."""
from pathlib import Path
from dataclasses import dataclass
import json, math, time, traceback
import numpy as np
import pandas as pd
from scipy.stats import rankdata, chi2
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import Parallel, delayed
from survivors.ensemble import ParallelBootstrapCRAID
from histogram_core import aggregate_histograms, vector_logrank_hist

METHODS = ['max', 'mean', 'maximin', 'rank', 'topsis', 'cosine', 'weighted_corr', 'ibs', 'iauc']

def outcome(y):
    if isinstance(y, pd.DataFrame):
        e = next((c for c in ['event', 'cens'] if c in y), None)
        t = next((c for c in ['duration', 'time'] if c in y), None)
        if e is None or t is None: raise ValueError('Need event/cens and duration/time columns')
        ev, tm = y[e].to_numpy(), y[t].to_numpy()
    elif getattr(y.dtype, 'names', None):
        ev = y['event' if 'event' in y.dtype.names else 'cens']
        tm = y['duration' if 'duration' in y.dtype.names else 'time']
    else:
        raise ValueError('Use a named DataFrame or structured array, not positional columns')
    if not np.all(np.isfinite(ev)) or not np.all(ev == np.floor(ev)) or np.any(ev < 0):
        raise ValueError('Event codes must be nonnegative integers; 0 is censoring')
    if not np.all(np.isfinite(tm)) or np.any(tm < 0): raise ValueError('Invalid times')
    return pd.DataFrame({'event': np.asarray(ev, int), 'duration': np.asarray(tm, float)})

class KM:
    def fit(self, durations, observed):
        order = np.argsort(durations)
        t, e = np.asarray(durations)[order], np.asarray(observed)[order]
        self.times, idx = np.unique(t, return_index=True)
        self.survival = np.cumprod(1 - np.add.reduceat(e.astype(float), idx) / (len(t) - idx))
        return self
    def predict(self, times):
        idx = np.searchsorted(self.times, times, side='right') - 1
        return np.where(idx < 0, 1., self.survival[np.maximum(idx, 0)])

def km_models(y, events):
    return [KM().fit(y.duration.to_numpy(), y.event.to_numpy() == e) for e in events]

def leaf_prediction(y, events, times, n):
    curves = np.stack([m.predict(times) for m in km_models(y, events)])
    return np.broadcast_to(curves, (n,) + curves.shape).copy()

def validate_prediction(y, pred, times, events):
    y = outcome(y); pred = np.asarray(pred, float); times = np.asarray(times, float)
    if pred.shape != (len(y), len(events), len(times)): raise ValueError('Expected (N,K,T)')
    if len(times) < 2 or np.any(np.diff(times) <= 0): raise ValueError('Time grid must increase')
    if not np.all(np.isfinite(pred)) or pred.min() < -1e-10 or pred.max() > 1+1e-10:
        raise ValueError('Invalid KM probabilities')
    if np.any(np.diff(pred, axis=2) > 1e-8): raise ValueError('KM curves must not increase')
    if not set(y.event.unique()).issubset({0, *events}): raise ValueError('Unmapped risk label')
    return y, pred, times

def ibscr(y, pred, times, events):
    """Supplied notebook's IBS_remain extension, on KM survival predictions.
    Before observed time: average (1-S_k)^2 over K. After event k: S_k^2.
    Censored observations leave the denominator at their censoring time.
    This is the course's remain convention, not an IPCW metric.
    """
    y, p, ts = validate_prediction(y, pred, times, events)
    before = ts[None, :] < y.duration.to_numpy()[:, None]
    total = np.where(before, np.mean((1-p)**2, axis=1), 0.)
    for j, e in enumerate(events):
        total += ((~before) & (y.event.to_numpy()[:, None] == e)) * p[:, j]**2
    eligible = before | (y.event.to_numpy()[:, None] > 0)
    denom = eligible.sum(axis=0)
    valid = denom > 0
    if valid.sum() < 2: return float('nan')
    bs = total[:, valid].sum(axis=0) / denom[valid]
    return float(np.trapz(bs, ts[valid]) / np.ptp(ts[valid]))

def iauccr(y, pred, times, events, tied_tol=1e-8):
    """Time-integrated course IAUC_CR: pooled comparable case-control pairs.
    Cases: event k by t; controls: duration > t. Smaller S_k means higher risk.
    Half credit for ties; integrate at actual eligible times, never fabricate 0.5.
    """
    y, p, ts = validate_prediction(y, pred, times, events)
    ev, tm = y.event.to_numpy(), y.duration.to_numpy()
    values, valid_times = [], []
    for ti, t in enumerate(ts):
        controls = tm > t
        numerator = denominator = 0.
        for j, e in enumerate(events):
            cases = (tm <= t) & (ev == e)
            if not cases.any() or not controls.any(): continue
            ctrl = np.sort(p[controls, j, ti]); case = p[cases, j, ti]
            lo = np.searchsorted(ctrl, case-tied_tol, side='left')
            hi = np.searchsorted(ctrl, case+tied_tol, side='right')
            numerator += (len(ctrl)-hi + .5*(hi-lo)).sum()
            denominator += len(ctrl)*len(case)
        if denominator:
            values.append(numerator/denominator); valid_times.append(t)
    if len(values) < 2: return float('nan')
    return float(np.trapz(values, valid_times) / np.ptp(valid_times))

def select_candidate(vectors, method, importance=None):
    """Select across candidate rows, keeping all K LR criteria until this step."""
    v = np.asarray(vectors, float)
    if not np.all(np.isfinite(v)): raise ValueError('Non-finite LR statistics')
    if method == 'max': score = v.max(axis=1)
    elif method == 'mean': score = v.mean(axis=1)
    elif method == 'maximin': score = v.min(axis=1)
    elif method == 'rank':
        # Rank 1 is best, so minimize mean rank (equivalently maximize its negative).
        score = -rankdata(-v, axis=0, method='average').mean(axis=1)
    else:
        z = v / np.maximum(np.linalg.norm(v, axis=0), 1e-12)
        ideal = z.max(axis=0)
        if method == 'topsis':
            good = np.linalg.norm(z-ideal, axis=1)
            bad = np.linalg.norm(z-z.min(axis=0), axis=1)
            score = bad / np.maximum(good+bad, 1e-12)
        elif method == 'cosine':
            score = (z @ ideal) / np.maximum(np.linalg.norm(z, axis=1)*np.linalg.norm(ideal), 1e-12)
        elif method == 'weighted_corr':
            w = np.ones(v.shape[1]) if importance is None else np.asarray(importance, float)
            if w.shape != (v.shape[1],) or np.any(w < 0) or w.sum() <= 0:
                raise ValueError('Invalid risk importance weights')
            centered = v-v.mean(axis=0)
            norms = np.linalg.norm(centered, axis=0)
            corr = centered.T @ centered / np.maximum(norms[:, None]*norms[None, :], 1e-12)
            redundancy = np.maximum(np.abs(corr).sum(axis=1), 1.)
            w = w/redundancy; w /= w.sum()
            score = z @ w
        else: raise ValueError(method)
    return int(np.argmax(score))

@dataclass
class TreeNode:
    number: int
    depth: int
    n: int
    models: list
    counts: list
    feature: int = -1
    threshold: float = float('nan')
    lr: object = None
    left: object = None
    right: object = None

class KMCompetingTree:
    def __init__(self, depth=3, min_samples_leaf=.05, method='max', events=None,
                 n_bins=50, feature_bins=100, random_state=42, importance=None):
        self.depth=depth; self.min_samples_leaf=min_samples_leaf; self.method=method
        self.events=events; self.n_bins=n_bins; self.feature_bins=feature_bins
        self.random_state=random_state; self.importance=importance
    def fit(self, X, y):
        if self.method not in METHODS: raise ValueError(self.method)
        self.columns_ = list(X.columns); X=np.asarray(X, float); y=outcome(y)
        if len(X)!=len(y) or not np.all(np.isfinite(X)): raise ValueError('Invalid X or row count')
        self.events_ = list(self.events) if self.events is not None else sorted(set(y.event)-{0})
        if not self.events_ or not set(y.event).issubset({0, *self.events_}): raise ValueError('Risk mapping mismatch')
        value=self.min_samples_leaf
        if isinstance(value, (float, np.floating)):
            if not 0 < value <= .5: raise ValueError('Leaf fraction must be in (0,.5]')
            self.min_leaf_=max(1, math.ceil(value*len(y)))
        else:
            self.min_leaf_=int(value)
            if self.min_leaf_ < 1: raise ValueError('Leaf count must be positive')
        self.node_count_=0; self.root_candidates_=None
        self.root_=self._build(X,y,0)
        return self
    def _candidates(self,X,y):
        # Global risk order remains fixed even if a node has no events of one type.
        event=np.zeros(len(y),np.int32)
        for j,e in enumerate(self.events_): event[y.event.to_numpy()==e]=j+1
        tm=y.duration.to_numpy(); edges=np.linspace(tm.min(),tm.max(),self.n_bins+1)
        ti=np.clip(np.searchsorted(edges,tm,side='right')-1,0,self.n_bins-1).astype(np.int32)
        candidates=[]
        for f in range(X.shape[1]):
            values=np.unique(X[:,f])
            if len(values)<2: continue
            if len(values)<=self.feature_bins+1: cuts=values[:-1]+np.diff(values)/2
            else: cuts=np.unique(np.quantile(X[:,f],np.linspace(0,1,self.feature_bins+2)[1:-1]))
            bi=np.searchsorted(cuts,X[:,f],side='left').astype(np.int32)
            th,ch=aggregate_histograms(bi,ti,event,len(cuts)+1,self.n_bins,len(self.events_))
            rt=th.sum(axis=0); rc=ch.sum(axis=0); lt=np.zeros_like(rt); lc=np.zeros_like(rc)
            for i,cut in enumerate(cuts):
                lt+=th[i]; rt-=th[i]; lc+=ch[i]; rc-=ch[i]
                if lt.sum()<self.min_leaf_ or rt.sum()<self.min_leaf_: continue
                lr=vector_logrank_hist(lt,rt,lc,rc).astype(float)
                if np.all(np.isfinite(lr)): candidates.append((f,float(cut),lr))
        return candidates
    def _metric_choice(self,candidates,X,y):
        # Inner holdout only: test set is never used for selecting a split.
        ids=np.arange(len(y)); tr,va=train_test_split(ids,test_size=.25,random_state=self.random_state)
        grid=np.linspace(0,float(y.iloc[tr].duration.max()),32)
        scored=[]
        for ci,(f,cut,_) in enumerate(candidates):
            mask=X[:,f]<=cut; a=tr[mask[tr]]; b=tr[~mask[tr]]
            if len(a)<2 or len(b)<2: continue
            pred=np.empty((len(va),len(self.events_),len(grid)))
            for train_ids,vm in [(a,mask[va]),(b,~mask[va])]:
                if vm.any(): pred[vm]=leaf_prediction(y.iloc[train_ids],self.events_,grid,int(vm.sum()))
            val=ibscr(y.iloc[va],pred,grid,self.events_) if self.method=='ibs' else iauccr(y.iloc[va],pred,grid,self.events_)
            if np.isfinite(val): scored.append(((-val if self.method=='ibs' else val),ci))
        return max(scored,key=lambda a:a[0])[1] if scored else None
    def _build(self,X,y,depth):
        node=TreeNode(self.node_count_,depth,len(y),km_models(y,self.events_),
                      [int((y.event==e).sum()) for e in self.events_])
        self.node_count_+=1
        if depth>=self.depth or len(y)<2*self.min_leaf_ or y.duration.nunique()<2: return node
        candidates=self._candidates(X,y)
        if not candidates: return node
        if depth==0: self.root_candidates_=candidates
        if self.method in ['ibs','iauc']: pick=self._metric_choice(candidates,X,y)
        else: pick=select_candidate([c[2] for c in candidates],self.method,self.importance)
        if pick is None: return node
        f,cut,lr=candidates[pick]
        if np.max(lr)<=0: return node
        mask=X[:,f]<=cut
        if min(mask.sum(),(~mask).sum())<self.min_leaf_: raise AssertionError('Leaf bound violated')
        node.feature=f; node.threshold=cut; node.lr=lr
        node.left=self._build(X[mask],y[mask].reset_index(drop=True),depth+1)
        node.right=self._build(X[~mask],y[~mask].reset_index(drop=True),depth+1)
        return node
    def predict(self,X,times):
        X=np.asarray(X[self.columns_],float)
        if not np.all(np.isfinite(X)): raise ValueError('Preprocess missing values before prediction')
        out=np.empty((len(X),len(self.events_),len(times)))
        def walk(node,ids):
            if not len(ids): return
            if node.left is None:
                out[ids]=np.stack([m.predict(times) for m in node.models]); return
            mask=X[ids,node.feature]<=node.threshold
            walk(node.left,ids[mask]); walk(node.right,ids[~mask])
        walk(self.root_,np.arange(len(X)))
        return out

class KMParallelBootstrapCRAID(ParallelBootstrapCRAID):
    """ParallelBootstrapCRAID adapter; every base model is the unchanged KM tree.
    Overrides binary-outcome OOB machinery of survivors for multiclass outcomes.
    """
    def __init__(self,n_estimators=10,n_jobs=2,random_state=42,**tree_params):
        self.n_estimators=n_estimators; self.n_jobs=n_jobs
        self.random_state=random_state; self.tree_params=tree_params
    def fit(self,X,y):
        y=outcome(y); params=dict(self.tree_params)
        params.setdefault('events',sorted(set(y.event)-{0}))
        self.events_=params['events']
        if self.n_estimators<1: raise ValueError('Empty ensemble')
        def build(seed):
            rng=np.random.default_rng(seed); ids=rng.integers(0,len(X),len(X))
            p=dict(params); p['random_state']=int(seed)
            return KMCompetingTree(**p).fit(X.iloc[ids].reset_index(drop=True),y.iloc[ids])
        seeds=np.random.SeedSequence(self.random_state).generate_state(self.n_estimators)
        self.models=Parallel(n_jobs=self.n_jobs,prefer='threads')(delayed(build)(s) for s in seeds)
        return self
    def predict(self,X,times,n_estimators=None):
        count=len(self.models) if n_estimators is None else n_estimators
        if not 1<=count<=len(self.models): raise ValueError('Invalid ensemble prefix')
        total=np.zeros((len(X),len(self.events_),len(times)))
        for m in self.models[:count]: total+=m.predict(X,times)
        return total/count

def prepare_split(X,y,seed=42):
    y=outcome(y); ids=np.arange(len(y))
    def split(ids,size,rs):
        e=y.iloc[ids].event
        strat=e if e.value_counts().min()>=2 else None
        return train_test_split(ids,test_size=size,random_state=rs,stratify=strat)
    train,test=split(ids,.2,seed); train,val=split(train,.25,seed+1)
    numeric=list(X.select_dtypes(include=np.number).columns)
    categorical=[c for c in X if c not in numeric]
    blocks=[]
    if numeric: blocks.append(('numeric',SimpleImputer(strategy='median'),numeric))
    if categorical: blocks.append(('categorical',make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)),categorical))
    pre=ColumnTransformer(blocks,verbose_feature_names_out=False)
    a=pre.fit_transform(X.iloc[train]); names=pre.get_feature_names_out()
    cv=lambda data: pd.DataFrame(data,columns=names)
    return (cv(a),cv(pre.transform(X.iloc[val])),cv(pre.transform(X.iloc[test])),
            y.iloc[train].reset_index(drop=True),y.iloc[val].reset_index(drop=True),y.iloc[test].reset_index(drop=True))

def plot_tree(model,path,max_depth=3):
    import matplotlib.pyplot as plt
    def leaves(node,depth):
        return 1 if node.left is None or depth>=max_depth else leaves(node.left,depth+1)+leaves(node.right,depth+1)
    width=leaves(model.root_,0); height=min(model.depth,max_depth)+1
    fig=plt.figure(figsize=(max(8,4*width),3.7*height))
    grid=np.linspace(0,max(m.times[-1] for m in model.root_.models),100)
    positions=[]
    def visit(node,x0,x1,d,parent=None,rule=''):
        x=(x0+x1)/2; y=1-(d+.5)/height
        ax=fig.add_axes([x-.43/width,y-.36/height,.86/width,.72/height])
        for e,m in zip(model.events_,node.models): ax.step(grid,m.predict(grid),where='post',label=f'Risk {e}')
        ax.set_ylim(-.03,1.03); ax.set_title(f'Node {node.number}; N={node.n}',fontsize=10)
        ax.legend(fontsize=6); ax.grid(alpha=.2); ax.set_xlabel('Time',fontsize=7)
        ax.tick_params(labelsize=6)
        if parent:
            fig.add_artist(plt.Line2D([parent[0],x],[parent[1]-.36/height,y+.36/height],transform=fig.transFigure,color='gray',lw=.8))
            fig.text((x+parent[0])/2,(y+parent[1])/2,rule,ha='center',fontsize=7)
        if node.left is not None and d<max_depth:
            f=model.columns_[node.feature]; mid=(x0+x1)/2
            # LR is a statistic; p-values here are exploratory, unadjusted for split search.
            j=int(np.argmax(node.lr)); stat=node.lr[j]
            detail=f'LR[{model.events_[j]}]={stat:.2f}'
            visit(node.left,x0,mid,d+1,(x,y),f'{f} <= {node.threshold:.3g}\n{detail}')
            visit(node.right,mid,x1,d+1,(x,y),f'{f} > {node.threshold:.3g}\n{detail}')
    visit(model.root_,0,1,0)
    fig.savefig(path,dpi=130,bbox_inches='tight'); plt.close(fig)

def plot_geometry(model,outdir):
    import matplotlib.pyplot as plt
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    candidates=model.root_candidates_
    if not candidates: return
    def project(v,kind):
        if len(v)<2: return np.zeros((len(v),2))
        if kind=='PCA':
            z=PCA(n_components=min(2,*v.shape),random_state=42).fit_transform(v)
            return np.pad(z,((0,0),(0,2-z.shape[1])))
        return TSNE(n_components=2,perplexity=min(20,max(1,(len(v)-1)/3)),init='random',learning_rate='auto',random_state=42).fit_transform(v)
    rows=[]
    for ci,(f,t,lr) in enumerate(candidates): rows.append({'candidate':ci,'feature':model.columns_[f],'threshold':t,**{f'LR_{e}':v for e,v in zip(model.events_,lr)}})
    pd.DataFrame(rows).to_csv(outdir/'split_vectors.csv',index=False)
    # For metric-directed trees, the geometry summaries use TOPSIS as an explicit reference.
    selection_method=model.method if model.method not in ['ibs','iauc'] else 'topsis'
    winners=[]
    for f in sorted(set(c[0] for c in candidates)):
        group=[c for c in candidates if c[0]==f]; v=np.array([c[2] for c in group]); thresholds=np.array([c[1] for c in group])
        pick=select_candidate(v,selection_method,model.importance); winners.append(group[pick])
        for kind in ['PCA','tSNE']:
            xy=project(v,kind); fig,ax=plt.subplots(figsize=(6,4))
            sc=ax.scatter(xy[:,0],xy[:,1],c=thresholds,cmap='viridis',s=20)
            ax.scatter(*xy[pick],s=140,facecolors='none',edgecolors='red',label='Selected')
            ax.set_title(f'{model.columns_[f]}: {kind}, {selection_method}'); ax.legend()
            fig.colorbar(sc,ax=ax,label='Feature split threshold'); fig.tight_layout()
            fig.savefig(outdir/f'feature_{f}_{kind}.png',dpi=130); plt.close(fig)
    for kind in ['PCA','tSNE']:
        v=np.array([c[2] for c in winners]); xy=project(v,kind)
        pick=select_candidate(v,selection_method,model.importance)
        fig,ax=plt.subplots(figsize=(7,5)); ax.scatter(xy[:,0],xy[:,1],c=np.arange(len(v)),cmap='tab20')
        for i,c in enumerate(winners): ax.annotate(model.columns_[c[0]],xy[i],fontsize=7)
        ax.scatter(*xy[pick],s=180,facecolors='none',edgecolors='red')
        ax.set_title(f'Best split per feature: {kind}, {selection_method}')
        fig.tight_layout(); fig.savefig(outdir/f'between_features_{kind}.png',dpi=130); plt.close(fig)

def run_experiments(outdir,full=False,freddie_csv=None):
    import data_gen_hazardous as dgh
    import matplotlib.pyplot as plt
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; n=50000 if full else 1200; seeds=[42,43,44] if full else [42]
    depths=[3,5,7,10]; ratios=[.1,.05,.01,.001]; sizes=[1,5,10,25,50] if full else [1,3,5]
    def save(): pd.DataFrame(rows).to_csv(outdir/'results.csv',index=False)
    for seed in seeds:
        datasets=[]
        for k in [2,3,4,5]:
            X,y=dgh.make_synthetic_competing_weibull(n_samples=n,n_events=k,n_features=10,complex_features=False,return_X_y=True,random_state=seed)
            datasets.append((f'synthetic_K{k}',X,outcome(y)))
        if freddie_csv:
            df=pd.read_csv(freddie_csv)
            datasets.append(('FreddieMac',df.drop(columns=['event','duration']),outcome(df[['event','duration']])))
        for name,X,y in datasets:
            print(f'Dataset {name}, seed={seed}, N={len(y)}',flush=True)
            a,v,b,ya,yv,yb=prepare_split(X,y,seed)
            events=sorted(set(ya.event)-{0})
            if not set(y.event).issubset({0,*events}): raise ValueError('Risk absent from training; change split')
            times=np.linspace(0,float(ya.duration.max()),60 if full else 30)
            best=None; bestval=float('inf')
            configs=[(d,r,m,'baseline') for d in depths for r in ratios for m in ['max','mean']]
            configs += [(d,r,m,'multicriteria') for d in (depths if full else [3]) for r in (ratios if full else [.05]) for m in METHODS[2:]]
            for depth,ratio,method,stage in configs:
                row=dict(dataset=name,seed=seed,n_events=len(events),stage=stage,depth=depth,leaf_ratio=ratio,method=method,n_estimators=1,n_samples=len(y))
                start=time.perf_counter()
                try:
                    model=KMCompetingTree(depth=depth,min_samples_leaf=ratio,method=method,events=events,random_state=seed,feature_bins=100 if full else 32).fit(a,ya)
                    pv=model.predict(v,times); pt=model.predict(b,times)
                    val=ibscr(yv,pv,times,events)
                    row.update(IBSCR=ibscr(yb,pt,times,events),IAUCCR=iauccr(yb,pt,times,events),validation_IBSCR=val,nodes=model.node_count_,min_leaf=model.min_leaf_,status='Success')
                    if np.isfinite(val) and val<bestval: bestval=val; best=model
                except Exception:
                    row.update(status='Failed',error=traceback.format_exc()); print(row['error'],flush=True)
                row['seconds']=time.perf_counter()-start; rows.append(row); save()
            if best is None: raise RuntimeError(f'All models failed for {name}')
            plot_tree(best,outdir/f'{name}_seed{seed}_KM_tree.png',max_depth=2)
            if seed==seeds[0] and name=='synthetic_K2': plot_geometry(best,outdir/'geometry')
            ensemble=KMParallelBootstrapCRAID(n_estimators=max(sizes),n_jobs=2,random_state=seed,depth=best.depth,min_samples_leaf=best.min_samples_leaf,method=best.method,events=events,feature_bins=best.feature_bins)
            start=time.perf_counter(); ensemble.fit(a,ya); fit_seconds=time.perf_counter()-start
            for count in sizes:
                pred=ensemble.predict(b,times,n_estimators=count)
                rows.append(dict(dataset=name,seed=seed,n_events=len(events),stage='ensemble',depth=best.depth,leaf_ratio=best.min_samples_leaf,method=best.method,n_estimators=count,n_samples=len(y),IBSCR=ibscr(yb,pred,times,events),IAUCCR=iauccr(yb,pred,times,events),status='Success',seconds=fit_seconds))
                save()
    df=pd.DataFrame(rows); good=df[df.status=='Success']
    for stage in ['baseline','multicriteria','ensemble']:
        data=good[(good.stage==stage)&good.dataset.str.startswith('synthetic')]
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        group='n_estimators' if stage=='ensemble' else 'method'
        for label,g in data.groupby(group):
            means=g.groupby('n_events')[['IBSCR','IAUCCR']].mean()
            for ax,metric in zip(axes,['IBSCR','IAUCCR']): ax.plot(means.index,means[metric],marker='o',label=str(label)); ax.set(xlabel='Number of risks',ylabel=metric); ax.legend(fontsize=7)
        fig.suptitle(f'{stage}; mean across configurations/seeds (see results.csv)'); fig.tight_layout(); fig.savefig(outdir/f'{stage}_quality.png',dpi=150); plt.close(fig)
    ens=good[good.stage=='ensemble']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,g in ens.groupby('dataset'):
        means=g.groupby('n_estimators')[['IBSCR','IAUCCR']].mean()
        for ax,metric in zip(axes,['IBSCR','IAUCCR']): ax.plot(means.index,means[metric],marker='o',label=name); ax.set(xlabel='Number of trees',ylabel=metric); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir/'ensemble_size_quality.png',dpi=150); plt.close(fig)
    metadata=dict(profile='full' if full else 'validation',n_samples=n,seeds=seeds,freddie_status='Executed' if freddie_csv else 'Not run: real data not supplied',prediction='Risk-specific KM survival',metric_convention='Course remain IBSCR; pooled-pair time-integrated IAUCCR; no IPCW',failures=int((df.status!='Success').sum()))
    (outdir/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    return df

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--full',action='store_true'); p.add_argument('--out',default='results'); p.add_argument('--freddie-csv')
    args=p.parse_args(); print(run_experiments(args.out,args.full,args.freddie_csv).groupby(['stage','status']).size())
