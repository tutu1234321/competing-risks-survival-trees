"""Numerical and end-to-end regression checks, runnable without experiment files."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from km_coursework import KM, KMCompetingTree, KMParallelBootstrapCRAID, METHODS, ibscr, iauccr, select_candidate
from histogram_core import lr_hist_statistic

def run_checks():
    passed=[]
    t=np.array([1.,1.,2.,3.,4.,5.]); e=np.array([1,0,1,0,1,1])
    grid=np.array([0.,1.,1.5,2.,3.,6.])
    ours=KM().fit(t,e).predict(grid)
    ref=KaplanMeierFitter().fit(t,e).predict(grid).to_numpy()
    np.testing.assert_allclose(ours,ref); passed.append('KM matches lifelines, including ties and censoring')
    np.testing.assert_allclose(KM().fit(t,np.zeros_like(e)).predict(grid),1)
    passed.append('Absent risk produces an all-one survival curve')
    ta=np.array([1,2,2,4]); tb=np.array([1,3,4,4]); ea=np.array([1,1,0,1]); eb=np.array([0,1,1,1])
    ha=np.bincount(ta,minlength=5).astype(float); hb=np.bincount(tb,minlength=5).astype(float)
    ca=np.bincount(ta,weights=ea,minlength=5); cb=np.bincount(tb,weights=eb,minlength=5)
    got=lr_hist_statistic(ha,hb,ca,cb)
    expected=logrank_test(ta,tb,event_observed_A=ea,event_observed_B=eb).test_statistic
    np.testing.assert_allclose(got,expected,rtol=1e-6); passed.append('Histogram LR agrees with reference at exact time bins')
    assert lr_hist_statistic(ha,hb,np.zeros(5),np.zeros(5))==0
    passed.append('Zero-event log-rank is finite zero')
    y=pd.DataFrame({'event':[1,0,2,1],'duration':[1.,4.,2.,3.]})
    ts=np.array([0.,1.,2.,3.]); pred=np.full((4,2,4),.5)
    structured=y.to_records(index=False)
    np.testing.assert_allclose(ibscr(y,pred,ts,[1,2]),ibscr(structured,pred,ts,[1,2]))
    np.testing.assert_allclose(iauccr(y,pred,ts,[1,2]),.5)
    passed.append('Named arrays and DataFrames agree; AUC gives half credit to ties')
    perfect=np.ones((4,2,4))
    for i,row in y.iterrows():
        if row.event: perfect[i,int(row.event)-1,ts>=row.duration]=0
    assert ibscr(y,perfect,ts,[1,2])==0
    assert iauccr(y,perfect,ts,[1,2])==1
    passed.append('Oracle KM-oriented predictions score IBSCR=0 and IAUCCR=1')
    v=np.array([[12.,1.],[1.,12.],[7.,7.]])
    for method in METHODS[:-2]: assert 0<=select_candidate(v,method)<3
    assert select_candidate(v,'maximin')==2
    # Symmetric ranks all average to 2: a stable first-candidate tie break is valid.
    assert select_candidate(v,'rank')==0
    passed.append('Maximin favors balance; average-rank ties have a stable tie break')
    rng=np.random.default_rng(42); X=pd.DataFrame(rng.normal(size=(160,3)),columns=['x','z','w'])
    yy=pd.DataFrame({'event':np.where(X.x>0,7,2),'duration':rng.uniform(1,10,len(X))})
    for method in METHODS:
        model=KMCompetingTree(depth=2,min_samples_leaf=.1,method=method,events=[2,7],feature_bins=6).fit(X,yy)
        assert model.node_count_ > 1, f'{method} incorrectly rejected every split'
        p=model.predict(X,np.linspace(0,11,10))
        assert p.shape==(160,2,10) and np.all(np.diff(p,axis=2)<=1e-10)
        assert model.min_leaf_==16
        def check_leaf(node):
            if node.left is None: assert node.n>=16
            else: check_leaf(node.left); check_leaf(node.right)
        check_leaf(model.root_)
    passed.append('All nine selectors train/predict with nonconsecutive risk labels and valid leaf bounds')
    ens=KMParallelBootstrapCRAID(n_estimators=3,n_jobs=2,random_state=9,depth=2,min_samples_leaf=.1,events=[2,7]).fit(X,yy)
    ts=np.linspace(0,11,10); expected=np.mean([m.predict(X,ts) for m in ens.models],axis=0)
    np.testing.assert_allclose(ens.predict(X,ts),expected)
    ens2=KMParallelBootstrapCRAID(n_estimators=3,n_jobs=2,random_state=9,depth=2,min_samples_leaf=.1,events=[2,7]).fit(X,yy)
    np.testing.assert_allclose(ens.predict(X,ts),ens2.predict(X,ts))
    passed.append('Parallel ensemble equals arithmetic mean and is reproducible')
    result={'status':'passed','checks':passed}
    Path(__file__).with_name('verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2)); return result

if __name__=='__main__': run_checks()
