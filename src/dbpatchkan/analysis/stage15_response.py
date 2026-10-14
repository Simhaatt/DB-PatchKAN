"""Past-only native-telemetry response comparison, executed on remote Colab.

Remote preparation uses backward signal joins and backward kinematic differences.
Inputs end at the cutoff; labels are strictly after it, at fixed 30/60/120-s
horizons. RWTH has no measured pack temperature, so that label remains masked.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import nnls
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dbpatchkan.analysis.stage15_review import update,save_json,digest
from dbpatchkan.data.rwth import matlab_datenum_to_datetime,rwth_vehicle_spec
from dbpatchkan.models import DBPatchKAN,DBPatchKANConfig
from dbpatchkan.models.kan import make_fusion_head

HORIZONS=(30,60,120)
TARGET_NAMES=("current_mean_a","current_abs_p95_a","voltage_mean_v",
              "pack_temperature_rise_c","net_energy_IV_kwh","high_current_exposure_s")
CONCEPT_NAMES=("speed_mean_mps","speed_std_mps","speed_last_mps",
    "acceleration_rms_mps2","jerk_rms_mps3","stop_fraction","high_accel_fraction",
    "current_last_a","current_squared_mean_a2","voltage_last_v","soc_last_pct",
    "pack_temperature_last_c","ambient_last_c","pack_spread_last_c","observed_history_fraction")


def read_signal(path,name):
    frame=pd.read_csv(path,sep='\t',usecols=['Datetime','Value'])
    frame['timestamp']=pd.to_datetime(frame.Datetime,utc=True,errors='coerce')
    frame[name]=pd.to_numeric(frame.Value,errors='coerce')
    return frame.dropna(subset=['timestamp',name]).groupby('timestamp',as_index=False)[name].median().sort_values('timestamp')


def causal_eve(raw):
    from dbpatchkan.data.eve import EVE_FILES,EVE_DOWNLOAD_ROOT
    import urllib.request
    needed=['speed','current','voltage','soc','ambient']+[f'module_temp_{i:02d}' for i in range(1,13)]
    raw.mkdir(parents=True,exist_ok=True)
    for name in needed:
        path=raw/f'{name}.tab'
        if not path.exists() or path.stat().st_size<100:
            url=f'{EVE_DOWNLOAD_ROOT}/{EVE_FILES[name]}'
            update(f'E4 EVE download {name}')
            started=time.time();last=[0.]
            def progress(block,size,total):
                if time.time()-last[0]>15:
                    update(f'E4 {name}: {block*size/2**20:.1f} MB received');last[0]=time.time()
            for attempt in range(3):
                try:
                    request=urllib.request.Request(url,headers={'User-Agent':'DBPatchKAN/0.1'})
                    with urllib.request.urlopen(request,timeout=120) as response, path.with_suffix('.partial').open('wb') as handle:
                        count=0;size=2**20;total=int(response.headers.get('Content-Length',0))
                        while True:
                            chunk=response.read(size)
                            if not chunk:break
                            handle.write(chunk);count+=1;progress(count,size,total)
                    path.with_suffix('.partial').replace(path);break
                except Exception:
                    if attempt==2:raise
            update(f'E4 EVE {name} ready in {(time.time()-started)/60:.1f} min')
    # Ceiling moves the observation to its next grid endpoint, never before it.
    frame=read_signal(raw/'speed.tab','speed_kmh')
    frame.timestamp=frame.timestamp.dt.ceil('5s')
    frame=frame.groupby('timestamp',as_index=False).last()
    for name,col,tolerance in [('current','current_source_a','30s'),('voltage','voltage_v','30s'),
                               ('soc','soc_pct','5min'),('ambient','ambient_c','90min')]:
        signal=read_signal(raw/f'{name}.tab',col)
        if name in ('current','voltage'):
            signal[f'{name}_observed_ns']=signal.timestamp.astype('int64')
        frame=pd.merge_asof(frame,signal,on='timestamp',direction='backward',tolerance=pd.Timedelta(tolerance))
        update(f'E4 EVE causal join: {name}')
    temperature=[];times=[]
    for i in range(1,13):
        col=f'temp_{i:02d}'
        signal=read_signal(raw/f'module_temp_{i:02d}.tab',col)
        signal.loc[~signal[col].between(-30,80),col]=np.nan
        signal[f'ts_{i:02d}']=signal.timestamp.astype('int64')
        frame=pd.merge_asof(frame,signal,on='timestamp',direction='backward',tolerance=pd.Timedelta('15min'))
        temperature.append(col);times.append(f'ts_{i:02d}')
    frame['pack_temp_c']=frame[temperature].mean(axis=1)
    frame['spread_c']=frame[temperature].max(axis=1)-frame[temperature].min(axis=1)
    frame['temp_source_ns']=frame[times].max(axis=1)
    frame['current_a']=frame.current_source_a # sign resolved on outer-training observations only
    frame['vehicle']='Renault-Zoe-Q90-1';frame['nominal_capacity_ah']=65.
    return frame


def make_examples(frame,dataset,*,max_per_vehicle=1500):
    """Disjoint input/maximum-future spans, 5-second backward-sampled grid."""
    d,b,c,y,metadata=[],[],[],[],[]
    frame=frame.sort_values('timestamp').drop_duplicates('timestamp').copy()
    # A source gap breaks the session. No resampling across parking gaps.
    frame['session']=frame.timestamp.diff().dt.total_seconds().gt(30).cumsum()
    for session,part in frame.groupby('session',sort=True):
        if len(part)<48:continue
        part=part.sort_values('timestamp')
        # Use last actual observation available at each endpoint; never interpolate.
        time_grid=pd.date_range(part.timestamp.min().ceil('5s'),part.timestamp.max().floor('5s'),freq='5s')
        if len(time_grid)<48:continue
        grid=pd.merge_asof(pd.DataFrame({'timestamp':time_grid}),part.drop(columns='session'),
                           on='timestamp',direction='backward',tolerance=pd.Timedelta('5s'))
        values=grid[['speed_kmh','current_a','voltage_v','soc_pct','pack_temp_c','spread_c','ambient_c']].to_numpy(float)
        speed=values[:,0]/3.6
        # Backward differences stop gradients looking across the prediction boundary.
        accel=np.diff(speed,prepend=np.nan)/5
        jerk=np.diff(accel,prepend=np.nan)/5
        vehicle=str(part.vehicle.iloc[0]);q=float(part.nominal_capacity_ah.iloc[0])
        for end in range(23,len(grid)-24,48):
            past=slice(end-23,end+1)
            v=speed[past];a=accel[past];j=jerk[past]
            batt=values[past]
            if np.isfinite(v).mean()<.9 or np.nanmax(v)<1.4 or np.isfinite(batt[:,1:4]).mean()<.85:
                continue
            if not np.isfinite(v[-1]) or not np.isfinite(batt[-1,1:4]).all():continue
            def finite_stat(x,op,default=np.nan):
                finite=x[np.isfinite(x)];return op(finite) if len(finite) else default
            concepts=np.array([finite_stat(v,np.mean),finite_stat(v,np.std),v[-1],
                finite_stat(a,lambda x:np.sqrt(np.mean(x*x))),finite_stat(j,lambda x:np.sqrt(np.mean(x*x))),
                np.mean(v<.3),np.mean(a>1.5),batt[-1,1],finite_stat(batt[:,1]**2,np.mean),
                batt[-1,2],batt[-1,3],batt[-1,4],batt[-1,6],batt[-1,5],np.isfinite(v).mean()],dtype='float32')
            driver=np.full((24,13),np.nan,dtype='float32');battery=np.full((24,11),np.nan,dtype='float32')
            driver[:,:3]=np.stack([v,a,j],axis=-1)
            battery[:,:6]=batt[:,[2,1,3,4,5,6]]
            driver[:,9]=np.isfinite(v);driver[:,10]=5;driver[:,11]=np.arange(24)/23
            driver[:,12]=1-np.isfinite(driver[:,:9]).mean(axis=1)
            battery[:,7]=np.isfinite(battery[:,:3]).mean(axis=1);battery[:,8]=5;battery[:,9]=np.arange(24)/23
            battery[:,10]=1-np.isfinite(battery[:,:7]).mean(axis=1)
            targets=[];cutoff=grid.timestamp.iloc[end]
            for horizon in HORIZONS:
                future=values[end+1:end+1+horizon//5]
                ii,vv,tt=future[:,1],future[:,2],future[:,4]
                valid=np.isfinite(ii)&np.isfinite(vv)
                if 'current_observed_ns' in grid:
                    source_time=grid.current_observed_ns.iloc[end+1:end+1+horizon//5].to_numpy()
                    voltage_time=grid.voltage_observed_ns.iloc[end+1:end+1+horizon//5].to_numpy()
                    valid &= (source_time>grid.timestamp.iloc[end].value)&(voltage_time>grid.timestamp.iloc[end].value)
                    ii=np.where(valid,ii,np.nan);vv=np.where(valid,vv,np.nan)
                measured_fraction=valid.mean()
                temp_ok=False
                if 'temp_source_ns' in grid and np.isfinite(tt[-1]):
                    last_temp=grid.temp_source_ns.iloc[end+horizon//5]
                    temp_ok=bool(np.isfinite(last_temp) and last_temp>cutoff.value)
                target=[finite_stat(ii,np.mean),finite_stat(np.abs(ii),lambda x:np.percentile(x,95)),
                    finite_stat(vv,np.mean),tt[-1]-batt[-1,4] if temp_ok else np.nan,
                    np.sum(ii[valid]*vv[valid])*5/3.6e6 if measured_fraction>=.9 else np.nan,
                    np.sum(np.abs(ii[valid])/q>=1)*5 if measured_fraction>=.9 else np.nan]
                if measured_fraction<.85: target=[np.nan]*6
                targets.extend(target)
            if np.isfinite(targets).sum()<12:continue
            d.append(driver);b.append(battery);c.append(concepts);y.append(targets)
            metadata.append({'dataset':dataset,'vehicle':vehicle,'session_key':f'{vehicle}:{session}',
                'input_start_ns':grid.timestamp.iloc[end-23].value,'cutoff_ns':cutoff.value,
                'future_end_ns':grid.timestamp.iloc[end+24].value,'nominal_capacity_ah':q})
    if not metadata:raise ValueError(f'No valid native response examples: {dataset}')
    table=pd.DataFrame(metadata);keep=[]
    for _,part in table.groupby('vehicle',sort=True):
        # Fixed evenly-spaced cap across chronology, chosen before targets are scored.
        indices=part.index.to_numpy();keep.extend(indices[np.linspace(0,len(indices)-1,min(len(indices),max_per_vehicle),dtype=int)])
    keep=np.array(sorted(keep));arrays={'driver':np.stack(d)[keep],'battery':np.stack(b)[keep],
        'concepts':np.stack(c)[keep],'targets':np.asarray(y,dtype='float32')[keep]}
    return arrays,table.iloc[keep].reset_index(drop=True)


def prepare(inputs,output,*,max_per_vehicle=1500,include_eve=True):
    output.mkdir(parents=True,exist_ok=True)
    rwth=[];tables=[]
    for file in sorted((inputs/'rwth_native').glob('*.parquet')):
        update(f'E4 native preparation: {file.stem}')
        cols=['time_num','Speed','Current','Voltage','SoC_Real','Temp_Ambient']
        raw=pq.read_table(file,columns=cols).to_pandas()
        f=pd.DataFrame({'timestamp':pd.to_datetime(matlab_datenum_to_datetime(raw.time_num),utc=True),
            'speed_kmh':pd.to_numeric(raw.Speed,errors='coerce'),
            'current_a':-pd.to_numeric(raw.Current,errors='coerce'),
            'voltage_v':pd.to_numeric(raw.Voltage,errors='coerce'),
            'soc_pct':pd.to_numeric(raw.SoC_Real,errors='coerce'),
            'ambient_c':pd.to_numeric(raw.Temp_Ambient,errors='coerce'),
            'pack_temp_c':np.nan,'spread_c':np.nan,'vehicle':file.stem,
            'nominal_capacity_ah':rwth_vehicle_spec(file.stem).nominal_capacity_ah})
        f.loc[~f.speed_kmh.between(0,200),'speed_kmh']=np.nan
        f.loc[~f.voltage_v.between(100,500),'voltage_v']=np.nan
        f.loc[f.current_a.abs().gt(1000),'current_a']=np.nan
        f.loc[~f.soc_pct.between(0,100),'soc_pct']=np.nan
        f=f.dropna(subset=['timestamp'])
        f['current_observed_ns']=f.timestamp.astype('int64')
        f['voltage_observed_ns']=f.timestamp.astype('int64')
        arrays,table=make_examples(f,'RWTH',max_per_vehicle=max_per_vehicle)
        rwth.append(arrays);tables.append(table)
        update(f'E4 {file.stem}: {len(table)} non-overlapping examples retained')
    if not rwth:raise FileNotFoundError('RWTH raw field parquets missing')
    combined={k:np.concatenate([x[k] for x in rwth]) for k in rwth[0]}
    np.savez_compressed(output/'rwth_future_response.npz',**combined)
    pd.concat(tables).to_csv(output/'rwth_future_response_metadata.csv',index=False)
    status={"RWTH":"prepared","EVE":"disabled"}
    if include_eve:
        try:
            frame=causal_eve(output/'eve_raw')
            frame.loc[~frame.speed_kmh.between(0,200),'speed_kmh']=np.nan
            frame.loc[~frame.voltage_v.between(100,500),'voltage_v']=np.nan
            frame.loc[frame.current_a.abs().gt(1000),'current_a']=np.nan
            frame.loc[~frame.soc_pct.between(0,100),'soc_pct']=np.nan
            arrays,table=make_examples(frame,'EVE',max_per_vehicle=6000)
            np.savez_compressed(output/'eve_future_response.npz',**arrays)
            table.to_csv(output/'eve_future_response_metadata.csv',index=False)
            status['EVE']='prepared'
        except Exception as error:
            status['EVE']=f'incomplete: {type(error).__name__}: {error}'
            update(f'E4 EVE pending: {error}. RWTH and the remaining analyses can proceed.')
    save_json(output/'E4_PREPARATION_AUDIT.json',{'status':status,'horizons_s':HORIZONS,'history_s':120,
        'cadence_s':5,'sampling':'last available observation with backward joins; no interpolation',
        'derivatives':'backward differences only','stride_s':240,'targets':TARGET_NAMES,
        'thermal_labels':'EVE measured module temperatures; RWTH temperature masked',
        'energy_target':'derived from measured I*V; not an independent energy-meter measurement',
        'high_current_target':'derived exposure with stated nominal capacity; not independent health ground truth',
        'max_examples_per_rwth_vehicle':max_per_vehicle})


class ResponseConceptModel(nn.Module):
    def __init__(self,head):
        super().__init__()
        self.config=DBPatchKANConfig(segment_length=24,patch_length=6,patch_stride=3,d_model=96,
            encoder_layers=2,n_heads=4,ff_dim=192,concept_names=CONCEPT_NAMES,fusion_head=head,
            kan_hidden=(16,8),mlp_hidden=(32,16),kan_grid_size=5)
        self.backbone=DBPatchKAN(self.config)
        self.response_head=make_fusion_head(head,15,18,hidden_dimensions=(16,8) if head=='kan' else (32,16),grid_size=5,dropout=.1)
    def forward(self,d,b):
        encoded=self.backbone.encode_segments(d,b,driver_mask=torch.isfinite(d),battery_mask=torch.isfinite(b))
        concepts=self.backbone.concept_predictor(encoded['embedding'])
        # Concept head is evaluated FP32, including all B-spline calculations.
        with torch.autocast(device_type=d.device.type,enabled=False):
            predictions=self.response_head(concepts.float())
        return predictions,concepts.float()


def norm_stats(x,axis):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=RuntimeWarning)
        center=np.nanmedian(x,axis=axis);scale=np.nanpercentile(x,75,axis=axis)-np.nanpercentile(x,25,axis=axis)
    center=np.nan_to_num(center).astype('float32');scale=np.where(np.isfinite(scale)&(scale>1e-5),scale,1).astype('float32')
    return center,scale


def infer(model,d,b,device,batch=256):
    model.eval();result=[]
    with torch.no_grad():
        for start in range(0,len(d),batch):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                pred,_=model(torch.from_numpy(d[start:start+batch]).to(device),torch.from_numpy(b[start:start+batch]).to(device))
            result.append(pred.float().cpu().numpy())
    return np.concatenate(result)


def constrain_responses(prediction):
    """Pre-specified physical output domains, applied to all predictor families."""
    prediction=prediction.copy()
    for h,horizon in enumerate(HORIZONS):
        prediction[:,h*6+1]=np.maximum(prediction[:,h*6+1],0)
        prediction[:,h*6+2]=np.clip(prediction[:,h*6+2],0,500)
        prediction[:,h*6+5]=np.clip(prediction[:,h*6+5],0,horizon)
    return prediction


def fit_neural(arrays,train,validation,test,head,seed,output,epochs,device,signature):
    started=time.time()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=ResponseConceptModel(head).to(device)
    # Remove unused health/reconstruction modules from optimization (kept only as backbone API).
    dcenter,dscale=norm_stats(arrays['driver'][train],(0,1));bcenter,bscale=norm_stats(arrays['battery'][train],(0,1))
    # Metadata channels are not scaled.
    dcenter[9:]=0;dscale[9:]=1;bcenter[7:]=0;bscale[7:]=1
    model.backbone.set_input_normalization(torch.from_numpy(dcenter),torch.from_numpy(dscale),torch.from_numpy(bcenter),torch.from_numpy(bscale))
    cc,cs=norm_stats(arrays['concepts'][train],0);yc,ys=norm_stats(arrays['targets'][train],0)
    concepts=(arrays['concepts']-cc)/cs;targets=(arrays['targets']-yc)/ys
    output.mkdir(parents=True,exist_ok=True);latest=output/'latest.pt';best=output/'model.pt'
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    history=[];best_loss=np.inf;patience=6;start_epoch=1
    generator=torch.Generator().manual_seed(seed+1001)
    if latest.exists():
        cp=torch.load(latest,map_location=device,weights_only=False)
        if cp['signature']!=signature:raise ValueError('Resume protocol differs; choose a new run folder')
        model.load_state_dict(cp['state']);optimizer.load_state_dict(cp['optimizer'])
        history=cp['history'];best_loss=cp['best_loss'];patience=cp['patience'];start_epoch=cp['epoch']+1
        torch.set_rng_state(cp['torch_rng'].cpu());np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng'])
        generator.set_state(cp['loader_rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([x.cpu() for x in cp['cuda_rng']])
    loader=DataLoader(TensorDataset(torch.from_numpy(arrays['driver'][train]),torch.from_numpy(arrays['battery'][train]),
        torch.from_numpy(targets[train]),torch.from_numpy(concepts[train])),batch_size=256,shuffle=True,num_workers=0,generator=generator)
    def huber(p,t):
        valid=torch.isfinite(t);return nn.functional.huber_loss(p[valid],t[valid]) if valid.any() else p.sum()*0
    for epoch in range(start_epoch,epochs+1):
        if patience<=0:break
        model.train();losses=[]
        for d,b,y,c in loader:
            d,b,y,c=[v.to(device) for v in (d,b,y,c)];optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                pred,concept=model(d,b)
            loss=huber(pred,y)+.2*huber(concept,c)+1e-4*model.response_head.regularization_loss()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
        predicted=infer(model,arrays['driver'][validation],arrays['battery'][validation],device)
        predicted=(constrain_responses(predicted*ys+yc)-yc)/ys
        err=np.abs(predicted-targets[validation]);count=np.isfinite(err).sum(axis=0)
        per_target=np.divide(np.nansum(err,axis=0),count,out=np.full(err.shape[1],np.nan),where=count>0)
        val=float(np.nanmean(per_target));history.append({'epoch':epoch,'train_loss':np.mean(losses),'validation_normalized_mae':val})
        if val<best_loss-1e-5:
            best_loss=val;patience=6
            torch.save({'state':model.state_dict(),'config':asdict(model.config),'head':head,'signature':signature,
                'target_center':yc,'target_scale':ys,'concept_center':cc,'concept_scale':cs,
                'train_indices':train,'validation_indices':validation,'test_indices':test},best)
        else:patience-=1
        torch.save({'state':model.state_dict(),'optimizer':optimizer.state_dict(),'history':history,'epoch':epoch,
            'best_loss':best_loss,'patience':patience,'signature':signature,'torch_rng':torch.get_rng_state(),
            'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'loader_rng':generator.get_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []},latest)
        pd.DataFrame(history).to_csv(output/'history.csv',index=False)
        if epoch==1 or epoch%5==0:update(f'E4 {output.parent.name}/{head} seed={seed} epoch={epoch}/{epochs}: validation={val:.4f}')
    cp=torch.load(best,map_location=device,weights_only=False);model.load_state_dict(cp['state'])
    prediction=constrain_responses(infer(model,arrays['driver'][test],arrays['battery'][test],device)*ys+yc)
    save_json(output/'compute_metrics.json',{'elapsed_current_session_minutes':(time.time()-started)/60,
        'best_validation_normalized_mae':best_loss,'epochs_in_history':len(history),
        'parameters':sum(p.numel() for p in model.parameters()),
        'peak_gpu_allocated_gb':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
        'precision':'BF16 encoders/FP32 spline head','device':str(device)})
    return prediction,model


def tabular_predictions(arrays,train,test,output):
    c=arrays['concepts'];y=arrays['targets'];n=len(test)
    # Context only has observed past battery state and ambient; no target-window state.
    context_idx=[7,8,9,10,11,12,13,14]
    x_context=c[:,context_idx];x_driver=c
    v=c[:,2];a=arrays['driver'][:,-1,1];voltage=np.maximum(c[:,9],100)
    physics=np.stack([v/voltage,v**3/voltage,v*a/voltage,c[:,7],c[:,11]-c[:,12],c[:,8]],axis=1)
    baseline=np.tile(np.stack([c[test,7],np.abs(c[test,7]),c[test,9],np.zeros(n),np.zeros(n),np.zeros(n)],axis=1),(1,3))
    for h,horizon in enumerate(HORIZONS):
        baseline[:,h*6+4]=c[test,7]*c[test,9]*horizon/3.6e6
        # Exact capacity for exposure is supplied separately after this helper.
    result={'State_persistence':baseline}
    # Effective longitudinal demand: P = a*v + b*v^3 + m_eff*v*accel + P_aux.
    # Nonnegative coefficients preserve the drag/rolling/inertial signs; negative
    # acceleration can still yield regenerative (negative) current.
    force_features=np.stack([v/voltage,v**3/voltage,v*a/voltage,1/voltage],axis=1)
    imputer=SimpleImputer(strategy='median',keep_empty_features=True).fit(force_features[train])
    ff=imputer.transform(force_features)
    pooled_x=[];pooled_y=[]
    for h in range(3):
        valid=np.isfinite(y[train,h*6])
        pooled_x.append(ff[train[valid]]);pooled_y.append(y[train[valid],h*6])
    motor=nnls(np.concatenate(pooled_x),np.concatenate(pooled_y))[0]
    current_prediction=ff[test]@motor
    r_x=[];r_y=[];thermal_x=[];thermal_y=[]
    for h,horizon in enumerate(HORIZONS):
        valid=np.isfinite(y[train,h*6])&np.isfinite(y[train,h*6+2])&np.isfinite(c[train,7])&np.isfinite(c[train,9])
        r_x.extend((-(y[train[valid],h*6]-c[train[valid],7])).tolist())
        r_y.extend((y[train[valid],h*6+2]-c[train[valid],9]).tolist())
        valid_t=np.isfinite(y[train,h*6+3])&np.isfinite(c[train,11])&np.isfinite(c[train,12])&np.isfinite(c[train,8])
        thermal_x.extend(np.stack([horizon*c[train[valid_t],8],-horizon*(c[train[valid_t],11]-c[train[valid_t],12])],axis=1).tolist())
        thermal_y.extend(y[train[valid_t],h*6+3].tolist())
    resistance=float(nnls(np.array(r_x).reshape(-1,1),np.array(r_y))[0][0]) if len(r_y)>=10 else 0.
    heat_cool=nnls(np.array(thermal_x),np.array(thermal_y))[0] if len(thermal_y)>=10 else np.zeros(2)
    volts=np.clip(c[test,9]-resistance*(current_prediction-c[test,7]),0,500)
    structured=np.full((n,18),np.nan)
    for h,horizon in enumerate(HORIZONS):
        structured[:,h*6]=current_prediction
        structured[:,h*6+1]=np.abs(current_prediction) # constant-demand tail estimate
        structured[:,h*6+2]=volts
        structured[:,h*6+3]=horizon*(heat_cool[0]*c[test,8]-heat_cool[1]*(c[test,11]-c[test,12]))
        structured[:,h*6+4]=current_prediction*volts*horizon/3.6e6
        # High-current duration uses known pack capacity in the caller.
    result['Longitudinal_Electrothermal_NNLS']=structured
    output.mkdir(parents=True,exist_ok=True)
    import joblib
    joblib.dump(imputer,output/'physical_feature_imputer.joblib')
    save_json(output/'PHYSICAL_BASELINE_PARAMETERS.json',{'current_power_coefficients':motor.tolist(),
        'feature_order':['v_over_V','v_cubed_over_V','v_accel_over_V','one_over_V'],
        'effective_resistance_ohm':resistance,'heating_K_per_A2_second':float(heat_cool[0]),
        'cooling_per_second':float(heat_cool[1]),'voltage_QC_clip_V':[0,500],
        'electrical_training_label_count':len(r_y),'thermal_training_label_count':len(thermal_y),
        'assumption':'locally sustained prior driver demand and linear lumped heat balance over 30-120s',
        'calibration':'outer-fold inner training only; effective fitted parameters, not independently measured pack constants'})
    for name,x,estimator in (
        ('Physics_feature_Ridge',physics,Ridge(alpha=1)),
        ('Context_HGB',x_context,HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=1,random_state=42)),
        ('Context_Driver_HGB',x_driver,HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=1,random_state=42))):
        prediction=np.full((n,18),np.nan)
        for j in range(18):
            valid=np.isfinite(y[train,j])
            if valid.sum()<10:continue
            model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),copy.deepcopy(estimator))
            model.fit(x[train[valid]],y[train[valid],j]);prediction[:,j]=model.predict(x[test])
            import joblib
            output.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,output/f'{name}_target_{j:02d}.joblib')
        result[name]=prediction
    return result


def run_response(prepared,output,seeds,epochs):
    if not torch.cuda.is_available():raise RuntimeError('E4 neural comparisons require the Colab GPU')
    device=torch.device('cuda');output.mkdir(parents=True,exist_ok=True)
    all_rows=[];splits=[]
    for dataset in ('rwth','eve'):
        path=prepared/f'{dataset}_future_response.npz'
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        meta=pd.read_csv(prepared/f'{dataset}_future_response_metadata.csv')
        if not (meta.input_start_ns<meta.cutoff_ns).all() or not (meta.cutoff_ns<meta.future_end_ns).all():
            raise ValueError('Past/future boundary audit failed')
        index=np.arange(len(meta));outer=[]
        if dataset=='rwth':
            for vehicle in sorted(meta.vehicle.unique()):outer.append((vehicle,index[meta.vehicle.ne(vehicle)],index[meta.vehicle.eq(vehicle)]))
        else:
            # Entire observation sessions stay in one fold. Future spans are purged at cutoffs.
            sessions=meta.groupby('session_key').cutoff_ns.min().sort_values().index.to_numpy()
            bounds=np.linspace(max(3,len(sessions)//2),len(sessions),4,dtype=int)
            for number,(start,end) in enumerate(zip(bounds[:-1],bounds[1:]),1):
                te=index[meta.session_key.isin(sessions[start:end])];tr=index[meta.session_key.isin(sessions[:start])]
                if len(te):tr=tr[meta.iloc[tr].future_end_ns.to_numpy()<meta.iloc[te].input_start_ns.min()]
                outer.append((f'temporal_{number}',tr,te))
        for fold,(name,outer_train,test) in enumerate(outer,1):
            if len(outer_train)<30 or len(test)<5:continue
            # Inner validation is identical across all models/seeds in this new matched experiment.
            if dataset=='rwth':
                a,b=next(GroupShuffleSplit(1,test_size=.2,random_state=42).split(outer_train,groups=meta.iloc[outer_train].vehicle))
                train,validation=outer_train[a],outer_train[b]
            else:
                sessions=meta.iloc[outer_train].groupby('session_key').cutoff_ns.min().sort_values().index.to_numpy()
                count=max(1,len(sessions)//5);validation=outer_train[meta.iloc[outer_train].session_key.isin(sessions[-count:])]
                train=outer_train[meta.iloc[outer_train].session_key.isin(sessions[:-count])]
                train=train[meta.iloc[train].future_end_ns.to_numpy()<meta.iloc[validation].input_start_ns.min()]
            if len(train)<20 or len(validation)<5:continue
            local={k:v.copy() for k,v in arrays.items()}
            multiplier=1.
            if dataset=='eve':
                moving=local['driver'][train,:,0]>2.8
                current=local['battery'][train,:,1]
                multiplier=-1. if np.nanmedian(current[moving])<0 else 1.
                local['battery'][:,:,1]*=multiplier;local['concepts'][:,7]*=multiplier
                local['targets'][:,0::6]*=multiplier;local['targets'][:,4::6]*=multiplier
            splits.append({'dataset':dataset,'fold':fold,'name':name,'training_rows':len(train),
                'validation_rows':len(validation),'test_rows':len(test),'current_sign_multiplier':multiplier,
                'training_vehicles':sorted(set(meta.iloc[train].vehicle)),'test_vehicles':sorted(set(meta.iloc[test].vehicle)),
                'train_indices':train.tolist(),'validation_indices':validation.tolist(),'test_indices':test.tolist()})
            update(f'E4 {dataset} fold {fold}/{len(outer)} {name}: train={len(train)}, validation={len(validation)}, test={len(test)}')
            preds=tabular_predictions(local,train,test,output/'models'/dataset/f'fold_{fold}_{name}'/'classical')
            for h,horizon in enumerate(HORIZONS):
                preds['State_persistence'][:,h*6+5]=(np.abs(local['concepts'][test,7])/meta.iloc[test].nominal_capacity_ah.to_numpy()>=1)*horizon
                preds['Longitudinal_Electrothermal_NNLS'][:,h*6+5]=(np.abs(preds['Longitudinal_Electrothermal_NNLS'][:,h*6])/meta.iloc[test].nominal_capacity_ah.to_numpy()>=1)*horizon
            def records(model,seed,prediction):
                prediction=constrain_responses(prediction)
                for i,j in enumerate(test):
                    for h,horizon in enumerate(HORIZONS):
                        for t,target in enumerate(TARGET_NAMES):
                            col=h*6+t
                            if np.isfinite(local['targets'][j,col]) and np.isfinite(prediction[i,col]):
                                all_rows.append({'dataset':dataset,'fold':fold,'model':model,'seed':seed,
                                    'example_id':int(j),'vehicle_key':meta.vehicle.iloc[j],'session_key':meta.session_key.iloc[j],
                                    'cutoff_ns':meta.cutoff_ns.iloc[j],'horizon_s':horizon,'target':target,
                                    'observed':local['targets'][j,col],'predicted':prediction[i,col]})
            for model,p in preds.items():records(model,-1,p)
            for seed in seeds:
                for head in ('mlp','kan'):
                    root=output/'models'/dataset/f'fold_{fold}_{name}'/f'{head}_seed_{seed}'
                    signature=f'{digest(path)}:seed={seed}:head={head}:epochs={epochs}:protocol=stage15v1'
                    saved=root/'predictions.npy'
                    complete=root/'COMPLETE.json'
                    if complete.exists() and json.loads(complete.read_text())['signature']==signature:
                        prediction=np.load(saved);update(f'E4 resume completed {dataset}/{name}/{head}/{seed}')
                    else:
                        prediction,model=fit_neural(local,train,validation,test,head,seed,root,epochs,device,signature)
                        np.save(saved,prediction);save_json(complete,{'signature':signature,'complete':True})
                    records('DualStream_Concept_'+head.upper(),seed,prediction)
            pd.DataFrame(all_rows).to_csv(output/'E4_future_response_seed_predictions.csv',index=False)
            save_json(output/'E4_SPLIT_LINEAGE.json',splits)
    frame=pd.DataFrame(all_rows)
    if frame.empty:raise ValueError('No response folds were eligible')
    keys=['dataset','fold','model','example_id','vehicle_key','session_key','cutoff_ns','horizon_s','target','observed']
    ens=frame.groupby(keys,as_index=False).predicted.mean()
    ens.to_csv(output/'E4_future_response_ensemble_predictions.csv',index=False)
    summaries=[];paired=[]
    for (dataset,target,horizon),part in ens.groupby(['dataset','target','horizon_s']):
        for model,p in part.groupby('model'):
            e=p.predicted-p.observed
            summaries.append({'dataset':dataset,'target':target,'horizon_s':horizon,'model':model,'n':len(p),
                'mae':e.abs().mean(),'rmse':np.sqrt((e*e).mean()),'bias':e.mean(),
                'vehicle_macro_mae':e.abs().groupby(p.vehicle_key).mean().mean()})
        for candidate,reference in [('Context_Driver_HGB','Context_HGB'),
            ('DualStream_Concept_KAN','Context_Driver_HGB'),('DualStream_Concept_KAN','DualStream_Concept_MLP'),
            ('DualStream_Concept_KAN','State_persistence'),('DualStream_Concept_KAN','Physics_feature_Ridge'),
            ('DualStream_Concept_KAN','Longitudinal_Electrothermal_NNLS')]:
            c=part[part.model.eq(candidate)];r=part[part.model.eq(reference)]
            m=c.merge(r,on=['example_id','fold'],suffixes=('','_r'),validate='one_to_one')
            if not len(m):continue
            cluster=m.vehicle_key if dataset=='rwth' else (m.cutoff_ns//(7*86400*10**9)).astype(str)
            diff=(m.predicted-m.observed).abs()-(m.predicted_r-m.observed).abs()
            d=diff.groupby(cluster).mean().to_numpy();rng=np.random.default_rng(42)
            boot=np.mean(rng.choice(d,(10000,len(d)),replace=True),axis=1)
            paired.append({'dataset':dataset,'target':target,'horizon_s':horizon,'candidate':candidate,'reference':reference,
                'clusters':len(d),'cluster_unit':'vehicle' if dataset=='rwth' else '7_day_time_block',
                'mean_delta_mae':d.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),
                'few_clusters':len(d)<10,'negative_favors_candidate':True})
    pd.DataFrame(summaries).to_csv(output/'E4_matched_future_response_metrics.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'E4_matched_response_cluster_comparisons.csv',index=False)
    save_json(output/'E4_SCOPE.json',{'future_response_only':True,'same_splits_and_targets':True,
        'new_experiment_posthoc_on_known_datasets':True,'driver_future_available':False,
        'concepts':CONCEPT_NAMES,'seed_values':seeds,'epochs_max':epochs,
        'physics_baseline':'nonnegative longitudinal power coefficients; nonnegative effective resistance; nonnegative lumped heating/cooling gains; inner-training calibration only. Additional physics-feature Ridge also retained.',
        'time_normalized_whole_trip_inputs_used':False,'physics_soft_penalty_used':False,
        'shared_output_bounds':'abs-current >= 0; voltage 0-500 V; high-current exposure 0-horizon seconds, applied to every family before scoring; signed current, energy and temperature change remain signed',
        'independent_health_validation':False,
        'EVE_uncertainty':'7-day time blocks on one vehicle; exploratory temporal uncertainty, not cross-vehicle generalization'})
    update('E4 future response matched comparisons complete')


def benchmark(inputs,output,prepared,response_output):
    """Benchmark real saved weights on the remote CPU; no energy-meter claim."""
    from dbpatchkan.analysis.stage13_persistence_forecast import PersistenceAnchoredPatchConcept
    from dbpatchkan.analysis.stage13b_recent_methods import make_model
    torch.set_num_threads(1);records=[]
    import psutil
    process=psutil.Process()
    paths=[('Fleet_KAN',inputs/'fleet_checkpoints/fold_1/kan_seed_42/model.pt','kan'),
           ('Fleet_matched_MLP',inputs/'benchmark_checkpoints/mlp/model.pt','mlp'),
           ('Fleet_TSLANet_2024',inputs/'benchmark_checkpoints/tslanet/model.pt','tslanet')]
    for name,path,head in paths:
        if not path.exists():continue
        model=PersistenceAnchoredPatchConcept(head=head) if head in ('kan','mlp') else make_model(head)
        state=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state']);model.eval()
        params=sum(x.numel() for x in model.parameters());bytes_=sum(x.numel()*x.element_size() for x in model.parameters())
        buffers=sum(x.numel()*x.element_size() for x in model.buffers())
        for batch in (1,64):
            sequence=torch.zeros(batch,128,8);physical=torch.zeros(batch,14);context=torch.zeros(batch,6)
            def forward():
                return model(sequence,physical,context) if head in ('kan','mlp') else model(sequence,torch.cat([physical,context],1))
            with torch.no_grad():
                for _ in range(10):forward()
                durations=[];rss_before=process.memory_info().rss;peak_rss=rss_before
                for _ in range(100):
                    start=time.perf_counter();forward();durations.append((time.perf_counter()-start)*1000)
                    peak_rss=max(peak_rss,process.memory_info().rss)
            records.append({'model':name,'parameters':params,'weight_bytes':bytes_,'buffer_bytes':buffers,
                'checkpoint_bytes':path.stat().st_size,'batch_size':batch,'precision':'float32','threads':1,
                'task':'charging_capacity_forecast','input_steps':128,'physical_sequence_channels':8,
                'latency_median_ms':np.median(durations),'latency_p95_ms':np.quantile(durations,.95),
                'rss_before_mb':rss_before/2**20,'peak_process_rss_mb':peak_rss/2**20,
                'throughput_samples_per_second':batch/(np.mean(durations)/1000)})
        update(f'E6 {name}: {params:,} parameters; remote CPU benchmark complete')
    # Also measure the full frozen Nissan dual-stream architecture, within its
    # own common input profile. Do not conflate segment latency with a full-trip bag.
    from dbpatchkan.analysis.stage8_final import _load_lightweight
    full_paths=[('Nissan_full_DB_PatchKAN',inputs/'benchmark_checkpoints/nissan_kan/model.pt'),
                ('Nissan_full_matched_MLP',inputs/'benchmark_checkpoints/nissan_mlp/model.pt')]
    for name,path in full_paths:
        model,_=_load_lightweight(path,torch.device('cpu'));model.eval()
        params=sum(x.numel() for x in model.parameters())
        for batch in (1,64):
            d=torch.zeros(batch,256,13)+model.driver_input_center
            b=torch.zeros(batch,256,11)+model.battery_input_center
            with torch.no_grad():
                for _ in range(10):model(d,b,domain='nissan')
                durations=[];rss_before=process.memory_info().rss;peak_rss=rss_before
                for _ in range(100):
                    start=time.perf_counter();model(d,b,domain='nissan');durations.append((time.perf_counter()-start)*1000)
                    peak_rss=max(peak_rss,process.memory_info().rss)
            records.append({'model':name,'parameters':params,'weight_bytes':sum(x.numel()*x.element_size() for x in model.parameters()),
                'buffer_bytes':sum(x.numel()*x.element_size() for x in model.buffers()),'checkpoint_bytes':path.stat().st_size,
                'batch_size':batch,'precision':'float32','threads':1,'task':'Nissan_health_one_segment',
                'input_steps':256,'physical_sequence_channels':24,'latency_median_ms':np.median(durations),
                'latency_p95_ms':np.quantile(durations,.95),'rss_before_mb':rss_before/2**20,
                'peak_process_rss_mb':peak_rss/2**20,'throughput_samples_per_second':batch/(np.mean(durations)/1000)})
        update(f'E6 {name}: full saved architecture CPU benchmark complete')
    pd.DataFrame(records).to_csv(output/'E6_remote_CPU_latency.csv',index=False)
    # Named-concept perturbations for all three saved KAN seed models in fold 1.
    # These are model response diagnostics; correlated concepts are not causal factors.
    from dbpatchkan.analysis.stage13_persistence_forecast import load_transition_cache,forecast_matrix,FORECAST_CONCEPTS
    loaded=load_transition_cache(inputs/'evbattery_stage13_transitions.npz')
    seq=loaded['sequence'];con=forecast_matrix(loaded['table']).to_numpy('float32');table=loaded['table']
    from sklearn.model_selection import StratifiedGroupKFold
    _,test=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(con,table.manufacturer_id.astype(str),table.vehicle_key))
    test=test[:64];effects=[]
    for seed in (42,52,62):
        model=PersistenceAnchoredPatchConcept(head='kan');cp=inputs/f'fleet_checkpoints/fold_1/kan_seed_{seed}/model.pt'
        state=torch.load(cp,map_location='cpu',weights_only=False);model.load_state_dict(state['model_state']);model.eval();s=state['scaling']
        xx=np.nan_to_num((seq[test]-s['sequence_center'])/np.array(s['sequence_scale'])).astype('float32')
        cc=np.nan_to_num((con[test]-s['concept_center'])/np.array(s['concept_scale'])).astype('float32')
        x=torch.from_numpy(xx);c=torch.from_numpy(cc)
        with torch.no_grad():
            base=model(x,c[:,:14],c[:,14:])[0].numpy()*s['delta_scale']
            for j,name in enumerate(FORECAST_CONCEPTS):
                changed=c.clone();changed[:,j]=0
                pred=model(x,changed[:,:14],changed[:,14:])[0].numpy()*s['delta_scale']
                effects.append({'seed':seed,'concept':name,'mean_absolute_prediction_change_ah':np.abs(pred-base).mean(),
                    'signed_prediction_change_ah':(pred-base).mean(),'samples':len(test)})
    pd.DataFrame(effects).to_csv(output/'E6_concept_perturbation_seed_stability.csv',index=False)
    cpu_name=platform.processor()
    info=Path('/proc/cpuinfo')
    if info.exists():
        cpu_name=next((line.split(':',1)[1].strip() for line in info.read_text().splitlines() if line.startswith('model name')),cpu_name)
    save_json(output/'E6_SCOPE.json',{'cpu':cpu_name,'platform':platform.platform(),
        'real_saved_checkpoint_weights':True,'timing_inputs':'same zero-normalized synthetic input shapes for repeatability',
        'memory':'weights, buffers, checkpoint bytes and sampled process RSS; allocator/process memory is not isolated model activation memory',
        'energy_joules':'not measured; no CPU power meter available',
        'not_user_device_latency':True,'explanation':'training-median concept replacement on fold-1 held-out vehicles; diagnostics rather than causal attribution'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--task',choices=['prepare','train','benchmark'],required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,52,62]);p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--skip-eve',action='store_true');p.add_argument('--max-per-vehicle',type=int,default=1500)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.task=='prepare':prepare(args.input_dir,args.prepared_dir,max_per_vehicle=args.max_per_vehicle,include_eve=not args.skip_eve)
    elif args.task=='train':run_response(args.prepared_dir,args.output_dir,args.seeds,args.epochs)
    else:benchmark(args.input_dir,args.output_dir,args.prepared_dir,args.output_dir)
    save_json(args.output_dir/f'E4_{args.task.upper()}_COMPLETE.json',{'complete':True})


if __name__=='__main__':main()
