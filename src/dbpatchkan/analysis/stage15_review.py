"""Remote CPU review analyses. Execute in Colab, using immutable prior results.

E1 is development-only. E2/E3 inspect saved predictions. E5 fits switching
thresholds on the ORIGINAL outer-training inner validation vehicles only.
No locked-test tensor, training, model selection or replacement is supported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dbpatchkan.data.dbsi import DBSIConfig, compute_dbsi

COMPONENTS = tuple(f"bsi_{x}" for x in
                   ("rate", "thermal", "soc", "cycle", "transient", "interaction"))
WEIGHTS = np.array([.25, .25, .15, .15, .10, .10])
HORIZONS = ((0, 500, "<=500km"), (500, 1000, "500-1000km"),
            (1000, 2000, "1000-2000km"), (2000, np.inf, ">2000km"))


def update(message):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def paired_clusters(frame, candidate, reference, *, cluster="vehicle_key", draws=10000):
    """Paired vehicle-macro MAE bootstrap + cluster sign-flip randomization.

    Positive differences favour the reference. Resampling seed predictions
    or individual overlapping windows as independent units is prohibited.
    """
    keys = ["transition_id", cluster]
    if frame.duplicated(keys + ["model"]).any():
        raise ValueError("Duplicate model/transition rows: ensemble seeds before inference")
    def select(name):
        p = frame[frame.model.eq(name)].copy()
        p["ae"] = (p.prediction_capacity_ah - p.future_capacity_ah).abs()
        return p[keys + ["ae"]]
    paired = select(candidate).merge(select(reference), on=keys, suffixes=("_c", "_r"),
                                     validate="one_to_one")
    paired = paired.dropna()
    veh = paired.groupby(cluster)[["ae_c", "ae_r"]].mean()
    if veh.empty:
        return {"candidate": candidate, "reference": reference, "vehicles": 0}
    difference = (veh.ae_c - veh.ae_r).to_numpy(float)
    rng = np.random.default_rng(42)
    boot = np.mean(rng.choice(difference, (draws, len(difference)), replace=True), axis=1)
    totals=paired.groupby(cluster).agg(sum_c=('ae_c','sum'),sum_r=('ae_r','sum'),n=('ae_c','size'))
    indices=rng.integers(0,len(totals),size=(draws,len(totals)))
    row_boot=(totals.sum_c.to_numpy()[indices].sum(axis=1)-totals.sum_r.to_numpy()[indices].sum(axis=1))/totals.n.to_numpy()[indices].sum(axis=1)
    signs = rng.choice([-1., 1.], size=(draws, len(difference)))
    null = np.mean(signs * difference, axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(difference.mean()) - 1e-14)) / (draws + 1)
    return {"candidate": candidate, "reference": reference,
            "rows": len(paired), "vehicles": len(veh),
            "candidate_vehicle_macro_mae": veh.ae_c.mean(),
            "reference_vehicle_macro_mae": veh.ae_r.mean(),
            "delta_mae": difference.mean(), "ci_low": np.quantile(boot, .025),
            "ci_high": np.quantile(boot, .975), "signflip_p": p,
            "vehicle_win_fraction": np.mean(difference < 0),
            "improvement_pct": 100*(veh.ae_r.mean()-veh.ae_c.mean())/max(veh.ae_r.mean(), 1e-12),
            "row_weighted_candidate_mae":paired.ae_c.mean(),
            "row_weighted_reference_mae":paired.ae_r.mean(),
            "row_weighted_delta_mae":(paired.ae_c-paired.ae_r).mean(),
            "row_weighted_ci_low":np.quantile(row_boot,.025),
            "row_weighted_ci_high":np.quantile(row_boot,.975),
            "row_weighted_improvement_pct":100*(paired.ae_r.mean()-paired.ae_c.mean())/max(paired.ae_r.mean(),1e-12),
            "few_clusters": len(veh) < 10, "negative_favors_candidate": True}


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    result = np.empty(len(p))
    result[order] = np.minimum(1, np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return result


def regression_rows(frame):
    records = []
    for model, part in frame.groupby("model"):
        error = part.prediction_capacity_ah - part.future_capacity_ah
        finite = np.isfinite(error)
        p = part.loc[finite]
        error = error.loc[finite]
        coverage = np.nan
        if {"interval_low_ah", "interval_high_ah"} <= set(p):
            valid = p.interval_low_ah.notna() & p.interval_high_ah.notna()
            if valid.any():
                coverage = p.loc[valid].future_capacity_ah.between(
                    p.loc[valid].interval_low_ah, p.loc[valid].interval_high_ah).mean()
        vehicle_mae = error.abs().groupby(p.vehicle_key).mean()
        records.append({"model": model, "rows": len(p), "vehicles": len(vehicle_mae),
                        "mae_ah": error.abs().mean(), "rmse_ah": np.sqrt((error**2).mean()),
                        "bias_ah": error.mean(), "vehicle_macro_mae_ah": vehicle_mae.mean(),
                        "interval_90_coverage": coverage})
    return records


def e2(inputs, output, draws):
    frame = pd.read_csv(inputs / "fleet_predictions.csv")
    groups = [("all", "all", frame)]
    for lo, hi, name in HORIZONS:
        groups.append(("horizon", name, frame[frame.forecast_horizon_km.gt(lo) &
                                              frame.forecast_horizon_km.le(hi)]))
    for manufacturer, part in frame.groupby("manufacturer_id"):
        groups.append(("manufacturer", str(manufacturer), part))
        for lo, hi, name in HORIZONS:
            groups.append(("manufacturer_horizon", f"{manufacturer}:{name}",
                           part[part.forecast_horizon_km.gt(lo) & part.forecast_horizon_km.le(hi)]))
    metrics, comparisons = [], []
    for family, subgroup, part in groups:
        if part.empty:
            continue
        metrics.extend({"family": family, "subgroup": subgroup, **x}
                       for x in regression_rows(part))
        for model in sorted(set(part.model)-{"Persistence"}):
            comparisons.append({"family": family, "subgroup": subgroup,
                                **paired_clusters(part, model, "Persistence", draws=draws)})
        update(f"E2: {family}/{subgroup} complete ({part.vehicle_key.nunique()} vehicles)")
    comparison = pd.DataFrame(comparisons)
    comparison["holm_p_within_family"] = comparison.groupby("family").signflip_p.transform(
        lambda x: holm(x.to_numpy()))
    pd.DataFrame(metrics).to_csv(output/"E2_subgroup_metrics.csv", index=False)
    comparison.to_csv(output/"E2_cluster_comparisons.csv", index=False)
    save_json(output/"E2_SCOPE.json", {
        "evidence": "post-hoc subgroup analysis of frozen out-of-fold predictions",
        "cluster_unit": "vehicle", "draws": draws,
        "estimands": "vehicle-macro primary; row-weighted MAE separately retains cluster-resampled row counts to audit the reported long-horizon percentage",
        "intervals": "coverage of previously saved bounds; averaging bounds is not new conformal calibration",
        "cannot_tune_gate_from_these_results": True,
        "manufacturer_ids": "anonymous released groups, not identified vehicle brands"})


def nissan_lineage(inputs):
    """Reconstruct exact target-row selection from the original health builder."""
    trips = pd.read_csv(inputs/"nissan_trip_features.csv")
    labels = pd.read_csv(inputs/"nissan_trip_labels.csv")
    from dbpatchkan.data.features import build_health_windows
    # Use the same original function for past-only trend and odometer QC.
    windows = build_health_windows(trips, collapse_plateaus=False)
    timeline = windows.sort_values("anchor_time_utc").reset_index(drop=True)
    timeline["anchor_time_utc"] = pd.to_datetime(timeline.anchor_time_utc, utc=True)
    lookup = timeline.set_index("trip_key")
    records = []
    for row in labels.itertuples():
        if row.trip_key not in lookup.index:
            continue
        anchor = lookup.loc[row.trip_key]
        for kind, variable, lo, hi, horizon in (
            ("loss_30d", "days", 23, 37, 30),
            ("loss_500km", "km", 400, 600, 500)):
            future = timeline[timeline.anchor_time_utc.gt(anchor.anchor_time_utc)].copy()
            if variable == "days":
                delta = (future.anchor_time_utc-anchor.anchor_time_utc).dt.total_seconds()/86400
            else:
                delta = future.cumulative_distance_km-anchor.cumulative_distance_km
            candidates = future.loc[delta.between(lo, hi)]
            if candidates.empty:
                continue
            selected = candidates.loc[(delta.loc[candidates.index]-horizon).abs().idxmin()]
            expected = max(0., float(anchor.soh_target_trend_pct-selected.soh_target_trend_pct))
            original = getattr(row, "target_"+kind+"_pp", np.nan)
            # NaN is significant: original plateau anchors without targets stay excluded.
            if not np.isfinite(original):
                continue
            match = bool(np.isclose(original, expected, atol=2e-5))
            records.append({"trip_key": row.trip_key, "target": kind,
                            "anchor_key": row.trip_key, "future_key": selected.trip_key,
                            "anchor_time": anchor.anchor_time_utc, "future_time": selected.anchor_time_utc,
                            "anchor_trend": anchor.soh_target_trend_pct,
                            "future_trend": selected.soh_target_trend_pct,
                            "target_loss": original, "reconstructed_loss": expected,
                            "target_matches": match, "split_role": row.split_role,
                            "transition_key": f"{row.trip_key}->{selected.trip_key}"})
    return pd.DataFrame(records), timeline, labels


def e3(inputs, output, draws, raw_dir=None, raw_members=10000):
    lineage, timeline, labels = nissan_lineage(inputs)
    lineage.to_csv(output/"E3_nissan_target_lineage.csv", index=False)
    saved = pd.read_csv(inputs/"nissan_locked_predictions.csv")
    # Only saved seed ensembles and strict persistence; no inference or fitting.
    candidates = [x for x in set(saved.model) if x.endswith("_Ensemble")]
    identities, comparisons = [], []
    for target in ("loss_30d", "loss_500km"):
        base = saved[saved.target.eq(target) & saved.model.eq("Strict_Pretest_Persistence")]
        mapping = base.merge(lineage[lineage.target.eq(target)], on=["target", "trip_key"],
                             validate="one_to_one")
        mapping.to_csv(output/f"E3_locked_{target}_lineage.csv", index=False)
        identities.append({"target": target, "saved_forecasts": len(base),
                           "mapped_forecasts": len(mapping),
                           "distinct_transition_pairs": mapping.transition_key.nunique(),
                           "distinct_future_checkpoints": mapping.future_key.nunique(),
                           "label_mapping_verified": bool(len(mapping)==len(base) and mapping.target_matches.all()),
                           "unique_observed_losses": base.observed.nunique(),
                           "zero_loss_count": int(base.observed.abs().le(1e-6).sum())})
        if len(mapping) != len(base) or not mapping.target_matches.all():
            update(f"E3 {target}: unresolved target lineage; interval inference withheld")
            continue
        for model in candidates:
            p = saved[saved.target.eq(target) & saved.model.eq(model)]
            paired = p.merge(base[["trip_key", "predicted"]], on="trip_key", suffixes=("", "_p"))
            paired = paired.merge(mapping[["trip_key", "transition_key", "future_key"]], on="trip_key")
            for cluster in ("transition_key", "future_key"):
                # Conservative future-checkpoint clustering captures shared destination labels.
                f = paired.rename(columns={"trip_key": "transition_id", "observed": "future_capacity_ah",
                                           "predicted": "prediction_capacity_ah", cluster: "vehicle_key"})
                b = f.copy(); b["model"]="Persistence"; b["prediction_capacity_ah"]=b.predicted_p
                comparison = paired_clusters(pd.concat([f,b]), model, "Persistence", draws=draws)
                comparisons.append({"target": target, "cluster": cluster, "unit":"SOH_percentage_points", **comparison})
    pd.DataFrame(identities).to_csv(output/"E3_nissan_independent_counts.csv", index=False)
    pd.DataFrame(comparisons).to_csv(output/"E3_checkpoint_cluster_bootstrap.csv", index=False)
    raw_soh = pd.to_numeric(timeline.soh_target_raw_pct)
    unique = np.sort(raw_soh.dropna().unique())
    save_json(output/"E3_nissan_label_summary.json", {
        "trips": len(timeline), "distinct_raw_soh_values": len(unique),
        "minimum_positive_observed_step_pp": float(np.min(np.diff(unique))) if len(unique)>1 else None,
        "raw_increases": int(raw_soh.diff().gt(1e-5).sum()),
        "raw_plateaus": int(raw_soh.diff().abs().le(1e-5).sum()),
        "trend_plateaus": int(timeline.soh_target_trend_pct.diff().abs().le(1e-5).sum()),
        "noise_interpretation": "BMS changes/increases are recorded; recalibration cannot be identified without independent capacity diagnostics",
        "single_vehicle": True, "checkpoint_clusters_are_not_independent_vehicles": True,
        "unmapped_targets": int((~lineage.target_matches).sum())})
    # A cell noise sweep is conditional sensitivity, not a measured noise distribution.
    fleet = pd.read_csv(inputs/"fleet_predictions.csv")
    proposed = fleet[fleet.model.eq("PersistenceAnchored_KAN")]
    base = fleet[fleet.model.eq("Persistence")]
    paired = proposed.merge(base[["transition_id","prediction_capacity_ah"]], on="transition_id",
                            suffixes=("","_p"), validate="one_to_one")
    rows=[]; rng=np.random.default_rng(42)
    for sd in (0., .005, .01, .05, .1, .25, .5):
        delta=[]
        for _ in range(200):
            # Share one perturbation per destination capacity checkpoint across models.
            y=paired.future_capacity_ah.to_numpy()+rng.normal(0,sd,len(paired))
            diff=np.abs(paired.prediction_capacity_ah-y)-np.abs(paired.prediction_capacity_ah_p-y)
            delta.append(pd.Series(diff).groupby(paired.vehicle_key.to_numpy()).mean().mean())
        rows.append({"assumed_target_noise_sd_ah":sd, "delta_mae_mean_ah":np.mean(delta),
                     "perturbation_ci_low":np.quantile(delta,.025),
                     "perturbation_ci_high":np.quantile(delta,.975),
                     "assumption_only":True})
    pd.DataFrame(rows).to_csv(output/"E3_assumed_label_noise_sensitivity.csv",index=False)
    raw_status="unavailable: original snippet archives were not found in remote Drive"
    if raw_dir:
        from dbpatchkan.data.evbattery import safe_load_evbattery_payload, _manufacturer_id
        import tarfile
        files=[p for p in Path(raw_dir).rglob('battery_dataset*.tar.gz') if p.is_file()]
        records=[]
        for path in sorted(files):
            update(f"E3 bounded original-label audit: {path.name}, maximum {raw_members} decoded snippets")
            # One bounded streaming pass; do not enumerate all archive members
            # or unpack the millions of legacy snippet files.
            manufacturer=_manufacturer_id(path);decoded=0
            with tarfile.open(path,'r|gz') as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith('.pkl'):continue
                    handle=archive.extractfile(member)
                    if handle is None:continue
                    _,meta=safe_load_evbattery_payload(handle.read());decoded+=1
                    capacity=float(meta.get('capacity',0) or 0)
                    if capacity>0:
                        records.append({'manufacturer_id':manufacturer,'vehicle_id':int(meta['car']),
                            'mileage':float(meta.get('mileage',np.nan)),
                            'charge_segment':str(meta.get('charge_segment','')),'capacity_ah':capacity})
                    if decoded==1 or decoded%2000==0:update(f'E3 {path.name}: decoded {decoded:,}')
                    if decoded>=raw_members:break
        if records:
            meta=pd.DataFrame(records)
            meta.to_csv(output/"E3_original_label_sample.csv",index=False)
            for grouping,name in ((["manufacturer_id","vehicle_id","mileage"],"checkpoint"),
                                  (["manufacturer_id","vehicle_id","charge_segment"],"charge_segment")):
                selected=meta if name=='checkpoint' else meta[meta.charge_segment.ne('')]
                dispersion=selected.groupby(grouping).capacity_ah.agg(['size','nunique','median','std','min','max'])
                dispersion.to_csv(output/f"E3_label_dispersion_{name}.csv")
            raw_status="bounded, archive-order sample; dispersion is descriptive, not independent measurement-error calibration"
        elif files:
            raw_status="unavailable: bounded original-archive sample contained no nonzero capacity labels; increase the sample in a separately documented audit"
    save_json(output/"E3_SCOPE.json", {"original_label_status":raw_status,
        "noise_sweep":"hypothetical zero-mean target perturbations; not measured manufacturer uncertainty",
        "operational_significance":"compare effect magnitude with repeated-label dispersion and application tolerance; neither establishes ground-truth measurement precision"})
    # Check actual exported target presence and logged losses, not just loss code.
    loss_rows=[]
    for path in (inputs/"training_evidence").rglob("history.json"):
        history=json.loads(path.read_text())
        def visit(obj,prefix=''):
            if isinstance(obj,dict):
                for key,value in obj.items():
                    yield from visit(value, prefix+'/'+key)
            elif isinstance(obj,list):
                for i,value in enumerate(obj): yield from visit(value,prefix+'/'+str(i))
            elif isinstance(obj,(int,float)) and ("physics" in prefix.lower() or "driver_id" in prefix.lower()):
                loss_rows.append({"file":str(path.relative_to(inputs)),"metric_path":prefix,"value":obj})
        visit_result=visit(history)
        list(visit_result)
    pd.DataFrame(loss_rows).to_csv(output/"E3_actual_logged_loss_terms.csv",index=False)
    objectives=[]
    loss_frame=pd.DataFrame(loss_rows)
    for config in (inputs/'training_evidence').rglob('run_config.json'):
        values=json.loads(config.read_text());training=values.get('training',{})
        history_path=str(config.with_name('history.json').relative_to(inputs))
        logged=loss_frame[loss_frame.file.eq(history_path)] if not loss_frame.empty else pd.DataFrame()
        def term(name):
            if logged.empty:return []
            return logged[logged.metric_path.str.contains(name,regex=False)].value.to_numpy().tolist()
        driver_loss=term('train_driver_id');physics_loss=term('train_physics')
        weight=training.get('driver_id_weight')
        objectives.append({'config':str(config.relative_to(inputs)),'driver_id_weight':weight,
            'physics_weight':training.get('physics_weight'),
            'logged_driver_loss_max':max(driver_loss) if driver_loss else np.nan,
            'logged_physics_loss_max':max(physics_loss) if physics_loss else np.nan,
            'driver_auxiliary_active_verified':bool(weight is not None and weight>0 and driver_loss and max(driver_loss)>0),
            'physics_inactive_in_logged_epochs':bool(physics_loss and max(physics_loss)==0)})
    pd.DataFrame(objectives).to_csv(output/'E3_training_objective_audit.csv',index=False)
    eve_path=inputs/'eve_saved_health_predictions.csv'
    if eve_path.exists():
        external=pd.read_csv(eve_path);eve=external[external.dataset.eq('EVE')].copy()
        intervals=pd.read_csv(inputs/'eve_original_health_intervals.csv')
        interval_meta=intervals[['checkpoint_id','previous_checkpoint_id','previous_soh_trend_pct','bms_soh_trend_pct']].copy()
        eve['row_id']=eve.row_id.astype(str)
        interval_meta['checkpoint_id']=interval_meta.checkpoint_id.astype(str)
        lineage_eve=eve.merge(interval_meta,left_on='row_id',right_on='checkpoint_id',how='left',validate='many_to_one')
        lineage_eve.to_csv(output/'E3_EVE_health_checkpoint_lineage.csv',index=False)
        counts=[]
        for name,p in lineage_eve.groupby('model'):
            counts.append({'model':name,'source_total_intervals':len(intervals),'heldout_interval_ids':p.row_id.nunique(),
                'finite_observed_loss_rows':int(np.isfinite(p.true_loss_pp).sum()),
                'finite_future_soh_rows':int(np.isfinite(p.true_future_soh_pct).sum()),
                'finite_anchor_and_future_rows':int((np.isfinite(p.anchor_soh_pct)&np.isfinite(p.true_future_soh_pct)).sum()),
                'unique_future_checkpoint_ids':p.checkpoint_id.nunique()})
        pd.DataFrame(counts).to_csv(output/'E3_EVE_target_eligibility_counts.csv',index=False)
    with np.load(inputs/"development_101_trips.npz",allow_pickle=False) as data:
        driver=data["driver_target"] if "driver_target" in data.files else np.array([])
        nonzero_driver=[x for x in loss_rows if 'driver_id' in x['metric_path'].lower() and x['value']>0]
        nonzero_physics=[x for x in loss_rows if 'physics' in x['metric_path'].lower() and x['value']>0]
        save_json(output/"E3_auxiliary_driver_audit.json", {
            "exported_driver_target":bool(len(driver)),"valid_driver_targets":int(np.sum(driver!=-100)),
            "distinct_valid_codes":np.unique(driver[driver!=-100]).tolist(),
            "nonzero_logged_driver_loss_entries":len(nonzero_driver),
            "nonzero_logged_physics_loss_entries":len(nonzero_physics),
            "interpretation":"target availability and nonzero logged loss together establish activity; absence of a log is unresolved, not proof of inactivity"})
    update("E3 saved-label and actual-loss audits complete")


def sensitivity_configs():
    cfg=DBSIConfig(); result={"reference":cfg}
    names=("discharge_c_lower","discharge_c_upper","regen_c_lower","regen_c_upper",
           "slew_lower_c_per_s","slew_upper_c_per_s","efc_rate_lower_per_h","efc_rate_upper_per_h")
    for factor in (.8,1.2):
        result[f"rate_slew_cycle_x{factor}"]=replace(cfg,**{x:getattr(cfg,x)*factor for x in names})
    for offset in (-5,5):
        result[f"temperature_offset_{offset:+d}"]=replace(cfg,
            hot_temp_lower_c=cfg.hot_temp_lower_c+offset,hot_temp_upper_c=cfg.hot_temp_upper_c+offset,
            cold_regen_start_c=cfg.cold_regen_start_c+offset,cold_regen_full_c=cfg.cold_regen_full_c+offset)
    result["capacity_350V_assumption"]=replace(cfg,nominal_capacity_ah=40000/350)
    for offset in (-5,5):
        result[f"SOC_activation_offset_{offset:+d}"]=replace(cfg,
            high_soc_lower_pct=cfg.high_soc_lower_pct+offset,
            low_soc_upper_pct=cfg.low_soc_upper_pct+offset)
    for factor in (.8,1.2):
        result[f"DoD_spread_x{factor}"]=replace(cfg,
            dod_lower_pct=cfg.dod_lower_pct*factor,dod_upper_pct=cfg.dod_upper_pct*factor,
            temp_spread_lower_c=cfg.temp_spread_lower_c*factor,temp_spread_upper_c=cfg.temp_spread_upper_c*factor)
    return result


def e1(inputs, output):
    import pyarrow.parquet as pq
    lineage,_,labels=nissan_lineage(inputs)
    development=labels[labels.split_role.isin(["train","validation"])].copy()
    keys=set(development.trip_key)
    telemetry_path=inputs/"nissan_telemetry_1hz.parquet"
    schema=set(pq.read_schema(telemetry_path).names)
    wanted={"trip_key","dt_s","dt_1hz_s","current_std_a","HV_Bat_Current_2",
            "pack_temp_avg_clean_c","pack_temp_spread_c","SOC","soc_pct","c_rate_slew",
            "ah_throughput_step","timestamp_utc","datetime_utc","segment_id","Timestamp"}
    telemetry=pq.read_table(telemetry_path,columns=sorted(schema&wanted),
                            filters=[("trip_key","in",sorted(keys))]).to_pandas()
    # Recompute c-rate and EFC for every changed capacity. Do not use cached c_dis/c_regen/efc.
    cfgs=sensitivity_configs(); records=[]
    for number,(name,cfg) in enumerate(cfgs.items(),1):
        for key,group in telemetry.groupby("trip_key",sort=False):
            group=group.copy()
            if 'datetime_utc' in group:group=group.sort_values('datetime_utc',kind='stable')
            dt=group['dt_1hz_s'] if 'dt_1hz_s' in group else group['dt_s']
            current=group['current_std_a'] if 'current_std_a' in group else -group['HV_Bat_Current_2']
            group['ah_throughput_step']=current.abs()*dt/3600
            difference=current.groupby(group.segment_id).diff() if 'segment_id' in group else current.diff()
            group['c_rate_slew']=difference.abs()/cfg.nominal_capacity_ah/dt.replace(0,np.nan)
            records.append({"variant":name,"trip_key":key,**compute_dbsi(group,cfg)})
        update(f"E1 ramps {number}/{len(cfgs)}: {name}")
    scores=pd.DataFrame(records)
    ref=scores[scores.variant.eq("reference")].set_index("trip_key")
    variants={name:part.set_index("trip_key")["bsi_intensity"] for name,part in scores.groupby("variant")}
    variants["equal_weights"]=100*ref[list(COMPONENTS)].mean(axis=1)
    for j,col in enumerate(COMPONENTS):
        variants[f"single_{col}"]=100*ref[col]
        w=WEIGHTS.copy(); w[j]=0; w/=w.sum()
        variants[f"without_{col}"]=100*ref[list(COMPONENTS)].dot(w)
    rng=np.random.default_rng(42)
    ranking=[]
    for run in range(100):
        w=WEIGHTS*rng.uniform(.8,1.2,6);w/=w.sum()
        name=f"perturbed_weight_{run:03d}"
        variants[name]=100*ref[list(COMPONENTS)].dot(w)
        ranking.append({"variant":name,**dict(zip(COMPONENTS,w))})
    save_json(output/"E1_FIXED_CONFIGS.json", {k:asdict(v) for k,v in cfgs.items()})
    pd.DataFrame(ranking).to_csv(output/"E1_weight_draws.csv",index=False)
    summaries=[]
    for name,series in variants.items():
        values=series.reindex(ref.index)
        corr=spearmanr(values,ref.bsi_intensity).statistic if values.nunique()>1 else np.nan
        top=max(1,int(np.ceil(.2*len(ref))))
        overlap=len(set(values.nlargest(top).index)&set(ref.bsi_intensity.nlargest(top).index))/top
        summaries.append({"variant":name,"trips":len(values),"spearman_vs_reference":corr,
            "top20pct_overlap":overlap,"mean":values.mean(),"std":values.std(),
            "min":values.min(),"max":values.max(),"fixed_low_fraction":values.lt(25).mean()})
    pd.DataFrame(summaries).to_csv(output/"E1_rank_distribution_robustness.csv",index=False)
    scores.to_csv(output/"E1_recomputed_components.csv",index=False)
    pd.DataFrame(variants).rename_axis("trip_key").to_csv(output/"E1_variant_scores.csv")
    published=development.set_index('trip_key')['bsi_intensity'].reindex(ref.index)
    pd.DataFrame({'native_export_reference':published,'recomputed_1hz_reference':ref.bsi_intensity,
                  'difference':ref.bsi_intensity-published}).to_csv(output/'E1_native_vs_1hz_reference.csv')
    # Limit downstream fits to planned variants; 100 random weights are ranking-only.
    joined=development.set_index("trip_key").join(ref[list(COMPONENTS)],rsuffix="_recomputed")
    development_keys=joined.index
    base_names=["calendar_age_days","odometer_clean_km","last_known_soh"]
    predictions=[]; learned=[];downstream_status=[]
    for target in ("loss_30d","loss_500km"):
        valid=lineage[lineage.target.eq(target)&lineage.trip_key.isin(keys)&lineage.future_key.isin(keys)&lineage.target_matches].copy()
        valid=valid.sort_values("anchor_time").reset_index(drop=True)
        if len(valid)<12:
            downstream_status.append({'target':target,'available_development_transitions':len(valid),
                'status':'insufficient labels for four temporal folds'})
            update(f"E1 {target}: only {len(valid)} development-only transitions; descriptive scores retained")
            continue
        bounds=np.linspace(max(6,len(valid)//2),len(valid),5,dtype=int)
        for fold,(start,end) in enumerate(zip(bounds[:-1],bounds[1:]),1):
            test=valid.iloc[start:end]; cutoff=pd.Timestamp(test.anchor_time.iloc[0])
            train=valid.iloc[:start]; train=train[pd.to_datetime(train.future_time,utc=True).lt(cutoff)]
            if len(train)<8 or test.empty:
                downstream_status.append({'target':target,'fold':fold,'training_transitions':len(train),
                    'evaluation_transitions':len(test),'status':'insufficient training labels after future-boundary purge'})
                continue
            downstream_status.append({'target':target,'fold':fold,'training_transitions':len(train),
                'evaluation_transitions':len(test),'status':'fit completed'})
            tr=joined.loc[train.trip_key];te=joined.loc[test.trip_key]
            model=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=1.))
            model.fit(tr[base_names],train.target_loss)
            import joblib
            model_dir=output/'E1_models'/target/f'fold_{fold}'
            model_dir.mkdir(parents=True,exist_ok=True)
            joblib.dump(model,model_dir/'aging_lagged_soh_ridge.joblib')
            base_tr=model.predict(tr[base_names]);base_te=model.predict(te[base_names])
            variant_weights={}
            variants_to_fit={k:v for k,v in variants.items() if not k.startswith("perturbed_weight")}
            for name,series in variants_to_fit.items():
                xtr=np.asarray(series.reindex(train.trip_key)).reshape(-1,1)
                xte=np.asarray(series.reindex(test.trip_key)).reshape(-1,1)
                # Additional stress coefficient is fit on outer training only.
                weight=nnls(np.nan_to_num(xtr),train.target_loss.to_numpy()-base_tr)[0]
                variant_weights[name]=weight.tolist()
                pred=base_te+np.nan_to_num(xte)@weight
                for i,row in enumerate(test.itertuples()):
                    predictions.append({"target":target,"fold":fold,"variant":name,"trip_key":row.trip_key,
                        "future_key":row.future_key,"observed":row.target_loss,"predicted":pred[i],
                        "base_prediction":base_te[i],"persistence_prediction":0.})
            xtr=ref.loc[train.trip_key,list(COMPONENTS)].to_numpy()
            xte=ref.loc[test.trip_key,list(COMPONENTS)].to_numpy()
            if len(train)<24 or train.future_key.nunique()<12:
                save_json(model_dir/'stress_weights_and_split.json',{'weights':variant_weights,'base_features':base_names,
                    'train_trip_keys':train.trip_key.tolist(),'test_trip_keys':test.trip_key.tolist(),
                    'joint_NNLS_status':'insufficient supervision: requires at least 24 training transitions and 12 future checkpoints'})
                downstream_status.append({'target':target,'fold':fold,'status':'six-component NNLS omitted: insufficient independent supervision'})
                update(f'E1 {target} fold {fold}: fixed-index sensitivity fits complete; joint weighting lacks independent supervision')
                continue
            w=nnls(xtr,train.target_loss.to_numpy()-base_tr)[0]
            variant_weights['training_only_NNLS_components']=w.tolist()
            save_json(model_dir/'stress_weights_and_split.json',{'weights':variant_weights,'base_features':base_names,
                'component_names':COMPONENTS,'train_trip_keys':train.trip_key.tolist(),'test_trip_keys':test.trip_key.tolist()})
            for col,value in zip(COMPONENTS,w): learned.append({"target":target,"fold":fold,"component":col,
                "coefficient_pp_per_unit":value,"normalized_weight":value/w.sum() if w.sum()>0 else np.nan})
            for i,row in enumerate(test.itertuples()):
                predictions.append({"target":target,"fold":fold,"variant":"training_only_NNLS_components",
                    "trip_key":row.trip_key,"future_key":row.future_key,"observed":row.target_loss,
                    "predicted":base_te[i]+xte[i]@w,"base_prediction":base_te[i],"persistence_prediction":0.})
            update(f"E1 downstream {target} fold {fold}/4: train={len(train)}, evaluation={len(test)}, future-boundary purged")
    pred=pd.DataFrame(predictions);pred.to_csv(output/"E1_downstream_predictions.csv",index=False)
    pd.DataFrame(downstream_status).to_csv(output/'E1_downstream_eligibility.csv',index=False)
    pd.DataFrame(learned).to_csv(output/"E1_training_only_nonnegative_weights.csv",index=False)
    if not pred.empty:
        pred["ae"]=(pred.observed-pred.predicted).abs()
        summary=pred.groupby(["target","variant"]).agg(n=("ae","size"),mae_pp=("ae","mean"))
        summary.to_csv(output/"E1_downstream_metrics.csv")
        paired=[]
        for (target,variant),part in pred.groupby(['target','variant']):
            for comparator in ('base_prediction','persistence_prediction'):
                diff=(part.observed-part.predicted).abs()-(part.observed-part[comparator]).abs()
                delta=diff.groupby(part.future_key).mean().to_numpy()
                boot=np.mean(rng.choice(delta,(2000,len(delta)),replace=True),axis=1)
                paired.append({'target':target,'variant':variant,'reference':comparator,'checkpoint_clusters':len(delta),
                    'delta_mae_pp':delta.mean(),'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),
                    'exploratory_single_vehicle':True,'few_checkpoint_clusters':len(delta)<10})
        pd.DataFrame(paired).to_csv(output/'E1_downstream_checkpoint_comparisons.csv',index=False)
    save_json(output/"E1_SCOPE.json", {"development_trips":len(development_keys),
        "heldout_training_used":False,"future_targets_restricted_to_development":True,
        "variant_weights_are_engineering_sensitivity":True,
        "learned_weights":"NNLS of outer-training residual after age/mileage/lagged-SOH Ridge; not causal shares of degradation",
        "charging_history_available":False,"calendar_age_is_days_since_logging_not_manufacture":True})


def e5(inputs,output,draws):
    """Reuse each original seed checkpoint and its original inner vehicle split."""
    import torch
    from torch.utils.data import DataLoader
    from dbpatchkan.analysis.stage13_persistence_forecast import (
        load_transition_cache,forecast_matrix,scale_fold,TransitionDataset,
        PersistenceAnchoredPatchConcept,predict,Scaling,FORECAST_PHYSICAL)
    loaded=load_transition_cache(inputs/"evbattery_stage13_transitions.npz")
    seq=loaded['sequence'];table=loaded['table'];groups=table.vehicle_key.to_numpy()
    con=forecast_matrix(table).to_numpy('float32');delta=table.capacity_delta_ah.to_numpy('float32')
    current=table.current_capacity_ah.to_numpy('float32')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splitter=StratifiedGroupKFold(5,shuffle=True,random_state=42)
    records=[];thresholds=[]
    for fold,(outer_train,outer_test) in enumerate(splitter.split(con,table.manufacturer_id.astype(str),groups),1):
        for seed in (42,52,62):
            cp=inputs/"fleet_checkpoints"/f"fold_{fold}"/f"kan_seed_{seed}"/"model.pt"
            state=torch.load(cp,map_location=device,weights_only=False)
            model=PersistenceAnchoredPatchConcept(head='kan').to(device)
            model.load_state_dict(state['model_state']);scaling=Scaling(**state['scaling'])
            # Reproduce the exact training split instead of reusing outer OOF labels.
            a,b=next(GroupShuffleSplit(1,test_size=.15,random_state=seed).split(outer_train,groups=groups[outer_train]))
            fit,validation=outer_train[a],outer_train[b]
            seq_scaled,con_scaled,y,check=scale_fold(seq,con,delta,fit)
            if not np.allclose(check.sequence_center,scaling.sequence_center,atol=1e-5):
                raise ValueError(f'Checkpoint normalization does not match original inner split: {cp}')
            for field in ('sequence_scale','concept_center','concept_scale','delta_center','delta_scale'):
                if not np.allclose(getattr(check,field),getattr(scaling,field),atol=1e-5):
                    raise ValueError(f'Checkpoint {field} differs from original inner split: {cp}')
            pc=len(FORECAST_PHYSICAL)
            def infer(indices):
                loader=DataLoader(TransitionDataset(seq_scaled,con_scaled[:,:pc],con_scaled[:,pc:],y,current,indices),
                                  batch_size=512,shuffle=False,num_workers=0)
                r=predict(model,loader,device,scaling)
                return r
            vr=infer(validation);idx=np.array([x[0] for x in vr],int);pv=np.array([x[2] for x in vr])
            threshold_choices=[0,500,1000,2000,np.inf];losses=[]
            for threshold in threshold_choices:
                pred=np.where(table.iloc[idx].forecast_horizon_km.to_numpy()>threshold,pv,current[idx])
                ae=pd.Series(np.abs(pred-table.iloc[idx].future_capacity_ah.to_numpy()))
                losses.append(ae.groupby(groups[idx]).mean().mean())
            # Tie breaks in favour of the simpler persistence rule.
            best=min(range(len(losses)),key=lambda i:(round(losses[i],10),-threshold_choices[i]))
            threshold=threshold_choices[best]
            thresholds.append({'fold':fold,'seed':seed,'threshold_km':threshold,
                'inner_validation_vehicles':len(set(groups[idx])),'inner_macro_mae':losses[best],
                'all_threshold_scores':json.dumps(dict(zip(map(str,threshold_choices),losses)))})
            r=infer(outer_test);idx=np.array([x[0] for x in r],int);p=np.array([x[2] for x in r])
            active=table.iloc[idx].forecast_horizon_km.to_numpy()>threshold
            for i,index in enumerate(idx):
                row=table.iloc[index]
                records.append({'transition_id':row.transition_id,'fold':fold,'seed':seed,'vehicle_key':row.vehicle_key,
                    'manufacturer_id':row.manufacturer_id,'forecast_horizon_km':row.forecast_horizon_km,
                    'future_capacity_ah':row.future_capacity_ah,'current_capacity_ah':row.current_capacity_ah,
                    'model':'Selective_Persistence_KAN','prediction_capacity_ah':p[i] if active[i] else current[index],
                    'ai_active':bool(active[i])})
            update(f'E5 fold {fold}/5 seed={seed}: inner-selected threshold={threshold} km')
    raw=pd.DataFrame(records);raw.to_csv(output/'E5_seed_switch_predictions.csv',index=False)
    pd.DataFrame(thresholds).to_csv(output/'E5_inner_selected_thresholds.csv',index=False)
    keys=['transition_id','fold','vehicle_key','manufacturer_id','forecast_horizon_km','future_capacity_ah','current_capacity_ah','model']
    ens=raw.groupby(keys,as_index=False).agg(prediction_capacity_ah=('prediction_capacity_ah','mean'),ai_fraction=('ai_active','mean'))
    prior=pd.read_csv(inputs/'fleet_predictions.csv')
    combined=pd.concat([ens,prior],ignore_index=True)
    ens.to_csv(output/'E5_ensemble_switch_predictions.csv',index=False)
    utilization=ens.copy()
    utilization['horizon_band']=pd.cut(utilization.forecast_horizon_km,[0,500,1000,2000,np.inf],
        labels=['<=500km','500-1000km','1000-2000km','>2000km'])
    utilization.groupby(['manufacturer_id','horizon_band'],observed=True).agg(
        transitions=('transition_id','size'),mean_AI_seed_fraction=('ai_fraction','mean')).to_csv(output/'E5_AI_utilization_by_subgroup.csv')
    pd.DataFrame(regression_rows(combined)).to_csv(output/'E5_comparison_metrics.csv',index=False)
    pd.DataFrame([paired_clusters(combined,'Selective_Persistence_KAN',ref,draws=draws)
                  for ref in ('Persistence','PersistenceAnchored_KAN','PersistenceAnchored_MLP','PersistenceAnchored_TSLANet_2024')
                  if ref in set(combined.model)]).to_csv(output/'E5_cluster_comparisons.csv',index=False)
    save_json(output/'E5_SCOPE.json',{'threshold_training':'original per-seed inner validation vehicles in each outer fold',
        'threshold_choices_km':[0,500,1000,2000,'infinity'],'outer_labels_used_for_threshold':False,
        'new_confirmatory_dataset':False,'evidence':'post-hoc operational experiment on already studied outer cohorts'})


def figures(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=output/'figures';path.mkdir(exist_ok=True)
    p=output/'E2_cluster_comparisons.csv'
    if p.exists():
        data=pd.read_csv(p);data=data[data.family.eq('horizon')&data.candidate.eq('PersistenceAnchored_KAN')]
        if not data.empty:
            fig,ax=plt.subplots(figsize=(7,4))
            ax.errorbar(data.delta_mae,np.arange(len(data)),xerr=[data.delta_mae-data.ci_low,data.ci_high-data.delta_mae],fmt='o',color='black')
            ax.axvline(0,color='gray',linestyle='--');ax.set_yticks(np.arange(len(data)),data.subgroup)
            ax.set_xlabel('Vehicle-macro MAE difference vs persistence (Ah)\nNegative favours learned forecasting')
            fig.tight_layout();fig.savefig(path/'E2_horizon_cluster_CI.png',dpi=300);plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--tasks',nargs='+',choices=['e1','e2','e3','e5','figures'],required=True)
    parser.add_argument('--bootstrap-draws',type=int,default=10000)
    parser.add_argument('--raw-label-dir',type=Path)
    parser.add_argument('--raw-label-members',type=int,default=10000)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    started=time.time()
    for task in args.tasks:
        update(f'START {task.upper()} on remote runtime')
        if task=='e1':e1(args.input_dir,args.output_dir)
        elif task=='e2':e2(args.input_dir,args.output_dir,args.bootstrap_draws)
        elif task=='e3':e3(args.input_dir,args.output_dir,args.bootstrap_draws,args.raw_label_dir,args.raw_label_members)
        elif task=='e5':e5(args.input_dir,args.output_dir,args.bootstrap_draws)
        else:figures(args.output_dir)
        save_json(args.output_dir/f'{task.upper()}_COMPLETE.json',{'complete':True,'elapsed_minutes':(time.time()-started)/60})
        update(f'DONE {task.upper()}')
    save_json(args.output_dir/'REMOTE_CPU_ENVIRONMENT.json',{'platform':platform.platform(),'python':platform.python_version()})


if __name__=='__main__':main()
