"""Stage 4N-B ensemble, uncertainty, baselines, and driver-response evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .nissan_hybrid import DRIVER_FEATURES, finite_sample_quantile, run_baselines


def regression_metrics(true: pd.Series, prediction: pd.Series) -> dict[str, float | int]:
    valid = true.notna() & prediction.notna()
    y = true[valid].to_numpy(float); p = prediction[valid].to_numpy(float)
    error = p - y
    return {
        "n": int(len(y)), "mae": float(np.mean(np.abs(error))) if len(y) else math.nan,
        "rmse": float(np.sqrt(np.mean(error**2))) if len(y) else math.nan,
        "bias": float(np.mean(error)) if len(y) else math.nan,
        "r2": float(r2_score(y,p)) if len(y) >= 2 else math.nan,
    }


def ensemble(run_root: Path, output: Path, coverage: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for path in sorted(run_root.glob("seed_*/predictions.csv")):
        frame = pd.read_csv(path); frame["seed"] = int(path.parent.name.split("_")[-1]); frames.append(frame)
    if len(frames) != 5: raise ValueError(f"Expected five seed outputs, found {len(frames)}")
    all_predictions = pd.concat(frames, ignore_index=True)
    all_predictions.to_csv(output / "all_seed_predictions.csv", index=False)
    keys = ["split","model_mode","trip_key","Trip"]
    pred_cols = [c for c in all_predictions if c.endswith("_pred") or c.startswith("stress_probability_") or c == "current_correction_pp"]
    truth_cols = [c for c in all_predictions if c.endswith("_true")]
    mean = all_predictions.groupby(keys)[pred_cols].mean().add_suffix("_mean").reset_index()
    std = all_predictions.groupby(keys)[pred_cols].std().add_suffix("_std").reset_index()
    truth = all_predictions.groupby(keys)[truth_cols].first().reset_index()
    result = truth.merge(mean,on=keys).merge(std,on=keys)
    cal_rows=[]
    for target in ("current_soh","loss_30d","loss_500km"):
        cal=result[(result.split=="calibration")&(result.model_mode=="corrected_hybrid")]
        y=pd.to_numeric(cal[f"{target}_true"],errors="coerce"); p=pd.to_numeric(cal[f"{target}_pred_mean"],errors="coerce")
        valid=y.notna()&p.notna(); q=finite_sample_quantile(np.abs(y[valid]-p[valid]),coverage)
        cal_rows.append({"target":target,"calibration_n":int(valid.sum()),"coverage_requested":coverage,"absolute_residual_quantile":q})
        result[f"{target}_interval_low"]=(result[f"{target}_pred_mean"]-q).clip(lower=0)
        result[f"{target}_interval_high"]=result[f"{target}_pred_mean"]+q
        if target=="current_soh": result[f"{target}_interval_high"]=result[f"{target}_interval_high"].clip(upper=100)
    calibration=pd.DataFrame(cal_rows); calibration.to_csv(output/"conformal_calibration.csv",index=False)
    result.to_csv(output/"ensemble_predictions_with_intervals.csv",index=False)
    return result,calibration


def evaluate(result: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows=[]
    for mode in ("corrected_hybrid","analytic_only"):
        val=result[(result.split=="validation")&(result.model_mode==mode)]
        for target in ("current_soh","loss_30d","loss_500km"):
            stats=regression_metrics(pd.to_numeric(val[f"{target}_true"],errors="coerce"),pd.to_numeric(val[f"{target}_pred_mean"],errors="coerce"))
            y=pd.to_numeric(val[f"{target}_true"],errors="coerce"); lo=val[f"{target}_interval_low"]; hi=val[f"{target}_interval_high"]
            valid=y.notna()&lo.notna()&hi.notna()
            stats["interval_coverage"]=float(((y[valid]>=lo[valid])&(y[valid]<=hi[valid])).mean()) if valid.any() else math.nan
            stats["mean_interval_width"]=float((hi[valid]-lo[valid]).mean()) if valid.any() else math.nan
            rows.append({"task":target,"model":mode,**stats})
        probs=val[[f"stress_probability_{x}_mean" for x in ("low","moderate","high")]].to_numpy()
        true=val.stress_class_true.astype(int); pred=probs.argmax(1)
        rows.append({"task":"stress_class","model":mode,"n":len(val),"macro_f1":f1_score(true,pred,average="macro"),"balanced_accuracy":balanced_accuracy_score(true,pred)})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/"corrected_ensemble_metrics.csv",index=False);return metrics


def driver_response_associations(table: pd.DataFrame, output: Path) -> pd.DataFrame:
    fit=table[table.split.eq("train")].sort_values("trip_end_utc").iloc[:65].copy()
    response=["c_rate_abs_p95","c_rate_abs_peak","pack_temp_rise_c","energy_kwh_per_100km","high_c_rate_time_h","c_rate_squared_dose_h","electrothermal_dose_h","c_rate_transient_dose","bsi_intensity"]
    controls=["ambient_mean_c","physics_duration_h","soc_start_pct","hvac_energy_kwh"]
    rows=[]
    for driver in DRIVER_FEATURES:
        for stress in response:
            x=pd.to_numeric(fit[driver],errors="coerce");y=pd.to_numeric(fit[stress],errors="coerce");valid=x.notna()&y.notna()
            if valid.sum()<10:continue
            rho,p=spearmanr(x[valid],y[valid]); X=fit.loc[valid,controls]
            residual=[]
            for target in (x[valid],y[valid]):
                model=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=10.0)); model.fit(X,target);residual.append(target-model.predict(X))
            partial,partial_p=spearmanr(*residual)
            rows.append({"driver_feature":driver,"battery_response":stress,"n":int(valid.sum()),"spearman_rho":rho,"p_value":p,"partial_spearman_rho":partial,"partial_p_value":partial_p,"controls":"ambient,duration,SOC,HVAC","claim":"driver_to_immediate_battery_stress_association"})
    result=pd.DataFrame(rows).sort_values("partial_spearman_rho",key=lambda x:x.abs(),ascending=False)
    result.to_csv(output/"driver_battery_response_associations.csv",index=False);return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True);parser.add_argument("--physics-table",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--coverage",type=float,default=.9)
    args=parser.parse_args();output=args.output_dir.resolve();output.mkdir(parents=True,exist_ok=True)
    table=pd.read_parquet(args.physics_table)
    result,calibration=ensemble(args.run_root,output,args.coverage)
    corrected=evaluate(result,output);baselines=run_baselines(table,output);associations=driver_response_associations(table,output)
    primary=corrected[(corrected.task=="current_soh")&(corrected.model=="corrected_hybrid")].iloc[0]
    age=baselines[(baselines.task=="current_soh")&(baselines.model=="AgeMileage-Ridge")].iloc[0]
    persistence=baselines[(baselines.task=="current_soh")&(baselines.model=="Persistence")].iloc[0]
    acceptance={"current_soh_hybrid_mae_pp":primary.mae,"age_mileage_mae_pp":age.mae,"persistence_mae_pp":persistence.mae,"beats_age_mileage":bool(primary.mae<age.mae),"beats_persistence":bool(primary.mae<persistence.mae),"locked_test_authorized":False,"test_used":False}
    (output/"ACCEPTANCE_DECISION.json").write_text(json.dumps(acceptance,indent=2),encoding="utf-8")
    report=f"""# Corrected Nissan physics-hybrid development report

Current-SOH validation MAE: **{primary.mae:.4f} pp**. Age/mileage baseline:
**{age.mae:.4f} pp**. Persistence: **{persistence.mae:.4f} pp**.

The current model {'passes' if acceptance['beats_age_mileage'] else 'does not pass'} the
pre-registered requirement to improve on the age/mileage learning baseline. The locked
test remains unavailable regardless of this development result until the result is
reviewed together with forecast scale, uncertainty coverage, and external-validation plans.

Driver-to-battery evidence is reported per physical response with context-adjusted
Spearman associations. No composite aggression score is used as causal evidence.
"""
    (output/"CORRECTED_NISSAN_REPORT.md").write_text(report,encoding="utf-8")
    summary={"acceptance":acceptance,"metrics":corrected.to_dict(orient="records"),"top_driver_response_associations":associations.head(20).to_dict(orient="records"),"test_used":False}
    (output/"CORRECTED_NISSAN_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(corrected.to_string(index=False));print(json.dumps(acceptance,indent=2))


if __name__=="__main__":main()
