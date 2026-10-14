"""Stage 13 persistence-anchored future-capacity forecasting on EVBattery.

The model forecasts the next distinct mileage checkpoint:

``future capacity = current measured capacity + learned signed residual``.

Signed residuals are retained because public partial-window capacity labels can
increase through noise, operating-condition variation, and recalibration.  A
soft penalty discourages implausibly large positive changes without forcing a
strictly monotonic observed label trajectory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.analysis.stage12_evbattery import PHYSICAL_CONCEPTS
from dbpatchkan.data.evbattery import load_evbattery_sample, sha256_file
from dbpatchkan.models.kan import make_fusion_head
from dbpatchkan.models.patchkan import AttentionPool, DBPatchKANConfig, PatchStreamEncoder


FORECAST_PHYSICAL: tuple[str, ...] = tuple(
    name for name in PHYSICAL_CONCEPTS if name != "duration_s"
)
FORECAST_CONTEXT: tuple[str, ...] = (
    "current_capacity_ah",
    "current_mileage",
    "forecast_horizon_km",
    "manufacturer_1",
    "manufacturer_2",
    "manufacturer_3",
)
FORECAST_CONCEPTS = FORECAST_PHYSICAL + FORECAST_CONTEXT
assert len(FORECAST_CONCEPTS) == 20


def stamp(message: str) -> None:
    print(time.strftime("[%H:%M:%S UTC]", time.gmtime()), message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transition_cache(source_cache: str | Path, output: str | Path) -> Path:
    """Create adjacent, distinct-mileage future-capacity transitions."""

    source_cache = Path(source_cache)
    loaded = load_evbattery_sample(source_cache)
    sequence = loaded["sequence"]
    metadata = loaded["concepts"].copy()
    metadata["manufacturer_id"] = pd.to_numeric(metadata["manufacturer_id"]).astype(int)
    metadata["vehicle_id"] = pd.to_numeric(metadata["vehicle_id"]).astype(int)
    metadata["mileage"] = pd.to_numeric(metadata["mileage"], errors="coerce")
    metadata["capacity_ah"] = pd.to_numeric(metadata["capacity_ah"], errors="coerce")
    metadata["vehicle_key"] = (
        metadata["manufacturer_id"].astype(str)
        + "_" + metadata["vehicle_id"].astype(str)
    )
    transition_sequences: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    transition_id = 0
    for vehicle_key, vehicle in metadata.groupby("vehicle_key", sort=True):
        checkpoints: list[dict[str, Any]] = []
        for mileage, group in vehicle.groupby("mileage", sort=True):
            capacity = float(group["capacity_ah"].median())
            # Select the released snippet closest to the checkpoint's median
            # capacity; all other physical concepts are robust checkpoint medians.
            representative = int((group["capacity_ah"] - capacity).abs().idxmin())
            checkpoint = {
                "mileage": float(mileage),
                "capacity_ah": capacity,
                "sequence_index": representative,
                "manufacturer_id": int(group["manufacturer_id"].iloc[0]),
                "vehicle_id": int(group["vehicle_id"].iloc[0]),
            }
            for concept in FORECAST_PHYSICAL:
                checkpoint[concept] = float(pd.to_numeric(group[concept], errors="coerce").median())
            checkpoints.append(checkpoint)
        checkpoints.sort(key=lambda item: item["mileage"])
        for current, future in zip(checkpoints[:-1], checkpoints[1:]):
            horizon = future["mileage"] - current["mileage"]
            if not np.isfinite(horizon) or horizon <= 0:
                continue
            transition_sequences.append(sequence[current["sequence_index"]])
            rows.append({
                "transition_id": transition_id,
                "manufacturer_id": current["manufacturer_id"],
                "vehicle_id": current["vehicle_id"],
                "vehicle_key": vehicle_key,
                "current_mileage": current["mileage"],
                "future_mileage": future["mileage"],
                "forecast_horizon_km": horizon,
                "current_capacity_ah": current["capacity_ah"],
                "future_capacity_ah": future["capacity_ah"],
                "capacity_delta_ah": future["capacity_ah"] - current["capacity_ah"],
                **{name: current[name] for name in FORECAST_PHYSICAL},
            })
            transition_id += 1
    table = pd.DataFrame(rows)
    arrays: dict[str, np.ndarray] = {
        "sequence": np.stack(transition_sequences).astype("float32"),
        "columns": np.asarray(table.columns, dtype="U80"),
        "source_cache_sha256": np.asarray([sha256_file(source_cache)], dtype="U64"),
    }
    for column in table:
        values = table[column]
        if pd.api.types.is_numeric_dtype(values):
            arrays[f"meta__{column}"] = values.to_numpy()
        else:
            arrays[f"meta__{column}"] = values.astype(str).to_numpy(dtype="U80")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    stamp(
        f"Saved {len(table):,} transitions from {table.vehicle_key.nunique()} vehicles; "
        f"median horizon={table.forecast_horizon_km.median():.1f} km"
    )
    return output


def load_transition_cache(path: str | Path) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=False) as archive:
        sequence = np.asarray(archive["sequence"], dtype="float32")
        columns = [str(value) for value in archive["columns"]]
        table = pd.DataFrame({column: archive[f"meta__{column}"] for column in columns})
        source_hash = str(archive["source_cache_sha256"][0])
    if len(sequence) != len(table):
        raise ValueError("Transition sequence/table row mismatch")
    return {"sequence": sequence, "table": table, "source_cache_sha256": source_hash}


def forecast_matrix(table: pd.DataFrame) -> pd.DataFrame:
    result = table.loc[:, FORECAST_PHYSICAL].apply(pd.to_numeric, errors="coerce")
    for name in ("current_capacity_ah", "current_mileage", "forecast_horizon_km"):
        result[name] = pd.to_numeric(table[name], errors="coerce")
    manufacturer = pd.to_numeric(table["manufacturer_id"], errors="coerce")
    for value in (1, 2, 3):
        result[f"manufacturer_{value}"] = manufacturer.eq(value).astype(float)
    return result.loc[:, FORECAST_CONCEPTS]


class TransitionDataset(Dataset):
    def __init__(self, sequence, physical, context, delta, current, indices):
        self.sequence = sequence
        self.physical = physical
        self.context = context
        self.delta = delta
        self.current = current
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return (
            torch.from_numpy(self.sequence[index]),
            torch.from_numpy(self.physical[index]),
            torch.from_numpy(self.context[index]),
            torch.tensor(self.delta[index], dtype=torch.float32),
            torch.tensor(self.current[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class PersistenceAnchoredPatchConcept(nn.Module):
    """Patch encoder with bounded concept denoising and residual KAN/MLP."""

    def __init__(self, *, head: str, d_model: int = 96, grid_size: int = 5):
        super().__init__()
        config = DBPatchKANConfig(
            driver_channels=1,
            battery_channels=8,
            segment_length=128,
            patch_length=16,
            patch_stride=8,
            d_model=d_model,
            encoder_layers=2,
            n_heads=4,
            ff_dim=2 * d_model,
            dropout=0.1,
            concept_names=FORECAST_CONCEPTS,
            protected_driver_channels=(0,),
            protected_battery_channels=tuple(range(8)),
        )
        self.encoder = PatchStreamEncoder(
            channels=8,
            protected_channels=tuple(range(8)),
            config=config,
        )
        self.pool = AttentionPool(d_model)
        self.physical_predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(d_model, len(FORECAST_PHYSICAL)),
        )
        self.concept_gate_logit = nn.Parameter(
            torch.full((len(FORECAST_PHYSICAL),), -2.0)
        )
        hidden = (16, 8) if head == "kan" else (32, 16)
        self.residual_head = make_fusion_head(
            head, len(FORECAST_CONCEPTS), 1,
            hidden_dimensions=hidden, grid_size=grid_size, dropout=0.1,
        )

    def forward(self, sequence, observed_physical, context):
        encoded = self.encoder(sequence)
        pooled, attention = self.pool(encoded["tokens"], encoded["token_valid"])
        predicted_physical = self.physical_predictor(pooled)
        gate = torch.sigmoid(self.concept_gate_logit).view(1, -1)
        physical = observed_physical + gate * (predicted_physical - observed_physical)
        residual = self.residual_head(torch.cat((physical, context), dim=1)).squeeze(-1)
        return residual, predicted_physical, attention

    def regularization_loss(self):
        return self.residual_head.regularization_loss()


@dataclass
class Scaling:
    sequence_center: list[float]
    sequence_scale: list[float]
    concept_center: list[float]
    concept_scale: list[float]
    delta_center: float
    delta_scale: float


def robust(values: np.ndarray, axis):
    center = np.nanmedian(values, axis=axis)
    scale = np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return np.asarray(center, dtype="float32"), np.asarray(scale, dtype="float32")


def scale_fold(sequence, concepts, delta, fit):
    seq_center, seq_scale = robust(sequence[fit], (0, 1))
    con_center, con_scale = robust(concepts[fit], 0)
    delta_center, delta_scale = robust(delta[fit], 0)
    seq = np.nan_to_num((sequence - seq_center) / seq_scale).astype("float32")
    con = np.nan_to_num((concepts - con_center) / con_scale).astype("float32")
    y = ((delta - float(delta_center)) / float(delta_scale)).astype("float32")
    info = Scaling(
        seq_center.tolist(), seq_scale.tolist(), con_center.tolist(), con_scale.tolist(),
        float(delta_center), float(delta_scale),
    )
    return seq, con, y, info


def predict(model, loader, device, scaling):
    model.eval(); records=[]
    with torch.no_grad():
        for sequence, physical, context, _, current, indices in loader:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                residual, _, _ = model(
                    sequence.to(device), physical.to(device), context.to(device)
                )
            delta = residual.float().cpu().numpy() * scaling.delta_scale + scaling.delta_center
            future = current.numpy() + delta
            records.extend(zip(indices.numpy().tolist(), delta.tolist(), future.tolist()))
    return records


def train_fold(
    sequence, concepts, delta, current, outer_train, outer_test, groups,
    *, head, seed, epochs, batch_size, device,
):
    set_seed(seed)
    split = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_rel, val_rel = next(split.split(outer_train, groups=groups[outer_train]))
    train_idx, val_idx = outer_train[train_rel], outer_train[val_rel]
    seq, con, y, scaling = scale_fold(sequence, concepts, delta, train_idx)
    physical = con[:, : len(FORECAST_PHYSICAL)]
    context = con[:, len(FORECAST_PHYSICAL) :]
    make = lambda idx, shuffle, size: DataLoader(
        TransitionDataset(seq, physical, context, y, current, idx),
        batch_size=size, shuffle=shuffle, num_workers=2,
        pin_memory=device.type == "cuda",
    )
    train_loader = make(train_idx, True, batch_size)
    val_loader = make(val_idx, False, 2 * batch_size)
    test_loader = make(outer_test, False, 2 * batch_size)
    model = PersistenceAnchoredPatchConcept(head=head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_state = None; best_mae = float("inf"); stale = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for seq_b, physical_b, context_b, y_b, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            seq_b=seq_b.to(device); physical_b=physical_b.to(device)
            context_b=context_b.to(device); y_b=y_b.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                residual, predicted_physical, _ = model(seq_b, physical_b, context_b)
                beta = max(1e-3, 0.25 / scaling.delta_scale)
                regression = nn.functional.smooth_l1_loss(
                    residual.float(), y_b.float(), beta=beta
                )
                concept_loss = nn.functional.smooth_l1_loss(
                    predicted_physical.float(), physical_b.float()
                )
                residual_ah = residual.float() * scaling.delta_scale + scaling.delta_center
                positive_penalty = torch.relu(residual_ah - 0.25).square().mean()
                loss = (
                    regression + 0.2 * concept_loss + 0.02 * positive_penalty
                    + 1e-4 * model.regularization_loss()
                )
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); losses.append(float(loss.detach()))
        val_records = predict(model, val_loader, device, scaling)
        idx = np.asarray([item[0] for item in val_records], dtype=int)
        delta_prediction = np.asarray([item[1] for item in val_records])
        val_mae = float(np.mean(np.abs(delta[idx] - delta_prediction)))
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "validation_mae_ah": val_mae})
        if val_mae < best_mae - 1e-4:
            best_mae = val_mae; stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            stamp(f"{head} seed={seed} epoch={epoch}: validation MAE={val_mae:.4f} Ah")
        if stale >= 12:
            break
    model.load_state_dict(best_state)
    # Recalculate conformal score from the selected checkpoint only.
    val_records = predict(model, val_loader, device, scaling)
    val_idx_selected = np.asarray([item[0] for item in val_records], dtype=int)
    val_future = np.asarray([item[2] for item in val_records])
    future_truth = current[val_idx_selected] + delta[val_idx_selected]
    q90 = float(np.quantile(np.abs(future_truth - val_future), 0.90, method="higher"))
    return predict(model, test_loader, device, scaling), pd.DataFrame(history), model, scaling, q90


def metric(frame: pd.DataFrame) -> dict[str, Any]:
    y=frame.future_capacity_ah.to_numpy(float); p=frame.prediction_capacity_ah.to_numpy(float)
    residual=p-y; vehicle=frame.assign(ae=np.abs(residual)).groupby("vehicle_key").ae.mean()
    output={
        "n":len(frame), "vehicles":frame.vehicle_key.nunique(),
        "mae_ah":float(np.mean(np.abs(residual))),
        "rmse_ah":float(np.sqrt(np.mean(residual**2))),
        "bias_ah":float(np.mean(residual)), "r2":float(r2_score(y,p)),
        "vehicle_macro_mae_ah":float(vehicle.mean()),
    }
    if {"interval_low_ah","interval_high_ah"}.issubset(frame):
        output["interval_90_coverage"] = float(
            ((y >= frame.interval_low_ah) & (y <= frame.interval_high_ah)).mean()
        )
        output["interval_mean_width_ah"] = float(
            (frame.interval_high_ah - frame.interval_low_ah).mean()
        )
    return output


def run_validation(cache, output_dir, *, folds, seeds, epochs, batch_size):
    loaded=load_transition_cache(cache); sequence=loaded["sequence"]; table=loaded["table"].copy()
    concepts=forecast_matrix(table).to_numpy("float32")
    delta=pd.to_numeric(table.capacity_delta_ah).to_numpy("float32")
    current=pd.to_numeric(table.current_capacity_ah).to_numpy("float32")
    future=pd.to_numeric(table.future_capacity_ah).to_numpy("float32")
    groups=table.vehicle_key.astype(str).to_numpy()
    strata=table.manufacturer_id.astype(int).astype(str).to_numpy()
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    predictions=[]; metrics=[]
    for fold,(train_idx,test_idx) in enumerate(splitter.split(concepts,strata,groups),1):
        stamp(f"Outer fold {fold}/{folds}: train vehicles={len(set(groups[train_idx]))}, test vehicles={len(set(groups[test_idx]))}")
        # Persistence requires only a prior checkpoint. Its interval calibration
        # uses outer-training transitions and never the held-out vehicle labels.
        q_persistence=float(np.quantile(np.abs(delta[train_idx]),0.90,method="higher"))
        baseline_predictions={"Persistence":current[test_idx]}
        scaler=RobustScaler().fit(concepts[train_idx]); xtr=scaler.transform(concepts[train_idx]); xte=scaler.transform(concepts[test_idx])
        ridge=Ridge(alpha=1.0).fit(xtr,delta[train_idx]); baseline_predictions["Anchored_Ridge"]=current[test_idx]+ridge.predict(xte)
        rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=0.8,n_jobs=-1,random_state=42).fit(xtr,delta[train_idx]); baseline_predictions["Anchored_RF"]=current[test_idx]+rf.predict(xte)
        for name,prediction in baseline_predictions.items():
            q=q_persistence
            frame=pd.DataFrame({"transition_id":table.iloc[test_idx].transition_id.to_numpy(),"fold":fold,"seed":42,"model":name,"manufacturer_id":table.iloc[test_idx].manufacturer_id.to_numpy(),"vehicle_id":table.iloc[test_idx].vehicle_id.to_numpy(),"vehicle_key":groups[test_idx],"forecast_horizon_km":table.iloc[test_idx].forecast_horizon_km.to_numpy(),"current_capacity_ah":current[test_idx],"future_capacity_ah":future[test_idx],"prediction_capacity_ah":prediction,"interval_low_ah":prediction-q,"interval_high_ah":prediction+q})
            predictions.append(frame); metrics.append({"fold":fold,"seed":42,"model":name,**metric(frame)})
        for seed in seeds:
            for head in ("mlp","kan"):
                records,history,model,scaling,q90=train_fold(sequence,concepts,delta,current,train_idx,test_idx,groups,head=head,seed=seed,epochs=epochs,batch_size=batch_size,device=device)
                indices=np.asarray([item[0] for item in records],int); prediction=np.asarray([item[2] for item in records])
                name=f"PersistenceAnchored_{head.upper()}"
                frame=pd.DataFrame({"transition_id":table.iloc[indices].transition_id.to_numpy(),"fold":fold,"seed":seed,"model":name,"manufacturer_id":table.iloc[indices].manufacturer_id.to_numpy(),"vehicle_id":table.iloc[indices].vehicle_id.to_numpy(),"vehicle_key":groups[indices],"forecast_horizon_km":table.iloc[indices].forecast_horizon_km.to_numpy(),"current_capacity_ah":current[indices],"future_capacity_ah":future[indices],"prediction_capacity_ah":prediction,"interval_low_ah":prediction-q90,"interval_high_ah":prediction+q90})
                predictions.append(frame); metrics.append({"fold":fold,"seed":seed,"model":name,**metric(frame)})
                run=output_dir/"models"/f"fold_{fold}"/f"{head}_seed_{seed}"; run.mkdir(parents=True,exist_ok=True)
                torch.save({"model_state":model.state_dict(),"scaling":asdict(scaling),"q90_ah":q90,"concepts":FORECAST_CONCEPTS,"target":"next_distinct_mileage_partial_capacity_ah"},run/"model.pt"); history.to_csv(run/"history.csv",index=False)
    pred=pd.concat(predictions,ignore_index=True); met=pd.DataFrame(metrics)
    pred.to_csv(output_dir/"stage13_predictions.csv",index=False); met.to_csv(output_dir/"stage13_fold_metrics.csv",index=False)
    summary=met.groupby("model",as_index=False).agg(mae_ah=("mae_ah","mean"),rmse_ah=("rmse_ah","mean"),r2=("r2","mean"),vehicle_macro_mae_ah=("vehicle_macro_mae_ah","mean"),interval_90_coverage=("interval_90_coverage","mean"),interval_mean_width_ah=("interval_mean_width_ah","mean"))
    summary.to_csv(output_dir/"stage13_model_summary.csv",index=False)
    # Seed-ensemble deep predictions and paired vehicle bootstrap.
    keys=["transition_id","fold","manufacturer_id","vehicle_id","vehicle_key","forecast_horizon_km","current_capacity_ah","future_capacity_ah"]
    base=pred[pred.model.isin(["Persistence","Anchored_Ridge","Anchored_RF"])]
    deep=pred[pred.model.str.startswith("PersistenceAnchored")]
    ensemble=deep.groupby(keys+["model"],as_index=False).agg(prediction_capacity_ah=("prediction_capacity_ah","mean"),interval_low_ah=("interval_low_ah","mean"),interval_high_ah=("interval_high_ah","mean"))
    combined=pd.concat([base[keys+["model","prediction_capacity_ah","interval_low_ah","interval_high_ah"]],ensemble],ignore_index=True)
    combined.to_csv(output_dir/"stage13_seed_ensemble_predictions.csv",index=False)
    ens_metrics=pd.DataFrame([{"model":name,**metric(frame)} for name,frame in combined.groupby("model")]); ens_metrics.to_csv(output_dir/"stage13_seed_ensemble_metrics.csv",index=False)
    combined["ae"]=(combined.prediction_capacity_ah-combined.future_capacity_ah).abs(); veh=combined.groupby(["model","vehicle_key"]).ae.mean().unstack("model")
    rng=np.random.default_rng(42); paired=[]
    for challenger in ["Persistence","Anchored_Ridge","Anchored_RF","PersistenceAnchored_MLP"]:
        d=(veh["PersistenceAnchored_KAN"]-veh[challenger]).dropna().to_numpy(); boot=np.mean(rng.choice(d,(50000,len(d)),replace=True),axis=1)
        paired.append({"proposed":"PersistenceAnchored_KAN","challenger":challenger,"vehicles":len(d),"mean_delta_mae_ah":d.mean(),"proposed_win_fraction":np.mean(d<0),"ci_low":np.quantile(boot,.025),"ci_high":np.quantile(boot,.975),"negative_favors_proposed":True})
    pd.DataFrame(paired).to_csv(output_dir/"stage13_paired_vehicle_bootstrap.csv",index=False)
    # Horizon bands are descriptive and fixed before inspecting predictions.
    bands=[(0,500,"<=500km"),(500,1000,"500-1000km"),(1000,2000,"1000-2000km"),(2000,np.inf,">2000km")]; horizon=[]
    for name,frame in combined.groupby("model"):
        for low,high,label in bands:
            part=frame[(frame.forecast_horizon_km>low)&(frame.forecast_horizon_km<=high)]
            if len(part): horizon.append({"model":name,"horizon_band":label,**metric(part)})
    pd.DataFrame(horizon).to_csv(output_dir/"stage13_horizon_metrics.csv",index=False)
    audit={"task":"next_distinct_mileage_partial_capacity_forecast","formula":"future_capacity=current_capacity+learned_signed_residual","vehicle_disjoint":True,"same_mileage_collapsed":True,"target_is_absolute_soh":False,"driver_telemetry_available":False,"transitions":len(table),"vehicles":int(table.vehicle_key.nunique()),"median_horizon_km":float(pd.to_numeric(table.forecast_horizon_km).median()),"folds":folds,"seeds":seeds,"concepts":list(FORECAST_CONCEPTS),"source_cache_sha256":loaded["source_cache_sha256"]}
    (output_dir/"STAGE13_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    stamp("Stage 13 persistence-anchored validation complete")


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build"); build.add_argument("--source-cache",required=True); build.add_argument("--output",required=True)
    run=sub.add_parser("validate"); run.add_argument("--cache",required=True); run.add_argument("--output-dir",required=True); run.add_argument("--folds",type=int,default=5); run.add_argument("--seeds",nargs="+",type=int,default=[42,52,62]); run.add_argument("--epochs",type=int,default=60); run.add_argument("--batch-size",type=int,default=512)
    args=parser.parse_args()
    if args.command=="build": build_transition_cache(args.source_cache,args.output)
    else: run_validation(args.cache,args.output_dir,folds=args.folds,seeds=args.seeds,epochs=args.epochs,batch_size=args.batch_size)


if __name__ == "__main__":
    main()
