"""Stage 12 vehicle-disjoint EVBattery capacity validation.

This stage evaluates the battery-stream concept architecture on real charging
records from 464 vehicles.  It deliberately calls the target *partial-window
capacity (Ah)* rather than SOH: the public release does not provide rated pack
capacity or a universally comparable 100%-SOH reference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import RobustScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dbpatchkan.data.evbattery import (
    build_evbattery_sample,
    load_evbattery_sample,
    save_evbattery_sample,
)
from dbpatchkan.models.kan import make_fusion_head
from dbpatchkan.models.patchkan import (
    AttentionPool,
    DBPatchKANConfig,
    PatchStreamEncoder,
)


PHYSICAL_CONCEPTS: tuple[str, ...] = (
    "average_cell_voltage_mean_v",
    "average_cell_voltage_std_v",
    "average_cell_voltage_rise_v",
    "charging_current_mean_a",
    "charging_current_std_a",
    "charging_current_peak_abs_a",
    "cell_voltage_spread_mean_v",
    "cell_voltage_spread_max_v",
    "cell_temperature_mean_c",
    "cell_temperature_rise_c",
    "cell_temperature_spread_mean_c",
    "cell_temperature_spread_max_c",
    "soc_start_pct",
    "soc_gain_pct",
    "duration_s",
)
CONTEXT_CONCEPTS: tuple[str, ...] = (
    "mileage",
    "manufacturer_1",
    "manufacturer_2",
    "manufacturer_3",
)
ALL_CONCEPTS = PHYSICAL_CONCEPTS + CONTEXT_CONCEPTS


def _stamp(message: str) -> None:
    print(time.strftime("[%H:%M:%S UTC]", time.gmtime()), message, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_cache(
    archives: list[Path],
    output: Path,
    *,
    max_members_per_archive: int,
    max_records_per_vehicle: int,
    seed: int,
) -> Path:
    """Stream a bounded, vehicle-balanced sample and cache contiguous arrays."""

    def progress(event: dict[str, Any]) -> None:
        _stamp(f"{event['archive']}: decoded {event['decoded']:,} pickle members")

    _stamp("Starting direct-from-tar EVBattery conversion")
    sample = build_evbattery_sample(
        archives,
        capacity_only=True,
        max_members_per_archive=max_members_per_archive,
        max_records_per_vehicle=max_records_per_vehicle,
        seed=seed,
        progress=progress,
    )
    save_evbattery_sample(output, sample)
    concepts = sample["concepts"]
    _stamp(
        f"Saved {len(concepts):,} labeled snippets from "
        f"{concepts[['manufacturer_id', 'vehicle_id']].drop_duplicates().shape[0]} vehicles"
    )
    return output


def design_matrix(concepts: pd.DataFrame) -> pd.DataFrame:
    """Return nineteen named concepts, including non-ordinal manufacturer context."""

    result = concepts.loc[:, PHYSICAL_CONCEPTS].apply(pd.to_numeric, errors="coerce")
    result["mileage"] = pd.to_numeric(concepts["mileage"], errors="coerce")
    manufacturer = pd.to_numeric(concepts["manufacturer_id"], errors="coerce")
    for value in (1, 2, 3):
        result[f"manufacturer_{value}"] = manufacturer.eq(value).astype(float)
    return result.loc[:, ALL_CONCEPTS]


def _vehicle_keys(concepts: pd.DataFrame) -> np.ndarray:
    return (
        concepts["manufacturer_id"].astype(int).astype(str)
        + "_"
        + concepts["vehicle_id"].astype(int).astype(str)
    ).to_numpy()


class _ArrayDataset(Dataset):
    def __init__(self, sequence, context, physical, target, indices):
        self.sequence = sequence
        self.context = context
        self.physical = physical
        self.target = target
        self.indices = np.asarray(indices, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return (
            torch.from_numpy(self.sequence[index]),
            torch.from_numpy(self.context[index]),
            torch.from_numpy(self.physical[index]),
            torch.tensor(self.target[index], dtype=torch.float32),
            torch.tensor(index, dtype=torch.long),
        )


class EVBatteryPatchConceptModel(nn.Module):
    """Battery Patch Transformer -> named concepts -> KAN/MLP capacity head."""

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
            concept_names=ALL_CONCEPTS,
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
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, len(PHYSICAL_CONCEPTS)),
        )
        # The measured concepts are independently calculated and therefore
        # remain the primary concept values.  The Transformer can learn only a
        # bounded denoising correction toward its sequence-derived estimate.
        # Initial sigmoid(-2) ~= 0.12 keeps the external model close to the
        # transparent observed-concept KAN at the start of training.
        self.concept_gate_logit = nn.Parameter(
            torch.full((len(PHYSICAL_CONCEPTS),), -2.0)
        )
        hidden = (16, 8) if head == "kan" else (32, 16)
        self.head = make_fusion_head(
            head,
            len(ALL_CONCEPTS),
            1,
            hidden_dimensions=hidden,
            grid_size=grid_size,
            dropout=0.1,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
        observed_physical: torch.Tensor,
    ):
        encoded = self.encoder(sequence)
        pooled, attention = self.pool(encoded["tokens"], encoded["token_valid"])
        predicted_physical = self.physical_predictor(pooled)
        gate = torch.sigmoid(self.concept_gate_logit).view(1, -1)
        physical = observed_physical + gate * (predicted_physical - observed_physical)
        concepts = torch.cat((physical, context), dim=1)
        prediction = self.head(concepts).squeeze(-1)
        return prediction, predicted_physical, attention

    def regularization_loss(self) -> torch.Tensor:
        if hasattr(self.head, "regularization_loss"):
            return self.head.regularization_loss()
        return next(self.parameters()).new_zeros(())


@dataclass
class FoldScaling:
    sequence_center: np.ndarray
    sequence_scale: np.ndarray
    concept_center: np.ndarray
    concept_scale: np.ndarray
    target_center: float
    target_scale: float


def _robust_center_scale(values: np.ndarray, axis: tuple[int, ...] | int):
    center = np.nanmedian(values, axis=axis)
    q25 = np.nanpercentile(values, 25, axis=axis)
    q75 = np.nanpercentile(values, 75, axis=axis)
    scale = np.where((q75 - q25) > 1e-6, q75 - q25, 1.0)
    return center.astype("float32"), scale.astype("float32")


def _scale_fold(sequence, matrix, target, fit_indices) -> tuple[np.ndarray, np.ndarray, np.ndarray, FoldScaling]:
    seq_center, seq_scale = _robust_center_scale(sequence[fit_indices], axis=(0, 1))
    concept_center, concept_scale = _robust_center_scale(matrix[fit_indices], axis=0)
    target_center_arr, target_scale_arr = _robust_center_scale(target[fit_indices], axis=0)
    seq = np.nan_to_num((sequence - seq_center) / seq_scale, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    concepts = np.nan_to_num((matrix - concept_center) / concept_scale, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
    y = ((target - float(target_center_arr)) / float(target_scale_arr)).astype("float32")
    scaling = FoldScaling(seq_center, seq_scale, concept_center, concept_scale, float(target_center_arr), float(target_scale_arr))
    return seq, concepts, y, scaling


def _predict(model, loader, device, scaling):
    model.eval()
    rows=[]
    with torch.no_grad():
        for sequence, context, physical, _, indices in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, _, _ = model(
                    sequence.to(device), context.to(device), physical.to(device)
                )
            values = prediction.float().cpu().numpy() * scaling.target_scale + scaling.target_center
            rows.extend(zip(indices.numpy().tolist(), values.tolist()))
    return rows


def train_deep_fold(
    sequence,
    matrix,
    target,
    outer_train,
    outer_test,
    groups,
    *,
    head: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
):
    set_seed(seed)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    sub_rel, val_rel = next(splitter.split(outer_train, groups=groups[outer_train]))
    subtrain = outer_train[sub_rel]
    validation = outer_train[val_rel]
    seq, concepts, y, scaling = _scale_fold(sequence, matrix, target, subtrain)
    physical = concepts[:, : len(PHYSICAL_CONCEPTS)]
    context = concepts[:, len(PHYSICAL_CONCEPTS) :]
    train_loader = DataLoader(_ArrayDataset(seq, context, physical, y, subtrain), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(_ArrayDataset(seq, context, physical, y, validation), batch_size=2 * batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(_ArrayDataset(seq, context, physical, y, outer_test), batch_size=2 * batch_size, shuffle=False, num_workers=2)
    model = EVBatteryPatchConceptModel(head=head).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best_state=None; best_mae=float("inf"); stale=0; history=[]
    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for seq_b, context_b, physical_b, y_b, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            seq_b=contextless_seq=seq_b.to(device); context_b=context_b.to(device); physical_b=physical_b.to(device); y_b=y_b.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                prediction, predicted_physical, _ = model(
                    contextless_seq, context_b, physical_b
                )
                regression = nn.functional.smooth_l1_loss(prediction.float(), y_b.float(), beta=max(1e-3, 1.0/scaling.target_scale))
                concept_loss = nn.functional.smooth_l1_loss(predicted_physical.float(), physical_b.float())
                loss = regression + 0.2 * concept_loss + 1e-4 * model.regularization_loss()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        val_rows=_predict(model,val_loader,device,scaling)
        val_idx=np.asarray([r[0] for r in val_rows]); val_pred=np.asarray([r[1] for r in val_rows])
        val_mae=float(mean_absolute_error(target[val_idx],val_pred)); history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_mae_ah":val_mae})
        if val_mae < best_mae - 1e-4:
            best_mae=val_mae; stale=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if epoch == 1 or epoch % 10 == 0: _stamp(f"{head} seed={seed} epoch={epoch}: validation MAE={val_mae:.4f} Ah")
        if stale >= 12: break
    model.load_state_dict(best_state)
    rows=_predict(model,test_loader,device,scaling)
    return rows, pd.DataFrame(history), model


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    y=frame["target_capacity_ah"].to_numpy(); p=frame["prediction_capacity_ah"].to_numpy(); residual=p-y
    vehicle_mae=frame.assign(abs_error=np.abs(residual)).groupby("vehicle_key")["abs_error"].mean()
    return {
        "n":len(frame), "vehicles":frame["vehicle_key"].nunique(),
        "mae_ah":float(mean_absolute_error(y,p)),
        "rmse_ah":float(mean_squared_error(y,p)**0.5),
        "bias_ah":float(np.mean(residual)),
        "r2":float(r2_score(y,p)),
        "vehicle_macro_mae_ah":float(vehicle_mae.mean()),
    }


def _persistence(concepts: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, target: np.ndarray) -> np.ndarray:
    train=concepts.iloc[train_idx]
    fallback=train.assign(target=target[train_idx]).groupby("manufacturer_id")["target"].median().to_dict()
    test=concepts.iloc[test_idx].copy(); test["_row"]=np.arange(len(test)); test["_target"]=target[test_idx]
    output=np.full(len(test),np.nan)
    for _, group in test.groupby(["manufacturer_id","vehicle_id"],sort=False):
        # One prior value per charge segment prevents overlapping windows from
        # leaking the simultaneous capacity label into persistence.
        segment=(group.groupby("charge_segment",sort=False).agg(mileage=("mileage","median"),capacity=("_target","median"),rows=("_row",list)).sort_values("mileage"))
        prior=segment["capacity"].shift()
        manufacturer=int(group["manufacturer_id"].iloc[0]); prior=prior.fillna(fallback[manufacturer])
        for rows,value in zip(segment["rows"],prior): output[np.asarray(rows,dtype=int)]=float(value)
    return output


def run_validation(cache: Path, output_dir: Path, *, folds: int, seeds: list[int], epochs: int, batch_size: int) -> None:
    loaded=load_evbattery_sample(cache); sequence=loaded["sequence"]; concepts=loaded["concepts"].copy()
    matrix_frame=design_matrix(concepts); matrix=matrix_frame.to_numpy(dtype="float32"); target=pd.to_numeric(concepts["capacity_ah"],errors="coerce").to_numpy(dtype="float32")
    groups=_vehicle_keys(concepts); strata=(concepts["manufacturer_id"].astype(int).astype(str)+"_"+concepts["anomaly_label"].astype(int).astype(str)).to_numpy()
    splitter=StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=42); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True,exist_ok=True); predictions=[]; metrics=[]
    for fold,(train_idx,test_idx) in enumerate(splitter.split(matrix,strata,groups),1):
        _stamp(f"Outer fold {fold}/{folds}: train vehicles={len(set(groups[train_idx]))}, test vehicles={len(set(groups[test_idx]))}")
        scaler=RobustScaler().fit(matrix[train_idx]); x_train=scaler.transform(matrix[train_idx]); x_test=scaler.transform(matrix[test_idx])
        baselines={
            "Mean":np.full(len(test_idx),float(np.median(target[train_idx]))),
            "Persistence_WarmStart":_persistence(concepts,train_idx,test_idx,target),
        }
        ridge=Ridge(alpha=1.0).fit(x_train,target[train_idx]); baselines["Ridge"]=ridge.predict(x_test)
        rf=RandomForestRegressor(n_estimators=300,min_samples_leaf=3,max_features=0.8,n_jobs=-1,random_state=42).fit(x_train,target[train_idx]); baselines["RandomForest"]=rf.predict(x_test)
        for model_name,values in baselines.items():
            frame=pd.DataFrame({"row_index":test_idx,"fold":fold,"seed":42,"model":model_name,"manufacturer_id":concepts.iloc[test_idx]["manufacturer_id"].to_numpy(),"vehicle_id":concepts.iloc[test_idx]["vehicle_id"].to_numpy(),"vehicle_key":groups[test_idx],"target_capacity_ah":target[test_idx],"prediction_capacity_ah":values})
            predictions.append(frame); metrics.append({"fold":fold,"seed":42,"model":model_name,**_metrics(frame)})
        for seed in seeds:
            for head in ("mlp","kan"):
                rows,history,model=train_deep_fold(sequence,matrix,target,train_idx,test_idx,groups,head=head,seed=seed,epochs=epochs,batch_size=batch_size,device=device)
                row_idx=np.asarray([r[0] for r in rows]); values=np.asarray([r[1] for r in rows])
                frame=pd.DataFrame({"row_index":row_idx,"fold":fold,"seed":seed,"model":f"BatteryPatchConcept_{head.upper()}","manufacturer_id":concepts.iloc[row_idx]["manufacturer_id"].to_numpy(),"vehicle_id":concepts.iloc[row_idx]["vehicle_id"].to_numpy(),"vehicle_key":groups[row_idx],"target_capacity_ah":target[row_idx],"prediction_capacity_ah":values})
                predictions.append(frame); metrics.append({"fold":fold,"seed":seed,"model":f"BatteryPatchConcept_{head.upper()}",**_metrics(frame)})
                run_dir=output_dir/"models"/f"fold_{fold}"/f"{head}_seed_{seed}"; run_dir.mkdir(parents=True,exist_ok=True); torch.save({"model_state":model.state_dict(),"concept_names":ALL_CONCEPTS,"target_definition":"manufacturer_partial_window_capacity_ah"},run_dir/"model.pt"); history.to_csv(run_dir/"history.csv",index=False)
    prediction_frame=pd.concat(predictions,ignore_index=True); metric_frame=pd.DataFrame(metrics)
    prediction_frame.to_csv(output_dir/"stage12_predictions.csv",index=False); metric_frame.to_csv(output_dir/"stage12_fold_metrics.csv",index=False)
    summary=(metric_frame.groupby("model",as_index=False).agg(mae_ah=("mae_ah","mean"),rmse_ah=("rmse_ah","mean"),r2=("r2","mean"),vehicle_macro_mae_ah=("vehicle_macro_mae_ah","mean")))
    summary.to_csv(output_dir/"stage12_model_summary.csv",index=False)
    audit={"target":"manufacturer_partial_window_capacity_ah","target_is_absolute_soh":False,"driver_telemetry_available":False,"vehicle_disjoint":True,"folds":folds,"seeds":seeds,"records":len(concepts),"vehicles":int(len(set(groups))),"manufacturers":sorted(pd.to_numeric(concepts["manufacturer_id"]).unique().astype(int).tolist()),"concepts":list(ALL_CONCEPTS)}
    (output_dir/"STAGE12_AUDIT.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    _stamp("Stage 12 validation complete")


def main() -> None:
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare"); prep.add_argument("--archives",nargs=3,required=True); prep.add_argument("--output",required=True); prep.add_argument("--max-members-per-archive",type=int,default=50_000); prep.add_argument("--max-records-per-vehicle",type=int,default=64); prep.add_argument("--seed",type=int,default=42)
    run=sub.add_parser("validate"); run.add_argument("--cache",required=True); run.add_argument("--output-dir",required=True); run.add_argument("--folds",type=int,default=5); run.add_argument("--seeds",nargs="+",type=int,default=[42,52,62]); run.add_argument("--epochs",type=int,default=60); run.add_argument("--batch-size",type=int,default=256)
    args=parser.parse_args()
    if args.command=="prepare": prepare_cache([Path(x) for x in args.archives],Path(args.output),max_members_per_archive=args.max_members_per_archive,max_records_per_vehicle=args.max_records_per_vehicle,seed=args.seed)
    else: run_validation(Path(args.cache),Path(args.output_dir),folds=args.folds,seeds=args.seeds,epochs=args.epochs,batch_size=args.batch_size)


if __name__ == "__main__":
    main()
