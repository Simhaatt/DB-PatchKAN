# Dataset acquisition and rights

MIT applies to the project's original software, not to source datasets. Obtain data from the original publisher, retain their citations/license terms, and verify hashes. No raw telemetry is included in this repository.

## Nissan Leaf

The project input is a user-provided CSV named `Nissan Leaf Real-World Data (1) (1).csv`: one 40-kWh Leaf, 127 trips and 56 distinct BMS SOH values. Original download provenance and redistribution permission have not been established for this public snapshot. It is deliberately not redistributed. Contact the authors for permitted access; do not substitute another Leaf dataset while claiming exact reproduction.

See `docs/data_contract.md` and `configs/default.yaml` for required columns, sentinel handling, current sign and nominal constants. Set `data.source_csv` to your permitted local file. Keep all segments of a trip in the same split; use only training trips for scaling/selection/clustering. No future health label may be substituted for a missing one.

## RWTH Electric Vehicle and Battery Data

Original record: [Dataset to “Battery Electric Vehicles in Commercial Fleets: Use profiles, battery aging, and open-access data”](https://doi.org/10.18154/RWTH-2024-01907), by Fabian Rücker, Jan Figgener, Ilka Schoeneberger and Dirk Uwe Sauer (2024).

Download `Electric_Vehicle_and_Battery_Data.zip` from the source, retain the publisher's terms, and extract it below `data/external/rwth/raw/` with its directory structure preserved. Use `scripts/build_rwth_stage4.py --help` for the expected paths. Field pack temperature is not present. Nine vehicles supply the driving study; controlled capacity checkpoints are sparse and are not the same target definition as BMS SOH. Published fixed-window capacity labels and independent integration audits are retained by the adapter.

## EVE Pilot

Original record: [The EVE Pilot: Usage Data from an Electric Car in France](https://doi.org/10.57745/5O6QIH). The project's adapter records Etalab Open License 2.0; verify terms on the original record before redistribution.

One Renault Zoe vehicle; irregular speed/current/voltage/SOC/ambient/SOH and module-temperature streams. `src/dbpatchkan/data/eve.py` contains the selected publisher datafile IDs and downloader. To acquire and harmonize:

```bash
python scripts/build_eve_stage5.py --raw-dir data/external/eve/raw --output-dir outputs/eve
```

Read the stage-specific causal join policy: historical whole-trip responses and Stage 15 past-only fixed-horizon responses are different experiments. Single-vehicle temporal folds are not fleet validation.

## EVBattery / Tsinghua

Original record: [EV field datasets Tsinghua](https://doi.org/10.17632/mcsh4hnb8b.1), version 1. The project records CC BY 4.0 for the release; retain the original attribution and confirm the publisher's terms.

Download the three archives and keep them compressed. The adapter uses a restricted legacy PyTorch decoder; **do not switch to unrestricted pickle loading**. Zero capacity labels are missing, not healthy or zero-capacity packs.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| battery_dataset1.tar.gz | 812110847 | BF56F38EF1F0C96BB184F085F261F72092D16366C5EFB7264DF55E68FBA16BE5 |
| battery_dataset2.tar.gz | 418251471 | 265E3A7B6CEDCD8A6C1844C20C9E69F0DE7DCDEE07F10816C29479791366107F |
| battery_dataset3.tar.gz | 177847087 | 0B258606D79D6392C0F8C76E74B1F544650AB7FBF29E758F078C2C6AB74630AD |

The raw release has 1,278,277 charging snippets. The project's deterministic sampling uses up to 200,000 visited members/archive, up to 64 records/vehicle and seed 42, producing 12,224 sampled labelled snippets from 191 vehicles in the recorded run. These yield 9,714 next-distinct-mileage transitions. Verify your reproduced audit counts; an altered sampling/cohort is a new experiment.

Signals have 128 steps and eight channels: average/max/min cell voltage, charging current, max/min cell temperature, SOC and timestamp. Labels are manufacturer partial-window capacity estimates in Ah. They are not independent laboratory full-capacity tests or absolute SOH. There are no driving trajectories in this dataset.

## Unused 300-EV sample

A one-vehicle telemetry sample was inspected but lacks an explicit defensible capacity/SOH target for the proposed health evaluation. It is not counted as a 300-vehicle external health validation and is not required for the main reproduction route.
