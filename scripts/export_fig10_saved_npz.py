"""Convert the already saved fitted Figure 10 CSV to the supplied renderer's NPZ.

No fitting, smoothing, interpolation or band recomputation is performed.
"""
import argparse
import csv
from pathlib import Path
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=root/'figures/data/Figure_10_data.csv')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.source.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    assert rows and all(r['Actual_Fitted_Data']=='YES' for r in rows)
    assert all(r['Output_Name']=='current_soh' for r in rows)
    payload={}
    for key in ('rate','cycle'):
        subset=[r for r in rows if r['Concept_Name']=='bsi_'+key]
        aggregate=sorted([r for r in subset if r['Curve_Type']=='Across-seed aggregate'],
                         key=lambda r:float(r['Normalized_Concept_Value']))
        x=np.array([float(r['Normalized_Concept_Value']) for r in aggregate])
        assert len(x)>1 and np.all(np.diff(x)>0)
        seed_ids=sorted({int(r['Seed']) for r in subset if r['Curve_Type']=='Fitted seed'})
        assert len(seed_ids)==5
        values=[]
        for seed in seed_ids:
            fitted=sorted([r for r in subset if r['Curve_Type']=='Fitted seed' and int(r['Seed'])==seed],
                          key=lambda r:float(r['Normalized_Concept_Value']))
            sx=np.array([float(r['Normalized_Concept_Value']) for r in fitted])
            assert np.array_equal(x,sx),'Seed grid mismatch: no automatic interpolation is permitted'
            values.append([float(r['Centered_KAN_Response_pp']) for r in fitted])
        payload[key+'_x']=x
        payload[key+'_seeds']=np.asarray(values)
        for name,column in [('lo','Lower_pp'),('hi','Upper_pp'),('mean','Centered_KAN_Response_pp')]:
            payload[key+'_'+name]=np.array([float(r[column]) for r in aggregate])
        assert all(np.isfinite(payload[key+'_'+n]).all() for n in ['x','seeds','lo','hi','mean'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**payload)
    print(f'Saved exact fitted curves and original band: {args.output}',flush=True)


if __name__=='__main__':main()
