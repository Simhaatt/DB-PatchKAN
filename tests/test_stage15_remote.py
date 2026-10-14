"""Run only on remote Colab for this delivery (local CPU is occupied)."""
import numpy as np
import pandas as pd
import pytest

from dbpatchkan.analysis.stage15_review import paired_clusters,holm,sensitivity_configs
from dbpatchkan.analysis.stage15_response import make_examples,ResponseConceptModel


def paired_example():
    rows=[]
    for i in range(6):
        for model,pred in [('Persistence',1.),('AI',.5)]:
            rows.append({'transition_id':i,'vehicle_key':f'v{i//2}',
                'model':model,'future_capacity_ah':0.,'prediction_capacity_ah':pred})
    return pd.DataFrame(rows)


def test_paired_bootstrap_clusters_not_rows():
    result=paired_clusters(paired_example(),'AI','Persistence',draws=200)
    assert result['vehicles']==3
    assert result['rows']==6
    assert np.isclose(result['delta_mae'],-.5)
    assert np.isclose(result['ci_low'],-.5)


def test_duplicates_fail_instead_of_pseudoreplication():
    frame=paired_example()
    with pytest.raises(ValueError,match='Duplicate'):
        paired_clusters(pd.concat([frame,frame.iloc[:1]]),'AI','Persistence')


def test_holm_restores_original_order():
    assert np.allclose(holm([.04,.001,.03]),[.06,.003,.06])


def test_sensitivity_never_inverts_ramps():
    for cfg in sensitivity_configs().values():
        assert cfg.discharge_c_upper>cfg.discharge_c_lower
        assert cfg.hot_temp_upper_c>cfg.hot_temp_lower_c
        assert cfg.cold_regen_start_c>cfg.cold_regen_full_c


def native_example():
    times=pd.date_range('2020-01-01',periods=180,freq='5s',tz='UTC')
    return pd.DataFrame({'timestamp':times,'speed_kmh':30.+np.sin(np.arange(180)),
        'current_a':20.,'voltage_v':350.,'soc_pct':50.,'pack_temp_c':np.nan,
        'spread_c':np.nan,'ambient_c':15.,'vehicle':'v1','nominal_capacity_ah':52.})


def test_future_perturbation_leaves_inputs_unchanged():
    frame=native_example();a,meta=make_examples(frame,'RWTH')
    changed=frame.copy();changed.loc[24:47,'speed_kmh']=80.;changed.loc[24:47,'current_a']=120.
    b,_=make_examples(changed,'RWTH')
    assert np.array_equal(a['driver'][0],b['driver'][0],equal_nan=True)
    assert np.array_equal(a['battery'][0],b['battery'][0],equal_nan=True)
    assert np.array_equal(a['concepts'][0],b['concepts'][0],equal_nan=True)
    assert not np.array_equal(a['targets'][0],b['targets'][0],equal_nan=True)
    assert meta.cutoff_ns.iloc[0]<meta.future_end_ns.iloc[0]
    assert np.isnan(a['targets'][:,3::6]).all() # no fabricated RWTH pack temperature


def test_stale_observations_cannot_become_future_labels():
    frame=native_example()
    frame['current_observed_ns']=frame.timestamp.iloc[0].value
    frame['voltage_observed_ns']=frame.timestamp.iloc[0].value
    with pytest.raises(ValueError,match='No valid native response examples'):
        make_examples(frame,'RWTH')


def test_response_models_mask_missing_channels():
    import torch
    from dbpatchkan.analysis.stage15_response import CONCEPT_NAMES
    data,_=make_examples(native_example(),'RWTH')
    for head in ('kan','mlp'):
        model=ResponseConceptModel(head).eval()
        with torch.no_grad():
            p,c=model(torch.from_numpy(data['driver'][:2]),torch.from_numpy(data['battery'][:2]))
        assert p.shape==(2,18)
        assert c.shape==(2,len(CONCEPT_NAMES))
        assert torch.isfinite(p).all()


def test_classical_and_physics_fits_do_not_use_evaluation_labels(tmp_path):
    from dbpatchkan.analysis.stage15_response import tabular_predictions
    frame=pd.concat([native_example()]*12,ignore_index=True)
    frame['timestamp']=pd.date_range('2020-01-01',periods=len(frame),freq='5s',tz='UTC')
    data,_=make_examples(frame,'RWTH')
    train=np.arange(25);test=np.arange(25,30)
    first=tabular_predictions(data,train,test,tmp_path/'original')
    changed={k:v.copy() for k,v in data.items()}
    changed['targets'][test]+=1000
    second=tabular_predictions(changed,train,test,tmp_path/'changed_test_labels')
    for model in first:
        assert np.allclose(first[model],second[model],equal_nan=True)
    import json
    physics=json.loads((tmp_path/'original/PHYSICAL_BASELINE_PARAMETERS.json').read_text())
    assert min(physics['current_power_coefficients'])>=0
    assert physics['effective_resistance_ohm']>=0
    assert physics['thermal_training_label_count']==0
