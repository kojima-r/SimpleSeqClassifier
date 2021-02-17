# SimpleSeqClassifier

```
nohup python model.py cv_after --config ./config_base.json --gpu 0 > log_cv_after.txt &
nohup python model.py cv_prev  --config ./config_base.json --gpu 1 > log_cv_after.txt &

nohup python model.py train_after --config ./config_base.json --gpu 0 > log_train_after.txt &
nohup python model.py train_prev  --config ./config_base.json --gpu 0 > log_train_prev.txt &

nohup python model.py test_after --config ./config_base_ex.json --data ./ROP_1h_ych_28w/rop_data_npy/dataset.json --gpu 0 > log_test_ex_after.txt &
nohup python model.py test_prev  --config ./config_base_ex.json --data ./ROP_1h_ych_28w/rop_data_npy/dataset.json --gpu 0 > log_test_ex_prev.txt &


```
