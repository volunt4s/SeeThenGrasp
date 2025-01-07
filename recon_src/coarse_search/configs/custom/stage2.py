import os

_base_ = os.path.join('..', 'default.py')

expname = 'scan'
basedir = os.path.join('.', 'logs', 'custom')
train_all = True
reso_level = 1
exp_stage = 'coarse'

data = dict(
    datadir=os.path.join('recon_src', 'coarse_search', 'data', 'stage2'),
    dataset_type='dtu',
    inverse_y=True,
    white_bkgd= False
)

coarse_train = dict(
    N_iters=40000
)