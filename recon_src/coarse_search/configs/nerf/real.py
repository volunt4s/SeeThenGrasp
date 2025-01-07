_base_ = '../default.py'

expname = 'dvgo_real'
basedir = './logs/'

data = dict(
    datadir='./data/',
    dataset_type='blender',
    white_bkgd=True,
)