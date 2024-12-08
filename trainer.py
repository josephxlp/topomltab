import os
import time
from os.path import join
import pandas as pd
import numpy as np 
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
import pickle
import joblib
from pprint import pprint
from glob import glob
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from upaths import RESAMPLE_MODELS_PATH,RESAMPLE_TILES_DPATH



def pickle_write(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def joblib_write(model, filename):
    joblib.dump(model, filename)

def pickle_read(filename):
    with open(filename, 'rb') as file:
        loaded_model_pickle = pickle.load(file) 
    return loaded_model_pickle

def joblib_read(filename):
    return joblib.load(filename) 

def performance(y_train, y_train_pred,y_vali, y_val_pred):
        # Calculate additional regression metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    val_mae = mean_absolute_error(y_vali, y_val_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_vali, y_val_pred)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_vali, y_val_pred)

    train_rmse = np.sqrt(train_mse)

    val_rmse = np.sqrt(val_mse)

    # Create a DataFrame to store the metrics
    metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'MSE', 'R2'],
    'Train': [train_rmse, train_mae, train_mse, train_r2],
    'Validation': [val_rmse, val_mae, val_mse, val_r2]
    })
    return metrics_df

def add_targets_to_df(df,rcols, ycols,label):
    for rcol, ycol in zip(rcols, ycols):
        #print(rcol,ycol)
        df[ycol] = df[rcol].subtract(df[label])
    return df


def train_ctb_baseline(prt_files, parameters, model_dir, Xgrid, mode='dev'):
    ti = time.perf_counter()
    
    # Extract parameters
    config = parameters['config']
    mx = config['processing_device']
    fname = config['fname']
    roi = config['roi']
    rnd_seed = config['rnd_seed']
    catboost_params = parameters['catboost_params']
    nboost = catboost_params['iterations']
    ycols = parameters['columns']['ycols']
    rcols = parameters['columns']['rcols']
    label = parameters['columns']['label']
    xcols = parameters['columns']['xcols']
    numF = len(xcols)
    MLCOLS = xcols + rcols + [label]

    print('Loading datasets...')
    df = load_dataset(prt_files, mode, MLCOLS)
    if df is None:
        return
    
    df = add_targets_to_df(df, rcols, ycols, label)
    dtrain, dvalid = train_test_split(df, test_size=0.15, random_state=42)
    numR = len(dtrain)

    print(f'Training @ {str(Xgrid)}...')
    for ycol in ycols:
        print(f'params @ {str(Xgrid)} ##{ycol}...nboost={nboost}')
        wdir_roi = join(model_dir, str(Xgrid), roi)
        os.makedirs(wdir_roi, exist_ok=True)
        
        outname = create_outname(fname, roi, nboost, ycol, numR, numF, rnd_seed, mode, mx)
        modelpath = join(wdir_roi, f'CTB_{outname}.cbm')

        if os.path.isfile(modelpath):
            print(f'Model already exists: {modelpath}')
            continue
        
        print(f'Training {outname}...')
        dvalid, dtrain = clean_data(dvalid, dtrain, ycol)
        
        X_train, y_train = dtrain.drop(ycol, axis=1), dtrain[ycol]
        X_vali, y_vali = dvalid.drop(ycol, axis=1), dvalid[ycol]

        train_data = Pool(X_train, label=y_train)
        val_data = Pool(X_vali, label=y_vali)
        
        model = train_model(train_data, val_data, catboost_params)
        save_model_and_metrics(model, modelpath, wdir_roi, outname, X_train, y_train, X_vali, y_vali, parameters)

    print_summary(ti, xcols, rcols, ycols)

def load_dataset(prt_files, mode, columns):
    if mode == 'dev':
        return pd.read_parquet(prt_files[0], columns=columns).astype('float32')
    elif mode == 'dep':
        return pd.read_parquet(prt_files, columns=columns).astype('float32')
    else:
        print('Modes available are dev and dep')
        return None

def create_outname(fname, roi, nboost, ycol, numR, numF, rnd_seed, mode, mx):
    return f'{fname}_{roi}_{nboost}_{ycol}_{numR}_{numF}_{rnd_seed}_{mode}_{mx}'

def clean_data(dvalid, dtrain, ycol):
    return dvalid.dropna(subset=[ycol]), dtrain.dropna(subset=[ycol])

def train_model(train_data, val_data, catboost_params):
    model = CatBoostRegressor(**catboost_params)
    model.fit(train_data, eval_set=val_data, verbose=100)
    return model

def save_model_and_metrics(model, modelpath, wdir_roi, outname, X_train, y_train, X_vali, y_vali, parameters):
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_vali)
    metrics_df = performance(y_train, y_train_pred, y_vali, y_val_pred)
    
    model.save_model(modelpath)
    pickle_write(model, join(wdir_roi, f'CTB_{outname}.pkl'))
    pickle_write(parameters, join(wdir_roi, f'PARAMS_{outname}.pkl'))
    metrics_df.to_csv(join(wdir_roi, f'{outname}.csv'))

def print_summary(ti, xcols, rcols, ycols):
    tf = time.perf_counter() - ti
    print('=' * 30)
    print(f'Run time = {tf / 60:.2f} min(s)')
    print('xcols', xcols)
    print('rcols', rcols)
    print('ycols', ycols)



# version 2: S1,S1, AUX
def define_parameters():
 
    columns = {
        'xcols': [ 'egm08', 'egm96', 'vv', 'vh', 'red', 'green','blue','hem'],
        'rcols': ['tdem','edemg','edemw'],
        'ycols': ['tdif','egdif','ewdif'],
        'label': 'lidar'
    }

    config = {
        'roi': 'MKD',  # TLS, MDT, RNG, ALL, ALL-r=[call all]
        'processing_device': 'gpu',  # 'gpu' or 'cpu',
        'rnd_seed': 123,
        'fname':'all'  # variables used for ML
    }

    # CatBoost parameters
    catboost_params = {
        'iterations': 5000,#100,1000,5000
        #'depth': 16,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 123,
        'od_type': 'Iter',
        'od_wait': 100,
        'task_type': 'GPU',  # Use 'GPU' for GPU, 'CPU' for CPU
        'devices': '0:1'  # Specify GPU devices if needed
    }

    # Adjust iterations based on od_wait
    catboost_params['iterations'] += catboost_params['od_wait']

    # Combine all settings into a single dictionary
    parameters = {
        'config': config,
        'columns': columns,
        'catboost_params': catboost_params
    }

    return parameters

ti = time.perf_counter()

Xgrid_list = [1000,500,90,30]# 12 too 
#Xgrid = 500  
parameters = define_parameters()
for Xgrid in Xgrid_list:
    prt_pattern = f"{RESAMPLE_TILES_DPATH}{Xgrid}/*/*_byldem.parquet"
    prt_files = glob(prt_pattern)
    print(len(prt_files))
    pprint(prt_files)
    train_ctb_baseline(prt_files, parameters, RESAMPLE_MODELS_PATH, Xgrid, mode = 'dep')


tf = time.perf_counter() - ti 
print('='*60)
print(f'Run time = {tf/60:.2f} min(s)')