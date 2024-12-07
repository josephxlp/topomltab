#step1 @tiliflist to parquet >> feedinto synthetic_tab
import os 
import glob
import numpy as np 
import pandas as pd 
from concurrent.futures import ProcessPoolExecutor
import rasterio
from upaths import RES_DPATH, tilenames_lidar
from upaths import var_ending_all, name_ending_all

def filter_files_by_endingwith(files, var_ending):
    filtered_files = [f for f in files if any(f.endswith(ending) for ending in var_ending)]
    print(f"Filtered files count: {len(filtered_files)}/{len(files)}")
    return filtered_files


def pathlist2df(vpaths):
    data_list = []

    for path in vpaths:
        with rasterio.open(path) as src:
            bname = str(os.path.basename(path)[:-4]).lower()
            data = src.read()  # Shape: (bands, height, width)
            #print(data.shape)
            for band_idx in range(data.shape[0]):
                flattened = data[band_idx].flatten().astype(np.float32)
                       
                if data.shape[0] > 1: 
                    bname = f"{bname}_B{band_idx+1}"
                
                data_list.append(pd.DataFrame({bname: flattened}))

    # Concatenate all DataFrames
    df = pd.concat(data_list, axis=1)
    return df

def tiflist2parquet(TILESX,tilename,name_ending_all):
    tile_dpath = f'{TILESX}/{tilename}'
    fparquet = f'{tile_dpath}/{tilename}_byldem.parquet'
    tile_files = glob.glob(f'{tile_dpath}/*.tif')
    tile_filex = filter_files_by_endingwith(tile_files, var_ending_all)

    if not os.path.isfile(fparquet):
        df = pathlist2df(tile_filex)
        df.columns = name_ending_all
        df = df.dropna(subset='lidar')
        df.to_parquet(fparquet)
    print('Alrady Created')



X = 30 
tilenames = tilenames_lidar
Xlist = [30,90,500,1000]
if __name__ == '__main__':
    # TILESX = f"{RES_DPATH}{X}"
    # with ProcessPoolExecutor(17) as PEX:
    #     for tilename in tilenames:
    #         #tiflist2parquet(TILESX,tilename,name_ending_all)
    #         PEX.submit(tiflist2parquet,TILESX,tilename,name_ending_all)
    for X in Xlist:
        TILESX = f"{RES_DPATH}{X}"
        with ProcessPoolExecutor(17) as PEX:
            for tilename in tilenames:
                #tiflist2parquet(TILESX,tilename,name_ending_all)
                PEX.submit(tiflist2parquet,TILESX,tilename,name_ending_all)


    print('Bring my TimerNotificationPackageHereToo')


