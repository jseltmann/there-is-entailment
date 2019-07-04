import pandas as pd
import numpy as np

import configparser
import os
import random
from textwrap import fill
import sys
from copy import deepcopy
import pickle
import random

config_path = os.environ.get('VISCONF')
if not config_path:
    # try default location, if not in environment
    default_path_to_config = '../../clp-vision/Config/gpu_potsdam.cfg'
    if os.path.isfile(default_path_to_config):
        config_path = default_path_to_config

assert config_path is not None, 'You need to specify the path to the config file via environment variable VISCONF.'        

config = configparser.ConfigParser()
with open(config_path, 'r', encoding='utf-8') as f:
    config.read_file(f)

corpora_base = config.get('DEFAULT', 'corpora_base')
preproc_path = config.get('DSGV-PATHS', 'preproc_path')
dsgv_home = config.get('DSGV-PATHS', 'dsgv_home')

sys.path.append(dsgv_home + '/Utils')
from utils import icorpus_code, plot_labelled_bb, get_image_filename, query_by_id
from utils import plot_img_cropped, plot_img_ax, invert_dict, get_a_by_b
sys.path.append(dsgv_home + '/WACs/WAC_Utils')
from wac_utils import create_word2den, is_relational
sys.path.append(dsgv_home + '/Preproc')
from sim_preproc import load_imsim, n_most_sim

sys.path.append('../Common')
from data_utils import load_dfs, plot_rel_by_relid, get_obj_bb, compute_distance_objs
from data_utils import get_obj_key, compute_relpos_relargs_row, get_all_predicate
from data_utils import compute_distance_relargs_row, get_rel_type, get_rel_instances
from data_utils import compute_obj_sizes_row

df_names = ['mscoco_bbdf', 'refcoco_refdf', 'refcocoplus_refdf', 'grex_refdf',
            'vgregdf', 'vgimgdf', 'vgobjdf', 'vgreldf',
            'vgpardf', 'cococapdf']

df = load_dfs(preproc_path, df_names)

df['vgpregdf'] = df['vgregdf'][df['vgregdf']['pphrase'].notnull() & 
                               (df['vgregdf']['pphrase'] != '')]
coco_sem_sim, coco_sem_map = load_imsim(os.path.join(preproc_path, 'mscoco_sim.npz'))
visg_sem_sim, visg_sem_map = load_imsim(os.path.join(preproc_path, 'visgen_sim.npz'))
coco_id2semsim = invert_dict(coco_sem_map)
visg_id2semsim = invert_dict(visg_sem_map)

coco_vis_sim, coco_vis_map = load_imsim(os.path.join(preproc_path, 'mscoco_vis_sim.npz'))
visg_vis_sim, visg_vis_map = load_imsim(os.path.join(preproc_path, 'visgen_vis_sim.npz'))
coco_id2vissim = invert_dict(coco_vis_map)
visg_id2vissim = invert_dict(visg_vis_map)

# intersecting visual genome and coco captions. Slow-ish.
caption_coco_iids = list(set(df['cococapdf']['image_id'].tolist()))
# regions for only those image for which we also have coco captions
visgencocap_regdf = df['vgregdf'].merge(pd.DataFrame(caption_coco_iids, columns=['coco_id']))
# coco_image_ids for images with both caption and region
vgcap_coco_iids = list(set(visgencocap_regdf['coco_id'].tolist()))
# visgen_image_ids for images with both caption and region
vgcap_vg_iids = list(set(visgencocap_regdf['image_id'].tolist()))

# map coco_ids to visgen_ids, and back
coco2vg = dict(visgencocap_regdf[['coco_id', 'image_id']].values)
vg2coco = dict([(v,k) for k,v in coco2vg.items()])

df['vgpardf']['coco_image_id'] = df['vgpardf']['image_id'].apply(lambda x: vg2coco.get(x, None))
df['cocoparcapdf'] = df['cococapdf'].merge(df['vgpardf'],
                                           left_on='image_id', right_on='coco_image_id')

def create_binary_dataset(data_filename, sim_rank=0, train_split=[0.9,0.1,0.1]):
    """
    Create a datafile for the caption-to-there-is 
    entailment as binary classification.
    
    Parameters:
    -----------
    data_filename : str
        Filename to which to save the data.
    sim_rank : int
        Number of images from the vis_sim and sem_sim relations to ignore.
        E.g., if sim_rank is five, don't use objects from the five most similar
        images as negative examples.
    train_split : [float]
        Fractions of data to use for train, dev, and test set.
    """
    #pos_dict = dict()
    data_points = []
    #neg_dict = dict()
    ic = icorpus_code['mscoco']
    
    for _, row in visgencocap_regdf.iterrows():
        vgii = row['image_id']
        cocoii = row['coco_id']

        # get objects in image
        obj_syns = set(df['vgobjdf'][df['vgobjdf']['image_id'] == vgii]['syn'].tolist())
        if None in obj_syns:
            #if an object has no syn_id ignore this caption
            continue

        phyps = set(df['vgobjdf'][df['vgobjdf']['image_id'] == vgii]['name'].tolist())
        #pos_dict[vgii] = phyps
        data_points += [(vgii, phyp, True) for phyp in phyps]


        ignore_sem_ids = n_most_sim(visg_sem_sim, visg_sem_map,
                                    visg_id2semsim[vgii], n=sim_rank)
        ignore_ids = list(map(lambda x: visg_sem_map, ignore_sem_ids))
        ignore_vis_ids = n_most_sim(visg_vis_sim, visg_vis_map,
                                    visg_id2vissim[vgii], n=sim_rank)
        ignore_ids += list(map(lambda x: visg_vis_map, ignore_vis_ids))
        ignore_ids.append(vgii)

        
        neg_num = len(phyps)
        nhyps = set()
        while len(nhyps) < neg_num:
            neg_sample = df['vgobjdf'].sample()
            negii = neg_sample['image_id']
            if negii in ignore_ids:
                continue
            neg_obj = neg_sample['syn']
            if neg_obj in obj_syns:
                continue
            neg_name = neg_sample['name']
            nhyps.add((vgii, neg_name, False))


        #total_len = len(all_hyps)
        #train_len = int(train_split[0] * total_len)
        #dev_len = int(train_split[1] * total_len)
        plen = len(phyps)
        train_len = int(train_split[0] * plen)
        dev_len = int(train_split[1] * plen)

        train_hyps = phyps[:train_len] + nhyps[:train_len]
        dev_hyps = phyps[train_len:dev_len] + nhyps[train_len:dev_len]
        test_hyps = phyps[train_len+dev_len:] + nhyps[train_len+dev_len:]

        data = dict()
        data['train_hyps'] = train_hyps
        data['dev_hyps'] = dev_hyps
        data['test_hyps'] = test_hyps

        with open(data_filename, "wb") as data_file:
            pickle.dump(data, data_file)


        
            
        



