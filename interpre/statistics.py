'''
Created on 25 Jun 2023

@author: Yang
'''
import warnings
import numpy as np
import pandas as pd

from support.files import parse_23910_clinicalid_from_slideid
from support.metadata import parse_flinc_clinical_elsx


def df_cd45_cg_tis_pct_fib_score_corr(ENV_task, slide_tis_pct_dict):
    '''
    count the correlation between tissue percentage and fibrosis score
    tissue percentage was checked for iso/gath tiles in specific cluster and group
    
    return a dataFrame
    '''
    
    # clinical metadata file name
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    clinical_fib_dict = clinical_label_dicts['fibrosis_score']
    
    tis_pct_fib_tuples = []
    for slide_id in slide_tis_pct_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        tis_pct_dict = slide_tis_pct_dict[slide_id]
        if clinical_id.startswith('HV') is False and clinical_id not in clinical_fib_dict.keys():
            warnings.warn('there is no such slide in the record!')
            continue
        
        fib_label = ''
        if clinical_id.startswith('HV'):
            fib_label = 'HV'
        else:
            fib_label = 'fib-' + str(clinical_fib_dict[clinical_id])
            
        tis_pct_fib_tuples.append([fib_label, tis_pct_dict[0], 'iso tiles']) 
        tis_pct_fib_tuples.append([fib_label, tis_pct_dict[1], 'gath tiles'])
        
    df_tis_pct_fib_elemts = pd.DataFrame(tis_pct_fib_tuples,
                                         columns=['fibrosis_score', 'tissue_percentage', 'groups'])
    return df_tis_pct_fib_elemts

def df_cd45_cg_tis_pct_3a_score_corr(ENV_task, slide_tis_pct_dict):
    '''
    count the correlation between tissue percentage and alt, ast, alp
    tissue percentage was checked for iso/gath tiles in specific cluster and group
    
    return a dataFrame
    '''
    
    # clinical metadata file name
    cmeta_name = 'FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'
    xlsx_path_clinical = '{}/{}'.format(ENV_task.META_FOLDER, cmeta_name)
    ''' clinical_label_dicts['col_label'][clinical_id] = score value / stage rank, etc. '''
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical, aim_columns=['alt', 'ast', 'alp'])
    clinical_alt_dict = clinical_label_dicts['alt']
    clinical_ast_dict = clinical_label_dicts['ast']
    clinical_alp_dict = clinical_label_dicts['alp']
    
    tis_pct_alt_tuples = []
    tis_pct_ast_tuples = []
    tis_pct_alp_tuples = []
    for slide_id in slide_tis_pct_dict.keys():
        clinical_id = parse_23910_clinicalid_from_slideid(slide_id)
        tis_pct_dict = slide_tis_pct_dict[slide_id]
        if clinical_id.startswith('HV'):
            continue
        
        alt_score = clinical_alt_dict[clinical_id] if clinical_id in clinical_alt_dict.keys() else None
        ast_score = clinical_ast_dict[clinical_id] if clinical_id in clinical_ast_dict.keys() else None
        alp_score = clinical_alp_dict[clinical_id] if clinical_id in clinical_alp_dict.keys() else None
            
        if alt_score is not None:       
            tis_pct_alt_tuples.append([alt_score, tis_pct_dict[0], 'iso tiles']) 
            tis_pct_alt_tuples.append([alt_score, tis_pct_dict[1], 'gath tiles'])
        if ast_score is not None:
            tis_pct_ast_tuples.append([ast_score, tis_pct_dict[0], 'iso tiles'])
            tis_pct_ast_tuples.append([ast_score, tis_pct_dict[1], 'gath tiles'])
        if alp_score is not None:       
            tis_pct_alp_tuples.append([alp_score, tis_pct_dict[0], 'iso tiles']) 
            tis_pct_alp_tuples.append([alp_score, tis_pct_dict[1], 'gath tiles'])
        
        
    df_tis_pct_alt_elemts = pd.DataFrame(tis_pct_alt_tuples,
                                         columns=['alt_score', 'tissue_percentage', 'groups'])
    df_tis_pct_ast_elemts = pd.DataFrame(tis_pct_ast_tuples,
                                         columns=['ast_score', 'tissue_percentage', 'groups'])
    df_tis_pct_alp_elemts = pd.DataFrame(tis_pct_alp_tuples,
                                         columns=['alp_score', 'tissue_percentage', 'groups'])
    return df_tis_pct_alt_elemts, df_tis_pct_ast_elemts, df_tis_pct_alp_elemts
    

if __name__ == '__main__':
    pass