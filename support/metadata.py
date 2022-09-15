'''
@author: Yang Hu
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl

from env_flinc_he_stea import ENV_FLINC_HE_STEA
from env_flinc_he_fib import ENV_FLINC_HE_FIB
from env_flinc_p62_stea import ENV_FLINC_P62_STEA
from env_flinc_p62_fib import ENV_FLINC_P62_FIB


def parse_flinc_clinical_elsx(xlsx_filepath, aim_columns=['steatosis_score', 'lobular_inflammation_score',
                                                          'ballooning_score', 'fibrosis_score']):
    '''
    read the new excel file (.xlsx) by row
    parse
    '''
    wb = openpyxl.load_workbook(xlsx_filepath)
    sheet = wb['clinical_data']
    
    table_heads_dict = {}
    nb_column = sheet. max_column
    for col_id in range(nb_column):
        table_heads_dict[sheet.cell(row=1, column=col_id + 1).value] = col_id
    print('<Get table heads and their column id>: \n', table_heads_dict.items() )
    
    label_dicts = {}
    for key in aim_columns:
        label_dicts[key] = {}
    for i, row in enumerate(sheet.iter_rows()):
        if i == 0 or row[1].value is None:
            continue
        for key in aim_columns:
            if row[table_heads_dict[key]].value != 'NA':
                label_dicts[key][row[0].value] = int(row[table_heads_dict[key]].value)
    print('<Get label dicts>, like[]: \n', label_dicts)
    
    return label_dicts

def count_clinical_labels(label_dicts, aim_label_names=['steatosis_score']):
    '''
    statistic the distribution of some clinical label
    '''

    count_dicts = []
    for i, label_name in enumerate(aim_label_names):
        annotation_dict = label_dicts[label_name]
        
        count_dict = {}
        for _, value in annotation_dict.items():
            if value not in count_dict.keys():
                count_dict[value] = 0
            else:
                count_dict[value] += 1
        print('label distribution for {}:'.format(label_name), count_dict)
        count_dicts.append(count_dict)
        
    return count_dicts

def count_flinc_stain_labels(slide_label_dict_list, stain_type, aim_label_name):
    '''
    statistic the distribution of clinical label for slides after filtering the stain (like HE)
    '''
    
    label_stat = {}
    for i, label_dict in enumerate(slide_label_dict_list):
        label = list(label_dict.items())[1][1]
        if 'Score-{}'.format(str(label)) not in label_stat.keys():
            label_stat['Score-{}'.format(str(label))] = 0
        else:
            label_stat['Score-{}'.format(str(label))] += 1
            
    labels, counts = [], []
    for item in label_stat.items():
        labels.append(item[0])
        counts.append(item[1])
    print(counts)
    
    def absolute_value(val):
        a  = np.round(val/100.*np.array(counts).sum(), 0)
        return int(a)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.pie(counts, labels=labels, autopct=absolute_value)
    ax.set_title('{}-{}'.format(stain_type, aim_label_name))
    plt.tight_layout()
    plt.savefig('{}-{}.png'.format(stain_type, aim_label_name), bbox_inches='tight')
    

def make_flinc_slide_label(ENV_task, label_dicts, xlsx_filepath):
    '''
    filter the specific stain type, map the slide_id to specific label_name
    '''
    
    aim_label_name = ENV_task.TASK_NAME # aim_label_name is the task_name here
    stain_type = ENV_task.STAIN_TYPE 
    annotation_dict = label_dicts[aim_label_name]
    
    wb = openpyxl.load_workbook(xlsx_filepath)
    sheet = wb['SlideList']
     
    nb_column = sheet. max_column
    slide_column = None
    subject_id_column = None
    stain_column = None
    for col_id in range(nb_column):
        if sheet.cell(row=1, column=col_id + 1).value == 'Slide':
            slide_column = col_id
        elif sheet.cell(row=1, column=col_id + 1).value == 'SubjectID':
            subject_id_column = col_id
        elif sheet.cell(row=1, column=col_id + 1).value == 'Stain':
            stain_column = col_id
            
    if slide_column is None or subject_id_column is None or stain_column is None:
        warnings.warn('miss critical column...')
        return

    slide_label_dict_list = []
    for i, row in enumerate(sheet.iter_rows()):
        if i == 0:
            continue
        if row[stain_column].value == stain_type:
            slide_idstr = 'Sl%03d' % row[slide_column].value
            if row[subject_id_column].value in annotation_dict.keys():
                slide_aim_label = annotation_dict[row[subject_id_column].value]
                slide_label_dict_list.append({'slide_id': slide_idstr, aim_label_name: slide_aim_label})
    print('<Get slide:{}_label:{}_dict>, length:{}\n like: \n'.format(stain_type, aim_label_name, len(slide_label_dict_list)), slide_label_dict_list)
    
#     csv_test_path = 'D:/workspace/Liver-path-V10/data/FLINC/meta/HE_steatosis_score.csv'
    csv_test_path = '{}/{}_{}.csv'.format(ENV_task.META_FOLDER, ENV_task.STAIN_TYPE, ENV_task.TASK_NAME)
    csv_to_df = pd.DataFrame(slide_label_dict_list)
    csv_to_df.to_csv(csv_test_path, index=False)
    print('<Make csv annotations file at: {}>'.format(csv_test_path))
    
    return slide_label_dict_list
    
    
def _load_clinical_labels():
    
    TASK_ENVS = [ENV_FLINC_HE_STEA, ENV_FLINC_HE_FIB, ENV_FLINC_P62_STEA, ENV_FLINC_P62_FIB]
    
    xlsx_path_clinical = '{}/FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    _ = count_clinical_labels(clinical_label_dicts, aim_label_names=['steatosis_score'])
    
    xlsx_path_slide_1 = '{}/FLINC_23910-157_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_2 = '{}/FLINC_23910-158_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_list = [xlsx_path_slide_1, xlsx_path_slide_1, xlsx_path_slide_2, xlsx_path_slide_2]
    
    for i, task_env in enumerate(TASK_ENVS):
        slide_label_dict_list = make_flinc_slide_label(task_env, clinical_label_dicts,
                                                       xlsx_filepath=xlsx_path_slide_list[i])
        count_flinc_stain_labels(slide_label_dict_list, task_env.STAIN_TYPE, task_env.TASK_NAME)


if __name__ == '__main__':
    _load_clinical_labels()
    
#     print(ENV_FLINC_HE_STEA.PROJECT_NAME)
