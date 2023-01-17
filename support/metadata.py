'''
@author: Yang Hu
'''

import csv
import os
import warnings

import openpyxl

from support.env_flinc_he import ENV_FLINC_HE_FIB
from support.env_flinc_he import ENV_FLINC_HE_STEA, ENV_FLINC_HE_STEA_C2
from support.env_flinc_psr import ENV_FLINC_PSR_FIB
from support.env_flinc_psr import ENV_FLINC_PSR_FIB_C3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




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
    plt.close(fig)
    

def extract_slideid_subid_for_stain(xlsx_filepath, stain_type='HE'):
    '''
    extract the dict for {slide_id: subject_id} with specific stain type from xls metafile
    '''
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
    
    slideid_subid_dict = {}
    for i, row in enumerate(sheet.iter_rows()):
        if i == 0:
            continue
        if row[stain_column].value is None:
            continue
        if row[stain_column].value.find(stain_type) != -1:
            slide_id = 'Sl%03d' % row[slide_column].value
            subject_id = row[subject_id_column].value
            slideid_subid_dict[slide_id] = subject_id
    print('extracted %d slides\' subject_id' % len(slideid_subid_dict.keys()) )
    return slideid_subid_dict
            

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
        if row[stain_column].value is None:
            continue
        if row[stain_column].value.find(stain_type) != -1:
            # print(row[stain_column].value)
            slide_idstr = 'Sl%03d' % row[slide_column].value
            # print(row[subject_id_column].value, type(row[subject_id_column].value), type(row[subject_id_column].value) == str)
            if row[subject_id_column].value in annotation_dict.keys():
                slide_aim_label = annotation_dict[row[subject_id_column].value]
                slide_label_dict_list.append({'slide_id': slide_idstr, aim_label_name: slide_aim_label})
                print('loaded subject_id: {} with label: {}'.format(row[subject_id_column].value, slide_aim_label) )
            elif type(row[subject_id_column].value) == str and row[subject_id_column].value.startswith('HV'):
                slide_label_dict_list.append({'slide_id': slide_idstr, aim_label_name: 0})
                print('loaded subject_id: {} with label: {}'.format(row[subject_id_column].value, 0) )
                
    print('<Get slide:{}_label:{}_dict>, length:{}\n like: \n'.format(stain_type, aim_label_name, len(slide_label_dict_list)), slide_label_dict_list)
    
#     csv_test_path = 'D:/workspace/Liver-path-V10/data/FLINC/meta/HE_steatosis_score.csv'
    csv_test_path = '{}/{}_{}.csv'.format(ENV_task.META_FOLDER, ENV_task.STAIN_TYPE, ENV_task.TASK_NAME)
    csv_to_df = pd.DataFrame(slide_label_dict_list)
    csv_to_df.to_csv(csv_test_path, index=False)
    print('<Make csv annotations file at: {}>'.format(csv_test_path))
    
    return slide_label_dict_list

def combine_slide_labels_group_cx(slide_label_dict_list, groups):
    '''
    combine the labels into 3 groups, cx means c? (c2, c3, ...)
    '''
    group_label_dict, group_label_count = {}, {}
    for group_value in groups.keys():
        group_label_count[group_value] = 0
        for item in groups[group_value]: # the list of group_items
            group_label_dict[item] = group_value
    print('group mapping as: {}'.format(group_label_dict))
    
    new_slide_label_dict_list = []
    label_dist_titles = list(slide_label_dict_list[0].keys())
    label_name = label_dist_titles[1] if label_dist_titles[0] == 'slide_id' else label_dist_titles[0]
    for dict_item in slide_label_dict_list:
        org_label = int(dict_item[label_name])
        new_slide_label_dict_list.append({'slide_id': dict_item['slide_id'], label_name: group_label_dict[org_label]} )
        group_label_count[group_label_dict[org_label] ] += 1
    print('Combine the labels into groups, with new dict:')
    for group_label in group_label_count.keys():
        print('group label -> {}, number: {}'.format(group_label, group_label_count[group_label]) )
    return new_slide_label_dict_list

def count_flinc_stain_amount(ENV_task, xlsx_filepath):
    '''
    counting the number of slides for each staining type
    '''
    stain_type = ENV_task.STAIN_TYPE
    
    wb = openpyxl.load_workbook(xlsx_filepath)
    sheet = wb['SlideList']
     
    nb_column = sheet. max_column
    stain_column = None
    for col_id in range(nb_column):
        if sheet.cell(row=1, column=col_id + 1).value == 'Stain':
            stain_column = col_id
            
    if stain_column is None:
        warnings.warn('miss stain column...')
        return
    
    nb_stain_slides = 0
    for i, row in enumerate(sheet.iter_rows()):
        if i == 0:
            continue
        if row[stain_column].value is None:
            continue
        if row[stain_column].value.find(stain_type) != -1:
            nb_stain_slides += 1
            
    print('count stain: %s with slides: %d' % (ENV_task.STAIN_TYPE, nb_stain_slides))
    return nb_stain_slides
    
    
def _load_clinical_labels():
    
    TASK_ENVS = [ENV_FLINC_HE_STEA, ENV_FLINC_HE_FIB, ENV_FLINC_PSR_FIB]
    # TASK_ENVS = [ENV_FLINC_PSR_FIB]
    
    xlsx_path_clinical = '{}/FLINC_clinical_data_DBI_2022-0715_EDG.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    clinical_label_dicts = parse_flinc_clinical_elsx(xlsx_path_clinical)
    _ = count_clinical_labels(clinical_label_dicts, aim_label_names=['steatosis_score'])
    
    xlsx_path_slide_1 = '{}/FLINC_23910-157_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_2 = '{}/FLINC_23910-158_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_list = [xlsx_path_slide_1, xlsx_path_slide_1, xlsx_path_slide_1, xlsx_path_slide_2, xlsx_path_slide_2]
    # xlsx_path_slide_list = [xlsx_path_slide_1]
    
    for i, task_env in enumerate(TASK_ENVS):
        slide_label_dict_list = make_flinc_slide_label(task_env, clinical_label_dicts,
                                                       xlsx_filepath=xlsx_path_slide_list[i])
        count_flinc_stain_labels(slide_label_dict_list, task_env.STAIN_TYPE, task_env.TASK_NAME)
        
def _prod_combine_labels():
    '''
    produce the combined label csv file from ENV_task, for C_ENV_task
    '''
    ENV_task = ENV_FLINC_PSR_FIB
    C_ENV_task = ENV_FLINC_PSR_FIB_C3
    groups = {0: [0, 1], 1: [2], 2: [3, 4]}
    # ENV_task = ENV_FLINC_HE_STEA
    # C_ENV_task = ENV_FLINC_HE_STEA_C2
    # groups = {0: [0, 1], 1: [2, 3]}
    
    slide_label_dict_list = query_task_label_dict_list_fromcsv(ENV_task)
    new_slide_label_dict_list = combine_slide_labels_group_cx(slide_label_dict_list, groups=groups)
    
    csv_test_path = '{}/{}_{}.csv'.format(C_ENV_task.META_FOLDER, C_ENV_task.STAIN_TYPE, C_ENV_task.TASK_NAME)
    csv_to_df = pd.DataFrame(new_slide_label_dict_list)
    csv_to_df.to_csv(csv_test_path, index=False)
    print('<Make combined csv annotations file at: {}>'.format(csv_test_path))
    return new_slide_label_dict_list
        
def _count_stain_amount():
    
    TASK_ENVS = [ENV_FLINC_HE_STEA, ENV_FLINC_HE_FIB, ENV_FLINC_PSR_FIB]
    
    xlsx_path_slide_1 = '{}/FLINC_23910-157_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_2 = '{}/FLINC_23910-158_withSubjectID.xlsx'.format(TASK_ENVS[0].META_FOLDER)
    xlsx_path_slide_list = [xlsx_path_slide_1, xlsx_path_slide_1, xlsx_path_slide_1, xlsx_path_slide_2, xlsx_path_slide_2]
    
    for i, task_env in enumerate(TASK_ENVS):
        _ = count_flinc_stain_amount(task_env, xlsx_filepath=xlsx_path_slide_list[i])
        

def auto_find_task_csvname(ENV_task):
    '''
    auto detect the task_csv_filename
    '''
    f_start_string = '{}_{}'.format(ENV_task.STAIN_TYPE, ENV_task.TASK_NAME)
    for f in os.listdir(ENV_task.META_FOLDER):
        if f.startswith(f_start_string) and f.endswith('.csv'):
            print('automatically find CSV file: {}'.format(f))
            task_csv_filename = f
            break
    return task_csv_filename

def query_task_label_dict_list_fromcsv(ENV_task, task_csv_filename=None):
    '''
    load the task label dict list (to maintain the sort of keys) from csv
    '''
    if task_csv_filename == None:
        task_csv_filename = auto_find_task_csvname(ENV_task)
    
    task_csv_filepath = os.path.join(ENV_task.META_FOLDER, task_csv_filename)
    slide_label_dict_list = []
    column_1, column_2 = 'slide_id', ''
    with open(task_csv_filepath, 'r', newline='') as task_csv_file:
        csv_reader = csv.reader(task_csv_file)
        for l, csv_line in enumerate(csv_reader):
            if l == 0: # skip the first line for title
                column_1, column_2 = csv_line[0], csv_line[1]
            else:
                slide_label_dict_list.append({column_1: csv_line[0], column_2: csv_line[1]})
    return slide_label_dict_list
        
def query_task_label_dict_fromcsv(ENV_task, task_csv_filename=None):
    '''
    load the task label dict from csv
    '''
    if task_csv_filename == None:
        task_csv_filename = auto_find_task_csvname(ENV_task)
            
    task_csv_filepath = os.path.join(ENV_task.META_FOLDER, task_csv_filename)
    task_label_dict = {}
    with open(task_csv_filepath, 'r', newline='') as task_csv_file:
        csv_reader = csv.reader(task_csv_file)
        for l, csv_line in enumerate(csv_reader):
            if l == 0: # skip the first line for title
                continue
            task_label_dict[csv_line[0]] = int(csv_line[1])
            
    return task_label_dict


if __name__ == '__main__':
    # _load_clinical_labels()
    # _count_stain_amount()
    _prod_combine_labels()
    
#     print(ENV_FLINC_HE_STEA.PROJECT_NAME)
