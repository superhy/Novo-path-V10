'''
@author: Yang Hu
'''
import os
import shutil


def move_file(src_path, dst_path, mode='move'):
    """
    move single file from src_path to dst_dir
    
    Args -> mode: 
        'move' -> move the file
        'copy' or other string -> copy the file (for test)
    """
    if mode == 'move':
        shutil.move(src_path, dst_path)
    else:
        shutil.copy(src_path, dst_path)
        
        
def clear_dir(abandon_dirs):
    """
    remove all the files in one folder from disk
    """
    print('remove the old dirs: {}.'.format(abandon_dirs))
    for dir in abandon_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            
def parse_filesystem_slide(slide_dir):
    """
    find all slides in one folder (include sub-folder)
    
    Return:
        slide_path_list: a list of file paths for all slides in the target folder
    """
    slide_path_list = []
    for root, dirs, files in os.walk(slide_dir):
        for f in files:
            if f.endswith('.svs') or f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.ndpi'):
                slide_path = os.path.join(root, f)
                print('found slide file from original folder: ' + slide_path)
                slide_path_list.append(slide_path)
                
    return slide_path_list

def parse_23910_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for 23910 cohort
    
    Args:
        slide_filepath: as name
        cut_range: filepath string cut range to get the TCGA case_id
    """
    case_id = slide_filepath[slide_filepath.find('_Sl') + 1: slide_filepath.find('.ndpi')]
    return case_id

def parse_visit_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath, for two visit cohorts
    
    Args:
        slide_filepath: as name
        cut_range: filepath string cut range to get the TCGA case_id
    """
    case_id = slide_filepath[slide_filepath.find('_images') + 8: slide_filepath.find('_V')]
    return case_id

def parse_slide_caseid_from_filepath(slide_filepath):
    """
    get the case id from slide's filepath
    
    Args:
        slide_filepath: as name
    """
    if slide_filepath.find('23910') != -1:
        case_id = parse_23910_slide_caseid_from_filepath(slide_filepath)
    elif slide_filepath.find('VISIT') != -1 or slide_filepath.find('Visit') != -1:
        case_id = parse_visit_slide_caseid_from_filepath(slide_filepath)
    else:
        raise NameError('cannot detect right dataset indicator!')
    
    return case_id

def move_he_slide_dir_2_dir(source_dir, target_dir, mode='move', 
                            sp_pattern=None, filter_annotated_slide=True):
    """
    """
    label_dict = None
    
    if label_dict is None or len(label_dict) == 0:
        filter_annotated_slide = False
        
    slide_path_list = parse_filesystem_slide(source_dir)
    ''' filtering the slide paths, with 
    '''
    # TODO:

if __name__ == '__main__':
    # test_s_filepath = 'D:/FLINC_dataset/transfer/23910-157_part/23910-157_Sl001.ndpi'
    # case_id = parse_23910_slide_caseid_from_filepath(test_s_filepath)
    # print(case_id)
    
    pass






