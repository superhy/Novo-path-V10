'''
@author: Yang Hu
'''
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import sys



'''
for different task id, with different meaning!

1: algorithm Try MIL
2: algorithm attention MIL
5. algorithm LCSB MIL
7. algorithm reLCSB MIL

For algorithm 1 (Try-MIL): <deprecated>
    10: topK MIL from Thomas J. Fuchs's paper
For algorithm 2 (attention-MIL):
    20: Attentional Pool MIL, encoder: ResNet18 <deprecated>
    21: Gated Attentional Pool MIL, encoder: ResNet18 <*>
    26: Gated Attentional Pool MIL, encoder: ViT-6-8 <*>
    27: Gated Attentional Pool MIL, encoder: ViT-9-12
    29: Gated Attentional Pool MIL, encoder: ViT-3-4-t
For algorithm 5 (LCSB MIL)
    50: LCSB with Attention Pool, encoder: ResNet18 <deprecated>
    51: LCSB with Gated Attention Pool, encoder: ResNet18 <*>
    56: LCSB with Gated Attention Pool, encoder: ViT-6-8 <*>
    57: LCSB with Gated Attention Pool, encoder: ViT-9-12
    59: LCSB with Gated Attention Pool, encoder: ViT-3-4-t
For algorithm 7 (reversed gradient LCSB MIL) <deprecated>
    70: reversed LCSB with Attention Pool, encoder: ResNet18
    71: reversed LCSB with Gated Attention Pool, encoder: ResNet18
    76: reversed LCSB with Gated Attention Pool, encoder: ViT-6-8
    77: reversed LCSB with Gated Attention Pool, encoder: ViT-9-12
    79: reversed LCSB with Gated Attention Pool, encoder: ViT-3-4-t
For algorithm 0 (pre-training of aggregator in MIL) <deprecated>
    01(1): pre-training the Attention Pool
    02(2): pre-training the Gated Attention Pool
    
For algorithm 8 (pre-training of encoder, usually ViT)
    80: self-supervised pre-train with ViT-6-8, by DINO
    81: self-supervised pre-train with ViT-6-8, by MAE <deprecated>
    82: self-supervised pre-train with ViT-9-12, by DINO
    89: self-supervised pre-train with ViT-3-4-t, by DINO
'''

'''
Experiments running process:

pre-processing part
1. task-1 in run_pre.py: copy slides from folder <transfer> to specific <tissues> folders (the folder we have full access to read and write)
2. task-2 metadata.py: produce annotations file (if necessary, combine annotations into groups)
3. task-3 in run_pre.py: split the train/test sets (in 5-folds for this moment) and generate the tiles-list pkl for them

training part:
3. running any task in above (except the deprecated)

testing part:
4. 
'''

'''
Experiments running steps, 2023.4.4:
    1. set ENV_task = env_flinc_cd45.ENV_FLINC_CD45_REG_PT, run run_pre.py in task_ids = [2], prepare the slide tiles 
    2. set ENV_task = ENV_FLINC_CD45_REG_PT, run run_main_82_5.py in task_ids = [82.5], dino pre-trian the reg_vit
    3. set ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U, run run_main_clst_12X.py in task_ids = [110], 
        perform the clustering with region context prior encoding
    PS: best to re-run run_pre.py for ENV_task = env_flinc_cd45.ENV_FLINC_CD45_U

'''


class Logger(object):

    def __init__(self, filename='running_log-default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    pass
    
    