import os
def mkdir(e, r, m, n):
    p=os.path.join(e, r, m, n)
    if not os.path.isdir(p):
        os.makedirs(p)

def create_dirs(exp_name, model_name):
    mkdir(exp_name, "Translation", model_name, 'translation_train_visual')
    mkdir(exp_name, "Translation", model_name, 'translation_checkpoints')