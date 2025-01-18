import os
import sys
import time

# Doing this before importing gpt_2_simple so it never displays tf2 behaviour 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import gpt_2_simple as gpt2
from fine_tuning_gpt2.query_splitter import split_data

def download_pretrained_model(name, force=False):
    if force or not os.path.isdir(f'./pre_trained_models/{name}'):
        gpt2.download_gpt2(
            model_dir='./pre_trained_models',
            model_name=name
        )

def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

# @timer_func
def fine_tuning(model_name, data_path, checkpoint_dir_name):
    file_name = data_path
    sess = gpt2.start_tf_sess()
    gpt2.finetune(sess,
                  dataset=file_name,
                  steps=-1,  # steps is
                  model_name=model_name,  # "355M" "774M" "128M"
                  #model_dir="models",
                  model_dir='./pre_trained_models',
                  combine=50000,
                  batch_size=1,
                  learning_rate=0.00001,
                  accumulate_gradients=5,
                  restore_from='latest',
                  run_name='run1',
                  checkpoint_dir=checkpoint_dir_name,
                  sample_every=100,
                  sample_length=1023,
                  sample_num=1,
                  #multi_gpu=False,
                  multi_gpu=True,
                  save_every=5000,
                  print_every=50,
                  max_checkpoints=50,
                  use_memory_saving_gradients=False,
                  only_train_transformer_layers=False,
                  optimizer='adam',
                  overwrite=False,
                  reuse=False)

def conversation(sess, checkpoint, runname, length):
    while True:
        inp = ""
        ques = input("Question : ")
        inp = '[YOU] : '+ques+'\n'+'[BOT] :'
        x = gpt2.generate(sess,
                          length=length,
                          temperature=0.6,
                          include_prefix=False,
                          run_name=runname,
                          checkpoint_dir=checkpoint,
                          prefix=inp,
                          nsamples=1,
                          return_as_list=True)[0]
        anwser = x.split('\n')
        print(anwser[1])

if __name__ == '__main__':
    
    orig_dataset_size = next((int(arg) for i, arg in enumerate(sys.argv) if i > 0 and sys.argv[i-1] == '--tr-size' and arg.isdigit()), 10000)
    orig_dataset_path = ""
    training_dir = ""
    question_dataset_file_name = f"data_for_question_model_dataset_{orig_dataset_size}.txt"
    question_dataset_path = os.path.join(training_dir, question_dataset_file_name)
    query_dataset_file_name = f"data_for_query_model_dataset_{orig_dataset_size}.txt"
    query_dataset_path = os.path.join(training_dir, query_dataset_file_name)
    if '--force-split' in sys.argv or not os.path.isfile(question_dataset_path) or not os.path.isfile(query_dataset_path):
        split_data(orig_dataset_path, training_dir, question_dataset_file_name, query_dataset_file_name)

    # Default: GPT2-small for question generation
    # Options for model_name: "117M" (small)  "355M" (medium) "774M" (large) "1558M" (extra-large) "128M" 
    model_name = "117M"
    tr_dataset_path = question_dataset_path
    model_size_suffix = 's'
    model_purpose_suffix = 'question'

    if '-s' in sys.argv:
        model_name = "117M"
        model_size_suffix = 's'

    if '-m' in sys.argv:
        model_name = "355M"
        model_size_suffix = 'm'
    
    if '-l' in sys.argv:
        model_name = "774M"
        model_size_suffix = 'l'
    
    if '-qs' in sys.argv:
        tr_dataset_path = question_dataset_path
        model_purpose_suffix = 'question'
    
    if '-qry' in sys.argv:
        tr_dataset_path = query_dataset_path
        model_purpose_suffix = 'query'
    
    # Download pre-trained gpt2 model if needed
    download_pretrained_model(model_name, force=('--download-pre-trained' in sys.argv))

    checkpoint_path = f"./gpt2_models/gpt2_{model_size_suffix}_{model_purpose_suffix}"
    fine_tuning(model_name, tr_dataset_path, checkpoint_path)
    
    # Test conversation
    """length = 100
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, checkpoint_dir='checkpoint_v3_1M', run_name='run1')
    conversation(sess, checkpoint_dir,
                 run_name, length)"""
