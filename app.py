import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import time, os, sys
import argparse
import json;

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_test, directories

import os
from flask import Flask, render_template,jsonify, request,send_from_directory,flash,redirect, url_for
from werkzeug import secure_filename
from pathlib import Path
import hashlib
import sys
tf.logging.set_verbosity(tf.logging.ERROR)

UPLOAD_FOLDER = 'upload_input/'
OUTPUT_FOLDER = 'output/final.png'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

def single_compress(config, args):
    start = time.time()
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)
    assert (ckpt.model_checkpoint_path), 'Missing checkpoint file!'

    if config.use_conditional_GAN:
        print('Using conditional GAN')
        paths, semantic_map_paths = np.array([args.image_path]), np.array([args.semantic_map_path])
    else:
        paths = np.array([args.image_path])

    gan = Model(config, paths, name='single_compress', dataset=args.dataset, evaluate=True)
    saver = tf.train.Saver()

    if config.use_conditional_GAN:
        feed_dict_init = {gan.path_placeholder: paths,
                          gan.semantic_map_path_placeholder: semantic_map_paths}
    else:
        feed_dict_init = {gan.path_placeholder: paths}

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        # Initialize variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        handle = sess.run(gan.train_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Most recent {} restored.'.format(ckpt.model_checkpoint_path))
        else:
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('Previous checkpoint {} restored.'.format(args.restore_path))

        sess.run(gan.train_iterator.initializer, feed_dict=feed_dict_init)
        eval_dict = {gan.training_phase: False, gan.handle: handle}

        if args.output_path is None:
            output = os.path.splitext(os.path.basename(args.image_path))
            save_path = os.path.join(directories.samples, '{}_compressed.pdf'.format(output[0]))
        else:
            save_path = args.output_path
        Utils.single_plot(0, 0, sess, gan, handle, save_path, config, single_compress=True)
        print('Reconstruction saved to', save_path)
    return

@app.route('/output')
def done1():
    return render_template('output.html');

@app.route('/contact')
def done2():
    return render_template('contact.html');


@app.route('/')
def home():
    return render_template('upload.html');

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class Complex:
    def __init__(self,image_path,output_path,dataset,restore_path,restore_last):
        self.image_path=image_path;
        self.output_path=output_path;
        self.dataset=dataset;
        self.restore_path=restore_path
        self.restore_last=restore_last        

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_hash = hashlib.md5(file.read()).hexdigest()
        print("image id: ", file_hash)
        
        save_path1 = os.path.join(app.config['UPLOAD_FOLDER'], file_hash + '.png')
        file.seek(0)
        file.save(save_path1)
        args=Complex(save_path1,OUTPUT_FOLDER,"cityscapes","path to model to be restored","restore last saved model")
        single_compress(config_test,args);
        return redirect(url_for('done1'))
		
if __name__ == '__main__':
   app.run()
   


