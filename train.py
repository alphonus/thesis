#!/usr/bin/python
# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys, os, time, random, argparse, timeit
import tensorflow as tf
import numpy as np
from itertools import islice
from functools import reduce

from model import build_network
from instance_loader import InstanceLoader
from util import load_weights, save_weights
from tabucol import tabucol, test


def run_training_batch(sess, model, batch, batch_i, epoch_i, time_steps, d, verbose=True):

    M, C, VC, cn_exists, n_vertices, n_edges, f = batch
    #Generate colors embeddings
    ncolors = np.sum(C)
    #We define the colors embeddings outside, randomly. They are not learnt by the GNN (that can be improved)
    colors_initial_embeddings = np.random.rand(ncolors,d)
    
    # Define feed dict
    feed_dict = {
        model['M']: M,
        model['VC']: VC,
        model['chrom_number']: C,
        model['time_steps']: time_steps,
        model['cn_exists']: cn_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges,
        model['colors_initial_embeddings']: colors_initial_embeddings
    }

    outputs = [model['train_step'], model['loss'], model['acc'], model['predictions'], model['TP'], model['FP'], model['TN'], model['FN']]
    

    # Run model
    loss, acc, predictions, TP, FP, TN, FN = sess.run(outputs, feed_dict = feed_dict)[-7:]

    if verbose:
        # Print stats
        print('{train_or_test} Epoch {epoch_i} Batch {batch_i}\t|\t(n,m,batch size)=({n},{m},{batch_size})\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Prediction)=({avg_sat:.4f},{avg_pred:.4f})'.format(
            train_or_test = 'Train',
            epoch_i = epoch_i,
            batch_i = batch_i,
            loss = loss,
            acc = acc,
            n = np.sum(n_vertices),
            m = np.sum(n_edges),
            batch_size = n_vertices.shape[0],
            avg_sat = np.mean(cn_exists),
            avg_pred = np.mean(np.round(predictions))
            ),
            flush = True
        )
    #end
    return loss, acc, np.mean(cn_exists), np.mean(predictions), TP, FP, TN, FN
#end


def run_test_batch(sess, model, batch, batch_i, time_steps, logfile, runtabu=True):

    M, n_colors, VC, cn_exists, n_vertices, n_edges, f = batch
    
    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    #open up the batch, which contains 2 instances
    for i in range(n_problems):
      n, m, c = n_vertices[i], n_edges[i], n_colors[i]
      conn = m / n
      n_acc = sum(n_vertices[0:i])
      c_acc = sum(n_colors[0:i])
      
      
      #subset adjacency matrix
      M_t = M[n_acc:n_acc+n, n_acc:n_acc+n]
      c = c if i % 2 == 0 else c + 1
      
      gnnpred = tabupred = 999
      for j in range(2, c + 5):
        n_colors_t = j
        cn_exists_t = 1 if n_colors_t >= c else 0
        VC_t = np.ones( (n,n_colors_t) )
        #Generate colors embeddings
        colors_initial_embeddings = np.random.rand(n_colors_t,d)
        
        feed_dict = {
            model['M']: M_t,
            model['VC']: VC_t,
            model['chrom_number']: np.array([n_colors_t]),
            model['time_steps']: time_steps,
            model['cn_exists']: np.array([cn_exists_t]),
            model['n_vertices']: np.array([n]),
            model['n_edges']: np.array([m]),
            model['colors_initial_embeddings']: colors_initial_embeddings
        }

        outputs = [model['loss'], model['acc'], model['predictions'], model['TP'], model['FP'], model['TN'], model['FN'] ]
        
        # Run model - chromatic number or more
        init_time = timeit.default_timer()
        loss, acc, predictions, TP, FP, TN, FN = sess.run(outputs, feed_dict = feed_dict)[-7:]
        elapsed_gnn_time  = timeit.default_timer() - init_time
        gnnpred = n_colors_t if predictions > 0.5 and n_colors_t < gnnpred else gnnpred
        
        # run tabucol
        if runtabu:
          init_time = timeit.default_timer()
          tabu_solution = tabucol(M_t, n_colors_t, max_iterations=1000)
          elapsed_tabu_time  = timeit.default_timer() - init_time
          tabu_sol = 0 if tabu_solution is None else 1
          tabupred = n_colors_t if tabu_sol == 1 and n_colors_t < tabupred else tabupred
      #end for
      logfile.write('{batch_i} {i} {n} {m} {conn} {tstloss} {tstacc} {cn_exists} {c} {gnnpred} {prediction} {gnntime} {tabupred} {tabutime}\n'.format(
        batch_i = batch_i,
        i = i,
        n= n,
        m = m,
        c = c,
        conn = conn,
        cn_exists = cn_exists_t,
        tstloss = loss,
        tstacc = acc,
        gnnpred = gnnpred, 
        prediction = predictions.item(),
        gnntime = elapsed_gnn_time,
        tabupred = tabupred if runtabu else 0,
        tabutime = elapsed_tabu_time if runtabu else 0
        )
      )
      logfile.flush()
    #end for batch
#end

def summarize_epoch(epoch_i, loss, acc, sat, pred, train=False):
    print('{train_or_test} Epoch {epoch_i} Average\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Pred)=({avg_sat:.4f},{avg_pred:.4f})'.format(
        train_or_test = 'Train' if train else 'Test',
        epoch_i = epoch_i,
        loss = np.mean(loss),
        acc = np.mean(acc),
        avg_sat = np.mean(sat),
        avg_pred = np.mean(pred)
        ),
        flush = True
    )
#end






from dataclasses import dataclass

class Config(dataclass):
    embed_size: int = 64
    timesteps: int = 32
    epochs: int = 10000
    batchsize: int = 8
    path: str = 'adversarial-training'
    seed: int = 42

def GNN_GCP(graph, n_colors, config:Config):
    """
    algorithm according to the model
    """

    # compute binary adjacency matrix from vertex to vertex

    #compute binary adjacency matrix from vertex to colors

    #compute initial vertex embeddings

    #compute initial color embedings

    #run T_MAX massage-passing iterations
    for t in range(config.timesteps):
        #Refine each vertex embedding with messages received from its neighbours and candidate colours

        #Refine each colour embedding with messages recieved from all vertices

        #translate vertex embeddings into logit probabilities

        #average logits and translate to probability
        pass