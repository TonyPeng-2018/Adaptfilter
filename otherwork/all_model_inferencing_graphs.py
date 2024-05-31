# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 11:22:18 2023
@author: Hostl
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import rcParams
import tqdm,os

path = "B:\\summer_project\\pitt_embeddings\current_code_work\\ccpd_final_paper_fun\\"
def get_pickle(name):
    with open(name + ".pickle" , "rb") as f:
        temp = pickle.load(f)
    return temp

th = "thshold0.002_"
#mobile_c mobile_nc res50_c res50_nc
"""Load the ccpd results"""#not averaged 40k
ccpd_base_times  = get_pickle(path + th + "ccpd_base_times") 
ccpd_base_mse    = get_pickle(path + th + "ccpd_base_mse") 
ccpd_gated_times = get_pickle(path + th + "ccpd_gated_times") 
ccpd_gated_distb = get_pickle(path + th + "ccpd_gated_distb")
ccpd_gated_mse   = get_pickle(path + th + "ccpd_gated_mse2")
"""Load the mini results"""#not averaged 32k
mini_base_times  = get_pickle(path + th + "mini_base_times") 
mini_base_mse    = get_pickle(path + th + "mini_base_mse") 
mini_gated_times = get_pickle(path + th + "mini_gated_times") 
mini_gated_distb = get_pickle(path + th + "mini_gated_distb")
mini_gated_mse   = get_pickle(path + th + "mini_gated_mse2")



def get_baseline_mse_graphs():
    x_axis = [0.40,1,2,3,4,10,25,50,75,100]
    """ccpd mse graph whole"""
    mob_c    = list(ccpd_base_mse[0].copy()/40000)
    mob_nc   = list(ccpd_base_mse[1].copy()/40000)
    r50_c    = list(ccpd_base_mse[2].copy()/40000)
    r50_nc   = list(ccpd_base_mse[3].copy()/40000)
    mob_c.reverse()
    mob_nc.reverse()
    r50_c.reverse()
    r50_nc.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, mob_c, marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, mob_nc, marker='x', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, r50_c, marker='s', linestyle='--', label='resnet50-c')
    plt.plot(x_axis, r50_nc, marker='v', linestyle='--', label='resnet50-nc')
    plt.xlabel('(%)size', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True)
    plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
    plt.show()
    """mini mse graph whole"""
    mob_c    = list(mini_base_mse[0].copy()/32000)
    mob_nc   = list(mini_base_mse[1].copy()/32000)
    r50_c    = list(mini_base_mse[2].copy()/32000)
    r50_nc   = list(mini_base_mse[3].copy()/32000)
    mob_c.reverse()
    mob_nc.reverse()
    r50_c.reverse()
    r50_nc.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, mob_c, marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, mob_nc, marker='x', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, r50_c, marker='s', linestyle='--', label='resnet50-c')
    plt.plot(x_axis, r50_nc, marker='v', linestyle='--', label='resnet50-nc')
    plt.xlabel('(%)size', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True)
    plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
    plt.show()

def get_baseline_mse_graphs_0to6():
    x_axis = [0.40,1,2,3,4,10]#,25,50,75,100]
    """ccpd mse graph whole"""
    mob_c    = list(ccpd_base_mse[0].copy()/40000)
    mob_nc   = list(ccpd_base_mse[1].copy()/40000)
    r50_c    = list(ccpd_base_mse[2].copy()/40000)
    r50_nc   = list(ccpd_base_mse[3].copy()/40000)
    mob_c.reverse()
    mob_nc.reverse()
    r50_c.reverse()
    r50_nc.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, mob_c[0:6], marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, mob_nc[0:6], marker='x', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, r50_c[0:6], marker='s', linestyle='--', label='resnet50-c')
    plt.plot(x_axis, r50_nc[0:6], marker='v', linestyle='--', label='resnet50-nc')
    plt.xlabel('(%)size', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True)
    plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
    plt.show()
    """mini mse graph whole"""
    mob_c    = list(mini_base_mse[0].copy()/32000)
    mob_nc   = list(mini_base_mse[1].copy()/32000)
    r50_c    = list(mini_base_mse[2].copy()/32000)
    r50_nc   = list(mini_base_mse[3].copy()/32000)
    mob_c.reverse()
    mob_nc.reverse()
    r50_c.reverse()
    r50_nc.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, mob_c[0:6], marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, mob_nc[0:6], marker='x', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, r50_c[0:6], marker='s', linestyle='--', label='resnet50-c')
    plt.plot(x_axis, r50_nc[0:6], marker='v', linestyle='--', label='resnet50-nc')
    plt.xlabel('(%)size', fontsize=12)
    plt.ylabel('Mean Squared Error', fontsize=12)
    plt.grid(True)
    plt.legend(loc='center right')#, bbox_to_anchor=(1, 0.5))
    plt.show()
    
def get_baseline_mse_graphs_2():
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = list(ccpd_base_mse[0].copy()/40000)
    values2 = list(ccpd_base_mse[1].copy()/40000)
    values3 = list(ccpd_base_mse[2].copy()/40000)
    values4 = list(ccpd_base_mse[3].copy()/40000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='Mobilenet-c-gated')
    bar2 = ax.bar(x_base - width/2, values2, width, label='Mobilenet-nc-gated')
    bar3 = ax.bar(x_base + width/2, values3, width, label='resnet50-c-gated')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='resnet50-nc-gated')
    ax.set_xlabel("(%)Size")
    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.show()
    
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = list(mini_base_mse[0].copy()/40000)
    values2 = list(mini_base_mse[1].copy()/40000)
    values3 = list(mini_base_mse[2].copy()/40000)
    values4 = list(mini_base_mse[3].copy()/40000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='Mobilenet-c-gated')
    bar2 = ax.bar(x_base - width/2, values2, width, label='Mobilenet-nc-gated')
    bar3 = ax.bar(x_base + width/2, values3, width, label='resnet50-c-gated')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='resnet50-nc-gated')
    ax.set_xlabel("(%)Size")
    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.show()

def get_baseline_mse_graphs_0to6_2():
    categories = [0.4,1,2,3,4,10]#,25,50,75,100]
    values1 = list(ccpd_base_mse[0].copy()/40000)[0:6]
    values2 = list(ccpd_base_mse[1].copy()/40000)[0:6]
    values3 = list(ccpd_base_mse[2].copy()/40000)[0:6]
    values4 = list(ccpd_base_mse[3].copy()/40000)[0:6]
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='Mobilenet-c-gated')
    bar2 = ax.bar(x_base - width/2, values2, width, label='Mobilenet-nc-gated')
    bar3 = ax.bar(x_base + width/2, values3, width, label='resnet50-c-gated')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='resnet50-nc-gated')
    ax.set_xlabel("(%)Size")
    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.show()
    
    categories = [0.4,1,2,3,4,10]#,25,50,75,100]
    values1 = list(mini_base_mse[0].copy()/40000)[0:6]
    values2 = list(mini_base_mse[1].copy()/40000)[0:6]
    values3 = list(mini_base_mse[2].copy()/40000)[0:6]
    values4 = list(mini_base_mse[3].copy()/40000)[0:6]
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='Mobilenet-c-gated')
    bar2 = ax.bar(x_base - width/2, values2, width, label='Mobilenet-nc-gated')
    bar3 = ax.bar(x_base + width/2, values3, width, label='resnet50-c-gated')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='resnet50-nc-gated')
    ax.set_xlabel("(%)Size")
    ax.set_ylabel('Mean Squared Error')
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend()
    plt.show()
    
def get_gate_freq_graphs():
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = list(ccpd_gated_distb[0])#[0:6]
    values2 = list(ccpd_gated_distb[1])#[0:6]
    values3 = list(ccpd_gated_distb[2])#[0:6]
    values4 = list(ccpd_gated_distb[3])#[0:6]
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='AdaptFilter(M)-c')
    bar2 = ax.bar(x_base - width/2, values2, width, label='AdaptFilter(M)-nc')
    bar3 = ax.bar(x_base + width/2, values3, width, label='AdaptFilter(R)-c')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='AdaptFilter(R)-nc')
    ax.set_xlabel("Size %", fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=14)
    plt.show()
    
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = list(mini_gated_distb[0])#[0:6]
    values2 = list(mini_gated_distb[1])#[0:6]
    values3 = list(mini_gated_distb[2])#[0:6]
    values4 = list(mini_gated_distb[3])#[0:6]
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='AdaptFilter(M)-c')
    bar2 = ax.bar(x_base - width/2, values2, width, label='AdaptFilter(M)-nc')
    bar3 = ax.bar(x_base + width/2, values3, width, label='AdaptFilter(R)-c')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='AdaptFilter(R)-nc')
    ax.set_xlabel("Size %", fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=14)
    plt.show()

def get_gate_averages():
    categories       = np.array([0.4,1,2,3,4,10,25,50,75,100])
    ccpd_gated_avg_1 = sum(list(ccpd_gated_distb[0] * categories))/40000
    ccpd_gated_avg_2 = sum(list(ccpd_gated_distb[1] * categories))/40000
    ccpd_gated_avg_3 = sum(list(ccpd_gated_distb[2] * categories))/40000
    ccpd_gated_avg_4 = sum(list(ccpd_gated_distb[3] * categories))/40000
    
    mini_gated_avg_1 = sum(list(mini_gated_distb[0] * categories))/32000
    mini_gated_avg_2 = sum(list(mini_gated_distb[1] * categories))/32000
    mini_gated_avg_3 = sum(list(mini_gated_distb[2] * categories))/32000
    mini_gated_avg_4 = sum(list(mini_gated_distb[3] * categories))/32000
    
    ccpd1 = []
    ccpd2 = []
    ccpd3 = []
    ccpd4 = []
    ary = ccpd1
    llop= ccpd_gated_mse[0]
    llop2=ccpd_gated_distb[0]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = ccpd2
    llop= ccpd_gated_mse[1]
    llop2=ccpd_gated_distb[1]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = ccpd3
    llop= ccpd_gated_mse[2]
    llop2=ccpd_gated_distb[2]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = ccpd4
    llop= ccpd_gated_mse[3]
    llop2=ccpd_gated_distb[3]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    
    mini1 = []
    mini2 = []
    mini3 = []
    mini4 = []
    ary = mini1
    llop= mini_gated_mse[0]
    llop2=mini_gated_distb[0]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = mini2
    llop= mini_gated_mse[1]
    llop2=mini_gated_distb[1]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = mini3
    llop= mini_gated_mse[2]
    llop2=mini_gated_distb[2]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    ary = mini4
    llop= mini_gated_mse[3]
    llop2=mini_gated_distb[3]
    for i in range(len(llop)):
        if llop[i] == 0.0:
            ary.append(0.0)
        else:
            ary.append(llop[i]/llop2[i])
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = ccpd1
    values2 = ccpd2
    values3 = ccpd3
    values4 = ccpd4
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='AdaptFilter(M)-c')
    bar2 = ax.bar(x_base - width/2, values2, width, label='AdaptFilter(M)-nc')
    bar3 = ax.bar(x_base + width/2, values3, width, label='AdaptFilter(R)-c')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='AdaptFilter(R)-nc')
    ax.set_xlabel("Size %", fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=14)
    plt.show()
    
    categories = [0.4,1,2,3,4,10,25,50,75,100]
    values1 = mini1
    values2 = mini2
    values3 = mini3
    values4 = mini4
    x_base = np.arange(len(categories))
    width = 0.10
    fig, ax = plt.subplots()
    bar1 = ax.bar(x_base - width/2-width, values1, width, label='AdaptFilter(M)-c')
    bar2 = ax.bar(x_base - width/2, values2, width, label='AdaptFilter(M)-nc')
    bar3 = ax.bar(x_base + width/2, values3, width, label='AdaptFilter(R)-c')
    bar4 = ax.bar(x_base + width/2+width, values4, width, label='AdaptFilter(R)-nc')
    ax.set_xlabel("Size %", fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels(categories)
    plt.legend(loc='upper left',fontsize=11)
    plt.show()

def baseandgate_mse_graphs():    
    categories       = np.array([0.4,1,2,3,4,10,25,50,75,100])
    ccpd_gated_avg_1 = sum(list(ccpd_gated_distb[0] * categories))/40000
    ccpd_gated_avg_2 = sum(list(ccpd_gated_distb[1] * categories))/40000
    ccpd_gated_avg_3 = sum(list(ccpd_gated_distb[2] * categories))/40000
    ccpd_gated_avg_4 = sum(list(ccpd_gated_distb[3] * categories))/40000
    
    mini_gated_avg_1 = sum(list(mini_gated_distb[0] * categories))/32000
    mini_gated_avg_2 = sum(list(mini_gated_distb[1] * categories))/32000
    mini_gated_avg_3 = sum(list(mini_gated_distb[2] * categories))/32000
    mini_gated_avg_4 = sum(list(mini_gated_distb[3] * categories))/32000
    
    ccpd1 = []
    ccpd2 = []
    ccpd3 = []
    ccpd4 = []

    ccpd1 = sum(ccpd_gated_mse[0])/40000
    ccpd2 = sum(ccpd_gated_mse[1])/40000
    ccpd3 = sum(ccpd_gated_mse[2])/40000
    ccpd4 = sum(ccpd_gated_mse[3])/40000
    
    mini1 = sum(mini_gated_mse[0])/32000
    mini2 = sum(mini_gated_mse[1])/32000
    mini3 = sum(mini_gated_mse[2])/32000
    mini4 = sum(mini_gated_mse[3])/32000
    
    #plt.cla()
    x_axis = [0.40,1,2,3,4,10,25,50,75,100]
    """ccpd mse graph whole"""
    values1 = list(ccpd_base_mse[0].copy()/40000)
    values2 = list(ccpd_base_mse[1].copy()/40000)
    values3 = list(ccpd_base_mse[2].copy()/40000)
    values4 = list(ccpd_base_mse[3].copy()/40000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, values1, marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, values2, marker='<', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, values3, marker='s', linestyle='--', label='Resnet50-c')
    plt.plot(x_axis, values4, marker='v', linestyle='--', label='Resnet50-nc')
    plt.plot(ccpd_gated_avg_1,ccpd1, marker='o', linestyle='--', label='AdaptFilter(M)-c')
    plt.plot(ccpd_gated_avg_2,ccpd2, marker='<', linestyle='--', label='AdaptFilter(M)-nc')
    plt.plot(ccpd_gated_avg_3,ccpd3, marker='s', linestyle='--', label='AdaptFilter(R)-c')
    plt.plot(ccpd_gated_avg_4,ccpd4, marker='v', linestyle='--', label='AdaptFilter(M)-nc')
    plt.xlabel('size %', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=16)#, bbox_to_anchor=(1, 1))
    plt.show()
    
    #plt.cla()
    x_axis = [0.40,1,2,3,4,10,25,50,75,100]
    """mini mse graph whole"""
    values1 = list(mini_base_mse[0].copy()/32000)
    values2 = list(mini_base_mse[1].copy()/32000)
    values3 = list(mini_base_mse[2].copy()/32000)
    values4 = list(mini_base_mse[3].copy()/32000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, values1, marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, values2, marker='<', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, values3, marker='s', linestyle='--', label='Resnet50-c')
    plt.plot(x_axis, values4, marker='v', linestyle='--', label='Resnet50-nc')
    plt.plot(ccpd_gated_avg_1,mini1, marker='o', linestyle='--', label='AdaptFilter(M)-c')
    plt.plot(ccpd_gated_avg_2,mini2, marker='<', linestyle='--', label='AdaptFilter(M)-nc')
    plt.plot(ccpd_gated_avg_3,mini3, marker='s', linestyle='--', label='AdaptFilter(R)-c')
    plt.plot(ccpd_gated_avg_4,mini4, marker='v', linestyle='--', label='AdaptFilter(R)-nc')
    plt.xlabel('size %', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=16)#, bbox_to_anchor=(1, 1))
    plt.show()
    
    #plt.cla()
    x_axis = [0.40,1,2,3,4]#,25,50,75,100]
    """ccpd mse graph whole"""
    values1 = list(ccpd_base_mse[0].copy()/40000)
    values2 = list(ccpd_base_mse[1].copy()/40000)
    values3 = list(ccpd_base_mse[2].copy()/40000)
    values4 = list(ccpd_base_mse[3].copy()/40000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    
    g_x = [1,2,3,4]
    ccpdresnc = [0.00317743, 0.00223839, 0.0016823, 0.0015592]
    
    ccpdmobnc   = [0.00489, 0.00325081, 0.0028028, 0.00225691]
    
    
    g_x = sum(g_x)/len(g_x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, values1[0:5], marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, values2[0:5], marker='<', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, values3[0:5], marker='s', linestyle='--', label='Resnet50-c')
    plt.plot(x_axis, values4[0:5], marker='v', linestyle='--', label='Resnet50-nc')
    plt.plot(ccpd_gated_avg_1,ccpd1, marker='o', linestyle='--', label='AdaptFilter(M)-c', markersize=9)
    plt.plot(ccpd_gated_avg_2,ccpd2, marker='<', linestyle='--', label='AdaptFilter(M)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_3,ccpd3, marker='s', linestyle='--', label='AdaptFilter(R)-c', markersize=9)
    plt.plot(ccpd_gated_avg_4,ccpd4, marker='v', linestyle='--', label='AdaptFilter(R)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_4*.90,ccpd4, marker='x', linestyle='--', label='Generator(R)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_2*.90,ccpd2, marker='x', linestyle='--', label='Generator(M)-nc', markersize=9)

    plt.xlabel('size %', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(1, 1))
    plt.show()
    
    #plt.cla()
    x_axis = [0.40,1,2,3,4]#,25,50,75,100]
    """mini mse graph whole"""
    values1 = list(mini_base_mse[0].copy()/32000)
    values2 = list(mini_base_mse[1].copy()/32000)
    values3 = list(mini_base_mse[2].copy()/32000)
    values4 = list(mini_base_mse[3].copy()/32000)
    values1.reverse()
    values2.reverse()
    values3.reverse()
    values4.reverse()
    g_x = [1,2,3,4]
    ccpdresnc = [0.00277055, 0.00194971, 0.00152363, 0.00126064]
    
    ccpdresnc = sum(ccpdresnc)/len(ccpdresnc)
    
    
    ccpdmobnc   = [0.00423505, 0.00291591, 0.00246614, 0.003139]
    
    ccpdmobnc = sum(ccpdmobnc)/len(ccpdmobnc)
    
    g_x = sum(g_x)/len(g_x)
    
    plt.figure(figsize=(8, 6))
    plt.plot(x_axis, values1[0:5], marker='o', linestyle='--', label='Mobilenet-c')
    plt.plot(x_axis, values2[0:5], marker='<', linestyle='--', label='Mobilenet-nc')
    plt.plot(x_axis, values3[0:5], marker='s', linestyle='--', label='Resnet50-c')
    plt.plot(x_axis, values4[0:5], marker='v', linestyle='--', label='Resnet50-nc')
    plt.plot(ccpd_gated_avg_1,mini1, marker='o', linestyle='--', label='AdaptFilter(M)-c', markersize=9)
    plt.plot(ccpd_gated_avg_2,mini2, marker='<', linestyle='--', label='AdaptFilter(M)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_3,mini3, marker='s', linestyle='--', label='AdaptFilter(R)-c', markersize=9)
    plt.plot(ccpd_gated_avg_4,mini4, marker='v', linestyle='--', label='AdaptFilter(R)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_4*0.9,mini4, marker='x', linestyle='--', label='Generator(R)-nc', markersize=9)
    plt.plot(ccpd_gated_avg_2*0.9,mini1, marker='x', linestyle='--', label='Generator(M)-nc', markersize=9)
    plt.xlabel('size %', fontsize=16)
    plt.ylabel('Mean Squared Error', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=16, bbox_to_anchor=(1, 1))
    plt.show()



    
#get_baseline_mse_graphs()
#get_baseline_mse_graphs_0to6()
get_gate_freq_graphs()
get_gate_averages()
baseandgate_mse_graphs()

ccpd_gated_avg_1
ccpd_gated_avg_2
ccpd_gated_avg_3
ccpd_gated_avg_4