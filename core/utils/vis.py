'''
Author: whj
Date: 2022-02-15 15:46:41
LastEditors: whj
LastEditTime: 2022-06-01 13:11:44
Description: file content
'''

import random
import pyvista as pv
import numpy as np
import csv
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d


colors = ['#000080',
          '#FF0000',
          '#FF00FF',
          '#00BFFF',
          '#DC143C',
          '#DAA520',
          '#DDA0DD',
          '#708090',
          '#556B2F',
          '#483D8B', 
          '#CD5C5C', 
          '#21618C', 
          '#1C2833', 
          '#4169E1', 
          '#1E90FF', 
          '#FFD700', 
          '#FF4500', 
          '#646464', 
          '#DC143C', 
          '#98FB98', 
          '#9370DB', 
          '#8B4513', 
          '#00FF00', 
          '#008080'
        ]


def get_label(filename, factor):
    retarr = np.zeros((0, 7))
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            items = row[0].split(',')
            retarr = np.insert(retarr, 0, np.asarray(items), 0)

    retarr[:, 0:6] = retarr[:, 0:6] * factor

    return retarr


def disp_model(filename):
    pv.set_plot_theme("document")
    mesh = pv.PolyData(filename+'.STL')

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, opacity=0.8, color='#FFFFFF')

    shapetypes = ['O ring', 'Through hole', 'Blind hole', 'Triangular passage', 'Rectangular passage', 'Circular through slot', 'Triangular through slot', 'Rectangular through slot', 'Rectangular blind slot', 'Triangular pocket', 'Rectangular pocket', 'Circular end pocket',
                  'Triangular blind step', 'Circular blind step', 'Rectangular blind step', 'Rectangular through step', '2-sides through step', 'Slanted through step', 'Chamfer', 'Round', 'Vertical circular end blind slot', 'Horizontal circular end blind slot', '6-sides passage', '6-sides pocket']

    items = get_label(filename+'.csv', 1000)

    flag = np.zeros(24)

    for i in range(items.shape[0]):
        if flag[int(items[i, 6])] == 0:
            plotter.add_mesh(pv.Cube((0, 0, 0), 0, 0, 0, (items[i, 0], items[i, 3], items[i, 1], items[i, 4], items[i, 2], items[i, 5])), opacity=1, color=colors[int(
                items[i, 6])], style='wireframe', line_width=2, label=shapetypes[int(items[i, 6])])
            flag[int(items[i, 6])] = 1
        else:
            plotter.add_mesh(pv.Cube((0, 0, 0), 0, 0, 0, (items[i, 0], items[i, 3], items[i, 1], items[i, 4],
                                                          items[i, 2], items[i, 5])), opacity=1, color=colors[int(items[i, 6])], style='wireframe', line_width=2)

    plotter.add_legend()
    plotter.show()


def plot3D_with_labels(lowDWeights, labels):
    fig = plt.figure()
    ax = mpl_toolkits.mplot3d.Axes3D(fig)
    X, Y, Z = lowDWeights[:, 0], lowDWeights[:, 1], lowDWeights[:, 2]
    for x, y, z, s in zip(X, Y, Z, labels):
        c = colors[int(s)]
        ax.text3D(x, y, z, s, backgroundcolor=c, color='#FFFFFF', fontsize=9)
    ax.set_xlim3d(X.min(), X.max())
    ax.set_ylim3d(Y.min(), Y.max())
    ax.set_zlim3d(Z.min(), Z.max())
    #plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.05)


def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = colors[int(s)]
        plt.text(x, y, s, backgroundcolor=c, color='#FFFFFF', fontsize=12)
    plt.xlim(X.min()-1, X.max()+1)
    plt.ylim(Y.min()-1, Y.max()+1)
    # plt.axis('off')
    #plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.05)
