import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss, title, fname):
    plt.cla()
    plt.clf()
    plt.axis('on')
    plt.title(title)
    plt.xlabel('Number of epochs')
    plt.ylabel('Loss')
    plt.plot(loss)
    plt.tight_layout()
    plt.savefig(fname)

def plot_bpd(bpd, title, fname):
    plt.cla()
    plt.clf()
    plt.axis('on')
    plt.title(title)
    plt.xlabel('Number of epochs')
    plt.ylabel('bits/dim')
    plt.plot(bpd)
    plt.tight_layout()
    plt.savefig(fname)