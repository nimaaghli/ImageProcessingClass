
# coding: utf-8

# # Nima Aghli 
# ## PA5

# In[2]:

get_ipython().magic('matplotlib inline')
#from pylab import *
import texttable as tt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.image as mpimg
from pylab import *
import pandas as pd
from scipy import misc
import scipy.fftpack as fftim
from scipy.misc.pilutil import Image
from PIL import Image
import cmath
from joblib import Parallel, delayed
import scipy.io as sio
from pims import ImageSequence
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage import color
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## 1-loading Frames And Masks Into Program

# In[3]:

DAGSeg = np.load('Masks/birdfall_masksDagSeg.npy')
KeySeg=np.load('Masks/birdfall_masksKeySeg.npy')
SaliencySeg=ImageSequence('Masks/SaliencySeg//birdfall2_00*.bmp')
SeamSeg=np.load('Masks/birdfall_masksSeamSeg.npy')
JOTSeg=np.load('Masks/birdfall_masksFromJOTS.npy')
GroundTruth=np.load('Masks/birdfall_masksGroundtruth.npy')


# In[4]:

framecount=20
fig = plt.figure(figsize=(9, 40))
plt.suptitle("GroundTruth", size=36)
for k in range(0,framecount):
    img=GroundTruth[:,:,k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray')) 


# In[5]:

fig = plt.figure(figsize=(9, 40))
plt.suptitle("DAGSeg", size=36)
for k in range(0,framecount):
    img=DAGSeg[:,:,k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray')) 


# In[6]:


fig = plt.figure(figsize=(9, 40))
plt.suptitle("KeySeg", size=36)

#fig.subplots_adjust(left=0.1, wspace=0.5)
#fig.suptitle('20 Object Porposals For 20 Frames')
for k in range(0,framecount):
    img=KeySeg[:,:,k]      
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray'))
    
   


# In[7]:

fig = plt.figure(figsize=(9, 40))
plt.suptitle("SeamSeg", size=36)
#fig.subplots_adjust(left=0.1, wspace=0.5)
#fig.suptitle('20 Object Porposals For 20 Frames')
for k in range(0,framecount):
    img=SeamSeg[:,:,k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray')) 


# In[8]:

fig = plt.figure(figsize=(9, 40))
plt.suptitle("JOTSeg", size=36)
#fig.subplots_adjust(left=0.1, wspace=0.5)
#fig.suptitle('20 Object Porposals For 20 Frames')
for k in range(0,framecount):
    img=JOTSeg[:,:,k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray')) 


# In[9]:

fig = plt.figure(figsize=(9, 40))
plt.suptitle("SaliencySeg", size=36)
#fig.subplots_adjust(left=0.1, wspace=0.5)
#fig.suptitle('20 Object Porposals For 20 Frames')
for k in range(0,framecount):
    img=SaliencySeg[k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray'), vmin = 0, vmax = 255) 


# In[10]:

Baseline = np.where(((DAGSeg +KeySeg +                             SeamSeg +                            JOTSeg)/5) > 0.5,1,0).astype('uint8')

res = np.array([GroundTruth,DAGSeg,KeySeg,SeamSeg,JOTSeg,Baseline])
titleSeg = ('GroundTruth', 'DagSeg', 'KeySeg', 'SeamSeg', 'JOTSeg', 'Baseline')


# In[11]:

fig = plt.figure(figsize=(9, 40))
#fig.subplots_adjust(left=0.1, wspace=0.5)
#fig.suptitle('20 Object Porposals For 20 Frames')
plt.suptitle("Baseline", size=36)
for k in range(0,framecount):
    img=Baseline[:,:,k]       
    ax = fig.add_subplot(framecount, 5, k+1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.imshow(img,cmap = plt.get_cmap('gray')) 



# In[13]:

error=np.zeros(len(res))
error = [[]]
for k in range(0,len(res)): 
    masksTemp=res[k]
    tempErr = np.round(np.size(                               np.where(                                        (masksTemp.flatten()                                          ^ GroundTruth.flatten())==1)                                        )/masksTemp.shape[2]).astype('uint8')
    error.append(tempErr.astype(int))


# In[14]:

tab = tt.Texttable()

x = [[]] # The empty row will have the header

for i in range(1,len(res)+1):
    x.append([titleSeg[i-1],error[i]])

tab.add_rows(x)
tab.set_cols_align(['l','r'])
tab.header(['Segmentation Algorithm', 'Error'])
print(tab.draw())


# In[35]:

#image = img_as_float(Baseline[:,:,2])
image = np.dstack([gray, gray, gray])
plt.imshow(image)
plt.show()
segments = slic(image, 100,compactness=0.1, enforce_connectivity=True)
plt.imshow(segments),plt.show()
plt.imshow(mark_boundaries(image, segments)),plt.show()


# In[ ]:



