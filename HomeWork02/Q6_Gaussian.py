
import numpy as np
import matplotlib.pyplot as plt

# =============================== making plots ready ============================================

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(12,10))
fig.suptitle('Plots in the 1st row are obtained using binclass.txt and plots in 2nd row obtained using binclassv2.txt')
ax1.set_title('Different Sigma')
ax2.set_title('Same Sigma')
ax3.set_title('Different Sigma')
ax4.set_title('Same Sigma')

for ax in fig.get_axes():
     ax.set(xlabel='x1', ylabel='x2')


#===============================================================================================

# ==================================== reading data from dataset ===============================
def data(file_name):

    data = np.genfromtxt(file_name,delimiter=',')
    x = data[:,:2]
    y = data[:,2]
    pc = y>0
    nc = y<0
    x2 = np.arange(np.min(x[:,1]),np.max(x[:,1]),0.05)
    x1 = np.arange(np.min(x[:,0]),np.max(x[:,0]),0.05)

    return (x,y,pc,nc,x2,x1)
#===============================================================================================

# ========================== Working with the Guassion ===========================================

def Gaussian(mu, sigma, x2, x1,axes1,axes2,pc, nc,x):

    # ---------    plotting data points-----------------------------------
    axes1.plot(x[pc,0], x[pc,1], 'r.')
    axes1.plot(x[nc,0], x[nc,1], 'b.')

    axes2.plot(x[pc,0], x[pc,1], 'r.')
    axes2.plot(x[nc,0], x[nc,1], 'b.')
    #-----------------------------------------------------------------------------

    #---------------------------- when sigma are different -----------------------

    mui = mu[0].reshape((mu[0].shape[0], 1))
    muj = mu[1].reshape((mu[1].shape[0], 1))

    axes1.contour(np.meshgrid(x1,x2)[0], np.meshgrid(x1,x2)[1], (sigma[0]**(-1*mui.shape[0]/2))*np.exp(((np.meshgrid(x1,x2)[0]-mui[0])**2 + 
                        (np.meshgrid(x1,x2)[1]-mui[1])**2)/(-2*sigma[0]))-
                  (sigma[1]**(-1*mui.shape[0]/2))*np.exp(((np.meshgrid(x1,x2)[0]-muj[0])**2 + 
                    (np.meshgrid(x1,x2)[1]-muj[1])**2)/(-2*sigma[1])), 0)

    #----------------------------- when sigma are same ---------------------------------

    axes2.contour(np.meshgrid(x1,x2)[0], np.meshgrid(x1,x2)[1],(sigma[0]**(-1*mui.shape[0]/2))*np.exp(((np.meshgrid(x1,x2)[0]-mui[0])**2 + 
                        (np.meshgrid(x1,x2)[1]-mui[1])**2)/(-2*sigma[0]))-
                  (sigma[0]**(-1*mui.shape[0]/2))*np.exp(((np.meshgrid(x1,x2)[0]-muj[0])**2 + 
                    (np.meshgrid(x1,x2)[1]-muj[1])**2)/(-2*sigma[0])), 0)


#=========================================================================================
    

x,y,pc,nc,x2,x1 = data('binclass.txt')

# ============================ Estimate MLE for binclass.txt =========================

dev1 = np.std(x[pc], axis=0)
dev2 = np.std(x[nc], axis=0)
mu=np.array([np.mean(x[pc], axis=0), np.mean(x[nc], axis=0)])
sig=np.array([[np.mean(dev1*dev1)], [np.mean(dev2*dev2)]])

#---------------------------------------------------------------------------------------
Gaussian(mu, sig, x2, x1,ax1,ax2,pc,nc,x)

x,y,pc,nc,x2,x1 = data('binclassv2.txt')
# ============================ Estimate MLE for binclassv2.txt =========================

dev1 = np.std(x[pc], axis=0)
dev2 = np.std(x[nc], axis=0)
mu=np.array([np.mean(x[pc], axis=0), np.mean(x[nc], axis=0)])
sig=np.array([[np.mean(dev1*dev1)], [np.mean(dev2*dev2)]])

#---------------------------------------------------------------------------------

Gaussian(mu, sig, x2, x1,ax3,ax4,pc,nc,x)

plt.show()
