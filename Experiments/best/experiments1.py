import pandas 
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

plt.style.use('seaborn-paper') 
path = './Experiments/best/search-SEE_Exp-20190621-064030/'

data_error = np.zeros((30,1))
data_complex = np.zeros((30,1))

data = pandas.read_csv(path+'Generation-init.csv')
error = data['136'].values
complexity = data['137'].values
if len(error) != 30:
    error = np.hstack([error,[np.mean(error) for _ in range(30 - len(error))]]).reshape(-1,1)
if len(complexity) != 30:
    complexity = np.hstack([complexity,[np.mean(complexity) for _ in range(30 - len(complexity))]]).reshape(-1,1)
data_error = np.hstack([data_error, error.reshape(-1,1)])
data_complex = np.hstack([data_complex, complexity.reshape(-1,1)])

for file in range(30):
    data = pandas.read_csv(path+'Generation-{0}.csv'.format(file))
    error = data['136'].values
    complexity = data['137'].values
    if len(error) != 30:
        error = np.hstack([error,[np.mean(error) for _ in range(30 - len(error))]]).reshape(-1,1)
    if len(complexity) != 30:
        complexity = np.hstack([complexity,[np.mean(complexity) for _ in range(30 - len(complexity))]]).reshape(-1,1)
    data_error = np.hstack([data_error, error.reshape(-1,1)])
    data_complex = np.hstack([data_complex, complexity.reshape(-1,1)])

data_error = data_error[:,1:]
data_complex = data_complex[:,1:]

data_best = np.min(data_error,0)
data_complex_best = np.mean(data_complex,0)

# fig =plt.figure(figsize=(5,4))
# index = np.array([[x]*30 for x in range(len(data_best))]).reshape(1,-1)
# plt.scatter(data_complex[:,0],data_error[:,0])
# #plt.scatter(data_complex[:,5],data_error[:,5])
# plt.scatter(data_complex[:,10],data_error[:,10])
# #plt.scatter(data_complex[:,15],data_error[:,15])
# #plt.scatter(data_complex[:,20],data_error[:,20])
# plt.scatter(data_complex[:,25],data_error[:,25])

baseline = np.array([75.96,75.96,75.96,76.06,76.24,76.59,76.72,76.83,76.95,77.06])
baseline = 100 - baseline



fig, axs = plt.subplots(1, 3, figsize=(14,5), constrained_layout=True)
X = [-5,1,5,8,14,20,23,29,35]

markerlist = ['o','*','^']
fontsize = 15
# axs[0].set_xlim(-2,32)
# for i in [0,10,30]:
axs[0].scatter(data_complex[:,0], data_error[:,0], alpha=1, edgecolors=None, label='Generation {0}'.format(0),marker='^',s=55)
axs[0].scatter(data_complex[:,10], data_error[:,10], alpha=1, edgecolors=None, label='Generation {0}'.format(10),marker='s',s=55)
axs[0].scatter(data_complex[:,30], data_error[:,30], alpha=1, edgecolors=None, label='Generation {0}'.format(30),marker='o',s=55)
    # axs[0].scatter(i*5,np.min(data_error[:,x[i]]),marker='x',c='black',s=30)
axs[0].legend(fontsize=fontsize)
axs[0].set_xlabel('The Compulational Complexity(FLOPs)',size=fontsize)
axs[0].set_ylabel('Test error on CIFAR-10',size=fontsize)
axs[0].tick_params(labelsize=fontsize)
axs[0].set_title('(a)',fontsize=fontsize)

basline, = axs[1].plot(X,[9.82]*len(X),'r--',label='Layer-based',linewidth=2)
basline, = axs[1].plot(X,[9.0]*len(X),'r-.',label='Connection-based',linewidth=2)
basline, = axs[1].plot(X,[8.9]*len(X),'r:',label='Module-based',linewidth=2)
wfbee, = axs[1].plot([0,5,10,15,20,25,30],[np.min(data_error[:,i]) for i in X[1:-1]],'bs-',label='WFBEE',linewidth=2)
# basline2, = axs[1].plot(X,[8.81]*len(X),'g--',label='Network in Network')
axs[1].set_xlim(-2,32)
axs[1].legend(fontsize=fontsize)
axs[1].set_xlabel('Generations',size=fontsize)
axs[1].set_ylabel('Test error on CIFAR-10',size=fontsize)
axs[1].tick_params(labelsize=fontsize)
# axs[0].boxplot(data_error[:,X[1:-1]],positions=[0,5,10,15,20,25,30],widths=3)
# for i in X[1:-1]
#     x = 

axs[2].boxplot(data_complex[:,[0,5,10,15,20,25,-1]],positions=[0,5,10,15,20,25,30],widths=3,boxprops={'linewidth':1.5},
                                                                                                            whiskerprops ={'linewidth':1.5},
                                                                                                            flierprops ={'linewidth':1.5},
                                                                                                            medianprops ={'linewidth':1.5},
                                                                                                            meanprops ={'linewidth':1.5})
axs[2].set_xlim(-2,32)
# wfbee, = axs[2].plot([0,5,10,15,20,25,30],[np.min(data_complex[:,i]) for i in [0,5,10,15,20,25,29]],'b--',label='WFBEE')
# axs[2].legend()
axs[2].set_xlabel('Generations',size=fontsize)
axs[2].set_ylabel('The Compulational Complexity(FLOPs)',size=fontsize)
axs[2].tick_params(labelsize=fontsize)
fig.show()
input()
