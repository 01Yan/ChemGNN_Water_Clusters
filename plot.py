import math

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.pyplot import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap,ListedColormap
from matplotlib import cm
from utils import *
from sklearn.metrics import r2_score
from matplotlib.colors import Normalize



def compute_correlation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar ** 2
        varY += difYYbar ** 2
    SST = math.sqrt(varX * varY)
    if SST == 0.0:
        return -1
    return SSR / SST

def regression_correlation(x, y):
    return np.corrcoef(x, y)[0, 1]

train_true_list=np.load('/root/autodl-tmp/soap/train/waterforce/test_true.npy')
train_pred_list=np.load('/root/autodl-tmp/soap/train/waterforce/test_pred.npy')
print(train_true_list)
print(train_pred_list)
print(len(train_true_list))
for i in range(len(train_true_list)):
    train_true_list[i]=reverse_min_max_scaler_1d(train_true_list[i])
    train_pred_list[i]=reverse_min_max_scaler_1d(train_pred_list[i])
error=[]
bins = np.arange(-361, -17, 18)  # 这将创建从-17到361（不包括361）的边界，间隔为10
hist, edges = np.histogram(train_true_list, bins=bins)
for i in range(len(hist)):
    print(f"Interval [{edges[i]}, {edges[i+1]}) has {hist[i]} points")
for i in range(len(train_true_list)):
    dif=abs(train_true_list[i]-train_pred_list[i])
    error.append(dif)
loss=np.mean(error)
# for i in range(len(train_true_list)):
#     dif=abs(train_true_list[i]-train_pred_list[i])
#     error.append(dif)
# loss=np.mean(error)

# r_train = compute_correlation(train_true_list, train_pred_list)
# r_train = regression_correlation(train_true_list, train_pred_list)
r_train = r2_score(train_true_list, train_pred_list)

max_true=max(train_true_list)
min_true=min(train_true_list)
fig,ax=plt.subplots(figsize=(14,9))
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': '24'}
#fit = np.polyfit(train_true_list, train_pred_list, 1)
#line_fn = np.poly1d(fit)
#y_line = train_true_list
train_true_list=np.array(train_true_list)
train_pred_list=np.array(train_pred_list)
xy=np.vstack([train_true_list,train_pred_list])
z=gaussian_kde(xy)(xy)
idx=z.argsort()
train_true_list=train_true_list[idx]
train_pred_list=train_pred_list[idx]
z=z[idx]
Colors=['lightsteelblue','b','crimson']
Cmap=LinearSegmentedColormap.from_list('mycmap',Colors)
sm=ax.scatter(train_true_list, train_pred_list,c=z,s=40,cmap=Cmap)

ax.scatter(train_true_list, train_pred_list, c='b', s=40)  # 使用蓝色作为点的颜色
# normalized_errors = (error - min(error)) / (max(error) - min(error))  # 归一化误差
# colors = cm.plasma(normalized_errors)  # 使用viridis颜色映射
# scatter = ax.scatter(train_true_list, train_pred_list, c=colors, s=40)
# cbar = plt.colorbar(scatter, ax=ax)
# cbar.set_label('Error Magnitude', rotation=270, labelpad=15)

# error = np.asarray(error)
# # 创建颜色映射器
# norm = Normalize(vmin=error.min(), vmax=error.max())
# mapper = cm.ScalarMappable(norm=norm, cmap=cm.plasma)
# # 使用误差映射颜色
# colors = mapper.to_rgba(error)
# # 绘制散点图
# scatter = ax.scatter(train_true_list, train_pred_list, c=colors, s=40)
# # 添加颜色条
# cbar = plt.colorbar(mapper, ax=ax)
# cbar.set_label('Error Magnitude', rotation=270, labelpad=15)


#ax.scatter(train_true_H, train_pred_H,s=40,c='b',label='H')
v=[0,10,20,30,40,50,60,70,80,90,100]
print(v)
# cm_1=plt.colorbar(sm)
# cm_1.ax.tick_params(labelsize=20)
# ax.scatter(train_true_list_B, train_pred_list_B,s=40, label='BETA')
# plt.plot(train_true_list, y_line, linewidth=1, c="black", linestyle="--")
ax.plot((0,1),(0,1),transform=ax.transAxes,linestyle='--',linewidth=1,c='black')
#x_major_locator=MultipleLocator(0.5)
#y_major_locator=MultipleLocator(0.5)
#x_minor_locator=MultipleLocator(0.1)
#y_minor_locator=MultipleLocator(0.1)
ax1=plt.gca()
#ax1.xaxis.set_major_locator(x_major_locator)
#ax1.yaxis.set_major_locator(y_major_locator)
#ax1.xaxis.set_minor_locator(x_minor_locator)
#ax1.yaxis.set_minor_locator(y_minor_locator)
ax.set_xlim(min_true-(max_true-min_true)/10, max_true+(max_true-min_true)/10)
ax.set_ylim(min_true-(max_true-min_true)/10, max_true+(max_true-min_true)/10)
ax.set_xlabel("True value", font)
ax.set_ylabel("Predicted value", font)
ax.tick_params(axis='x', which='major', direction='in', labelsize=20,length=10)
ax.tick_params(axis='y', which='major', direction='in', labelsize=20,length=10)
ax.tick_params(axis='x', which='minor', direction='in', labelsize=20,length=5)
ax.tick_params(axis='y', which='minor', direction='in', labelsize=20,length=5)
#plt.legend({"LOSS:{0:.4f}".format(np.mean(loss))},loc= 'upper left',  prop = font,markerscale=2,frameon=False)
plt.title('ENERGY Regression', fontsize=30)
plt.suptitle("R^2={0:.4f},MAE:{1:.2e}".format(r_train, loss), fontsize=25,x=0.5, y=0.2)
#plt.suptitle("Regression: R^2={0:.4f}".format(r_train ** 2.0), fontsize=25,x=0.6, y=0.2)
#plt.title('global/ceal', size=25, x=0.2, y=0.9)
#plt.suptitle('testing set', size=25, x=0.75, y=0.2)
#plt.legend(fontsize=25,frameon=False,loc='center right',bbox_to_anchor=(0.4,0.8))
plt.savefig('/root/autodl-tmp/soap/logs', dpi=500)
plt.show()
