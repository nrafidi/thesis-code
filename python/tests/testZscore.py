import numpy as np
from scipy import stats
import scipy.io

a = np.array([[[1, 2, 90], [8, 5, 6]], [[8, 9, 10], [11, 12, 13]]])
b = np.array([[[1, 2, 90], [8, 5, 6]], [[8, 9, 10], [11, 12, 13]]])
c = np.array([[[1, 2, 90], [8, 5, 6]], [[8, 9, 10], [11, 12, 13]]])
d = np.array([[[1, 2, 90], [8, 5, 6]], [[8, 9, 10], [11, 12, 13]]])
e =  np.array([[[1, 2, 90], [8, 5, 6]], [[8, 9, 10], [11, 12, 13]]])

reshaped_a = np.reshape(b, (2, -1), 'F')
reshaped_b = np.reshape(c, (2, -1), 'F')
reshaped_c = np.reshape(d, (2, -1), 'F')

zscored_a = stats.zscore(reshaped_b,axis=0,ddof=1)
zscored_b = stats.zscore(reshaped_c,axis=0,ddof=1)

rereshaped_zscored_a = np.reshape(zscored_b, (2, 2, 3), 'F');
just_zscored_a = stats.zscore(e, axis=0,ddof=1);

mean_a = np.mean(a,axis=0)
std_a = np.std(a,axis=0,ddof=1)

demean_a = a - mean_a
z_a = demean_a/std_a

scipy.io.savemat('test3Dnparray.mat',mdict={'a':a,'reshaped_a':reshaped_a,'zscored_a':zscored_a,'rereshaped_zscored_a':rereshaped_zscored_a,'just_zscored_a':just_zscored_a,'mean_a':mean_a,'std_a':std_a,'demean_a':demean_a,'z_a':z_a})
