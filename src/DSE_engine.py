
import math
import csv
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mticker

# My axis should display 10⁻¹ but you can switch to e-notation 1.00e+01
def log_tick_formatter(val, pos=None):
    return f"$2^{{{int(val)}}}$"  # remove int() if you don't use MaxNLocator
    # return f"{10**val:.2e}"      # e-Notation


fig = plt.figure()
ax = plt.axes(projection='3d')

#coefficients
l1 = 5 #mmult
l2 = 112
r1 = 470 #470
r2 = 21715
r3 = 9220
L = 2 #layer of GNN

# U250 available resources
DSP = 12288
LUT = 1728000

method = 'ns' #ss or ns.

#Sample Inputs
NS = [10,25] #sample size for each layer
Vt = 1024 #number of target vertices
f = 256
Sl = [256,256,128]#layer sampling sample size
SB = 2750 #sampling budget for subgraph


def check_resource(n,m):
    #check LUT
    LUT_usage = r1*m + r2*n + r3*n*0.5*math.log2(n)
    #check DSP
    DSP_usage = l1*m + l2*n
    
    if LUT_usage < LUT*0.8 and DSP_usage < DSP:
        if LUT_usage/LUT > 0.5 or DSP_usage/DSP > 0.5:
            print('LUT:', LUT_usage/LUT, ', DSP:', DSP_usage/DSP)
        return True
    else:
        return False


def throughput(n,m):
    if method == 'ns':
        B = [102483, 9000, 1024]
        E = [137420, 8505]
    else:
        B = [2750, 2750, 2750]
        E = [54510, 54510]
    f_l = 128
    t_prop = 0
    total_BL = B[0]+B[1]+B[2]
    for i in range(1,L+1):
        a = 0.3 if i==1 else 0.9
        t_load = (B[i-1]* f_l * 4)/(77*1e9*a)
        t_compute = (E[i-1]*f_l)/(n*16*300*1e6)
        t_update = (B[i]*f_l*f_l)/(m*300*1e6)
        t_prop += max(t_compute,t_load,t_update)
    
    return total_BL/t_prop*4

#construct search space

n_max = 1
m_max = 1

best_n = 0
best_m = 0
max_val = 0

#n_max
while r2*n_max + r3*0.5*n_max*math.log2(n_max) < LUT:
    n_max += 1
if round(DSP/l2) < n_max:
    n_max = round(DSP/l2)

#m_max
if round(DSP/l1) < round(LUT/r1):
    m_max = round(DSP/l1)
else:
    m_max = round(LUT/r1)

print('n max:', n_max)
print('m max:', m_max)

# open the file in the write mode
f = open('dse.csv', 'w')
# create the csv writer
writer = csv.writer(f)

fields = ['n', 'm', 'throughput']
writer.writerow(fields) 
xdata = []
ydata = []
zdata = []

#DSE start
n = 1
while n <= n_max:
    m = 1
    while m <= m_max:
        row_info = []
        valid = check_resource(n,m)
        if valid and throughput(n,m) > max_val:
            max_val = throughput(n,m)
            best_n = n
            best_m = m
        if valid:
            print(n,m,'{:.2e}'.format(throughput(n,m)))
            xdata.append(math.log2(n))
            ydata.append(math.log2(m))
            zdata.append(throughput(n,m)/1e6)
        row_info = [str(n),str(m),str('{:.2e}'.format(throughput(n,m)))]
        writer.writerow(row_info) 
        m = 2*m
    n = 2*n
f.close()


# Data for a three-dimensional line
zline = np.linspace(0, 30, 1000)
xline = np.linspace(0, 5, 1000)
yline = np.linspace(0, 11, 1000)

# Data for three-dimensional scattered points
su = ax.scatter3D(xdata, ydata, zdata, s=20, c = zdata  , cmap=cm.coolwarm, antialiased=False)
ax.set_xlabel('n')  # Add an x-label to the axes.
ax.set_ylabel('m')  # Add a y-label to the axes.
ax.set_zlabel('Throughput (MNVTPS)') 

ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

fig.colorbar(su, shrink=0.5, aspect=5, pad=0.15)
plt.savefig('result.pdf')

print(best_n, best_m)