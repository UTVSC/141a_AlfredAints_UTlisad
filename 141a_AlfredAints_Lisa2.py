objekt = "6918-1-2a"
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import statistics

import math

def steepest_rise(reaallist, imaginaarlist, ):
    max_slope = 0  
    max_index = 0  
    noise_tolerance = 0.2  
    min_points = 3  

    for i in range(len(reaallist) - 6): 
        x_segment = reaallist[i:i + 7]
        y_segment = imaginaarlist[i:i + 7]

        
        if objekt == '6920-1-6a' and max(x_segment) >= 1500:
            continue  
        if objekt == "6923-1-6a":
            if max(x_segment) >= 1500 or min(x_segment) < 500:
                continue  
        
        n = 0
        sum_x = 0
        sum_y = 0
        sum_xy = 0
        sum_x2 = 0
        
        last_y = float('inf')  

        for j in range(len(y_segment)):
            if y_segment[j] < last_y:  
                n += 1
                sum_x += x_segment[j]
                sum_y += y_segment[j]
                sum_xy += x_segment[j] * y_segment[j]
                sum_x2 += x_segment[j] ** 2
                last_y = y_segment[j]  

            else:
                
                n = 0
                sum_x = 0
                sum_y = 0
                sum_xy = 0
                sum_x2 = 0
                last_y = float('inf')  

       
        if n < min_points:
            continue

        k = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)

        if abs(k) > max_slope * (1 - noise_tolerance):
            if abs(k) > max_slope:  
                max_slope = abs(k)
                max_index = i

    x_steepest = reaallist[max_index:max_index + 7]
    y_steepest = imaginaarlist[max_index:max_index + 7]

    return x_steepest, y_steepest
def steepest_trend_line(x_list, y_list, num_points=12):
    
    max_negative_slope = float('inf')  
    best_x_segment = []
    best_y_segment = []
    best_a = None
    best_b = None


    for i in range(len(x_list) - (num_points - 1)):  
        x_segment = x_list[i:i + num_points]
        y_segment = y_list[i:i + num_points]

        slope, intercept = np.polyfit(x_segment, y_segment, 1)

        if slope < max_negative_slope:
            max_negative_slope = slope
            best_x_segment = x_segment
            best_y_segment = y_segment
            best_a = slope
            best_b = intercept

    last_x = best_x_segment[-1]  
    last_y = best_a * last_x + best_b  

    trend_x = np.linspace(last_x, 0, 100) 
    trend_y = best_a * trend_x + best_b  

    return trend_x, trend_y, best_a, best_b
    
    
    


num = 320 #Algtemperatuur

lopp = "K_"+objekt+"-CfT.txt"
alg = "E:/CfT/"+objekt+"-CfT/"
data = {}
data2 = {}
data3 = {}
data4 = {}
steepest_segments = {}
Rslist = []
templist = []
sageduslist = []
funktsioon = []
L = 0.0000012



while num >= 20:   #Lõpptemperatuur
    num_str = str(num)
    CfTfail = alg + num_str + lopp
    templist.append(int(num_str))

    reaallist = []
    imaginaarlist = []
    sageduslist = []
    
    with open(CfTfail, "r") as andmefail:
        andmed  = andmefail.readlines()

        for line in andmed:
            tulbad = line.strip().split()
            Zreaal = float(tulbad[4])
            Zimaginaar = float(tulbad[5])
            sagedus = float(tulbad[1])
            reaallist.append(Zreaal)
            imaginaarlist.append(Zimaginaar)
            sageduslist.append(sagedus)

            x_steepest, y_steepest = steepest_rise(reaallist, imaginaarlist)

            steepest_segments[num_str] = (x_steepest, y_steepest)

            
        n = len(x_steepest)  # Punktide arv
        sum_x = np.sum(x_steepest)  # Kõigi x väärtuste summa
        sum_y = np.sum(y_steepest)  # Kõigi y väärtuste summa
        sum_xy = np.sum(np.multiply(x_steepest, y_steepest))  # Kõigi x ja y korrutiste summa
        sum_x2 = np.sum(np.square(x_steepest))  # Kõigi x väärtuste ruutude summa

        k = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        b = (sum_y - k * sum_x) / n
        #print(f"Sirge võrrand on: y = {k}x + {b}, ja temperatuur on {num}")
        Rs = abs(b)/k
        Rslist.append(Rs)
        mahtuvuslist = []
        m2list = []
        wlist = []
       

    data[num_str] = (reaallist, imaginaarlist)
    
    with open("E:/Cft/"+objekt+"-Cft/andmed.txt", "w") as g:
        for f,z, zz in zip(sageduslist, reaallist, imaginaarlist): #z reaalarv, zz imaginaararv
            w = 2*3.14* f
            C = (w*L-zz)/(((z-Rs)**2 + (w*L-zz)**2)*w)
            m2list.append(abs(C))
            mahtuvuslist.append(C)
            wlist.append(w)
        funktsioon = []
        sageduslist= sageduslist[1:-1]
    for i in range(1, len(wlist) - 1):
        dC_dW = (mahtuvuslist[i + 1] - mahtuvuslist[i - 1]) / (wlist[i + 1] - wlist[i - 1])
        W_dCdW = wlist[i] * dC_dW
        funktsioon.append(W_dCdW)
    logsageduslist = [np.log10(q) for q in sageduslist ] 
    data4[num_str] = (logsageduslist, funktsioon)
    data3[num_str]=(sageduslist, funktsioon)
    data2[num_str]= (sageduslist, m2list)      
    num -= 10


data5= {}
for num_str, (logsageduslist, funktsioon) in data4.items():

    max_index = np.argmax(funktsioon) 


    peak_x = logsageduslist[max_index]
    peak_y = funktsioon[max_index]


    start_index = max(0, max_index - 4)  
    end_index = min(len(funktsioon), max_index + 5)  


    selected_x = logsageduslist[start_index:end_index]
    selected_y = funktsioon[start_index:end_index]  
    data5[num_str] = (selected_x, selected_y)
    
 
    
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(18, 12))




for num_str, (reaallist, imaginaarlist) in data.items():
    ax1.plot(reaallist, imaginaarlist, marker='.', linestyle="none", label=f'{num_str}K')

for num_str, (x_steepest, y_steepest) in steepest_segments.items():
    ax1.plot(x_steepest, y_steepest, marker='o', linestyle='-', markersize=5, label=f'Steepest {num_str}K')

ax1.grid(True)
ax1.set_xlabel("Reaal")
ax1.set_ylabel("imaginaar")
ax2.grid(True)
ax2.plot(templist, Rslist, marker='.', linestyle='none', color='r')
ax2.set_xlabel("Temperatuur/K")
ax2.set_ylabel("Rs/Ω")





for num_str, (sageduslist, m2list) in data2.items():
    ax3.plot(sageduslist, m2list[1:-1], marker='.', linestyle='none', label=f'{num_str}K')
ax3.set_xscale("log")
ax3.set_xlabel("Sagedus/Hz")
ax3.set_ylabel("Mahtuvus/C")
ax3.grid(True)

for num_str, (sageduslist, funktsioon) in data3.items():
    ax4.plot(sageduslist, funktsioon, linestyle='none', marker='.', label=f'{num_str}K')
ax4.set_xscale("log")
ax4.set_xlabel("Sagedus/Hz")
ax4.set_ylabel("W*ΔC/ΔW")
ax4.grid(True)
target = [ "160"]

punktid = []

for num_str, (logsageduslist, funktsioon) in data4.items():
    if num_str in target:
        ax5.plot(logsageduslist, funktsioon, linestyle="none", marker='.', color="blue")
        ax5.set_xscale("log") 
        ax5.grid(True)
for num_str, (selected_x, selected_y) in data5.items():
    
    mean, std_dev = norm.fit(selected_x)  

    x_curve = np.linspace(min(selected_x), max(selected_x), 1000)  
    y_curve = norm.pdf(x_curve, mean, std_dev)  


    y_curve_scaled = y_curve * (max(selected_y) / max(y_curve))

    max_index = np.argmax(y_curve_scaled)  
    max_x = x_curve[max_index]  

    punktid.append(max_x) 
    if num_str in target:
        ax5.plot(x_curve, y_curve_scaled, label=f'Fitted Normal Distribution\nMean: {mean:.2f}, Std Dev: {std_dev:.2f}', color='red')
ax5.set_xlabel("Sagedus/Hz")
ax5.set_ylabel("W*ΔC/ΔW")
oiged = [10**a for a in punktid]
lny = [np.log(W0 / t**2) for t, W0 in zip(templist, oiged)]
lnx = [1 / t for t in templist]

ax6.plot(lnx, lny, linestyle="none", marker="o")
num_points= 8

trend_x, trend_y, best_a, best_b = steepest_trend_line(lnx, lny,  num_points)

ax6.plot(trend_x, trend_y, linestyle="--", color="red")
ax6.set_xlabel("1/t")
ax6.set_ylabel("ln(W0/t**2)")
best_a =int(best_a)
boltz= 1.38*10**(-23)
Ev = 1.60217663*10**(-19)
aktenergia =(-best_a*boltz*1000)/Ev
print(aktenergia, "meV")





gfile =  "E:/Cft/ax2.txt"          
with open(gfile, 'w') as d:
    d.write("lnx\tlny\ttemplist\tRslist\n")

    num_points = len(lnx)

    for i in range(num_points):
        d.write(f"{lnx[i]}\t{lny[i]}\t{Rslist[i]}\n")










plt.tight_layout()
plt.grid(True)
plt.show()