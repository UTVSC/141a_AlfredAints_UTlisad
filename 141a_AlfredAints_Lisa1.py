objekt = "6918-1-2a" # Objekti nimi
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import linregress
a1 = 0
b1 = 0

def jarsk(Voclist, tolerants=0.05):
    mxlask = 0
    lasupunktid = 7  
    lasupunktidlist = []  
    yloik = None

    
    for i in range(len(Voclist) - lasupunktid + 1):
        
        y_vals = Voclist[i:i + lasupunktid]
        x_vals = np.array([320 - 10* j for j in range(lasupunktid)])

        
        slope, intercept, _, _, _ = linregress(x_vals, y_vals)

        
        fitted_y_vals = slope * x_vals + intercept

        total_deviation = np.sum(np.abs(y_vals - fitted_y_vals))

       
        avg_fitted_y = np.mean(fitted_y_vals)
        tolerance_limit = tolerants * avg_fitted_y

        
        if total_deviation > tolerance_limit:
            continue
        
       
        if abs(slope) > mxlask:
            mxlask = abs(slope)
            lasupunktidlist = y_vals
            yloik = intercept

            
            

    return mxlask, lasupunktidlist, yloik

MPP = 0
num = 320

lopp = "K_"+objekt+"-IVT.txt"
alg = "E:/IVT/"+objekt+"-IVT/"
data = {}
FFlist = []
efektiivsuselist = []    
templist = []
FFlist = []
Isclist = []
Voclist = []

with open(alg+objekt+"-IVT.txt", "a") as f:  # Fail kuhu kirjutab

    while num >= 20:
        num_str = str(num)
        kokku = alg + num_str + lopp
        pingelist = []
        voolulist = []
        templist.append(int(num_str))
        
        with open(kokku, "r") as file:
            andmed = file.readlines()
        num -= 10
        
        lines = iter(andmed)

        for line in lines:
            tulbad = line.strip().split()

            pinge = float(tulbad[1])
            vool = float(tulbad[2])
            
            voolulist.append(vool)
            pingelist.append(pinge)

        
            if pinge < 0 : 
                rida = next(lines, None)
                tulbad2 = rida.strip().split()
                pinge2 = float(tulbad2[1])
                
                
                vool2 = float(tulbad2[2])
                voolulist.append(vool2)
                pingelist.append(pinge2)
                
                if pinge2 > 0:
                    voimsus = pinge2*vool2
                    
                    

                    if voimsus < 0 and voimsus < MPP: # Kontrollib kas esimene punkt valgustatud olekus võib olla MPP
                        MPP = voimsus
                        Vmp = pinge2
                        Imp = vool2
                        k = (vool2 - vool)/(pinge2 - pinge)  #Leiab Isc
                        Isc = abs(vool - k*pinge)
                        

                        while voimsus < 0:
                            rida2 = next(lines, None)
                            tulbad3 = rida2.strip().split()
                            prevpinge = pinge
                            prevvool = vool
                            pinge = float(tulbad3[1])
                            vool = float(tulbad3[2])
                            voolulist.append(vool)
                            pingelist.append(pinge)
                            voimsus = pinge*vool
                            if voimsus < MPP:
                                MPP = voimsus
                                Vmp = pinge
                                Imp = vool
                    

                    
                    k2 = (vool - prevvool)/(pinge - prevpinge)
                    
                    vabaliige2 = (prevvool -k2*prevpinge)
                    Voc = -vabaliige2/k2
                    FF = (Vmp*abs(Imp)) / (Voc*Isc)
                    FFlist.append(FF)
                    Jmp = abs(Imp)/0.046           #Ühik on A/cm2
                    efektiivsus = Jmp*Vmp/0.1*100         #0.1 W/cm2
                    efektiivsuselist.append(efektiivsus)
                    Voclist.append(Voc)
                    Isclist.append(Isc)
                    
                    
                    #f.write(f" {num_str}K, Isc = {round(Isc, 8)} A, Voc = {round(Voc, 8)} V, "
                            #f"Voc lähenduskõvera vabaliige = {round(vabaliige2, 8)}, "
                            #f"MPP = {round(abs(MPP), 8)} W, Vmp = {round(Vmp, 8)} V, "
                            #f"Imp = {round(abs(Imp), 8)} A, FF = {round(FF, 8)}%, "
                            #f"Efektiivsus = {round(efektiivsus, 8)}%\n\n")
                            
                    #print("Isc -", Isc, "Voc -", Voc, "Voc lähenduskõvera vabaliige - ", vabaliige2, "MPP - ", abs(MPP),
                        #"Vmp -", Vmp, "Imp -", abs(Imp), "FF - ", FF, "%", "Efektiivsus-", efektiivsus, "%")
        data[num_str] = (pingelist, voolulist)
        Vmp = 0        
        Imp =0
        MPP= 0



mxlask, lasupunktidlist, yloik = jarsk(Voclist)

steepest_indices = [Voclist.index(point) for point in lasupunktidlist]


x_first = steepest_indices[0]
y_first = lasupunktidlist[0]
gfile = "E:/IVT/6923.txt"

max_length = max(len(FFlist), len(efektiivsuselist), len(Voclist), len(Isclist), len(templist))
columns = zip(
    FFlist + [None] * (max_length - len(FFlist)),
    efektiivsuselist + [None] * (max_length - len(efektiivsuselist)),
    Voclist + [None] * (max_length - len(Voclist)),
    Isclist + [None] * (max_length - len(Isclist)),
    templist + [None] * (max_length - len(templist))
)



with open(gfile, 'w') as g:
    for row in columns:
        g.write('\t'.join(str(value) if value is not None else '' for value in row) + '\n')

print("Voc kui temperatuur 0", yloik)
           
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 8))


for num_str, (pingelist, voolulist) in data.items():
    ax1.plot(pingelist, voolulist, marker='.', linestyle='None', label=f'{num_str}K')  

ax1.set_title("IV kover")
ax1.set_xlabel("Pinge/V")
ax1.set_ylabel("Vool/A")
ax1.grid(True)


ax2.plot(templist, FFlist, marker="o", linestyle='None')
ax2.set_title("FF-i sõltuvus temperatuurist")
ax2.set_xlabel("Temperatuur/K")
ax2.set_ylabel("FF/%")
ax2.grid(True)
ax2.set_xticks(templist[::2])

ax3.plot(templist, efektiivsuselist, marker="o", linestyle='None',)
ax3.set_title("Efektiivsuse sõltuvus temperatuurist")
ax3.set_xlabel("Temperatuur/K")
ax3.set_ylabel("Efektiivsus/%")
ax3.set_xticks(templist[::2])
ax3.grid(True)

ax4.plot(templist, Voclist, marker="o", linestyle='None')
ax4.scatter([templist[i] for i in steepest_indices], lasupunktidlist, color="red", s=50, zorder=5, label="Steepest Points")
ax4.set_title("Voc sõltuvus temperatuurist")
ax4.set_xlabel("Temperatuur/K")
ax4.set_ylabel("Voc/V")
ax4.set_xticks(templist[::2])
ax4.grid(True)
ax4.plot([templist[x_first], 0], [y_first, yloik], color="blue", linestyle="--", label="Steepest Slope Line")

ax5.plot(templist, Isclist, marker="o", linestyle='None')
ax5.set_title("Isc sõltuvus temperatuurist")
ax5.set_xlabel("Temperatuur/K")
ax5.set_ylabel("Isc/A")
ax5.set_xticks(templist[::2])
ax5.grid(True)

ax6.axis('off')

plt.tight_layout()
plt.show()

