
import numpy as np



drug = np.loadtxt('weidu_bian/drug_25.txt',encoding='utf-8')
protern = np.loadtxt('weidu_bian/protein_25.txt',encoding='utf-8')



interaction =np.loadtxt('data_yuce/mat_drug_protein.txt',encoding='utf-8')


idx_zheng=np.array(np.where(interaction==1))
# np.savetxt('bian.txt',idx_zheng,fmt="%d")

n= 0
all = 0
for x,y in zip(idx_zheng[0],idx_zheng[1]):
    try:
        qq = np.hstack((drug[x], protern[y]))
        if n == 0:
            all = qq
        else:
            all = np.vstack((all,qq))
        n+=1
    except:
        print(n)
        continue

np.savetxt('weidu_bian/feature50.txt',all,fmt='%f')

