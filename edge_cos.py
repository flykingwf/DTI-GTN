import numpy as np

drug = np.loadtxt('drug_75.txt',encoding='utf-8')
protein = np.loadtxt('protein_75.txt',encoding='utf-8')
interaction =np.loadtxt('mat_drug_protein.txt',encoding='utf-8')
dui = []
idx_zheng=np.array(np.where(interaction==1))
for i,j in enumerate(range(len(idx_zheng[1]))):
    dui.append([idx_zheng[0, j]+1, idx_zheng[1, j]+1,i])

new = []
fff = [[],[]]
nnnnnnnnnnn = 0
for x in dui:
    a = x[0]
    b = x[1]
    n1 = x[2]
    for y in dui:
        c = y[0]
        d = y[1]
        n2 = y[2]
        if a==c and b==d:
            continue
        elif a==c or  b==d:
            temp1 = np.hstack((drug[a-1],protein[b-1]))
            temp2 = np.hstack((drug[c-1],protein[d-1]))
            aaaa = np.linalg.norm(temp1)
            bbbb = np.linalg.norm(temp2)
            cos = np.dot(temp1,temp2)/(aaaa*bbbb)
            if cos > 0.0:
                fff[0].append(n1)
                fff[1].append(n2)
                nnnnnnnnnnn+=1

fff = np.array(fff)
np.savetxt('bian_0.0.txt', fff, fmt='%d', delimiter=' ')






