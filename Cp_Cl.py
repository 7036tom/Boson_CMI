# coding=utf-8

import sys
import numpy as np
import pandas
from sklearn.metrics import roc_auc_score, roc_curve


Ch_pred1 = str(sys.argv[1])
Ch_pred2 = str(sys.argv[2])
Labels = str(sys.argv[3])
weights = str(sys.argv[4])

# load dataset
dataframe_P1 = pandas.read_csv(Ch_pred1, header=None)
Prediction_1 = dataframe_P1.values

dataframe_P2 = pandas.read_csv(Ch_pred2, header=None)
Prediction_2 = dataframe_P2.values

dataframe_Y = pandas.read_csv(Labels, header=None)
Y = dataframe_Y.values

dataframe_weights = pandas.read_csv(weights, header=None)
weights_Y = dataframe_weights.values

print(Prediction_1[0])
print(Prediction_2[0])
print(Y[0])

Prediction1_AUC = np.copy(Prediction_1)
Prediction2_AUC = np.copy(Prediction_2)

for i in range(len(Prediction_1)):
	Prediction1_AUC[i]= Prediction_1[i]*2-1
	Prediction2_AUC[i]= Prediction_2[i]*2-1
		

c, r = Y.shape
Y_AUC = Y.reshape(c,)

# Statistiques
# Précision des classifieurs :

AUC1 = roc_auc_score(Y_AUC, Prediction1_AUC, average='macro', sample_weight=weights_Y)
print("L'AUC du modèle 1 est de : "+str(AUC1))

Vrais_positifs1 = 0.0
Vrais_negatifs1 = 0.0
total_positifs1 = 0.0

Vrais_positifs2 = 0.0
Vrais_negatifs2 = 0.0
total_positifs2 = 0.0

for i in range(len(Y)):
	if (Y[i] == 1):
		total_positifs1 += 1
		if (Prediction_1[i]>0.5):
			Vrais_positifs1 += 1
	if (Y[i] == 0 and Prediction_1[i] < 0.5):
		Vrais_negatifs1 += 1

Vrais_negatifs1 = Vrais_negatifs1 / (len(Y)-total_positifs1)
Vrais_positifs1 = Vrais_positifs1 / (total_positifs1)

print("Le premier classifieur prédit justement la nature de "+str(Vrais_positifs1)+" des elements de la classe+")
print("Le premier classifieur prédit justement la nature de "+str(Vrais_negatifs1)+" des elements de la classe- \n")


AUC2 = roc_auc_score(Y_AUC, Prediction2_AUC, average='macro', sample_weight=weights_Y)
print("L'AUC du modèle 2 est de : "+str(AUC2))

for i in range(len(Y)):
	if (Y[i] == 1):
		total_positifs2 += 1
		if (Prediction_2[i]>0.5):
			Vrais_positifs2 += 1
	if (Y[i] == 0 and Prediction_2[i] < 0.5):
		Vrais_negatifs2 += 1

Vrais_negatifs2 = Vrais_negatifs2 / (len(Y)-total_positifs2)
Vrais_positifs2 = Vrais_positifs2 / (total_positifs2)

print("Le premier classifieur prédit justement la nature de "+str(Vrais_positifs2)+" des elements de la classe+")
print("Le premier classifieur prédit justement la nature de "+str(Vrais_negatifs2)+" des elements de la classe- \n")

# Pourcentage de la base sur laquelle les classifieurs sont d'accords :
en_accords = 0.0

for i in range(len(Y)):
	if ((Prediction_1[i]<0.5 and Prediction_2[i]<0.5) or (Prediction_1[i]>=0.5 and Prediction_2[i]>=0.5)):
		en_accords += 1

en_accords /= len(Y)
en_desaccords = 1 - en_accords

print("Les deux classifieurs sont en accord sur "+str(en_accords)+" de la base. \n")


# Pourcentage de la base ou au moins l'un des classifieurs a raison :

un_a_raison = 0.0

for i in range(len(Y)):
	if (abs(Prediction_1[i]-Y[i])<0.5 or abs(Prediction_2[i]-Y[i])<0.5):
		un_a_raison += 1

un_a_raison /= len(Y)

print("Sur "+str(un_a_raison)+" de la base, au moins l'un des classifieurs à raison.\n")


# Pourcentage de la base sur lequel le classifieur 1 a raison, mais pas le classifieur 2

un_seul_a_raison = 0.0

for i in range(len(Y)):
	if (abs(Prediction_1[i]-Y[i])<0.5 and abs(Prediction_2[i]-Y[i])>0.5):
		un_seul_a_raison += 1

un_seul_a_raison /= (en_desaccords*len(Y))

print(str(un_seul_a_raison)+": Pourcentage des évenements douteux ou seul le classifieur 1 a raison.\n")


# 5 exemples ou seul le classifieur un a raison.
print("5 exemples ou seul le classifieur un a raison.")
k = 0 
for i in range(len(Y)):
	if (abs(Prediction_1[i]-Y[i])<0.5 and abs(Prediction_2[i]-Y[i])>0.5):
		print("Classifieur 1 : "+str(Prediction_1[i])+"\nClassifieur 2 : "+str(Prediction_2[i])+"\n")
		k = k + 1
		if (k >= 5):
			break

# 5 exemples ou seul le classifieur deux a raison.
print("5 exemples ou seul le classifieur deux a raison.")
k = 0 
for i in range(len(Y)):
	if (abs(Prediction_1[i]-Y[i])>0.5 and abs(Prediction_2[i]-Y[i])<0.5):
		print("Classifieur 1 : "+str(Prediction_1[i])+"\nClassifieur 2 : "+str(Prediction_2[i])+"\n")
		k = k + 1
		if (k >= 5):
			break


# Let's try to combine these two models

max_p1 = max(Prediction_1)
max_p2 = max(Prediction_2)

for i in range(len(Prediction_1)):
	Prediction_1[i]=Prediction_1[i]/max_p1
	Prediction_2[i]=Prediction_2[i]/max_p2

Prediction_3 = (Prediction_1+Prediction_2)/2

Prediction3_AUC = np.copy(Prediction_3)

np.savetxt("combi.csv", Prediction_3, delimiter=",")

for i in range(len(Prediction_3)):
	Prediction3_AUC[i]= Prediction_3[i]*2-1
	


AUC3 = roc_auc_score(Y_AUC, Prediction3_AUC, average='macro', sample_weight=weights_Y)
print("L'AUC de la combinaison est de : "+str(AUC3))

print("voulez vous conserver les prédictions du nouveau modèle ? (o/n)")
R_user = raw_input()
if (R_user == 'o'):
	print("Nom du fichier ?")
	nom = raw_input()
	np.savetxt(nom+'.csv', Prediction_3, delimiter=",")