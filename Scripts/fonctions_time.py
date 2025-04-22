import numpy as np
import csv
import os 

from config import chemin_tt_espace



def time_to_point(a,b,v,F=1): # aller du pt a au pt b avec la vitesse initiale v
    x0,y0=a
    xf,yf=b
    X=x0-xf
    Y=y0-yf
    k4=1
    k3=0
    k2=-4*(v[0]**2+v[1]**2)/F**2
    k1=-8*(v[0]*X+v[1]*Y)/F**2
    k0=-4*(X**2+Y**2)/F**2
    times=np.roots([k4,k3,k2,k1,k0])
    for i in range(4): # on garde la racine réelle et positive
        if times[i].imag==0:
            if times[i]>0:
                return times[i].real
    print('error')
    return times[0]



def frappe(match, point):
    with open(os.path.join(chemin_tt_espace,"example",match,point,point+"_annotation.csv"), newline='') as fichier :
        set_point_annotation=[]
        reader=csv.reader(fichier)
        for ligne in reader :
            set_point_annotation.append(ligne)

    time_frappe=[] # joueur qui frappe, frame de frappe, coordonnées de la balle fichier annotation
    for i in range(1, len(set_point_annotation)):
        if set_point_annotation[i][8]=='' :
            time_frappe.append([set_point_annotation[i][0], int(set_point_annotation[i][21]), (float(set_point_annotation[i][18]), -float(set_point_annotation[i][19])), set_point_annotation[i][4]])
        elif set_point_annotation[i][8]!='' :
            frappe_service=[set_point_annotation[i][0], int(set_point_annotation[i][21]), (float(set_point_annotation[i][18]), -float(set_point_annotation[i][19])), set_point_annotation[i][4]]
    
    return time_frappe, frappe_service