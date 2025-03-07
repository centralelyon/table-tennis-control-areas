import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

from config import chemin_tt_espace,USER_PREFERENCE
import os

from fonctions_time import time_to_point
from fonctions_positions import coordonnees,orientation

import cv2
import pandas as pd
import shutil

from scipy.spatial import ConvexHull


def path(a,b,v,F=5,nb_points=100): # calcul des positions de la trajectoire entre points a et b avec vitesse initiale v
    tf= time_to_point(a,b,v,F)
    Fx=2*(b[0]-a[0]-v[0]*tf)/tf**2
    Fy=2*(b[1]-a[1]-v[1]*tf)/tf**2
    dt=tf/nb_points
    t=dt
    lx=[a[0]]
    ly=[a[1]]
    for i in range(nb_points):
        x=a[0]+v[0]*t+Fx*t**2/2
        y=a[1]+v[1]*t+Fy*t**2/2
        lx.append(x)
        ly.append(y)
        t=t+dt
    return(lx,ly,Fx,Fy,tf)



def print_path(match,i,a,b,v,F=5): # afficher la trajectoire
    a, b, v = [x/100 for x in a], [x/100 for x in b], [x/100 for x in v]
    lx,ly,Fx,Fy,tf=path(a,b,v,F)
    plt.clf()
    plt.figure(figsize=(8, 8))
    field=plt.imread('{}/environnement_table.png'.format(chemin_tt_espace))
    plt.imshow(field, extent=[0,8,0,8])
    plt.plot(lx,ly) # positions prises par le joueur (ie sa trajectoire)
    plt.plot(a[0],a[1], 'ro') # pos init    
    plt.plot(b[0],b[1], 'bo') # pos fin
    plt.arrow(a[0],a[1],Fx/10,Fy/10,shape='full',lw=1.5,head_width=0.2) # flèche de force
    plt.arrow(a[0],a[1],v[0],v[1],shape='full',lw=1.5,head_width=0.2, color='g') # flèche de vitesse initiale
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig('{}/output/match_{}/inertia/inertia_frame_{}.png'.format(chemin_tt_espace, match, i))
    plt.close()



def heatmap(compet, match, set, point, tlim, que_taille_zone='non', num_vitesse=1): # taille de la zone de chaleur du joueurA + création des zones de chaleur frame par frame de chaque joueur pour un point donné dans un match donné (image stockée dans répertoire 'heatmap' dans le dossier du match dans tt-espace)
# si que_taille_zone=='oui', il n'y a pas de génération des images donc réduction du temps de calcul (passe d'environ 2 min à environ 1 min)
# tlim correspond au temps seuil : si le joueur atteint un point dans l'espace en + de tlim alors on considère que ce point ne va pas être atteint donc il ne sera pas coloré en rouge.
    if not os.path.isdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe")):
        os.mkdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe"))
    if not os.path.isdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap")):
        os.mkdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap"))
    if not os.path.isdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap_sans")):
        os.mkdir(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap_sans"))
    chemin_csv = os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"set_"+str(set)+"_point_"+str(point)+"_annotation.csv")
    df = pd.read_csv(chemin_csv)
    
    pos_init_b, vitesse_b, pos_init_jA, vitesse_jA, pos_init_jB, vitesse_jB = coordonnees(compet, match, set, point, lissage='oui', num_vitesse=num_vitesse)
    orientationJA,orientationJB = orientation(compet, match, set, point, pos_init_jA, lissage='non', joint=[11,14], multiplicateur=50)
    print(len(vitesse_jA))
    #vitesse_jA,vitesse_jB = orientationJA,orientationJB # Si on veut que ce soit l'orientation qui soit utilisée
    vitesse_jA = [[(vitesse_jA[i][0]+vitesse_jA[i][0]*orientationJA[i][0]/50)/2,(vitesse_jA[i][1]+vitesse_jA[i][1]*orientationJA[i][1]/50)/2] for i in range(len(vitesse_jA))] # Si on veut que l'orientation et la vitesse soient pris en compte
    vitesse_jB = [[(vitesse_jB[i][0]+vitesse_jB[i][0]*orientationJB[i][0]/50)/2,(vitesse_jB[i][1]+vitesse_jB[i][1]*orientationJA[i][1]/50)/2] for i in range(len(vitesse_jB))] # Si on veut que l'orientation et la vitesse soient pris en compte
    print(len(vitesse_jA))
    #print(vitesse_jA)

    # TEST ANNULATION DU RECUL
    for i in range(len(vitesse_jA)):
        if np.sign(vitesse_jA[i][1]) != np.sign(orientationJA[i][1]):  # Vérifie si les signes de y sont différents
            vitesse_jA[i][1] -= vitesse_jA[i][1]*0.8#+= np.sign(orientationJA[i][1])*50
            
    for i in range(len(vitesse_jB)):
        if np.sign(vitesse_jB[i][1]) != np.sign(orientationJB[i][1]):  # Vérifie si les signes de y sont différents
            vitesse_jB[i][1] -= vitesse_jB[i][1]*0.8 #+= np.sign(orientationJB[i][1])*50


    field=Image.open('{}/environnement_table.png'.format(chemin_tt_espace)).convert('RGB')
    W, h = field.size
    c=10

    filtres=[]
    for f in range(len(pos_init_b)):
        filtres.append(np.zeros((int(W/c),int(h/c))))
    
    tmin = np.inf

    surfaces_jA=[0]*len(pos_init_b)

    dist_front=200

    for f in range (len(pos_init_b)):
        debordement_jA=[]
        for i in range(int(W/c)):
            if 0<pos_init_jA[f][1]<h/2:  # jA en haut
                for j in range(int(h/2/c)):
                    ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                    filtres[f][i][j]=ta
                    if ta<tlim:
                        surfaces_jA[f]+=1
                        if i==0 and 'gauche' not in debordement_jA :
                            debordement_jA.append('gauche')
                        if j==0 and 'haut' not in debordement_jA :
                            debordement_jA.append('haut')
                        if i==W/c - 1 and 'droite' not in debordement_jA :
                            debordement_jA.append('droite')
                        if j==h/2/c - 1 and 'bas' not in debordement_jA :
                            debordement_jA.append('bas')
                for j in range(int(h/2/c), int(h/c)):
                    tb=time_to_point(pos_init_jB[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jB[f], F=500)
                    filtres[f][i][j]=tb
            elif h/2<pos_init_jA[f][1]<h: # jA en bas
                for j in range(int(h/2/c)):
                    tb=time_to_point(pos_init_jB[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jB[f], F=500)
                    filtres[f][i][j]=tb
                for j in range(int(h/2/c), int(h/c)):
                    ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                    filtres[f][i][j]=ta
                    if ta<tlim:
                        surfaces_jA[f]+=1
                        if i==0 and 'gauche' not in debordement_jA :
                            debordement_jA.append('gauche')
                        if j==h/2/c + 1 and 'haut' not in debordement_jA :
                            debordement_jA.append('haut')
                        if i==W/c - 1 and 'droite' not in debordement_jA :
                            debordement_jA.append('droite')
                        if j==h/c - 1 and 'bas' not in debordement_jA :
                            debordement_jA.append('bas')
        tminf=np.min(filtres[f])
        if tminf<tmin:
            tmin=tminf
        
        if debordement_jA!=[]: # il y a débordement du cadre (ie de la zone de jeu de 8m par 4m) de la zone accessible en moins de tlim
            if 0<pos_init_jA[f][1]<h/2:  # jA en haut
                if 'gauche' and 'haut' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c/2), 1):
                        for j in range(int(-dist_front/c), 0, 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(int(h/2/c)):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'haut' and 'droite' in debordement_jA :
                    for i in range(int(W/c/2), int(W/c + dist_front/c),1):
                        for j in range(int(-dist_front/c), 0, 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c)):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'droite' and 'bas' in debordement_jA :
                    for i in range(int(W/c/2), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c), int(h/2/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(0, int(h/2/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'bas' and 'gauche' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/2/c), 1):
                        for j in range(int(h/2/c), int(h/2/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(0, int(h/2/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'gauche' in debordement_jA :
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(int(-dist_front/c), int(h/2/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'haut' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c + dist_front/c), 1):
                        for j in range(int(-dist_front/c), 0, 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'droite' in debordement_jA :
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(int(-dist_front/c), int(h/2/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'bas' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c), int(h/2/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1

            elif h/2<pos_init_jA[f][1]<h: # jA en bas
                if 'gauche' and 'haut' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c/2), 1):
                        for j in range(int(h/2/c - dist_front/c), int(h/2/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(int(h/2/c), int(h/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'haut' and 'droite' in debordement_jA :
                    for i in range(int(W/c/2), int(W/c + dist_front/c),1):
                        for j in range(int(h/2/c - dist_front/c), int(h/2/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c), int(h/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'droite' and 'bas' in debordement_jA :
                    for i in range(int(W/c/2), int(W/c + dist_front/c), 1):
                        for j in range(int(h/c), int(h/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c), int(h/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'bas' and 'gauche' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/2/c), 1):
                        for j in range(int(h/c), int(h/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(int(h/2/c), int(h/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'gauche' in debordement_jA :
                    for i in range(int(-dist_front/c), 0, 1):
                        for j in range(int(h/2/c - dist_front/c), int(h/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'haut' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c - dist_front/c), int(h/2/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'droite' in debordement_jA :
                    for i in range(int(W/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/2/c - dist_front/c), int(h/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
                elif 'bas' in debordement_jA :
                    for i in range(int(-dist_front/c), int(W/c + dist_front/c), 1):
                        for j in range(int(h/c), int(h/c + dist_front/c), 1):
                            ta=time_to_point(pos_init_jA[f], [(i+1/2)*c, (j+1/2)*c], vitesse_jA[f], F=500)
                            if ta<tlim:
                                surfaces_jA[f]+=1
        #print(field)
        """if f in df['time_frappe'].values:
            row = df[df['time_frappe'] == f]
            x = row['coor_frappe_x'].values[0]
            y = row['coor_frappe_y'].values[0]
            #print('oui')
            field.putpixel((int(x+(324+152/2)),int(y+(263+274/2))),(0,0,0))"""
            #cv2.circle(field,(x+(324+152/2),y+(263+274/2)),5,(0,0,0),-1)

    liste_valeurs = []
    liste_valeurs_frappe = []
    if que_taille_zone=='non':
        for f in range(len(pos_init_b)):
            field_frame = field.copy()
            px_field = field_frame.load()
            for i in range(int(W/c)):
                for j in range(int(h/c)):
                    if filtres[f][i][j]>tlim :
                        g, b = 255, 255
                    else:
                        g = 255/(tlim-tmin)*(filtres[f][i][j]-tmin)
                        b = 255/(tlim-tmin)*(filtres[f][i][j]-tmin)
                    for l in range(c):
                        for k in range(c):
                            field_frame.putpixel([i*c+l,j*c+k], (255, int(g), int(b)))
                            px_field=field_frame.load()
            if f in df['time_frappe'].values:
                row = df.iloc[min(df['time_frappe'].tolist().index(f)+1,len(df['time_frappe'].values)-1)]#[df['time_frappe'] == max(0,.values.index(f))]
                x = row['coor_frappe_x']#.values[0]
                y = row['coor_frappe_y']#.values[0]
                draw = ImageDraw.Draw(field_frame)
                center = (min(int(x + (324 + 152 / 2)),799), -min(int(y + (263 + 274 / 2)),799))
                radius = 5  # Rayon du cercle
                liste_valeurs.append(field_frame.getpixel(center))
                #print(field.width,field.height)
                #print("radius",[center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius])
                draw.ellipse(
                    [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
                    fill=(0, 0, 0)  # Couleur noire
                )
            field_frame.save('{}/output/match_{}/heatmap/heatmap_frame_{}.png'.format(chemin_tt_espace, match,f))
            field_frame.save(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap_sans","image_"+str(f)+".png"))
            
            field_frame_np = np.array(field_frame)
            heatmap = field_frame_np[:, :, 2].copy()
            heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            heatmap_colored = np.full((h, W, 3), 255, dtype=np.uint8) #field_frame_np.copy()#cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2BGR)
            # Détection des contours pour des plages de densité
            contours = []
            thresholds = [50, 100, 150, 200]  # Seuils de densité
            for t in thresholds:
                # Appliquer un seuil pour créer une image binaire
                _, binary = cv2.threshold(heatmap_normalized, t, 255, cv2.THRESH_BINARY)
                blurred_image = cv2.GaussianBlur(binary, (5, 5), 0)

                # Détecter les contours avec Canny
                edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
                
                # Trouver les contours
                contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours.append((t, contour))
                #print(contour)
                #binary_image_pillow = Image.fromarray(binary)
                #binary_image_pillow.save('{}/output/match_{}/heatmap/binaire_heatmap_frame_{}_{}.png'.format(chemin_tt_espace, match,f,t))
                cv2.drawContours(heatmap_colored, contour, -1, (0, 0, 0), 1)
            heatmap_with_contours = Image.fromarray(cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(heatmap_with_contours)
            if f in df['time_frappe'].values:
                d.ellipse(
                    [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
                    fill=(0, 0, 0)  # Couleur noire
                )
            heatmap_with_contours.save('{}/output/match_{}/heatmap/zone_heatmap_frame_{}.png'.format(chemin_tt_espace, match,f))
            
            image1 = np.array(field_frame)  # Première image (fond)
            image2 = np.array(heatmap_with_contours)  # Deuxième image (pixels à ajouter)

            # Détecter les pixels non blancs dans l'image2
            mask = ~(np.all(image2 == [255, 255, 255], axis=-1))  # Masque des pixels non blancs

            # Remplacer les pixels correspondants dans image1
            image1[mask] = image2[mask]

            # Convertir l'image résultante en PIL
            result_image = Image.fromarray(image1)
            
            result_image.save(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"heatmap","image_"+str(f)+".png"))
            

            width, height = 800, 800

            joueurA = "ALEXIS-LEBRUN"
            effet = "topspin"

            IMAGE_PATH = '{}/output/match_{}/heatmap/heatmap_frame_{}.png'.format(chemin_tt_espace, match,f)
            # Chargement de l'image
            image = cv2.imread(IMAGE_PATH)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Détection des pixels non blancs
            non_white_pixels = np.column_stack(np.where(gray < 250))[:]  # Limite de traitement

            pipeline_path = USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"]
            df_revers = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_revers_{effet}_enveloppe.csv"))[['X', 'Y']]#.values.astype(np.int32)
            df_coup_droit = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_coup_droit_{effet}_enveloppe.csv"))[['X', 'Y']]#.values.astype(np.int32)
            #df_tous = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_tous_enveloppe.csv"))[['X', 'Y']]#.values.astype(np.int32)
            premier = True

            # Pré-allocation de l'image résultat
            result = np.zeros_like(image, dtype=np.uint8)
            #result_gris = np.zeros_like(image, dtype=np.uint16)


            # Fonction d'application des enveloppes convexes
            def appliquer_enveloppe(image_result, points, x, y, couleur):
                """Applique une enveloppe convexe centrée en (x, y)"""
                points_trans = np.copy(points)
                points_trans[:, 0] = -points_trans[:, 0] + x
                points_trans[:, 1] += y
                hull = cv2.convexHull(points_trans)
                cv2.drawContours(image_result, [hull], -1, couleur, thickness=-1)


            def sliding_window_max(image, window_size=10):
                """
                Applique une fenêtre glissante de taille donnée sur une image et crée une nouvelle image
                où chaque pixel prend la valeur maximale des pixels correspondants sur les canaux RGB.
                
                :param image: Image d'entrée (numpy array de forme (H, W, 3))
                :param window_size: Taille de la fenêtre glissante (par défaut 10x10)
                :return: Image transformée
                """
                # Obtenir les dimensions de l'image
                h, w, c = image.shape
                output = np.zeros((h, w, c), dtype=np.uint8)
                
                # Appliquer un filtre maximum par bloc de 10x10
                for i in range(0, h, window_size):
                    for j in range(0, w, window_size):
                        # Extraire la région de la fenêtre
                        region = image[i:i+window_size, j:j+window_size]
                        # Calculer le maximum par canal
                        max_value = np.max(region, axis=(0, 1))
                        # Remplir la nouvelle image avec cette valeur
                        output[i:i+window_size, j:j+window_size] = max_value

                return output

            c = 10
            intensite = 1#int(c/2)
            # Traitement de tous les pixels
            for i, (y, x) in enumerate(non_white_pixels):
                if y < 400 and pos_init_jA[f][1] < 400 and y%c == 0 and x%c == 0:
                    overlay = np.zeros_like(image, dtype=np.uint8)
                    appliquer_enveloppe(overlay, df_revers.values.astype(np.int32), x, y, (0, intensite, 0))  # Vert foncé
                    appliquer_enveloppe(overlay, df_coup_droit.values.astype(np.int32), x, y, (intensite, 0, 0))  # Bleu foncé
                    #appliquer_enveloppe(result_gris, df_tous, x, y, (100, 100, 100))  # Bleu foncé
                    result = cv2.add(result, overlay)  # Superposition efficace
                
                elif y > 400 and pos_init_jA[f][1] > 400 and y%c == 0 and x%c == 0:
                    overlay = np.zeros_like(image, dtype=np.uint8)
                    if premier:
                        df_revers["X"] = -df_revers["X"]
                        df_revers["Y"] = -df_revers["Y"]
                        df_coup_droit["X"] = -df_coup_droit["X"]
                        df_coup_droit["Y"] = -df_coup_droit["Y"]
                        premier = False
                    appliquer_enveloppe(overlay, df_revers.values.astype(np.int32), x, y, (0, intensite, 0))  # Vert foncé
                    appliquer_enveloppe(overlay, df_coup_droit.values.astype(np.int32), x, y, (intensite, 0, 0))  # Bleu foncé
                    #appliquer_enveloppe(result_gris, df_tous, x, y, (100, 100, 100))  # Bleu foncé
                    result = cv2.add(result, overlay)  # Superposition efficace
            
            result = sliding_window_max(result, window_size=c)
            #mask = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
            #result[mask] = [255, 255, 255]

            chemin_enregistrement = os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe",str(f)+".png")
            cv2.imwrite(chemin_enregistrement, result)
            if f in df['time_frappe'].values:
                row = df.iloc[min(df['time_frappe'].tolist().index(f),len(df['time_frappe'].values)-1)]#[df['time_frappe'] == max(0,.values.index(f))]
                x = row['coor_frappe_x']#.values[0]
                y = row['coor_frappe_y']#.values[0]
                center = (min(int(x + (324 + 152 / 2)),799), min(int(y + (263 + 274 / 2)),799))
                #print(row["time_frappe"],y,center)
                cv2.circle(result, center, 2, (255,255,255), -1)
                chemin_enregistrement2 = os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe",str(f)+"_frappe.png")
                cv2.imwrite(chemin_enregistrement2, result)


            im = Image.open(chemin_enregistrement).convert('RGB')
            im_np = np.array(im)
            heatmap = im_np[:, :, 0].copy()
            heatmap_normalized_r = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            heatmap = im_np[:, :, 1].copy()
            heatmap_normalized_g = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            heatmap = im_np[:, :, 2].copy()
            heatmap_normalized_b = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
            #im2 = im.copy()
            if f in df['time_frappe'].values:
                row = df.iloc[min(df['time_frappe'].tolist().index(f),len(df['time_frappe'].values)-1)]#[df['time_frappe'] == max(0,.values.index(f))]
                x = row['coor_frappe_x']#.values[0]
                y = row['coor_frappe_y']#.values[0]
                center = (min(int(x + (324 + 152 / 2)),799), min(int(y + (263 + 274 / 2)),799))
                radius = 5  # Rayon du cercle
                #liste_valeurs_frappe.append(im2.getpixel(center))
                val = '(' + str(heatmap_normalized_r[center[1], center[0]]) + ',' + str(heatmap_normalized_g[center[1], center[0]]) + ',' + str(heatmap_normalized_b[center[1], center[0]]) + ')'
                liste_valeurs_frappe.append(val)


            '''
            # Création d'une image blanche
            image = np.ones((height, width, 3), dtype=np.uint8) * 255

            coup = "coup_droit"
            effet = "topspin"
            joueurA = "ALEXIS-LEBRUN"
            if os.path.isfile(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_tous_enveloppe.csv")):
                df_enveloppe = pd.read_csv(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_tous_enveloppe.csv"))  # Remplace par le bon nom de fichier
                if pos_init_jA[f][1] < 400:
                    df_enveloppe["X"] = -df_enveloppe["X"]
                    df_enveloppe["X"] += pos_init_jA[f][0]
                    df_enveloppe["Y"] += pos_init_jA[f][1]
                else:
                    df_enveloppe["X"] += pos_init_jA[f][0]
                    df_enveloppe["Y"] = -df_enveloppe["Y"]
                    df_enveloppe["Y"] += pos_init_jA[f][1]   

                # Convertir les colonnes en array NumPy
                points = df_enveloppe[['X', 'Y']].values.astype(np.int32)  # Conversion en entier


                # Calcul de l'enveloppe convexe
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]  # Sommets de l'enveloppe convexe

                # Convertir les points pour OpenCV (format requis)
                hull_pts_cv2 = hull_points.reshape((-1, 1, 2))  # Reshape pour OpenCV

                # Colorier l'intérieur de l'enveloppe en bleu
                cv2.fillPoly(image, [hull_pts_cv2], (128,128,128))


            for coup in ["coup_droit","revers"]:
                if coup == "coup_droit":
                    couleur = (255, 150, 150)
                else:
                    couleur = (150, 255, 150)

                if os.path.isfile(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_enveloppe.csv")):
                    df_enveloppe = pd.read_csv(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_enveloppe.csv"))  # Remplace par le bon nom de fichier
                    if pos_init_jA[f][1] < 400:
                        df_enveloppe["X"] = -df_enveloppe["X"]
                        df_enveloppe["X"] += pos_init_jA[f][0]
                        df_enveloppe["Y"] += pos_init_jA[f][1]
                    else:
                        df_enveloppe["X"] += pos_init_jA[f][0]
                        df_enveloppe["Y"] = -df_enveloppe["Y"]
                        df_enveloppe["Y"] += pos_init_jA[f][1]   

                    # Convertir les colonnes en array NumPy
                    points = df_enveloppe[['X', 'Y']].values.astype(np.int32)  # Conversion en entier


                    # Calcul de l'enveloppe convexe
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]  # Sommets de l'enveloppe convexe

                    # Convertir les points pour OpenCV (format requis)
                    hull_pts_cv2 = hull_points.reshape((-1, 1, 2))  # Reshape pour OpenCV

                    # Colorier l'intérieur de l'enveloppe en bleu
                    cv2.fillPoly(image, [hull_pts_cv2], couleur)

                """for qualite in [1,"nulle"]:
                    if os.path.isfile(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_qualite_"+str(qualite)+"_enveloppe.csv")):
                        df_enveloppe = pd.read_csv(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_qualite_"+str(qualite)+"_enveloppe.csv"))
                        if pos_init_jA[f][1] < 400:
                            df_enveloppe["X"] = -df_enveloppe["X"]
                            df_enveloppe["X"] += pos_init_jA[f][0]
                            df_enveloppe["Y"] += pos_init_jA[f][1]
                        else:
                            df_enveloppe["X"] += pos_init_jA[f][0]
                            df_enveloppe["Y"] = -df_enveloppe["Y"]
                            df_enveloppe["Y"] += pos_init_jA[f][1]
                        if coup == "coup_droit":
                            if qualite == "nulle":
                                couleur = (255, 50, 50)
                            else:
                                couleur = (255, 200, 200)
                        else:
                            if qualite == "nulle":
                                couleur = (50, 255, 50)
                            else:
                                couleur = (200, 255, 200)
                    
                        if qualite == "nulle":
                            print(enveloppe)
                        points = df_enveloppe[['X', 'Y']].values.astype(np.int32)  # Conversion en entier
                        # Calcul de l'enveloppe convexe
                        hull = ConvexHull(points)
                        hull_points = points[hull.vertices]  # Sommets de l'enveloppe convexe
                        # Convertir les points pour OpenCV (format requis)
                        hull_pts_cv2 = hull_points.reshape((-1, 1, 2))  # Reshape pour OpenCV
                        # Colorier l'intérieur de l'enveloppe en bleu
                        cv2.fillPoly(image, [hull_pts_cv2], couleur)"""

            # Affichage de l'image
            cv2.imwrite(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe",str(f)+".png"), image)'''

    #print(liste_valeurs)
    df['zone_modele'] = liste_valeurs
    df.to_csv(chemin_csv.replace(".csv","_zone_frappe_orientation_"+str(num_vitesse)+".csv"), index=False)
    
    df['zone_modele'] = liste_valeurs_frappe
    df.to_csv(chemin_csv.replace(".csv","_zone_frappe_modele_frappe_"+str(num_vitesse)+".csv"), index=False)
    return surfaces_jA



def creer_dossier_heatmap_qualite_coup(csv_qualite,match,joueur1,joueur2):
    """
        Fonction permettant de créer des dossiers en fonction de la qualité des coups du type de coup et du joueur. Dans chaque dossier on met les images des heatmap 
        au moment de la frappe
        Entrée:
                - Le chemin du csv "_annotation_qualite.csv"
    """

    
    if not os.path.isdir(os.path.join("qualite_coup")):
        os.mkdir(os.path.join("qualite_coup"))

    if not os.path.isdir(os.path.join("qualite_coup",match)):
        os.mkdir(os.path.join("qualite_coup",match))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1)):
        os.mkdir(os.path.join("qualite_coup",match,joueur1))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2)):
        os.mkdir(os.path.join("qualite_coup",match,joueur2))

    
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"topspin")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"topspin"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"topspin")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"topspin"))

    
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"poussette")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"poussette"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"poussette")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"poussette"))

        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"bloc")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"bloc"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"bloc")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"bloc"))
        

    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"flip")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"flip"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"flip")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"flip"))



        
    
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"topspin","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"topspin","bon"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"topspin","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"topspin","bon"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"topspin","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"topspin","neutre"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"topspin","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"topspin","neutre"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"topspin","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"topspin","mauvais"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"topspin","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"topspin","mauvais"))

    
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"poussette","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"poussette","neutre"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"poussette","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"poussette","neutre"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"poussette","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"poussette","mauvais"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"poussette","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"poussette","mauvais"))

        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"bloc","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"bloc","bon"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"bloc","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"bloc","bon"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"bloc","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"bloc","neutre"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"bloc","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"bloc","neutre"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"bloc","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"bloc","mauvais"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"bloc","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"bloc","mauvais"))

        
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"flip","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"flip","bon"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"flip","bon")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"flip","bon"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"flip","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"flip","neutre"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"flip","neutre")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"flip","neutre"))
        
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur1,"flip","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur1,"flip","mauvais"))
    if not os.path.isdir(os.path.join("qualite_coup",match,joueur2,"flip","mauvais")):
        os.mkdir(os.path.join("qualite_coup",match,joueur2,"flip","mauvais"))
        


    df_qualite = pd.read_csv(csv_qualite)
    df_enrichi = pd.read_csv(csv_qualite.replace("_annotation_qualite.csv","_annotation_enrichi.csv"))

    if 'num_point' in df_enrichi.columns:
        # Ajouter la colonne 'x' du premier DataFrame au second
        df_qualite['num_point'] = df_enrichi['num_point'].values
        df_qualite['num_coup'] = df_enrichi['num_coup'].values


    for joueur in [joueur1,joueur2]:
        for effet_coup in ["topspin","bloc","flip","poussette"]:
            df_filtre = df_qualite[(df_qualite["nom"] == joueur) & (df_qualite["effet_coup"] == effet_coup) & (df_qualite["qualite_coup"] == 1)]
            

            liste_debut = df_qualite[df_qualite["num_coup"] == 1]["debut"].tolist()
            print(df_qualite)
            for i in range(len(df_filtre)):
                #print(df_filtre.iloc[i]["num_point"])
                chemin_image_source = os.path.join("output","match_"+match,"point_"+str(df_filtre.iloc[i]["num_point"]),"image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png")
                #print(chemin_image_source)
                shutil.copy(chemin_image_source, os.path.join("qualite_coup",match,joueur,effet_coup,"bon","point_"+str(df_filtre.iloc[i]["num_point"])+"_image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png"))


                
            df_filtre = df_qualite[(df_qualite["nom"] == joueur) & (df_qualite["effet_coup"] == effet_coup) & (df_qualite["qualite_coup"] == 0)]
            

            liste_debut = df_qualite[df_qualite["num_coup"] == 1]["debut"].tolist()
            print(df_qualite)
            for i in range(len(df_filtre)):
                #print(df_filtre.iloc[i]["num_point"])
                chemin_image_source = os.path.join("output","match_"+match,"point_"+str(df_filtre.iloc[i]["num_point"]),"image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png")
                #print(chemin_image_source)
                shutil.copy(chemin_image_source, os.path.join("qualite_coup",match,joueur,effet_coup,"neutre","point_"+str(df_filtre.iloc[i]["num_point"])+"_image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png"))

                
            df_filtre = df_qualite[(df_qualite["nom"] == joueur) & (df_qualite["effet_coup"] == effet_coup) & (df_qualite["qualite_coup"] == -1)]
            

            liste_debut = df_qualite[df_qualite["num_coup"] == 1]["debut"].tolist()
            print(df_qualite)
            for i in range(len(df_filtre)):
                #print(df_filtre.iloc[i]["num_point"])
                chemin_image_source = os.path.join("output","match_"+match,"point_"+str(df_filtre.iloc[i]["num_point"]),"image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png")
                #print(chemin_image_source)
                shutil.copy(chemin_image_source, os.path.join("qualite_coup",match,joueur,effet_coup,"mauvais","point_"+str(df_filtre.iloc[i]["num_point"])+"_image_"+str(df_filtre.iloc[i]["time_frappe"]-liste_debut[df_filtre.iloc[i]["num_point"]])+".png"))

    
if __name__ == "__main__":
    creer_dossier_heatmap_qualite_coup("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/ALEXIS-LEBRUN_vs_MA-LONG_annotation_qualite.csv",
                                       "ALEXIS-LEBRUN_vs_MA-LONG","ALEXIS-LEBRUN","MA-LONG")