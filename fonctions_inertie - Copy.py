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



def heatmap(compet, match, set, point, tlim, que_taille_zone='non', num_vitesse=1):
    """
    Génère une heatmap pour un point donné dans un match de tennis de table.
    
    :param compet: Nom de la compétition.
    :param match: Nom du match.
    :param set: Numéro du set.
    :param point: Numéro du point.
    :param tlim: Temps seuil pour considérer un point atteignable.
    :param que_taille_zone: Si 'oui', ne génère pas les images pour accélérer le calcul.
    :param num_vitesse: Indice de la vitesse à utiliser.
    """

    base_path = os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"], compet, match, "clips", f"set_{set}_point_{point}")
    enveloppe_path = os.path.join(base_path, "enveloppe")
    os.makedirs(enveloppe_path, exist_ok=True)
    
    chemin_csv = os.path.join(base_path, f"set_{set}_point_{point}_annotation.csv")
    df = pd.read_csv(chemin_csv)

    # Récupération des données
    pos_init_b, vitesse_b, pos_init_jA, vitesse_jA, pos_init_jB, vitesse_jB = coordonnees(compet, match, set, point, lissage='oui', num_vitesse=num_vitesse)
    orientationJA, orientationJB = orientation(compet, match, set, point, pos_init_jA, lissage='non', joint=[11, 14], multiplicateur=50)

    # Prise en compte de l'orientation dans la vitesse
    vitesse_jA = [(vx + vx * oJA[0] / 50) / 2, (vy + vy * oJA[1] / 50) / 2] for (vx, vy), oJA in zip(vitesse_jA, orientationJA)]
    vitesse_jB = [(vx + vx * oJB[0] / 50) / 2, (vy + vy * oJB[1] / 50) / 2] for (vx, vy), oJB in zip(vitesse_jB, orientationJB)]

    # Annulation du recul si l'orientation et la vitesse sont opposées
    for i in range(len(vitesse_jA)):
        if np.sign(vitesse_jA[i][1]) != np.sign(orientationJA[i][1]):
            vitesse_jA[i][1] *= 0.2  # Réduction de 80%
        if np.sign(vitesse_jB[i][1]) != np.sign(orientationJB[i][1]):
            vitesse_jB[i][1] *= 0.2  # Réduction de 80%

    # Chargement de l'image du terrain
    field = Image.open(f'{chemin_tt_espace}/environnement_table.png').convert('RGB')
    W, h = field.size
    c = 10  # Taille de la grille

    filtres = np.zeros((len(pos_init_b), int(W/c), int(h/c)))

    surfaces_jA = np.zeros(len(pos_init_b))
    tmin = np.inf
    dist_front = 200  # Distance d'expansion en cas de débordement

    for f in range(len(pos_init_b)):
        debordement_jA = set()

        for i in range(int(W/c)):
            for j in range(int(h/c)):
                pos = [(i + 0.5) * c, (j + 0.5) * c]
                if pos_init_jA[f][1] < h/2:  # Joueur A en haut
                    ta = time_to_point(pos_init_jA[f], pos, vitesse_jA[f], F=500)
                    tb = time_to_point(pos_init_jB[f], pos, vitesse_jB[f], F=500) if j >= h/2/c else np.inf
                else:  # Joueur A en bas
                    ta = time_to_point(pos_init_jA[f], pos, vitesse_jA[f], F=500) if j >= h/2/c else np.inf
                    tb = time_to_point(pos_init_jB[f], pos, vitesse_jB[f], F=500)

                filtres[f][i][j] = min(ta, tb)
                
                if ta < tlim:
                    surfaces_jA[f] += 1
                    if i == 0:
                        debordement_jA.add('gauche')
                    if j == 0:
                        debordement_jA.add('haut')
                    if i == int(W/c) - 1:
                        debordement_jA.add('droite')
                    if j == int(h/c) - 1:
                        debordement_jA.add('bas')

        tmin = min(tmin, np.min(filtres[f]))

        # Gestion des débordements
        for direction in debordement_jA:
            if direction == 'gauche':
                i_range = range(int(-dist_front/c), 0)
                j_range = range(int(-dist_front/c), int(h/c + dist_front/c))
            elif direction == 'droite':
                i_range = range(int(W/c), int(W/c + dist_front/c))
                j_range = range(int(-dist_front/c), int(h/c + dist_front/c))
            elif direction == 'haut':
                i_range = range(int(-dist_front/c), int(W/c + dist_front/c))
                j_range = range(int(-dist_front/c), 0)
            elif direction == 'bas':
                i_range = range(int(-dist_front/c), int(W/c + dist_front/c))
                j_range = range(int(h/c), int(h/c + dist_front/c))
            
            for i in i_range:
                for j in j_range:
                    ta = time_to_point(pos_init_jA[f], [(i + 0.5) * c, (j + 0.5) * c], vitesse_jA[f], F=500)
                    if ta < tlim:
                        surfaces_jA[f] += 1

    # Génération des images si nécessaire
    if que_taille_zone == 'non':
        for f in range(len(pos_init_b)):
            field_frame = field.copy()
            px_field = field_frame.load()
            for i in range(int(W/c)):
                for j in range(int(h/c)):
                    time_value = filtres[f][i][j]
                    if time_value < tlim:
                        intensity = int(255 * (1 - time_value / tlim))
                        px_field[i*c, j*c] = (255, intensity, intensity)

            field_frame.save(os.path.join(enveloppe_path, f"heatmap_{f}.png"))



            

            width, height = 800, 800

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
            cv2.imwrite(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips","set_"+str(set)+"_point_"+str(point),"enveloppe",str(f)+".png"), image)
    #print(liste_valeurs)
    df['zone_modele'] = liste_valeurs
    df.to_csv(chemin_csv.replace(".csv","_zone_frappe_orientation_"+str(num_vitesse)+".csv"), index=False)
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