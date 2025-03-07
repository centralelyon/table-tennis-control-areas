import numpy as np
import csv
import cv2
import os
import json
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import traja
import alphashape
from shapely.geometry import Polygon, MultiPolygon, Point
from scipy.interpolate import splprep, splev
from math import sqrt

from config import chemin_pipeline, chemin_tt_espace, USER_PREFERENCE

from fonctions_time import time_to_point, frappe



def coordonnees(compet, match, set, point, lissage='non',num_vitesse=1): # dans le repère tkinter (x vertical, y horizontal descendant, en haut à gauche de l'image)
    """
    Entrée:
            - compet
            - match
            - set
            - point
            - lissage
            - num_vitesse (1 ou 2 qui indique quelle vitesse on veut, celle se basant sur la frame précédente (1) ou celle qui se base sur la précédente et la suivante (2))
    Sortie:
            - pos_init_b (position de la balle)
            - vitesse_b (vitesse de la balle)
            - pos_init_jA (position du joueur A)
            - vitesse_jA (vitesse du joueur A)
            - pos_init_jB (position du joueur B)
            - vitesse_jB (vitesse du joueur B)
    """
    nom_point='set_{}_point_{}'.format(set, point)

    cap=cv2.VideoCapture('{}/{}/{}/clips/{}/{}.mp4'.format(chemin_pipeline, compet, match, nom_point, nom_point))
    FPS=cap.get(cv2.CAP_PROP_FPS)

    with open('{}/{}/{}/clips/{}/csv_json_openpose/{}_zone_joueur_avec_pos_balle_3D.csv'.format(chemin_pipeline, compet, match, nom_point, nom_point), newline='') as fichier_openpose :
        openpose=[]
        reader=csv.reader(fichier_openpose)
        for ligne in reader :
            openpose.append(ligne)
    
    coord_jA=[]
    coord_jB=[]
    coord_balle=[]
    for i in range(1,len(openpose)):
        if openpose[i][1]=='0' : #jA
            coord_jA.append([float(openpose[i][2]), -float(openpose[i][3])])
        elif openpose[i][1]=='1': #jB
            coord_jB.append([float(openpose[i][2]), -float(openpose[i][3])])
        elif openpose[i][1]=='4': #balle
            if openpose[i][2]=='': # si jamais on n'a pas les coordonnées de la balle, on la met hors champ
                coord_balle.append([850,850])
            else:
                coord_balle.append([float(openpose[i][2]), -float(openpose[i][3])])
    
    pos_init_b, vitesse_b, pos_init_jA, vitesse_jA, pos_init_jB, vitesse_jB = [],[[0,0]],[],[[0,0]],[],[[0,0]]
    for i in range(len(coord_balle)):
        pos_init_b.append([coord_balle[i][0], coord_balle[i][1]]) 
        pos_init_jA.append([coord_jA[i][0], coord_jA[i][1]])
        pos_init_jB.append([coord_jB[i][0], coord_jB[i][1]])
    for i in range(1,len(coord_balle)-1):
        vitesse_b.append([(coord_balle[i][0]-coord_balle[i-1][0])*FPS, -(coord_balle[i][1]-coord_balle[i-1][1])*FPS])
        #print((coord_jA[i][0]-coord_jA[i-1][0])*FPS,(((coord_jA[i][0]-coord_jA[i-1][0])+(coord_jA[i+1][0]-coord_jA[i][0]))/2)*FPS)
        if num_vitesse == 1:
            vitesse_jA.append([(coord_jA[i][0]-coord_jA[i-1][0])*FPS, -(coord_jA[i][1]-coord_jA[i-1][1])*FPS])
            vitesse_jB.append([(coord_jB[i][0]-coord_jB[i-1][0])*FPS, -(coord_jB[i][1]-coord_jB[i-1][1])*FPS])
        else:
            vitesse_jA.append([(((coord_jA[i][0]-coord_jA[i-1][0])+(coord_jA[i+1][0]-coord_jA[i][0]))/2)*FPS, -(((coord_jA[i][1]-coord_jA[i-1][1])+(coord_jA[i+1][1]-coord_jA[i][1]))/2)*FPS])
            vitesse_jB.append([(((coord_jB[i][0]-coord_jB[i-1][0])+(coord_jB[i+1][0]-coord_jB[i][0]))/2)*FPS, -(((coord_jB[i][1]-coord_jB[i-1][1])+(coord_jB[i+1][1]-coord_jB[i][1]))/2)*FPS])
    vitesse_jA.append([(coord_jA[len(coord_balle)-1][0]-coord_jA[len(coord_balle)-1-1][0])*FPS, -(coord_jA[len(coord_balle)-1][1]-coord_jA[len(coord_balle)-1-1][1])*FPS])
    vitesse_jB.append([(coord_jB[len(coord_balle)-1][0]-coord_jB[len(coord_balle)-1-1][0])*FPS, -(coord_jB[len(coord_balle)-1][1]-coord_jB[len(coord_balle)-1-1][1])*FPS])
    for i in range(len(coord_balle)):
        pos_init_b[i]=[pos_init_b[i][0]+400,-pos_init_b[i][1]+400]
        pos_init_jA[i]=[pos_init_jA[i][0]+400,-pos_init_jA[i][1]+400]
        pos_init_jB[i]=[pos_init_jB[i][0]+400,-pos_init_jB[i][1]+400]

    if lissage=='oui':
        cap=cv2.VideoCapture('{}/{}/{}/clips/set_{}_point_{}/set_{}_point_{}.mp4'.format(chemin_pipeline, compet, match, set, point, set, point))
        FPS=cap.get(cv2.CAP_PROP_FPS) # pour récupérer le nombre de fps. avant sur yt : 30. maintenant : 25. comme c'est susceptible de changer j'utilise une fonction qui récupère cette valeur automatiquement lorsqu'on lui donne la vidéo originale.

        # lissage de la trajectoire des joueurs par le filtre de Savitzky-Golay
        dico_pos_jA = pd.DataFrame(pos_init_jA, columns=['x', 'y'])
        dico_pos_jB = pd.DataFrame(pos_init_jB, columns=['x', 'y'])

        pos_init_jA_lissee=traja.trajectory.smooth_sg(dico_pos_jA, w=11)
        pos_init_jB_lissee=traja.trajectory.smooth_sg(dico_pos_jB, w=11)

        pos_init_jA = [[x, y] for x, y in zip(pos_init_jA_lissee['x'], pos_init_jA_lissee['y'])]
        pos_init_jB = [[x, y] for x, y in zip(pos_init_jB_lissee['x'], pos_init_jB_lissee['y'])]

        for i in range(1,len(pos_init_b)-1):
            if num_vitesse == 1:
                vitesse_jA[i]=[(pos_init_jA[i][0]-pos_init_jA[i-1][0])*FPS, (pos_init_jA[i][1]-pos_init_jA[i-1][1])*FPS]
                vitesse_jB[i]=[(pos_init_jB[i][0]-pos_init_jB[i-1][0])*FPS, (pos_init_jB[i][1]-pos_init_jB[i-1][1])*FPS]
            else:
                vitesse_jA[i]=[(((pos_init_jA[i][0]-pos_init_jA[i-1][0])+(pos_init_jA[i+1][0]-pos_init_jA[i][0]))/2)*FPS, (((pos_init_jA[i][1]-pos_init_jA[i-1][1])+(pos_init_jA[i+1][1]-pos_init_jA[i][1]))/2)*FPS]
                vitesse_jB[i]=[(((pos_init_jB[i][0]-pos_init_jB[i-1][0])+(pos_init_jB[i+1][0]-pos_init_jB[i][0]))/2)*FPS, (((pos_init_jB[i][1]-pos_init_jB[i-1][1])+(pos_init_jB[i+1][1]-pos_init_jB[i][1]))/2)*FPS]
            #print((pos_init_jA[i][0]-pos_init_jA[i-1][0])*FPS,(((pos_init_jA[i][0]-pos_init_jA[i-1][0])+(pos_init_jA[i+1][0]-pos_init_jA[i][0]))/2)*FPS)

    return pos_init_b, vitesse_b, pos_init_jA, vitesse_jA, pos_init_jB, vitesse_jB


def orientation(compet, match, set, point, pos_init_jA, lissage='non', joint=[11,14], multiplicateur=1): # dans le repère tkinter (x vertical, y horizontal descendant, en haut à gauche de l'image)
    """
    Fonction copiée et modifiée de coordonnees() afin de sortir l'orientation des joueurs en fonction des joints
    Entrée:
            - compet
            - match
            - set
            - point
            - pos_init_jA (sert à savoir quel joueur est de quel côté)
            - lissage
            - joints (un couple)
    Sortie:
            - pos_init_jA (position du joueur A)
            - vitesse_jA (orientation du joueur A)
            - pos_init_jB (position du joueur B)
            - vitesse_jB (orientation du joueur B)
    """
    nom_point='set_{}_point_{}'.format(set, point)

    cap=cv2.VideoCapture('{}/{}/{}/clips/{}/{}.mp4'.format(chemin_pipeline, compet, match, nom_point, nom_point))
    FPS=cap.get(cv2.CAP_PROP_FPS)
    csv_pose = pd.read_csv('{}/{}/{}/clips/{}/csv_json_openpose/{}_openpose.csv'.format(chemin_pipeline, compet, match, nom_point, nom_point))
    """with open('{}/{}/{}/clips/{}/csv_json_openpose/{}set_1_point_0_openpose.csv'.format(chemin_pipeline, compet, match, nom_point, nom_point), newline='') as fichier_openpose :
        openpose=[]
        reader=csv.reader(fichier_openpose)
        for ligne in reader :
            openpose.append(ligne)"""
    
    point_table = recuperation_points_table(os.path.join(chemin_pipeline, compet, match, match+"_perspective.json"))
    point_table = np.array(point_table, dtype=np.float32)
    point_table_vue_dessus = np.array([[324,263],[476,263],[476,537],[324,537]])
    H, _ = cv2.findHomography(point_table, point_table_vue_dessus)
    H_inv, _ = cv2.findHomography(point_table_vue_dessus,point_table)


    csv_pose_filtre = csv_pose[csv_pose["est_joueur"] == 1]

    df_frame1 = csv_pose_filtre[csv_pose_filtre["frame"] == 1]

    # Extraire les coordonnées de chaque joueur
    #joueurs = df_frame1.groupby("numero_pers")[["x", "y"]].mean()  # On prend la moyenne si plusieurs entrées par joueur

    # Coordonnées de pos_init_jA
    xA, yA = pos_init_jA[1]


    xA, yA = cv2.perspectiveTransform(np.array([[[xA, yA]]], dtype=np.float32), H_inv)[0][0]
    # Extraction des coordonnées transformées
    #x_transforme, y_transforme = point_transforme[0][0]

    df_frame1["distance"] = np.sqrt((df_frame1["x"] - xA)**2 + (df_frame1["y"] - yA)**2)
    # Trouver le joueur le plus proche et le plus éloigné
    liste_num_joueur = df_frame1["numero_pers"].unique()
    num_joueur1 = df_frame1.loc[df_frame1["distance"].idxmin(), "numero_pers"]
    if num_joueur1 == liste_num_joueur[0]:
        num_joueur2 = liste_num_joueur[1]
    else:
        num_joueur2 = liste_num_joueur[0]
    #num_joueur2 = df_frame1.loc[df_frame1["distance"].idxmax(), "numero_pers"]

    csv_pose_filtre_joueur1 = csv_pose_filtre[csv_pose_filtre["numero_pers"] == num_joueur1]
    csv_pose_filtre_joueur2 = csv_pose_filtre[csv_pose_filtre["numero_pers"] == num_joueur2]

    nb_frame = csv_pose_filtre_joueur1["frame"].max()
        

    orientationJA = []
    orientationJB = []

    


    for i in range(0,nb_frame+1):
        if len(csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[0]), "x"].values) > 0:
            x_new = csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[0]), "x"].values
        if len(csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[0]), "y"].values) > 0:
            y_new = csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[0]), "y"].values
        nouveau_point = np.array([[[x_new[0], y_new[0]]]], dtype=np.float32)
        #nouveau_point = np.expand_dims(nouveau_point, axis=0)  # Nécessaire pour OpenCV
        point_transforme = cv2.perspectiveTransform(nouveau_point, H)
        # Extraction des coordonnées transformées
        x_transforme, y_transforme = point_transforme[0][0]
        point1_JA = [x_transforme,y_transforme]

        if len(csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[1]), "x"].values) > 0:
            x_new = csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[1]), "x"].values
        if len(csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[1]), "y"].values) > 0:
            y_new = csv_pose_filtre_joueur1.loc[(csv_pose_filtre_joueur1["frame"] == i) & (csv_pose_filtre_joueur1["joint"] == joint[1]), "y"].values
        nouveau_point = np.array([[[x_new[0], y_new[0]]]], dtype=np.float32)
        #nouveau_point = np.expand_dims(nouveau_point, axis=0)  # Nécessaire pour OpenCV
        point_transforme = cv2.perspectiveTransform(nouveau_point, H)
        # Extraction des coordonnées transformées
        x_transforme, y_transforme = point_transforme[0][0]
        point2_JA = [x_transforme,y_transforme]
        
        vecteur_perpendiculaire_JA = calculer_vecteur_perpendiculaire(point1_JA,point2_JA)


        if len(csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[0]), "x"].values) > 0:
            x_new = csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[0]), "x"].values
        if len(csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[0]), "y"].values) > 0:
            y_new = csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[0]), "y"].values
        nouveau_point = np.array([[[x_new[0], y_new[0]]]], dtype=np.float32)
        #nouveau_point = np.expand_dims(nouveau_point, axis=0)  # Nécessaire pour OpenCV
        point_transforme = cv2.perspectiveTransform(nouveau_point, H)
        # Extraction des coordonnées transformées
        x_transforme, y_transforme = point_transforme[0][0]
        point1_JB = [x_transforme, y_transforme]
        
        if len(csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[1]), "x"].values) > 0:
            x_new = csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[1]), "x"].values
        if len(csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[1]), "y"].values) > 0:
            y_new = csv_pose_filtre_joueur2.loc[(csv_pose_filtre_joueur2["frame"] == i) & (csv_pose_filtre_joueur2["joint"] == joint[1]), "y"].values
        nouveau_point = np.array([[[x_new[0], y_new[0]]]], dtype=np.float32)
        #nouveau_point = np.expand_dims(nouveau_point, axis=0)  # Nécessaire pour OpenCV
        point_transforme = cv2.perspectiveTransform(nouveau_point, H)
        # Extraction des coordonnées transformées
        x_transforme, y_transforme = point_transforme[0][0]
        point2_JB = [x_transforme, y_transforme]
        
        vecteur_perpendiculaire_JB = calculer_vecteur_perpendiculaire(point1_JB,point2_JB)
        
        vecteur_perpendiculaire_JA[0] *= multiplicateur
        vecteur_perpendiculaire_JA[1] *= multiplicateur

        
        vecteur_perpendiculaire_JB[0] *= multiplicateur
        vecteur_perpendiculaire_JB[1] *= multiplicateur
        
        orientationJA.append(vecteur_perpendiculaire_JA)
        orientationJB.append(vecteur_perpendiculaire_JB)

        # Taille de l'image
        width, height = 1280, 720
        chemin_dossier_enregistrement = os.path.join("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/test_orientation")
        if not os.path.isdir(chemin_dossier_enregistrement):
            os.mkdir(chemin_dossier_enregistrement)
        taille_dossier = len(os.listdir(chemin_dossier_enregistrement))

        if os.path.isfile(os.path.join(chemin_dossier_enregistrement.replace("test_orientation","images_video"),str(taille_dossier)+".jpg")):
            image = cv2.imread(os.path.join(chemin_dossier_enregistrement.replace("test_orientation","images_video"),str(taille_dossier)+".jpg"))


            # Dimensions de l'image
            height, width, _ = image.shape


            # Tracé des cercles
            point1_JA = [int(v) for v in point1_JA]
            point2_JA = [int(v) for v in point2_JA]
            point1_JB = [int(v) for v in point1_JB]
            point2_JB = [int(v) for v in point2_JB]
            vecteur_perpendiculaire_JA = [int(v) for v in vecteur_perpendiculaire_JA]
            vecteur_perpendiculaire_JB = [int(v) for v in vecteur_perpendiculaire_JB]
            cv2.circle(image, point1_JA, 30, (0, 0, 255), 2)  # Rouge
            cv2.circle(image, point2_JA, 30, (0, 0, 255), 2)
            cv2.line(image, point1_JA, point2_JA, (0, 0, 0), 2)
            cv2.circle(image, point1_JB, 30, (0, 0, 255), 2)  # Rouge
            cv2.circle(image, point2_JB, 30, (0, 0, 255), 2)
            cv2.line(image, point1_JB, point2_JB, (0, 0, 0), 2)


            # Calcul du milieu entre les deux points
            midpoint = ((point1_JA[0] + point2_JA[0]) // 2, (point1_JA[1] + point2_JA[1]) // 2)
            arrow_end = (midpoint[0]+vecteur_perpendiculaire_JA[0], midpoint[1]+vecteur_perpendiculaire_JA[1])
            cv2.arrowedLine(image, midpoint, arrow_end, (0, 255, 0), 2, tipLength=0.2)
            
            # Calcul du milieu entre les deux points
            midpoint = ((point1_JB[0] + point2_JB[0]) // 2, (point1_JB[1] + point2_JB[1]) // 2)
            arrow_end = (midpoint[0]+vecteur_perpendiculaire_JB[0], midpoint[1]+vecteur_perpendiculaire_JB[1])
            cv2.arrowedLine(image, midpoint, arrow_end, (0, 255, 0), 2, tipLength=0.2)



            cv2.imwrite(os.path.join(chemin_dossier_enregistrement,str(taille_dossier)+".jpg"), image)
        
    
    return orientationJA,orientationJB

def recuperation_points_table(json_path):
    """
    Fonction permettant de ressortir les points enregistrés pour l'homographie (généralement les coins de la table)
    Entrée: Le chemin du json contenant les points pour l'homographie
    Sortie: un éléments contenant les points
    """

    with open(json_path) as json_file:
        json_course = json.load(json_file)

    # we convert to a flat array [[20,10],[80,10],[95,90],[5,90]]
    scr_pct1 = json_course['calibration']['srcPct1']
    src_pts1 = np.float32(list(map(lambda x: list(x.values()), scr_pct1)))

    src_pts1 = np.float32(json_course['homography']['srcPts'])

    return(src_pts1)

def calculer_vecteur_perpendiculaire(point1,point2):
    """
        Fonction permettant de calculer le vecteur perpendiculaire entre 2 points
    """
    # Coordonnées des deux points
    xA, yA = point1[0], point1[1]  # Point A
    xB, yB = point2[0], point2[1]  # Point B
    # Calcul du vecteur perpendiculaire
    vecteur_perpendiculaire = np.array([-(yB - yA), xB - xA])  # Ou (yB - yA, -(xB - xA))

    # Normalisation et mise à l'échelle du vecteur perpendiculaire
    norme = np.linalg.norm(vecteur_perpendiculaire)
    if norme != 0:
        vecteur_perpendiculaire = vecteur_perpendiculaire / norme  # Normalisation
        vecteur_perpendiculaire *= 2  # Agrandissement pour visibilité

    return(vecteur_perpendiculaire)

def ajout_vecteurs_trajectoires(compet, match, set, point, images, faire_orientation=False): # sur les frames d'un point, ajout de la trajectoire des 2 joueurs (ie positions prises aux frames précédentes) + leur vecteur vitesse à l'instant t

    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    if game[0][0]=='playerA' and game[1][0]=='playerB':
        joueurA=game[0][1]
        joueurB=game[1][1]
    else:
        joueurB=game[0][1]
        joueurA=game[1][1]

    os.makedirs('{}/output/match_{}/point_{}'.format(chemin_tt_espace, match,point), exist_ok='True')

    coord_balle, vitesse_balle, coord_jA, vitesse_jA, coord_jB, vitesse_jB = coordonnees(compet, match, set, point) # dans ce code pas besoin des vitesses
    orientationJA,orientationJB = orientation(compet, match, set, point, coord_jA, lissage='non', joint=[11, 14], multiplicateur=50)

    size=5

    if images[0]==None:
        image_init = Image.open(images[1]).convert('RGB')

    for i in range(len(coord_balle)):
        if images[0]==None:
            image=image_init.copy()
        else:
            image = Image.open(os.path.join(images[0],images[1].format(i))).convert('RGB')
        draw = ImageDraw.Draw(image)
        xjA,yjA,xjB,yjB,xb,yb = coord_jA[i][0], coord_jA[i][1], coord_jB[i][0], coord_jB[i][1], coord_balle[i][0], coord_balle[i][1]
        draw.ellipse((xb - size, yb - size, xb + size, yb + size), fill="black")
        draw.ellipse((xjA - size, yjA - size, xjA + size, yjA + size), fill="green")
        #draw.text((xjA+size,yjA+size), 'joueurA', fill='green')
        draw.ellipse((xjB - size, yjB - size, xjB + size, yjB + size), fill="blue")
        #draw.text((xjB+size,yjB+size), 'joueurB', fill='blue')
        draw.text((50,50), 'frame {}'.format(i), fill='black')
        draw.text((20,20), 'point {}'.format(point), fill='black')

        image.save('{}/output/match_{}/point_{}/image_{}.png'.format(chemin_tt_espace, match, point, i))


    for i in range(1,len(coord_balle)):
        queue_fleche_b, tete_fleche_b = [coord_balle[i][0], coord_balle[i][1]], [(coord_balle[i][0]-coord_balle[i-1][0])*3 +coord_balle[i][0], (coord_balle[i][1]-coord_balle[i-1][1])*3 + coord_balle[i][1]] # coordonnées de la tête et de la queue du vecteur. le x3 pour que le vecteur ne soit pas trop petit (valeur choisie arbitrairement)
        queue_fleche_jA, tete_fleche_jA = [coord_jA[i][0], coord_jA[i][1]], [(coord_jA[i][0]-coord_jA[i-1][0])*3 +coord_jA[i][0], (coord_jA[i][1]-coord_jA[i-1][1])*3 + coord_jA[i][1]]
        queue_fleche_jB, tete_fleche_jB = [coord_jB[i][0], coord_jB[i][1]], [(coord_jB[i][0]-coord_jB[i-1][0])*3 +coord_jB[i][0], (coord_jB[i][1]-coord_jB[i-1][1])*3 + coord_jB[i][1]]

        if faire_orientation:
            tete_fleche_jA = [coord_jA[i][0]+orientationJA[i][0], coord_jA[i][1]+orientationJA[i][1]]
            tete_fleche_jB = [coord_jB[i][0]+orientationJB[i][0], coord_jB[i][1]+orientationJB[i][1]]
            #tete_fleche_jA = [coord_jA[i][0], coord_jA[i][1]]
            #tete_fleche_jB = [coord_jB[i][0], coord_jB[i][1]]
            #print([orientationJA[i][0], orientationJA[i][1]])
        image=cv2.imread('{}/output/match_{}/point_{}/image_{}.png'.format(chemin_tt_espace, match, point, i))
        image=cv2.arrowedLine(image, (int(queue_fleche_b[0]),int(queue_fleche_b[1])), (int(tete_fleche_b[0]), int(tete_fleche_b[1])), color=(0,0,0), thickness=2) # ne pas oublier de convertir les coordonnées de l'image (repère au centre de la table) en coordonnées python (repère en au à gauche de la zone)
        image=cv2.arrowedLine(image, (int(queue_fleche_jA[0]),int(queue_fleche_jA[1])), (int(tete_fleche_jA[0]), int(tete_fleche_jA[1])), color=(35,125,55), thickness=2) #BGR les couleurs
        image=cv2.arrowedLine(image, (int(queue_fleche_jB[0]),int(queue_fleche_jB[1])), (int(tete_fleche_jB[0]), int(tete_fleche_jB[1])), color=(255,0,0), thickness=2)


        #for j in range(1,i+1): # ajout de la trajectoire en trait clair
        #    image=cv2.line(image, (int(coord_jA[j][0]), int(coord_jA[j][1])), (int(coord_jA[j-1][0]), int(coord_jA[j-1][1])), color=(120,190,110), thickness=2)
        #    image=cv2.line(image, (int(coord_jB[j][0]), int(coord_jB[j][1])), (int(coord_jB[j-1][0]), int(coord_jB[j-1][1])), color=(255,200,200), thickness=2)

        cv2.imwrite('{}/output/match_{}/point_{}/image_{}.png'.format(chemin_tt_espace, match, point, i), image)

def atteintes(compet, match, precision_espace='non', precision_frappe='non'): # positions relatives sur un match complet de la balle au moment de la frappe par rapport au joueur qui la frappe (en mettant 'oui' pour les précisions, on peut récupérer le pourcentage de frappes effectuées dans la zone rouge de la heatmap (inertie) ou pourcentage de frappes dans la banane)
    with open('{}/{}/{}/{}_annotation_enrichi.csv'.format(chemin_pipeline, compet, match, match), newline='') as fichier_annotations :
        annotations=[]
        reader=csv.reader(fichier_annotations)
        for ligne in reader :
            annotations.append(ligne)

    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    if annotations[0][26]=='probleme_annotation':
        C=1
    else:
        C=0

    set_point=[]
    set_point.append([1,0])
    for i in range(2,len(annotations)):
        if annotations[i][52+C]!=annotations[i-1][52+C]:
            set_point.append([int(annotations[i][5]), int(annotations[i][52+C])])

    cote_j=[[1, annotations[1][62+C], annotations[1][63+C]]] # [[set, cote jA, cote jB], ...]
    for i in range(2, len(annotations)):
        if annotations[i][5]!=annotations[i-1][5] :
            cote_j.append([annotations[i][5], annotations[i][62+C], annotations[i][63+C]])

    coord_jA=[] # xj, yj, xb, yb ; /!!!/ on transpose toutes les coordonnées en y<0 (comme si les 2 joueurs sont en bas, plus simple par la suite quand on fait les visualisations comme si je joueur était en bas)
    coord_jB=[]

    tlim=0.7
    compteur_zone_espace=0  # ces compteurs serviront à calculer la précision (pourcentage_precision = nb_frappes_dans_zone_prédite / nb_total_frappes)
    compteur_zone_frappe=0
    compteur_frappe_tot=0


    for i in range(len(set_point)):
        chemin='{}/{}/{}/clips/set_{}_point_{}'.format(chemin_pipeline, compet, match, set_point[i][0], set_point[i][1])

        with open('{}/set_{}_point_{}_annotation.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_annotation=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_annotation.append(ligne)

        time_frappe = frappe(compet, match, set_point[i][0], set_point[i][1])[0]
        
        for j in range(len(time_frappe)):
            if time_frappe[j][0]==joueurA:
                compteur_frappe_tot+=1

        with open('{}/csv_json_openpose/set_{}_point_{}_zone_joueur_avec_pos_balle.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_pose=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_pose.append(ligne)
        
        nb_frappes_par_pt=int((len(set_point_pose)-1)/3)
        coord_jA_pt, coord_jB_pt = [], []

        if set_point_pose[1][1]=='0' and set_point_pose[2][1]=='1': # parfois dans le fichier csv zone_joueur_avec_pos_balle les coordonnées sont données dans cet ordre : 0,1,4 et parfois : 1,0,4
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='': #souvent à partir de la dernière frappe il n'y a plus les coordonnées de la balle donc je prends celles à la frame précédente
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                            coord_jA_pt.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                            coord_jA_pt.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                            coord_jB_pt.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                            coord_jB_pt.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
        
        elif set_point_pose[1][1]=='1' and set_point_pose[2][1]=='0':
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                            coord_jA_pt.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                            coord_jA_pt.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                            coord_jB_pt.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19])])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])
                            coord_jB_pt.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19])])

        
        if precision_espace=='oui' or precision_frappe=='oui' :
            time_frappe_jA, time_frappe_jB = [], []
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    time_frappe_jA.append(int(time_frappe[j][1]))
                else:
                    time_frappe_jB.append(int(time_frappe[j][1]))

            if cote_j[set_point[i][0]-1][1]=='bas':
                vitesse_jA=coordonnees(compet, match, set_point[i][0], set_point[i][1])[3]
            elif cote_j[set_point[i][0]-1][1]=='haut':
                vitesse_jA=[]
                for j in range(nb_frappes_par_pt):
                    vitesse_jA.append([-coordonnees(compet, match, set_point[i][0], set_point[i][1])[3][j][0], -coordonnees(compet, match, set_point[i][0], set_point[i][1])[3][j][1]])

            if precision_espace=='oui': # pour faire stat sur nb de frappes dans la forme arrondie qui considère uniquement l'inertie
                for j in range(len(time_frappe_jA)):
                    time_2_point=time_to_point([coord_jA_pt[j][0], coord_jA_pt[j][1]], [coord_jA_pt[j][2], coord_jA_pt[j][3]], vitesse_jA[time_frappe_jA[j]], F=500)
                    if time_2_point<tlim:
                        compteur_zone_espace+=1

                pourcentage_precision_espace=compteur_zone_espace/compteur_frappe_tot
        
            if precision_frappe=='oui': # pour faire stat sur nb de frappes dans la forme de banane qui uniquement la forme de la zone de frappe (générée grâce aux données)
                coord_banane=banane_alexis()
                banane=Polygon(coord_banane)
                for j in range(len(time_frappe_jA)):
                    if banane.contains(Point((int(coord_jA_pt[j][2]-coord_jA_pt[j][0]), int(coord_jA_pt[j][3]-coord_jA_pt[j][1])))):
                        compteur_zone_frappe+=1
                pourcentage_precision_frappe=compteur_zone_frappe/compteur_frappe_tot

    atteintes_jA=[] # x_joueur-balle, y_joueur-balle au moment de la frappe
    atteintes_jB=[]

    for i in range(len(coord_jA)):
        atteintes_jA.append([float(coord_jA[i][2]-coord_jA[i][0]), float(coord_jA[i][3]-coord_jA[i][1])])
    for i in range(len(coord_jB)):
        atteintes_jB.append([float(coord_jB[i][2]-coord_jB[i][0]), float(coord_jB[i][3]-coord_jB[i][1])])
    
    if precision_espace=='non' : 
        if precision_frappe=='non':
            return atteintes_jA, atteintes_jB
        elif precision_frappe=='oui':
            return atteintes_jA, atteintes_jB, pourcentage_precision_frappe
    elif precision_espace=='oui' :
        if precision_frappe=='non':
            return atteintes_jA, atteintes_jB, pourcentage_precision_espace
        elif precision_frappe=='oui' :
            return atteintes_jA, atteintes_jB, pourcentage_precision_espace, pourcentage_precision_frappe



def atteintes_3D(compet, match, precision_espace='non', precision_frappe='non'): # idem que fonctions au dessus, sauf que les positions de la balle sont prises dans le fichier set_point_zone_joueur_avec_pos_balle_3D.csv alors qu'au dessus on avait la position de la balle via l'annotation
    with open('{}/{}/{}/{}_annotation_enrichi.csv'.format(chemin_pipeline, compet, match, match), newline='') as fichier_annotations :
        annotations=[]
        reader=csv.reader(fichier_annotations)
        for ligne in reader :
            annotations.append(ligne)

    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    if annotations[0][26]=='probleme_annotation':
        C=1
    else:
        C=0

    set_point=[]
    set_point.append([1,0])
    for i in range(2,len(annotations)):
        if annotations[i][52+C]!=annotations[i-1][52+C]:
            set_point.append([int(annotations[i][5]), int(annotations[i][52+C])])

    cote_j=[[1, annotations[1][62+C], annotations[1][63+C]]] # [[set, cote jA, cote jB], ...]
    for i in range(2, len(annotations)):
        if annotations[i][5]!=annotations[i-1][5] :
            cote_j.append([annotations[i][5], annotations[i][62+C], annotations[i][63+C]])

    coord_jA=[] # xj, yj, xb, yb ; /!!!/ on transpose toutes les coordonnées en y<0 (comme si les 2 joueurs sont en bas, plus simple par la suite quand on fait les visualisations comme si je joueur était en bas)
    coord_jB=[]

    tlim=0.7
    compteur_zone_espace=0  # ces compteurs serviront à calculer la précision (pourcentage_precision = nb_frappes_dans_zone_prédite / nb_total_frappes)
    compteur_zone_frappe=0
    compteur_frappe_tot=0


    for i in range(len(set_point)):
        chemin='{}/{}/{}/clips/set_{}_point_{}'.format(chemin_pipeline, compet, match, set_point[i][0], set_point[i][1])

        with open('{}/set_{}_point_{}_annotation.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_annotation=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_annotation.append(ligne)

        time_frappe = frappe(compet, match, set_point[i][0], set_point[i][1])[0]

        for j in range(len(time_frappe)):
            if time_frappe[j][0]==joueurA:
                compteur_frappe_tot+=1

        with open('{}/csv_json_openpose/set_{}_point_{}_zone_joueur_avec_pos_balle_3D.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_pose=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_pose.append(ligne)
        
        nb_frappes_par_pt=int((len(set_point_pose)-1)/3)
        coord_jA_pt, coord_jB_pt = [], []

        if set_point_pose[1][1]=='0' and set_point_pose[2][1]=='1': # parfois dans le fichier csv zone_joueur_avec_pos_balle les coordonnées sont données dans cet ordre : 0,1,4 et parfois : 1,0,4
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='': #souvent à partir de la dernière frappe il n'y a plus les coordonnées de la balle donc je prends celles à la frame précédente
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jA_pt.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jA_pt.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jB_pt.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jB_pt.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
        
        elif set_point_pose[1][1]=='1' and set_point_pose[2][1]=='0':
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jA_pt.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jA_pt.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jB_pt.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_pose[time_frappe[j][1]*3+3][2]), -float(set_point_pose[time_frappe[j][1]*3+3][3])])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])
                            coord_jB_pt.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_pose[time_frappe[j][1]*3+3][2]), float(set_point_pose[time_frappe[j][1]*3+3][3])])

        
        if precision_espace=='oui' or precision_frappe=='oui' :
            time_frappe_jA, time_frappe_jB = [], []
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    time_frappe_jA.append(int(time_frappe[j][1]))
                else:
                    time_frappe_jB.append(int(time_frappe[j][1]))

            if cote_j[set_point[i][0]-1][1]=='bas':
                vitesse_jA=coordonnees(compet, match, set_point[i][0], set_point[i][1])[3]
            elif cote_j[set_point[i][0]-1][1]=='haut':
                vitesse_jA=[]
                for j in range(nb_frappes_par_pt):
                    vitesse_jA.append([-coordonnees(compet, match, set_point[i][0], set_point[i][1])[3][j][0], -coordonnees(compet, match, set_point[i][0], set_point[i][1])[3][j][1]])

            if precision_espace=='oui': # pour faire stat sur nb de frappes dans la forme arrondie qui considère uniquement l'inertie
                for j in range(len(time_frappe_jA)):
                    time_2_point=time_to_point([coord_jA_pt[j][0], coord_jA_pt[j][1]], [coord_jA_pt[j][2], coord_jA_pt[j][3]], vitesse_jA[time_frappe_jA[j]], F=500)
                    if time_2_point<tlim:
                        compteur_zone_espace+=1

                pourcentage_precision_espace=compteur_zone_espace/compteur_frappe_tot
        
            if precision_frappe=='oui': # pour faire stat sur nb de frappes dans la forme de banane qui uniquement la forme de la zone de frappe (générée grâce aux données)
                coord_banane=banane_alexis()
                banane=Polygon(coord_banane)
                for j in range(len(time_frappe_jA)):
                    if banane.contains(Point((int(coord_jA_pt[j][2]-coord_jA_pt[j][0]), int(coord_jA_pt[j][3]-coord_jA_pt[j][1])))):
                        compteur_zone_frappe+=1
                pourcentage_precision_frappe=compteur_zone_frappe/compteur_frappe_tot

    atteintes_jA=[] # x_joueur-balle, y_joueur-balle au moment de la frappe
    atteintes_jB=[]

    for i in range(len(coord_jA)):
        atteintes_jA.append([float(coord_jA[i][2]-coord_jA[i][0]), float(coord_jA[i][3]-coord_jA[i][1])])
    for i in range(len(coord_jB)):
        atteintes_jB.append([float(coord_jB[i][2]-coord_jB[i][0]), float(coord_jB[i][3]-coord_jB[i][1])])
    
    if precision_espace=='non' : 
        if precision_frappe=='non':
            return atteintes_jA, atteintes_jB
        elif precision_frappe=='oui':
            return atteintes_jA, atteintes_jB, pourcentage_precision_frappe
    elif precision_espace=='oui' :
        if precision_frappe=='non':
            return atteintes_jA, atteintes_jB, pourcentage_precision_espace
        elif precision_frappe=='oui' :
            return atteintes_jA, atteintes_jB, pourcentage_precision_espace, pourcentage_precision_frappe



def atteintes_lateralite(compet, match): # idem que les 2 fonctions du dessus sauf qu'en plus on sait si le joueur a frappé la balle en coup droit ou en revers (3ème élément de chaque sous-liste)
    with open('{}/{}/{}/{}_annotation_enrichi.csv'.format(chemin_pipeline, compet, match, match), newline='') as fichier_annotations :
        annotations=[]
        reader=csv.reader(fichier_annotations)
        for ligne in reader :
            annotations.append(ligne)

    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    if annotations[0][26]=='probleme_annotation':
        C=1
    else:
        C=0

    set_point=[]
    set_point.append([1,0])
    for i in range(2,len(annotations)):
        if annotations[i][52+C]!=annotations[i-1][52+C]:
            set_point.append([int(annotations[i][5]), int(annotations[i][52+C])])

    cote_j=[[1, annotations[1][62+C], annotations[1][63+C]]] # [[set, cote jA, cote jB], ...]
    for i in range(2, len(annotations)):
        if annotations[i][5]!=annotations[i-1][5] :
            cote_j.append([annotations[i][5], annotations[i][62+C], annotations[i][63+C]])

    coord_jA=[] # xj, yj, xb, yb, latéralité (CD / revers)
    coord_jB=[]
    for i in range(len(set_point)):
        chemin='{}/{}/{}/clips/set_{}_point_{}'.format(chemin_pipeline, compet, match, set_point[i][0], set_point[i][1])
        
        with open('{}/set_{}_point_{}_annotation.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_annotation=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_annotation.append(ligne)

        time_frappe = frappe(compet, match, set_point[i][0], set_point[i][1])[0]
        
        with open('{}/csv_json_openpose/set_{}_point_{}_zone_joueur_avec_pos_balle.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_pose=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_pose.append(ligne)

        if set_point_pose[1][1]=='0' and set_point_pose[2][1]=='1':
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='': #souvent à partir de la dernière frappe il n'y a plus les coordonnées de la balle donc je prends celles à la frame précédente
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][3]])
        
        elif set_point_pose[1][1]=='1' and set_point_pose[2][1]=='0':
            for j in range(len(time_frappe)):
                if time_frappe[j][0]==joueurA:
                    if cote_j[set_point[i][0]-1][1]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                    elif cote_j[set_point[i][0]-1][1]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                elif time_frappe[j][0]==joueurB:
                    if cote_j[set_point[i][0]-1][2]=='bas':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][3]])
                    elif cote_j[set_point[i][0]-1][2]=='haut':
                        if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                            coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][3]])


    atteintes_jA=[] # x_joueur-balle, y_joueur-balle au moment de la frappe, latéralité
    atteintes_jB=[]

    for i in range(len(coord_jA)):
        atteintes_jA.append([float(coord_jA[i][2]-coord_jA[i][0]), float(coord_jA[i][3]-coord_jA[i][1]), coord_jA[i][4]])
    for i in range(len(coord_jB)):
        atteintes_jB.append([float(coord_jB[i][2]-coord_jB[i][0]), float(coord_jB[i][3]-coord_jB[i][1]), coord_jB[i][4]])

    return atteintes_jA, atteintes_jB



def nuage_frappes(compet, match):
    with open('{}/{}/{}/{}_annotation_enrichi.csv'.format(chemin_pipeline, compet, match, match), newline='') as fichier_annotations :
        annotations=[]
        reader=csv.reader(fichier_annotations)
        for ligne in reader :
            annotations.append(ligne)

    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
        joueurA=game[0][1]
        joueurB=game[1][1]

    if annotations[0][26]=='probleme_annotation':
        C=1
    else:
        C=0

    im_jA = Image.new('RGB', (800, 400), 'white')
    im_jB = Image.new('RGB', (800, 400), 'white')
    draw_jA = ImageDraw.Draw(im_jA)
    draw_jB = ImageDraw.Draw(im_jB)
    draw_jA.ellipse([400-10, 200-10, 400+10, 200+10], fill='black')
    draw_jB.ellipse([400-10, 200-10, 400+10, 200+10], fill='black')

    set_point=[]
    set_point.append([1,0])
    for i in range(2,len(annotations)):
        if annotations[i][52+C]!=annotations[i-1][52+C]:
            set_point.append([int(annotations[i][5]), int(annotations[i][52+C])])

    cote_j=[[1, annotations[1][62+C], annotations[1][63+C]]] # [[set, cote jA, cote jB], ...]
    for i in range(2, len(annotations)):
        if annotations[i][5]!=annotations[i-1][5] :
            cote_j.append([annotations[i][5], annotations[i][62+C], annotations[i][63+C]])

    coord_jA=[] # xj, yj, xb, yb, latéralité (CD / revers)
    coord_jB=[]

    for i in range(len(set_point)):
        chemin='{}/{}/{}/clips/set_{}_point_{}'.format(chemin_pipeline, compet, match, set_point[i][0], set_point[i][1])
        
        with open('{}/set_{}_point_{}_annotation.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
            set_point_annotation=[]
            reader=csv.reader(fichier)
            for ligne in reader :
                set_point_annotation.append(ligne)

        time_frappe=[] # joueur qui frappe, frame de frappe, latéralité
        for j in range(1, len(set_point_annotation)):
            if set_point_annotation[j][8]=='': # on ne prend pas les services
                time_frappe.append([set_point_annotation[j][0], int(set_point_annotation[j][21]), set_point_annotation[j][4]])
        
        if os.path.exists('{}/csv_json_openpose/set_{}_point_{}_zone_joueur_avec_pos_balle_3D.csv'.format(chemin, set_point[i][0], set_point[i][1])): # parfois le fichier 3d n'a pas été créé
            
            with open('{}/csv_json_openpose/set_{}_point_{}_zone_joueur_avec_pos_balle_3D.csv'.format(chemin, set_point[i][0], set_point[i][1]), newline='') as fichier :
                set_point_pose=[]
                reader=csv.reader(fichier)
                for ligne in reader :
                    set_point_pose.append(ligne)

            if set_point_pose[1][1]=='0' and set_point_pose[2][1]=='1':
                for j in range(len(time_frappe)):
                    if time_frappe[j][0]==joueurA:
                        if cote_j[set_point[i][0]-1][1]=='bas':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='': #souvent à partir de la dernière frappe il n'y a plus les coordonnées de la balle donc je prends celles à la frame précédente
                                coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                        elif cote_j[set_point[i][0]-1][1]=='haut':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                    elif time_frappe[j][0]==joueurB:
                        if cote_j[set_point[i][0]-1][2]=='bas':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                        elif cote_j[set_point[i][0]-1][2]=='haut':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][2]])
            
            elif set_point_pose[1][1]=='1' and set_point_pose[2][1]=='0':
                for j in range(len(time_frappe)):
                    if time_frappe[j][0]==joueurA:
                        if cote_j[set_point[i][0]-1][1]=='bas':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jA.append([float(set_point_pose[time_frappe[j][1]*3+2][2]),-float(set_point_pose[time_frappe[j][1]*3+2][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                        elif cote_j[set_point[i][0]-1][1]=='haut':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jA.append([-float(set_point_pose[time_frappe[j][1]*3+2][2]),float(set_point_pose[time_frappe[j][1]*3+2][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                    elif time_frappe[j][0]==joueurB:
                        if cote_j[set_point[i][0]-1][2]=='bas':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jB.append([float(set_point_pose[time_frappe[j][1]*3+1][2]),-float(set_point_pose[time_frappe[j][1]*3+1][3]), float(set_point_annotation[j+2][18]), -float(set_point_annotation[j+2][19]), time_frappe[j][2]])
                        elif cote_j[set_point[i][0]-1][2]=='haut':
                            if set_point_pose[time_frappe[j][1]*3+3][2]!='':
                                coord_jB.append([-float(set_point_pose[time_frappe[j][1]*3+1][2]),float(set_point_pose[time_frappe[j][1]*3+1][3]), -float(set_point_annotation[j+2][18]), float(set_point_annotation[j+2][19]), time_frappe[j][2]])


        atteintes_jA=[] # x_joueur-balle, y_joueur-balle au moment de la frappe, latéralité
        atteintes_jB=[]

        for i in range(len(coord_jA)):
            atteintes_jA.append([float(coord_jA[i][2]-coord_jA[i][0]), float(coord_jA[i][3]-coord_jA[i][1]), coord_jA[i][4]])
        for i in range(len(coord_jB)):
            atteintes_jB.append([float(coord_jB[i][2]-coord_jB[i][0]), float(coord_jB[i][3]-coord_jB[i][1]), coord_jB[i][4]])


        height, width = 400, 800
        xj, yj = width/2, height/2

        for i in range(len(atteintes_jA)):
            xb, yb = atteintes_jA[i][0] +xj, -atteintes_jA[i][1] +yj
            if atteintes_jA[i][2]=='revers':
                draw_jA.text((xb,yb), 'x', font=ImageFont.truetype("arial.ttf", 20), fill='red')
            elif atteintes_jA[i][2]=='coup_droit':
                draw_jA.text((xb,yb), 'x', font=ImageFont.truetype("arial.ttf", 20), fill='blue')

        for i in range(len(atteintes_jB)):
            xb, yb = atteintes_jB[i][0] +xj, -atteintes_jB[i][1] +yj
            if atteintes_jB[i][2]=='revers':
                draw_jB.text((xb,yb), 'x', font=ImageFont.truetype("arial.ttf", 20), fill='red')
            elif atteintes_jB[i][2]=='coup_droit':
                draw_jB.text((xb,yb), 'x', font=ImageFont.truetype("arial.ttf", 20), fill='blue')

        im_jA.save('{}/output/match_{}/nuage/nuage_jA.png'.format(chemin_tt_espace, match))
        im_jB.save('{}/output/match_{}/nuage/nuage_jB.png'.format(chemin_tt_espace, match))



def banane_alexis(surface='non'): # on récupère le contour de la zone de frappe (nommée 'banane' en raison de sa forme) d'Alexis Lebrun sur le match contre Ma Long à Singapour (cette banane a pour vocation d'être utilisée sur d'autres matchs avec Alexis, on a choisi arbitrairement ce match pour la tracer)
    compet='2024_WttSmash_Singapour'
    match='ALEXIS-LEBRUN_vs_MA-LONG'
    atteintes_jA, atteintes_jB = atteintes(compet, match)

    xj, yj = 400, 200 # inutile. j'ai décalé parce qu'initialement je voulais afficher la banane dans l'environnement de la table pour avoir une idée de l'échelle mais comme à la fin je renvoie la position relative de la banane de soustrait finalement xj et yj. je n'ai pas modifié le code parce que ma banane n'avait plus la même forme et je n'ai pas eu le temps de regarder de plus près.

    balles_frappees_jA, balles_frappees_jB = [], []
    for i in range(len(atteintes_jA)):
        balles_frappees_jA.append(np.array([atteintes_jA[i][0]+xj, atteintes_jA[i][1]+yj])) # /!/ j'enlève le - sur les y parce qu'en utilisant pyplot, le référentiel est en bas à gauche et non en haut à gauche pour pour Pillow
    for i in range(len(atteintes_jB)):
        balles_frappees_jB.append(np.array([atteintes_jB[i][0]+xj, atteintes_jB[i][1]+yj]))
    balles_frappees_jA=np.array(balles_frappees_jA)
    balles_frappees_jB=np.array(balles_frappees_jB)

    alpha = 0.033
    hull_points_jA = alphashape.alphashape(balles_frappees_jA, alpha)

    contour_points_liste = []
    if isinstance(hull_points_jA, Polygon):
        contour_points_liste.append(np.array(hull_points_jA.exterior.coords))
    elif isinstance(hull_points_jA, MultiPolygon):
        for poly in hull_points_jA.geoms:
            contour_points_liste.append(np.array(poly.exterior.coords)) # grâce à cette fonction .exteriot.coords on récup les coordonnées du contour du nuage de points
    
    contour_points=[] # dans ce bloc on choisit manuellement certains points pour avoir une forme plus englobante que celle obtenue avec exterior.coords
    for i in [0,2,3,7,12,14,18,25,26,29,34]: # pts retirés pour affiner la queue de la banane : 20,23 # les pts que je veux garder pour avoir une forme englobante et lisse (en prenant ces pts, la fonction splprep qui lisse par spline une forme 2D crée une forme qui recouvre tous les pts de mon nuage)
        contour_points.append([contour_points_liste[1][i][0], contour_points_liste[1][i][1]]) # la banane c'est le deuxième polygone créé par la fonction alphashape (le premier polygone est un cluster de points dont les positions doivent être fausses (mauvaise annotation) car sur ou derrière le joueur)
    contour_points=np.array(contour_points)
    tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=2.0, per=True) # pour lisser la forme
    u_nouv = np.linspace(u.min(), u.max(), 1000)
    x, y = splev(u_nouv, tck)

    if surface=='oui':
        surface_banane=Polygon([(x[i], y[i]) for i in range(len(x))]).area # /!/ en cm^2

    coord_spline=[] # coordonnées absolues du contour de la banane (ie le pt le plus à droite est à (x_droite, y_droite) du joueur)
    for i in range(len(x)):
        coord_spline.append([x[i]-xj, y[i]-yj])
    
    if surface=='non':
        return coord_spline
    elif surface=='oui':
        return coord_spline, surface_banane/10000



def banane_alexis_cd_r(): # en complément de la fonction du dessus : la banane est la zone globale dans laquelle se trouve toutes les frappes du match. or on peut raffiner cette zone puisqu'il y a 4 sous-zones qui se distinguent : 2 zones revers et 2 zones coup droit. cette fonction donne les coordonnées du contour de ces 4 zones.
    compet='2024_WttSmash_Singapour'
    match='ALEXIS-LEBRUN_vs_MA-LONG'
    atteintes_jA = atteintes_lateralite(compet, match)[0]

    cluster_cd, cluster_r = [], []
    for i in range(len(atteintes_jA)):
        if atteintes_jA[i][2]=='coup_droit':
            cluster_cd.append([atteintes_jA[i][0], atteintes_jA[i][1]])
        elif atteintes_jA[i][2]=='revers':
            cluster_r.append([atteintes_jA[i][0], atteintes_jA[i][1]])
    cluster_cd, cluster_r = np.array(cluster_cd), np.array(cluster_r)

    alpha=0.033
    hull_cd_jA = alphashape.alphashape(cluster_cd, alpha)
    hull_r_jA = alphashape.alphashape(cluster_r, 0.05)

    contour_hull_cd_jA, contour_hull_r_jA = [], []
    for contour in [contour_hull_cd_jA, contour_hull_r_jA]:
        if contour==contour_hull_cd_jA:
            hull=hull_cd_jA
        else:
            hull=hull_r_jA
        if isinstance(hull, Polygon):
            contour.append(np.array(hull.exterior.coords))
        elif isinstance(hull, MultiPolygon):
            for poly in hull.geoms:
                contour.append(np.array(poly.exterior.coords))

    L_cd_1=[0,4,6,7,8] # sélection manuelle des points les plus pertinents à garder pour former un contour englobant toutes les positions de frappe
    L_cd_2=[2,4,6,7]
    L_cd_3=[0,1,3,5,7,9]
    L_r_0=[0,2,6,7,11,13,16]
    L_r_1=[0,3,5,6,9,11,13,14]
    L_cd_2=[L_cd_2[i]+len(contour_hull_cd_jA[1]) for i in range(len(L_cd_2))]
    L=[L_cd_1+L_cd_2, L_cd_3, L_r_0, L_r_1]
    contour_hull_cd_jA[1]=contour_hull_cd_jA[1].tolist()
    contour_hull_cd_jA[2]=contour_hull_cd_jA[2].tolist()
    contours=[contour_hull_cd_jA[1]+contour_hull_cd_jA[2], contour_hull_cd_jA[3], contour_hull_r_jA[0], contour_hull_r_jA[1]]

    x, y = [], []

    for i in range(len(L)):
        contour_points=[]
        list=L[i]
        for j in range(len(list)):
            contour_points.append(contours[i][list[j]])
        contour_points=np.array(contour_points)
        tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=2.0, per=True)
        u_nouv = np.linspace(u.min(), u.max(), 1000)
        x_hull, y_hull = splev(u_nouv, tck)
        x.append(x_hull)
        y.append(y_hull)

    coord_hull_cd=[[x[0], y[0]], [x[1], y[1]]]
    coord_hull_r=[[x[2], y[2]], [x[3], y[3]]]

    return coord_hull_cd, coord_hull_r



def angles_avant_bras(compet, match, set, point): # angles de chaque joueur au cours d'un point entre ses avant-bras et l'axe des y (dans la longueur de la table). trouvé par calcul avec l'openpose par Aymeric (il doit y avoir une erreur dans son code puisque les angles ne sont pas du tout conformes à ce qu'on peut observer)
    with open('{}/{}/{}/clips/set_{}_point_{}/csv_json_openpose/set_{}_point_{}_angles.csv'.format(chemin_pipeline, compet, match, set, point, set, point), newline='') as fichier :
        set_point_angle=[]
        reader=csv.reader(fichier)
        for ligne in reader :
            set_point_angle.append(ligne)

    angles_jA, angles_jB = [[], []], [[], []] # angles_jA = [[angles gauches], [angles droits]]
    for i in range(1,len(set_point_angle)):
        if set_point_angle[i][3]=='':
            if set_point_angle[i][1]=='1':
                angles_jA[0].append(None)
            else:
                angles_jB[0].append(None)
        else:
            if set_point_angle[i][1]=='1':
                angles_jA[0].append(float(set_point_angle[i][3]))
            else:
                angles_jB[0].append(float(set_point_angle[i][3]))
        if set_point_angle[i][2]=='':
            if set_point_angle[i][1]=='1':
                angles_jA[1].append(None)
            else:
                angles_jB[1].append(None)
        else:
            if set_point_angle[i][1]=='1':
                angles_jA[1].append(float(set_point_angle[i][2]))
            else:
                angles_jB[1].append(float(set_point_angle[i][2]))
    
    return angles_jA, angles_jB



def angles_manuel(compet, match, set, point): # en regardant la vidéo d'un point on peut manuellement dresser la liste des angles (suite au fait que ceux d'Aymeric semblent faux). on note dans un fichier csv le num de frame et une valeur d'angle à des frames remarquables (en début de coup puis fin de coup), et ensuite on fait (dans la fonction) une interpolation linéaire. ce code pas utilisé finalement a juste été écrit pour le bras droit du jA (peut être adpaté en qq secondes aux 2 joueurs, pour leurs 2 bras)
    with open('{}/{}/{}/clips/set_{}_point_{}/csv_json_openpose/set_{}_point_{}_angles_manuel.csv'.format(chemin_pipeline, compet, match, set, point, set, point), newline='') as fichier :
        csv_angles_jA_droite=[]
        reader=csv.reader(fichier)
        for ligne in reader :
            csv_angles_jA_droite.append(ligne)

    angles_jA_droite=[]
    for i in range(1, len(csv_angles_jA_droite)):
        if csv_angles_jA_droite[i][1]!='':
            angles_jA_droite.append(int(float(csv_angles_jA_droite[i][1])))
        else:
            angles_jA_droite.append(np.nan)

    angles_jA_droite=np.array(angles_jA_droite)
    non_none_indices = np.where(~np.isnan(angles_jA_droite))[0]
    none_indices = np.where(np.isnan(angles_jA_droite))[0]
    non_none_values = angles_jA_droite[non_none_indices]
    interpolated_values = np.interp(none_indices, non_none_indices, non_none_values)
    angles_jA_droite[none_indices] = interpolated_values
    angles_jA_droite=angles_jA_droite.tolist()
    angles_jA_droite=[int(angles_jA_droite[i]) for i in range(len(angles_jA_droite))]

    return angles_jA_droite



def bdl(compet, match, set, point):
    h=800
    pos_init_b, vitesse_b, pos_init_jA, vitesse_jA, pos_init_jB, vitesse_jB = coordonnees(compet, match, set, point, lissage='oui')

    if 0<pos_init_jA[0][1]<h/2:
        cote_jA='en haut'
    else:
        cote_jA='en bas'

    bdl_jA, bdl_jB = [], []
    cote_balle_jA, cote_balle_jB = [], []
    for f in range(len(pos_init_b)):
        x1, y1 = pos_init_b[f][0], pos_init_b[f][1]
        x2, y2 = pos_init_jA[f][0], pos_init_jA[f][1]
        x3, y3 = pos_init_jB[f][0], pos_init_jB[f][1]
        vx, vy = vitesse_b[f][0], vitesse_b[f][1]
        if vx!=0 and vy!=0 :
            if cote_jA=='en bas':
                if vy>0: # la balle descend et comme le jA est en bas, elle est pour lui
                    d_jA = abs((x1-x2)*vy - (y1-y2)*vx)/sqrt(vx**2 + vy**2)
                    d_jB = None
                    delta_h_jA = (y2-y1)*vx/vy + x1 - x2 # si ce delta (entre la position de la balle lorsqu'elle sera à hauteur du joueur (ie y égaux) et le joueur qui va la frappe) est positif, alors la balle passera à droite, sinon, elle passe à gauche du joueur
                    delta_h_jB = None
                elif vy<0:
                    d_jA = None
                    d_jB = abs((x1-x3)*vy - (y1-y3)*vx)/sqrt(vx**2 + vy**2)
                    delta_h_jA = None
                    delta_h_jB = (y3-y1)*vx/vy + x1 - x3
            elif cote_jA=='en haut':
                if vy<0:
                    d_jA = abs((x1-x2)*vy - (y1-y2)*vx)/sqrt(vx**2 + vy**2)
                    d_jB = None
                    delta_h_jA = (y2-y1)*vx/vy + x1 - x2
                    delta_h_jB = None
                elif vy>0:
                    d_jA = None
                    d_jB = abs((x1-x3)*vy - (y1-y3)*vx)/sqrt(vx**2 + vy**2)
                    delta_h_jA = None
                    delta_h_jB = (y3-y1)*vx/vy + x1 - x3
            bdl_jA.append(d_jA)
            bdl_jB.append(d_jB)
            cote_balle_jA.append(delta_h_jA)
            cote_balle_jB.append(delta_h_jB)
        else: # au début du clip, il y a qq frames où la balle n'est pas encore en mouvement (avant le service) donc on n'étudie pas de bdl encore
            bdl_jA.append(None)
            bdl_jB.append(None)
            cote_balle_jA.append(None)
            cote_balle_jB.append(None)
    
    return bdl_jA, cote_balle_jA, bdl_jB, cote_balle_jB