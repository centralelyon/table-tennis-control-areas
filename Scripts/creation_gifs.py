import matplotlib.pyplot as plt
import numpy as np
import json
from math import sqrt
import cv2
import os
import imageio.v2 as iio
from PIL import Image, ImageDraw

from config import chemin_tt_espace

from Scripts.fonctions_inertie import heatmap
from Scripts.fonctions_positions import coordonnees, angles_avant_bras, ajout_vecteurs_trajectoires, banane_alexis, banane_alexis_cd_r
from Scripts.fonctions_time import frappe
from Scripts.generation_grille_table import quadrillage_sur_image



def gif_evol_surf_jA(compet, match, set, point):
    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    pos_init_b = coordonnees(compet, match, set, point, lissage='oui')[0]

    nb_pts=len(pos_init_b)
    time_points = np.array([f*1/25 for f in range(len(pos_init_b))]) # on a la valeur de la surface atteignable en moins de tlim pour chaque frame et 1 frame dure 1/25s d'où cette discrétisation du temps
    xmax=(len(time_points)-1)/25
    time_frappe = frappe(compet, match, set, point)[0]
    frappe_service = frappe(compet, match, set, point)[1]

    c=10 #fonctions_inertie.py > heatmap
    surfaces_jA = heatmap(compet, match, set, point, 0.8, que_taille_zone='oui')
    surface_points = []
    for f in range(len(surfaces_jA)):
        surface_points.append(surfaces_jA[f] * c**2 * 10**(-4)) # la liste surfaces_jA contient le nb de carrés de discrétisation de côté c (=10cm) à chaque frame du pt. il suffit donc de multiplier chaque elt de la liste par la surface d'un carré (10^-4 car on transpose en m²)

    liste_x=[]
    for i in range(1, nb_pts+1):
        x=i/nb_pts*time_points[len(time_points)-1]
        liste_x.append(x)
        plt.figure(figsize=(8, 5))
        plt.plot(liste_x, surface_points[:len(liste_x)], label='A_jA(t)')
        plt.xlabel('Temps [s]')
        plt.ylabel('Surface [m²]')
        plt.title("Surface de la zone d'atteignabilité de la balle en fonction du temps")
        plt.xlim(0, xmax)
        plt.ylim(0, 15)
        for j in range(len(time_frappe)):
            if time_frappe[j][1]<x*25:
                if time_frappe[j][0]==joueurA:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--')
                else:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--')
        if frappe_service[1]<x*25:
            if frappe_service[0]==joueurA :
                plt.axvline(x=frappe_service[1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
            elif frappe_service[0]==joueurB :
                plt.axvline(x=frappe_service[1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
        plt.legend(loc='lower left')
        plt.savefig('{}/output/match_{}/evolutions_temporelles/surface_inertie/A_jA_{}.png'.format(chemin_tt_espace, match, i))
        plt.close()

    frames = np.stack([iio.imread('{}/output/match_{}/evolutions_temporelles/surface_inertie/A_jA_{}.png'.format(chemin_tt_espace, match, i)) for i in range(1, nb_pts+1)], axis = 0)
    iio.mimwrite('{}/output/match_{}/point_{}/evol_temp_surface_inertie_jA.gif'.format(chemin_tt_espace, match, point), frames, fps=nb_pts*25/len(pos_init_b))



def gif_evol_vit_jA(compet, match, set, point):
    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    pos_init_b = coordonnees(compet, match, set, point, lissage='oui')[0]
    vitesse_jA = coordonnees(compet, match, set, point, lissage='oui')[3]

    time_points = np.array([f*1/25 for f in range(len(pos_init_b))]) # on a la valeur de la surface atteignable en moins de tlim pour chaque frame et 1 frame dure 1/25s d'où cette discrétisation du temps
    xmax=(len(time_points)-1)/25
    nb_pts=len(pos_init_b)
    time_frappe = frappe(compet, match, set, point)[0]
    frappe_service = frappe(compet, match, set, point)[1]

    norme_vitesse_jA=[]
    for i in range(len(vitesse_jA)):
        vitesse=sqrt(vitesse_jA[i][0]**2 + vitesse_jA[i][1]**2)
        vitesse=vitesse*10**(-2) # pour passer de cm/s à m/s
        norme_vitesse_jA.append(vitesse)

    liste_x=[]
    for i in range(1, nb_pts+1):
        x=i/nb_pts*time_points[len(time_points)-1]
        liste_x.append(x)
        plt.figure(figsize=(8, 5))
        plt.plot(liste_x, norme_vitesse_jA[:len(liste_x)], label='v_jA(t)')
        plt.xlabel('Temps [s]')
        plt.ylabel('Vitesse jA [m/s]')
        plt.title('Vitesse absolue de {} en fonction du temps'.format(joueurA))
        plt.xlim(0, xmax)
        plt.ylim(-2, 6)
        for j in range(len(time_frappe)):
            if time_frappe[j][1]<x*25:
                if time_frappe[j][0]==joueurA:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--')
                else:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--')
        if frappe_service[1]<x*25:
            if frappe_service[0]==joueurA :
                plt.axvline(x=frappe_service[1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
            elif frappe_service[0]==joueurB :
                plt.axvline(x=frappe_service[1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
        plt.legend(loc='upper left')
        plt.savefig('{}/output/match_{}/evolutions_temporelles/vitesse/v_jA_{}.png'.format(chemin_tt_espace, match, i))
        plt.close()

    frames = np.stack([iio.imread('{}/output/match_{}/evolutions_temporelles/vitesse/v_jA_{}.png'.format(chemin_tt_espace, match, i)) for i in range(1, nb_pts+1)], axis = 0)
    iio.mimwrite('{}/output/match_{}/point_{}/evol_temp_vitesse_jA.gif'.format(chemin_tt_espace, match, point), frames, fps=nb_pts*25/len(pos_init_b))



def gif_evol_surf_vit_jA(compet, match, set, point):
    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]

    pos_init_b = coordonnees(compet, match, set, point, lissage='oui')[0]
    vitesse_jA = coordonnees(compet, match, set, point, lissage='oui')[3]

    time_points = np.array([f*1/25 for f in range(len(pos_init_b))]) # on a la valeur de la surface atteignable en moins de tlim pour chaque frame et 1 frame dure 1/25s d'où cette discrétisation du temps
    xmax=(len(time_points)-1)/25
    nb_pts=len(pos_init_b)
    time_frappe = frappe(compet, match, set, point)[0]
    frappe_service = frappe(compet, match, set, point)[1]

    c=10 #fonctions_inertie.py > heatmap
    surfaces_jA = heatmap(compet, match, set, point, 0.8, que_taille_zone='oui')
    surface_points = []
    for f in range(len(surfaces_jA)):
        surface_points.append(surfaces_jA[f] * c**2 * 10**(-4)) # la liste surfaces_jA contient le nb de carrés de discrétisation de côté c (=10cm) à chaque frame du pt. il suffit donc de multiplier chaque elt de la liste par la surface d'un carré (10^-4 car on transpose en m²)

    norme_vitesse_jA=[]
    for i in range(len(vitesse_jA)):
        vitesse=sqrt(vitesse_jA[i][0]**2 + vitesse_jA[i][1]**2)
        vitesse=vitesse*10**(-2) # pour passer de cm/s à m/s
        norme_vitesse_jA.append(vitesse)

    norme_vitesse_jA_bis=[]
    surface_points_bis=[]
    moy_vit=sum(norme_vitesse_jA)/len(norme_vitesse_jA)
    moy_surf=sum(surface_points)/len(surface_points)
    for i in range(len(norme_vitesse_jA)):
        norme_vitesse_jA_bis.append(norme_vitesse_jA[i]/moy_vit)
        surface_points_bis.append(surface_points[i]/moy_surf)

    for i in range(1, nb_pts+1):
        plt.figure(figsize=(8, 5))
        plt.plot(time_points[:i], surface_points_bis[:i], label='évol surface')
        plt.plot(time_points[:i], norme_vitesse_jA_bis[:i], color='lawngreen', label='évol vitesse jA')
        plt.xlabel('Temps [s]')
        plt.xlim(0, xmax)
        plt.ylim(-1, 3)
        if frappe_service[1]<i:
            if frappe_service[0]==joueurA :
                plt.axvline(x=frappe_service[1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
            elif frappe_service[0]==joueurB :
                plt.axvline(x=frappe_service[1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
        for j in range(len(time_frappe)):
            if time_frappe[j][1]<i:
                if time_frappe[j][0]==joueurA:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--')
                else:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--')
        plt.legend(loc='upper left')
        plt.savefig('{}/output/match_{}/evolutions_temporelles/combinaison/evol_jA_{}.png'.format(chemin_tt_espace, match, i))
        plt.close()

    frames = np.stack([iio.imread('{}/output/match_{}/evolutions_temporelles/combinaison/evol_jA_{}.png'.format(chemin_tt_espace, match, i)) for i in range(1, nb_pts+1)], axis = 0)
    iio.mimwrite('{}/output/match_{}/point_{}/evol_temp_surf_vit_jA.gif'.format(chemin_tt_espace, match, point), frames, fps=nb_pts*25/len(pos_init_b))



def gif_evol_angle_jA(compet, match, set, point):
    with open('{}/{}/{}/{}_game.json'.format(chemin_pipeline, compet, match, match),  newline='') as fichier_game:
        data=json.load(fichier_game)
        game=[[cle,val] for cle,val in data.items()]
    joueurA=game[0][1]
    joueurB=game[1][1]
    lateralite_jA=game[2][1]

    pos_init_b = coordonnees(compet, match, set, point, lissage='oui')[0]

    time_points = np.array([f*1/25 for f in range(len(pos_init_b))]) # on a la valeur de la surface atteignable en moins de tlim pour chaque frame et 1 frame dure 1/25s d'où cette discrétisation du temps
    xmax=(len(time_points)-1)/25
    nb_pts=len(pos_init_b)
    time_frappe = frappe(compet, match, set, point)[0]
    frappe_service = frappe(compet, match, set, point)[1]

    angles_jA=angles_avant_bras(compet, match, set, point)[0] # en deuxième position il y a les angles du jB

    for i in range(1, nb_pts+1):
        plt.figure(figsize=(8, 5))
        if lateralite_jA=='droitier':
            plt.plot(time_points[:i], angles_jA[1][:i], label='angle_jA(t)')
        elif lateralite_jA=='gaucher':
            plt.plot(time_points[:i], angles_jA[0][:i], label='angle_jA(t)')
        plt.xlabel('Temps [s]')
        plt.xlim(0, xmax)
        plt.ylim(-5, 5)
        if frappe_service[1]<i:
            if frappe_service[0]==joueurA :
                plt.axvline(x=frappe_service[1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
            elif frappe_service[0]==joueurB :
                plt.axvline(x=frappe_service[1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
        for j in range(len(time_frappe)):
            if time_frappe[j][1]<i:
                if time_frappe[j][0]==joueurA:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--', label='{}'.format(joueurA))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='r', linestyle='--')
                else:
                    if j==0:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--', label='{}'.format(joueurB))
                    else:
                        plt.axvline(x=time_frappe[j][1]/25, color='k', linestyle='--')
        plt.legend(loc='upper left')
        plt.savefig('{}/output/match_{}/evolutions_temporelles/angle/angle_jA_{}.png'.format(chemin_tt_espace, match, i))
        plt.close()

    frames = np.stack([iio.imread('{}/output/match_{}/evolutions_temporelles/angle/angle_jA_{}.png'.format(chemin_tt_espace, match, i)) for i in range(1, nb_pts+1)], axis = 0)
    iio.mimwrite('{}/output/match_{}/point_{}/evol_temp_angle_jA.gif'.format(chemin_tt_espace, match, point), frames, fps=nb_pts*25/len(pos_init_b))



def gif_point(match, point, avec_heatmap='non', avec_banane='non', num_vitesse=1): # /!\ avec_banane='oui' ssi le joueurA est Alexis Lebrun (car la banane utilisée est la sienne). si d'autres bananes sont générées à posteriori, le code peut être simplement décliné pour des joueurs donnés.
    
    # dans un premier temps, on crée chaque image du point en ajoutant : quadrillage et image table en filigrane, positions des joueurs+balle, vitesses joueurs+balle, trajets des joueurs, position balle à la frappe (issue du fichier d'annot), banane si désirée, carte de chaleur des joueurs si désirée
    
    pos_init_b = coordonnees(match, point, lissage='oui', num_vitesse=num_vitesse)[0]

    if avec_heatmap=='non':
        ajout_vecteurs_trajectoires(match, point, [None, os.path.join(chemin_tt_espace,"environnement_table.png")])
    elif avec_heatmap=='oui':
        heatmap(match, point, 0.8, que_taille_zone='non', num_vitesse=num_vitesse)
        os.makedirs(os.path.join(chemin_tt_espace,"example",match,point,"output","heatmap","quadrillage"), exist_ok='True')
        for f in range(len(pos_init_b)):
            quadrillage_sur_image(os.path.join(chemin_tt_espace,"example",match,point,"heatmap","heatmap_frame_"+str(f)+".png"), os.path.join(chemin_tt_espace,"example",match,point,"output","heatmap","quadrillage"), 'heatmap_quadrillage_{}'.format(f))
        #ajout_vecteurs_trajectoires(compet, match, set, point, ['{}/output/match_{}/heatmap/quadrillage'.format(chemin_tt_espace, match), 'heatmap_quadrillage_{}.png'],True)
        ajout_vecteurs_trajectoires(match, point, [os.path.join(chemin_tt_espace,"example",match,point,"heatmap"), "image_{}.png"],False)

    if avec_banane=='oui': # dessin zone de frappe
        pos_init_jA = coordonnees(match, point, lissage='oui')[2]

        banane=banane_alexis(match) # /!\ pour la banane, les coordonnées sont donnnées comme : [[x, y], [x, y], ...] // clusters :  [[liste_des_x], [liste_des_y]]
        clusters_cd, clusters_r = banane_alexis_cd_r()

        if 0<pos_init_jA[0][1]<800/2: # jA en haut
            for i in range(len(pos_init_jA)):
                position_banane=[(pos_init_jA[i][0]-banane[j][0], pos_init_jA[i][1]+banane[j][1]) for j in range(len(banane))]
                position_cd_1=[(pos_init_jA[i][0]-clusters_cd[0][0][j], pos_init_jA[i][1]+clusters_cd[0][1][j]) for j in range(len(clusters_cd[0][0]))]
                position_cd_2=[(pos_init_jA[i][0]-clusters_cd[1][0][j], pos_init_jA[i][1]+clusters_cd[1][1][j]) for j in range(len(clusters_cd[1][0]))]
                position_r_1=[(pos_init_jA[i][0]-clusters_r[0][0][j], pos_init_jA[i][1]+clusters_r[0][1][j]) for j in range(len(clusters_r[0][0]))]
                position_r_2=[(pos_init_jA[i][0]-clusters_r[1][0][j], pos_init_jA[i][1]+clusters_r[1][1][j]) for j in range(len(clusters_r[1][0]))]
                image = Image.open(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(i)+".png")).convert('RGB')
                draw=ImageDraw.Draw(image)
                draw.polygon(position_banane, outline='dimgrey')
                draw.polygon(position_cd_1, outline='dimgrey')
                draw.polygon(position_cd_2, outline='dimgrey')
                draw.polygon(position_r_1, outline='dimgrey')
                draw.polygon(position_r_2, outline='dimgrey')
                image.save(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(i)+".png"))
        else :
            for i in range(len(pos_init_jA)):
                position_banane=[(pos_init_jA[i][0]+banane[j][0], pos_init_jA[i][1]-banane[j][1]) for j in range(len(banane))]
                position_cd_1=[(pos_init_jA[i][0]+clusters_cd[0][0][j], pos_init_jA[i][1]-clusters_cd[0][1][j]) for j in range(len(clusters_cd[0]))]
                position_cd_2=[(pos_init_jA[i][0]+clusters_cd[1][0][j], pos_init_jA[i][1]-clusters_cd[1][1][j]) for j in range(len(clusters_cd[1]))]
                position_r_1=[(pos_init_jA[i][0]+clusters_r[0][0][j], pos_init_jA[i][1]-clusters_r[0][1][j]) for j in range(len(clusters_r[0]))]
                position_r_2=[(pos_init_jA[i][0]+clusters_r[1][0][j], pos_init_jA[i][1]-clusters_r[1][1][j]) for j in range(len(clusters_r[1]))]
                image = Image.open(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(i)+".png")).convert('RGB')
                draw=ImageDraw.Draw(image)
                draw.polygon(position_banane, outline='dimgrey')
                draw.polygon(position_cd_1, outline='dimgrey')
                draw.polygon(position_cd_2, outline='dimgrey')
                draw.polygon(position_r_1, outline='dimgrey')
                draw.polygon(position_r_2, outline='dimgrey')
                image.save(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(i)+".png"))


    # ajout de la position de la frappe d'après le fichier 'annotation' (pour comparaison avec la position de la balle donnée par 'zone_joueur_avec_pos_balle_3D')

    time_frappe = frappe(match, point)[0]
    frappe_service = frappe( match, point)[1]

    for i in range(len(time_frappe)):
        image = Image.open(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(time_frappe[i][1])+".png")).convert('RGB')
        draw=ImageDraw.Draw(image)
        x = int(time_frappe[i][2][0]) + 400
        y = -int(time_frappe[i][2][1]) + 400
        draw.ellipse((x-10, y-10, x+10, y+10), fill='brown')
        image.save(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(time_frappe[i][1])+".png"))
    # ajout de la position de la frappe de SERVICE d'après 'annotation.csv' (n'est pas comprise dans 'time_frappe')
    image = Image.open(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(frappe_service[1])+".png")).convert('RGB')
    draw=ImageDraw.Draw(image)
    x = int(frappe_service[2][0]) + 400
    y = -int(frappe_service[2][1]) + 400
    draw.ellipse((x-10, y-10, x+10, y+10), fill='brown')
    image.save(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(frappe_service[1])+".png"))


    # seconde étape : on crée le gif à partir des images créées au dessus.
    
    cap=cv2.VideoCapture(os.path.join(chemin_tt_espace,"example",match,point,point+".mp4"))
    FPS=cap.get(cv2.CAP_PROP_FPS)
    frames = np.stack([iio.imread(os.path.join(chemin_tt_espace,"example",match,point,"output","image_"+str(i)+".png")) for i in range(len(pos_init_b))], axis = 0)
    iio.mimwrite(os.path.join(chemin_tt_espace,"example",match,point,"output", 'gif.gif'), frames, fps=FPS)
    # je ne supprime pas les images constitutives du gif, elle sont tjr dispo dans le même dossier que le gif

#gif_point("2024_WttSmash_Singapour", "ALEXIS-LEBRUN_vs_MA-LONG", 2, 31, avec_heatmap='oui', avec_banane='oui')