import cv2
import numpy as np
import pandas as pd
import os
from config import USER_PREFERENCE  # Import direct
import matplotlib.pyplot as plt


import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def tracer_courbe_air_surface_frappe(chemin_point, thresholds):
    """
    Fonction permettant de tracer l'air de la surface de frappe en fonction des seuils
    """
    csv_annotation = pd.read_csv(os.path.join(chemin_point, os.path.split(chemin_point)[1] + "_annotation.csv"))
    liste_frappes = csv_annotation["time_frappe"].to_list()
    liste_rebonds = csv_annotation["debut"].to_list()


    csv_position = pd.read_csv(os.path.join(chemin_point, "csv_json_openpose",os.path.split(chemin_point)[1] + "_zone_joueur_avec_pos_balle.csv"))
    
    liste_images = os.listdir(os.path.join(chemin_point, "enveloppe"))
    liste_images = [element for element in liste_images if "frappe" not in element]
    liste_images.sort()
    liste_images.sort(key=len)

    liste_x_bleu = []
    liste_x_vert = []
    
    for seuil in thresholds:
        liste_x_bleu_seuil = []
        liste_x_vert_seuil = []
        
        for image in liste_images:
            im = cv2.imread(os.path.join(chemin_point, "enveloppe", image))

            # Canal bleu
            heatmap_bleu = im[:, :, 0].copy()
            canal_bleu = ((heatmap_bleu - heatmap_bleu.min()) / (heatmap_bleu.max() - heatmap_bleu.min()) * 255).astype(np.uint8)
            nombre_pixels_bleus = np.sum(canal_bleu > seuil)
            liste_x_bleu_seuil.append(nombre_pixels_bleus)
            
            # Canal vert
            heatmap_vert = im[:, :, 1].copy()
            canal_vert = ((heatmap_vert - heatmap_vert.min()) / (heatmap_vert.max() - heatmap_vert.min()) * 255).astype(np.uint8)
            nombre_pixels_verts = np.sum(canal_vert > seuil)
            liste_x_vert_seuil.append(nombre_pixels_verts)

        liste_x_bleu.append(liste_x_bleu_seuil)
        liste_x_vert.append(liste_x_vert_seuil)
    
    # Création de la figure avec deux graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 ligne, 2 colonnes

    # Graphique 1 : Evolution de la zone de frappe (canal bleu)
    for i, liste in enumerate(liste_x_bleu):
        axes[0].plot(liste, label=f"Seuil {thresholds[i]}")

    for x in liste_frappes:
        axes[0].axvline(x=x, color='r', linestyle='--', linewidth=2)
        
    for x in liste_rebonds:
        axes[0].axvline(x=x, color='b', linestyle='--', linewidth=2)

    axes[0].set_title("Évolution de la zone de frappe (Canal Bleu)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Nb pixels")
    axes[0].legend()
    axes[0].grid(True)

    # Graphique 2 : Evolution de la zone de frappe (canal vert)
    for i, liste in enumerate(liste_x_vert):
        axes[1].plot(liste, label=f"Seuil {thresholds[i]}")

    for x in liste_frappes:
        axes[1].axvline(x=x, color='r', linestyle='--', linewidth=2)
        
    for x in liste_rebonds:
        axes[1].axvline(x=x, color='b', linestyle='--', linewidth=2)

    axes[1].set_title("Évolution de la zone de frappe (Canal Vert)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Nb pixels")
    axes[1].legend()
    axes[1].grid(True)

    # Affichage des graphiques
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    compet = "2024_WttSmash_Singapour"
    match = "ALEXIS-LEBRUN_vs_MA-LONG"
    point = "set_1_point_0"
    joueurA = "ALEXIS-LEBRUN"
    effet = "topspin"
    chemin_point = os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"clips",point)
    thresholds = [100, 150, 200, 250]
    tracer_courbe_air_surface_frappe(chemin_point,thresholds)