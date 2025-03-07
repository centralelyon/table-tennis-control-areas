"""def reformater_texte(texte):
    # Séparer le texte en éléments à partir des virgules
    mots = texte.split(",")
    
    # Ajouter un indice et reformater chaque mot
    mots_reformates = [f"{i} {mot.strip()}" for i, mot in enumerate(mots)]
    
    # Joindre les mots avec des retours à la ligne
    return "\n".join(mots_reformates)

# Texte fourni
texte = "nom,debut,fin,genre,lateralite,set,systeme,coup,type_service,type_coup,zone_jeu,faute,effet_coup,coor_balle_x,coor_balle_y,coor_balle_z,joueur_frappe,joueur_sur,coor_frappe_x,coor_frappe_y,coor_frappe_z,time_frappe,premier_rebond_x,premier_rebond_y,premier_rebond_z,time_premier_rebond,probleme_annotation,num_coup,winner,premier_topspin,serveur,relanceur,pos_balle_x_prece,pos_balle_y_prece,pos_balle_z_prece,pos_joueur_0_x,pos_joueur_0_y,pos_joueur_0_z,pos_joueur_1_x,pos_joueur_1_y,pos_joueur_1_z,pos_joueur_2_x,pos_joueur_2_y,pos_joueur_2_z,pos_joueur_3_x,pos_joueur_3_y,pos_joueur_3_z,dist_equipe_1_x,dist_equipe_1_y,dist_equipe_2_x,dist_equipe_2_y,dist_eucl_equipe_1,dist_eucl_equipe_2,num_point,service_lateralite,service_zone,remise_lateralite,remise_effet_coup,remise_zone,faute_du_point,faute_lateralite,nb_coup,valeur_derniere_zone,pos_joueur_0_debut_x,pos_joueur_0_debut_y,pos_joueur_0_debut_z,pos_joueur_1_debut_x,pos_joueur_1_debut_y,pos_joueur_1_debut_z,pos_joueur_0_frappe_x,pos_joueur_0_frappe_y,pos_joueur_0_frappe_z,pos_joueur_1_frappe_x,pos_joueur_1_frappe_y,pos_joueur_1_frappe_z,pos_joueur_0_fin_x,pos_joueur_0_fin_y,pos_joueur_0_fin_z,pos_joueur_1_fin_x,pos_joueur_1_fin_y,pos_joueur_1_fin_z,cote_jA,cote_jB,derniere_zone"

# Appeler la fonction et afficher le résultat
resultat = reformater_texte(texte)
print(resultat)"""

"""
import cv2
import numpy as np
import pandas as pd
from config import chemin_tt_espace,USER_PREFERENCE
import os


import cv2
import pandas as pd
import shutil

from scipy.spatial import ConvexHull
import time

# Charger l'image
image = cv2.imread("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/heatmap/image_0.png")  # Remplace par ton image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Définir une image d'accumulation des superpositions
result = np.zeros_like(image, dtype=np.uint8)  # Utilisation de 16 bits pour éviter la saturation rapide

# Détecter les pixels non blancs
non_white_pixels = np.column_stack(np.where(gray < 250))  # Seuil pour détecter les pixels non blancs

# Paramètres du contour convexe
rayon = 10  # Taille du contour convexe

# Générer le contour convexe autour de chaque pixel
compet = "2024_WttSmash_Singapour"
match = "ALEXIS-LEBRUN_vs_MA-LONG"
joueurA = "ALEXIS-LEBRUN"
coup = "revers"
effet = "topspin"



start_time = time.time()  # Début du chronomètre
df = pd.read_csv(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_enveloppe.csv"))  # Remplace par le bon nom de fichier


coup = "coup_droit"
dfr = pd.read_csv(os.path.join(USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"],compet,match,"enveloppe",match+"_"+joueurA+"_"+coup+"_"+effet+"_enveloppe.csv"))  # Remplace par le bon nom de fichier
        
i = 0
for y, x in non_white_pixels:
    if i > 1000000:
        break
    if y < 400:
        overlay = np.zeros_like(image, dtype=np.uint8)
        
        df_enveloppe = df.copy()
        
        df_enveloppe["X"] = -df_enveloppe["X"]
        df_enveloppe["X"] += x
        df_enveloppe["Y"] += y

        # Convertir les colonnes en array NumPy
        points = df_enveloppe[['X', 'Y']].values.astype(np.int32)  # Conversion en entier


        

        # Calcul de l'enveloppe convexe
        #hull = ConvexHull(points)
        #hull_points = points[hull.vertices]  # Sommets de l'enveloppe convexe

        # Convertir les points pour OpenCV (format requis)
        #hull_pts_cv2 = hull_points.reshape((-1, 1, 2))  # Reshape pour OpenCV

        # Colorier l'intérieur de l'enveloppe en bleu
        #cv2.fillPoly(image, [hull_pts_cv2], couleur)
        hull = cv2.convexHull(points)
        cv2.drawContours(overlay, [hull], -1, (0, 10, 0), thickness=-1)  # Dessin en bleu (BGR)
        result = cv2.add(result, overlay)




        overlay = np.zeros_like(image, dtype=np.uint8)
        
        dfr_enveloppe = dfr.copy()
        
        dfr_enveloppe["X"] = -dfr_enveloppe["X"]
        dfr_enveloppe["X"] += x
        dfr_enveloppe["Y"] += y

        # Convertir les colonnes en array NumPy
        points = dfr_enveloppe[['X', 'Y']].values.astype(np.int32)  # Conversion en entier
        hull = cv2.convexHull(points)
        cv2.drawContours(overlay, [hull], -1, (10, 0, 0), thickness=-1)  # Dessin en bleu (BGR)
        result = cv2.add(result, overlay)


        i += 1

end_time = time.time()  # Fin du chronomètre
execution_time = end_time - start_time  
print(execution_time)
# Normaliser pour éviter la saturation et améliorer la visibilité
#overlay = np.clip(overlay, 0, 255).astype(np.uint8)

# Fusionner avec l’image originale
#output = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)

# Sauvegarder et afficher
cv2.imwrite("test_sup_enveloppes_frappe.png", result)


"""





import cv2
import numpy as np
import pandas as pd
import os
import time
from config import USER_PREFERENCE  # Import direct

# 📌 Paramètres
IMAGE_PATH = "C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/heatmap/image_0.png"
compet = "2024_WttSmash_Singapour"
match = "ALEXIS-LEBRUN_vs_MA-LONG"
joueurA = "ALEXIS-LEBRUN"
effet = "topspin"
max_points = 1_000_000  # Limite de pixels à traiter

# 📌 Chargement de l'image
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 📌 Détection des pixels non blancs
non_white_pixels = np.column_stack(np.where(gray < 250))[:max_points]  # Limite de traitement

# 📌 Chargement des enveloppes convexes sous format NumPy (évite Pandas dans la boucle)
pipeline_path = USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"]
df_revers = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_revers_{effet}_enveloppe.csv"))[['X', 'Y']]#.values.astype(np.int32)
df_coup_droit = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_coup_droit_{effet}_enveloppe.csv"))[['X', 'Y']]#$.values.astype(np.int32)
df_tous = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_tous_enveloppe.csv"))[['X', 'Y']]#.values.astype(np.int32)
premier = True
# 📌 Pré-allocation de l'image résultat
result = np.zeros_like(image, dtype=np.uint8)
result_gris = np.zeros_like(image, dtype=np.uint8)


# 📌 Fonction d'application des enveloppes convexes
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

# 📌 Exécution optimisée
start_time = time.time()
c = 10
intensite = 1#int(c/2)
# Traitement de tous les pixels
for i, (y, x) in enumerate(non_white_pixels):
    if y < 400 and y%c == 0 and x%c == 0 and 1==0:
        overlay = np.zeros_like(image, dtype=np.uint8)
        appliquer_enveloppe(overlay, df_revers.values.astype(np.int32), x, y, (0, intensite, 0))  # Vert foncé
        appliquer_enveloppe(overlay, df_coup_droit.values.astype(np.int32), x, y, (intensite, 0, 0))  # Bleu foncé
        #appliquer_enveloppe(result_gris, df_tous, x, y, (100, 100, 100))  # Bleu foncé
        result = cv2.add(result, overlay)  # Superposition efficace
    elif y > 400 and y%c == 0 and x%c == 0:
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
        

"""
mask_blue = result[:, :, 0] != 0
mask_green = result[:, :, 1] != 0

# Appliquer l'opération 255 - pixel uniquement aux pixels non nuls
result[:, :, 1][mask_blue] = 255 - result[:, :, 0][mask_blue]  # Canal Bleu
result[:, :, 2][mask_blue] = 255 - result[:, :, 0][mask_blue]  # Canal Bleu

result[:, :, 1][mask_green] = 255 - result[:, :, 0][mask_green]
result[:, :, 2][mask_green] = 255 - result[:, :, 0][mask_green]"""

execution_time = time.time() - start_time
output = sliding_window_max(result, window_size=c)
print(f"Temps d'exécution : {execution_time:.6f} secondes")
cv2.imwrite("test_sup_enveloppes_frappe.png", output)
zedez

heatmap = output[:, :, 1].copy()
heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
h,w = heatmap_normalized.shape
heatmap_colored = np.full((h, w, 3), 255, dtype=np.uint8) #field_frame_np.copy()#cv2.cvtColor(heatmap_normalized, cv2.COLOR_GRAY2BGR)
contours = []
thresholds = [100, 150, 200, 250]  # Seuils de densité
for t in thresholds:
    # Appliquer un seuil pour créer une image binaire
    _, binary = cv2.threshold(heatmap_normalized, t, 255, cv2.THRESH_BINARY)
    blurred_image = cv2.GaussianBlur(binary, (5, 5), 0)

    # Détecter les contours avec Canny
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
    
    # Trouver les contours
    contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.append((t, contour))
    cv2.drawContours(heatmap_colored, contour, -1, (0, 0, 0), 1)
    cv2.drawContours(output, contour, -1, (0, 0, 0), 1)
heatmap = output[:, :, 0].copy()
heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
contours = []
thresholds = [100, 150, 200, 250]  # Seuils de densité
for t in thresholds:
    # Appliquer un seuil pour créer une image binaire
    _, binary = cv2.threshold(heatmap_normalized, t, 255, cv2.THRESH_BINARY)
    blurred_image = cv2.GaussianBlur(binary, (5, 5), 0)

    # Détecter les contours avec Canny
    edges = cv2.Canny(blurred_image, threshold1=100, threshold2=200)
    
    # Trouver les contours
    contour, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.append((t, contour))
    cv2.drawContours(heatmap_colored, contour, -1, (0, 0, 0), 1)
    cv2.drawContours(output, contour, -1, (0, 0, 0), 1)

cv2.imwrite("test_sup_enveloppes_frappe_zones.png", heatmap_colored)
cv2.imwrite("test_sup_enveloppes_frappe_zones_avec_tous.png", output)
dez

# Trouver les pixels noirs (0,0,0) et les remplacer par du blanc (255,255,255)
mask = (result[:, :, 0] == 0) & (result[:, :, 1] == 0) & (result[:, :, 2] == 0)
result[mask] = [255, 255, 255]

mask = (result_gris[:, :, 0] == 0) & (result_gris[:, :, 1] == 0) & (result_gris[:, :, 2] == 0)
result_gris[mask] = [255, 255, 255]

masque = (result[:, :, 0] < 255) | (result[:, :, 1] < 255) | (result[:, :, 2] < 255)

# Copier uniquement les pixels non blancs sur l'image cible
result_gris[masque] = result[masque]

# 📌 Sauvegarde de l'image
cv2.imwrite("test_sup_enveloppes_frappe.png", result_gris)

'''
import cv2
import numpy as np
import pandas as pd
import os
import time
import threading
from config import USER_PREFERENCE  # Import direct

# 📌 Paramètres
IMAGE_PATH = "C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/heatmap/image_0.png"
compet = "2024_WttSmash_Singapour"
match = "ALEXIS-LEBRUN_vs_MA-LONG"
joueurA = "ALEXIS-LEBRUN"
effet = "topspin"
max_points = 1_000_000  # Limite de pixels à traiter
num_threads = 4  # Nombre de threads à utiliser

# 📌 Chargement de l'image
image = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 📌 Détection des pixels non blancs
non_white_pixels = np.column_stack(np.where(gray < 250))[:max_points]  # Limite de traitement

# 📌 Chargement des enveloppes convexes sous format NumPy (évite Pandas dans la boucle)
pipeline_path = USER_PREFERENCE["pipeline"]["LOCAL-PIPELINE-TT"]
df_revers = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_revers_{effet}_enveloppe.csv"))[['X', 'Y']].values.astype(np.int32)
df_coup_droit = pd.read_csv(os.path.join(pipeline_path, compet, match, "enveloppe", f"{match}_{joueurA}_coup_droit_{effet}_enveloppe.csv"))[['X', 'Y']].values.astype(np.int32)

# 📌 Pré-allocation de l'image résultat (évite les conflits d'écriture)
results = [np.zeros_like(image, dtype=np.uint8) for _ in range(num_threads)]

# 📌 Fonction d'application des enveloppes convexes
def appliquer_enveloppe(points, x, y):
    """Retourne une enveloppe convexe transformée"""
    points_trans = np.copy(points)
    points_trans[:, 0] = -points_trans[:, 0] + x
    points_trans[:, 1] += y
    return cv2.convexHull(points_trans)

# 📌 Fonction de traitement des pixels en parallèle
def traiter_pixels(thread_id, pixels):
    """Traite un sous-ensemble de pixels dans un thread spécifique"""
    
    result = np.zeros_like(image, dtype=np.uint8)
    for y, x in pixels:
        if y >= 400:
            continue
        local_result = np.zeros_like(image, dtype=np.uint8)
        hull1 = appliquer_enveloppe(df_revers, x, y)
        hull2 = appliquer_enveloppe(df_coup_droit, x, y)

        cv2.drawContours(local_result, [hull1], -1, (0, 10, 0), thickness=-1)  # Vert foncé
        cv2.drawContours(local_result, [hull2], -1, (10, 0, 0), thickness=-1)  # Bleu foncé
        result = cv2.add(result, local_result) 

    results[thread_id] = result  # Sauvegarde du résultat du thread

# 📌 Exécution optimisée avec threading
start_time = time.time()

# Découpage des pixels en sous-ensembles pour chaque thread
split_pixels = np.array_split(non_white_pixels, num_threads)
threads = []

for i in range(num_threads):
    thread = threading.Thread(target=traiter_pixels, args=(i, split_pixels[i]))
    threads.append(thread)
    thread.start()

# Attente de la fin de tous les threads
for thread in threads:
    thread.join()

# Fusion des résultats
result = np.sum(results, axis=0, dtype=np.uint8)
#result = result = np.zeros_like(image, dtype=np.uint8)
#for im in results:
#    result = cv2.add(result, im) 

execution_time = time.time() - start_time
print(f"Temps d'exécution : {execution_time:.6f} secondes")

# 📌 Sauvegarde de l'image
cv2.imwrite("test_sup_enveloppes_frappe.png", result)
'''