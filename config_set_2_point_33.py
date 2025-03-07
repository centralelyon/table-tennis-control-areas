import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import imageio

from mathutils import Matrix, Vector

[324, 263], [476, 263], [476, 537], [324, 537]
table = np.array([[821.33, 363.55], [845.91, 459.84], [436.37, 456.65], [466.52, 362.25]], dtype=np.float32)  # Points sur image1
table_proj_sol = np.array([[807.0560241572812, 450.72322462289463], [850.8468941551321, 554.8325167532398], 
                           [454.57147485880085, 555.396836955502], [463.99038077237475, 448.45869007450403]], dtype=np.float32)
pts_image2 = np.array([[324, 263], [476, 263], [476, 537], [324, 537]], dtype=np.float32)  # Points sur image2



def supperposition(chemin_image1,chemin_image2,chemin_enveloppe,pts_image1_table,pts_image1_table_proj,pts_image2,chemin_sauvegarde,faire_enveloppe):
    """
        Fonction qui fait supperposition entre deux image en plaçant en fonction de la position de la table
    """
    # 1️⃣ Charger les images
    image1 = cv2.imread(chemin_image1)  # Image de base
    image2 = cv2.imread(chemin_image2)  # Image à transformer
    image_enveloppe = cv2.imread(chemin_enveloppe)  
    
    image2[263:537,324:476] = [255, 255, 255]
    image3 = cv2.imread(chemin_image2)
    image2_inv = np.full_like(image3, [255, 255, 255], dtype=np.uint8)
    # Copier uniquement la région [263:537, 324:476] depuis l'image originale
    image2_inv[263:537, 324:476] = image3[263:537, 324:476]

    # Convertir en format PIL pour gérer la transparence
    image1_pil = Image.open(chemin_image1).convert("RGBA")


    # 3️⃣ Calculer la matrice d’homographie
    H, _ = cv2.findHomography(pts_image2, pts_image1_table_proj, cv2.RANSAC)
    H_inv, _ = cv2.findHomography(pts_image2, pts_image1_table, cv2.RANSAC)

    h, w, _ = image1.shape
    image2_warped = cv2.warpPerspective(image2, H, (w, h))
    image2_inv_warped = cv2.warpPerspective(image2_inv, H_inv, (w, h))

    h_enveloppe, w_enveloppe, _ = image_enveloppe.shape
    image_enveloppe_warped = cv2.warpPerspective(image_enveloppe, H_inv, (w, h))

    # Définir les 6 points du polygone
    points = np.array([[pts_image1_table[0][0], pts_image1_table[0][1]], [pts_image1_table[1][0], pts_image1_table[1][1]],  
                       [pts_image1_table[2][0], pts_image1_table[2][1]], [pts_image1_table[3][0], pts_image1_table[3][1]]], np.int32)
    # Créer un masque noir (même taille que l'image, mais en niveaux de gris)
    masque = np.zeros(image2_warped.shape[:2], dtype=np.uint8)
    # Dessiner un polygone blanc sur le masque
    cv2.fillPoly(masque, [points], 255)
    # Appliquer le masque : rendre les pixels blancs là où le masque est blanc
    image2_warped[masque == 255] = [255, 255, 255]
    
    points = np.array([[pts_image1_table_proj[0][0], pts_image1_table_proj[0][1]], [pts_image1_table_proj[1][0], pts_image1_table_proj[1][1]],  
                       [pts_image1_table_proj[2][0], pts_image1_table_proj[2][1]], [pts_image1_table_proj[3][0], pts_image1_table_proj[3][1]]], np.int32)
    # Créer un masque noir (même taille que l'image, mais en niveaux de gris)
    masque = np.zeros(image2_warped.shape[:2], dtype=np.uint8)
    # Dessiner un polygone blanc sur le masque
    cv2.fillPoly(masque, [points], 255)
    # Appliquer le masque : rendre les pixels blancs là où le masque est blanc
    image2_warped[masque == 255] = [255, 255, 255]        

    if image_enveloppe_warped.shape != image2_warped.shape:
        image_enveloppe_warped = cv2.resize(image_enveloppe_warped, (image2_warped.shape[1], image2_warped.shape[0]))
    
    if faire_enveloppe:
        mask = np.any(image_enveloppe_warped != [255, 255, 255], axis=-1)  # Vrai si le pixel n'est pas blanc

        # Appliquer les pixels de image2 sur image1
        image2_warped[mask] = image_enveloppe_warped[mask]

    # Convertir en format PIL
    image2_warped_pil = Image.fromarray(cv2.cvtColor(image2_warped, cv2.COLOR_BGR2RGBA))
    image2_inv_warped_pil = Image.fromarray(cv2.cvtColor(image2_inv_warped, cv2.COLOR_BGR2RGBA))
    image_enveloppe_warped_pil = Image.fromarray(cv2.cvtColor(image_enveloppe_warped, cv2.COLOR_BGR2RGBA))

    

    # 5️⃣ Rendre le blanc transparent et appliquer la transparence
    image2_data = np.array(image2_warped_pil)
    r, g, b, a = image2_data[:, :, 0], image2_data[:, :, 1], image2_data[:, :, 2], image2_data[:, :, 3]
    mask = (r > 254) & (g > 254) & (b > 254)  # Détecte le blanc
    image2_data[mask] = [0, 0, 0, 0]  # Rendre transparent
    
    image2_inv_data = np.array(image2_inv_warped_pil)
    r, g, b, a = image2_inv_data[:, :, 0], image2_inv_data[:, :, 1], image2_inv_data[:, :, 2], image2_inv_data[:, :, 3]
    mask = (r > 254) & (g > 254) & (b > 254)  # Détecte le blanc
    image2_inv_data[mask] = [0, 0, 0, 0]  # Rendre transparent

    # 🔹 Appliquer une transparence de 50%
    image2_data[:, :, 3] = (image2_data[:, :, 3] * 0.5).astype(np.uint8)

    image2_transparent = Image.fromarray(image2_data)

    
    image2_inv_data[:, :, 3] = (image2_inv_data[:, :, 3] * 0.5).astype(np.uint8)

    image2_inv_transparent = Image.fromarray(image2_inv_data)

    # 6️⃣ Superposition des images
    image1_pil.paste(image2_transparent, (0, 0), image2_transparent)
    image1_pil.paste(image2_inv_transparent, (0, 0), image2_inv_transparent)






    # 7️⃣ Sauvegarder et afficher le résultat
    image1_pil.save(chemin_sauvegarde)
    #image1_pil.show()


def compute_average_frame(video_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo")

    # Initialiser l'accumulateur d'images
    frame_count = 0
    avg_frame = None
    
    while True:
        ret, frame = cap.read()
        print(frame_count)
        if not ret:
            break  # Fin de la vidéo

        frame = frame.astype(np.float32)  # Convertir en float pour l'addition
        if avg_frame is None:
            avg_frame = frame
        else:
            avg_frame += frame  # Additionner les images

        frame_count += 1

    cap.release()

    if frame_count == 0:
        raise ValueError("Aucune image dans la vidéo")

    # Calculer la moyenne et convertir en format uint8 (image standard)
    avg_frame /= frame_count
    avg_frame = np.clip(avg_frame, 0, 255).astype(np.uint8)

    # Sauvegarder l'image si un chemin est donné
    if save_path:
        cv2.imwrite(save_path, avg_frame)
        print(f"Image moyenne enregistrée à : {save_path}")

    return avg_frame

def compute_median_frame(video_path, save_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Aucune image dans la vidéo")

    # Convertir en tableau NumPy et calculer la médiane sur l'axe temporel
    frames_array = np.stack(frames, axis=3)  # Empilement des images
    median_frame = np.median(frames_array, axis=3).astype(np.uint8)  # Médiane par canal

    # Sauvegarder l'image si un chemin est donné
    if save_path:
        cv2.imwrite(save_path, median_frame)
        print(f"Image médiane enregistrée à : {save_path}")

    return median_frame


def subtract_images(image_path1, image_path2, save_path=None):
    # Charger les images
    img1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
    img2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        raise ValueError("Impossible de charger l'une des images.")

    # Vérifier que les images ont la même taille
    if img1.shape != img2.shape:
        raise ValueError("Les images doivent avoir la même taille pour être soustraites.")

    # Soustraction des images
    result = cv2.absdiff(img1, img2)  # absdiff garantit des valeurs positives

    # Sauvegarde si un chemin est donné
    if save_path:
        cv2.imwrite(save_path, result)
        print(f"Image soustraite enregistrée à : {save_path}")

    return result

def fusionner_images(image1_path, image2_path, image3_path, output_path, seuil_noir=40):
    """
        3 images de même dimension, on utilise l'image deux pour créer un masque avec tous les pixels qui ne sont pas noirs, ensuite on utilise ce masque pour 
        récupérer sur l'image 3 les éléments correspondants au masque et coller sur l'image 1
        Entrée:
                - Image avec l'affichage du modèle sur l'image de la frappe
                - Image contenant une soustraction de fond (créée à l'aide d'une image moyenne/médiane) qui sera utilisée comme masque
                - Image originale de la frappe
                - Chemin pour sauvegarder l'image
        Sortie:
                - Enregistrement de l'image
    """
    # Charger les images
    image1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)  # PNG (4 canaux)
    image2 = cv2.imread(image2_path, cv2.IMREAD_COLOR)      # JPG (3 canaux)
    image3 = cv2.imread(image3_path, cv2.IMREAD_COLOR)      # JPG (3 canaux)

    # Vérifier que les dimensions sont identiques
    if image1.shape[:2] != image2.shape[:2] or image1.shape[:2] != image3.shape[:2]:
        raise ValueError("Toutes les images doivent avoir la même dimension")

    # Si image1 a 4 canaux (RGBA), on enlève le canal alpha
    if image1.shape[2] == 4:
        alpha_channel = image1[:, :, 3]  # Canal alpha
        image1 = image1[:, :, :3]  # Convertir en RGB

    # Détection des pixels non proches du noir dans image2
    gris = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)  # Conversion en niveaux de gris
    mask = (gris > seuil_noir).astype(np.uint8) * 255  # Masque : blanc si pixel pas proche du noir

    # Extraire les pixels de l'image 3 correspondant au masque
    elements_extraits = cv2.bitwise_and(image3, image3, mask=mask)

    # Appliquer le masque inversé sur l'image 1 pour garder ses pixels d'origine
    masque_inverse = cv2.bitwise_not(mask)
    image1_fond = cv2.bitwise_and(image1, image1, mask=masque_inverse)

    # Fusionner l'image 1 et les éléments extraits de l'image 3
    image_finale = cv2.add(image1_fond, elements_extraits)

    # Ajouter le canal alpha d'origine si existant
    if 'alpha_channel' in locals():
        image_finale = np.dstack((image_finale, alpha_channel))

    # Sauvegarder l'image finale en PNG pour conserver la transparence
    cv2.imwrite(output_path, image_finale, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    return image_finale

def mp4_to_gif(video_path, gif_path, fps=15):
    """
    Convertit une vidéo MP4 en GIF animé avec une meilleure qualité et une boucle infinie.

    :param video_path: Chemin de la vidéo MP4 en entrée.
    :param gif_path: Chemin où enregistrer le GIF de sortie.
    :param fps: Nombre d'images par seconde pour le GIF (par défaut 15).
    :param quality: Qualité de compression du GIF (0-100, 100 = meilleure qualité).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Erreur : Impossible d'ouvrir la vidéo.")
        return

    frames = []
    
    # Lire les frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV lit en BGR, conversion en RGB
        frames.append(frame_rgb)

    cap.release()  # Fermer la vidéo

    if not frames:
        print("❌ Erreur : Aucun frame extrait.")
        return

    # Sauvegarder en GIF avec boucle infinie et qualité améliorée
    imageio.mimsave(gif_path, frames, fps=fps, loop=0, quantizer='nq', palettesize=256)

    print(f"✅ GIF enregistré avec succès : {gif_path}")


def draw_points_on_image(image_path, points, output_path, point_color=(0, 255, 0), point_radius=5):
    """
    Affiche une liste de points sur une image et enregistre le résultat.

    Arguments :
    - image_path : chemin de l'image d'entrée.
    - points : liste de tuples (x, y) représentant les coordonnées des points à afficher.
    - output_path : chemin où enregistrer l'image avec les points affichés.
    - point_color : couleur des points (B, G, R) (par défaut : vert).
    - point_radius : rayon des points affichés.

    Retour :
    - Image avec les points affichés.
    """
    # Charger l'image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {image_path}")

    # Dessiner chaque point sur l'image
    for (x, y) in points:
        cv2.circle(image, (int(x), int(y)), point_radius, point_color, -1)  # -1 remplit le cercle

    # Sauvegarder l'image
    cv2.imwrite(output_path, image)

    # Afficher l'image avec les points
    #cv2.imshow("Image avec points", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def compute_intrinsic_matrix(focale_mm, sensor_width_mm, img_width_px, img_height_px, cx, cy):
    """
    Calcule la matrice intrinsèque K de la caméra en estimant la hauteur du capteur.

    Arguments :
    - focale_mm : distance focale en mm
    - sensor_width_mm : largeur du capteur en mm
    - img_width_px : largeur de l'image en pixels
    - img_height_px : hauteur de l'image en pixels
    - cx, cy : centre optique en pixels

    Retourne :
    - Matrice intrinsèque K (3x3)
    """
    # Calcul de la hauteur du capteur en mm (en supposant le même ratio que l'image)
    sensor_height_mm = sensor_width_mm * (img_height_px / img_width_px)

    # Calcul des focales en pixels
    fx = (focale_mm * img_width_px) / sensor_width_mm
    fy = (focale_mm * img_height_px) / sensor_height_mm

    # Matrice intrinsèque
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])
    return K


def project_3d_to_2d(K, R, t, point_3D):
    """
    Projette un point 3D dans l'image en utilisant la matrice de projection de la caméra.

    Arguments :
    - K : Matrice intrinsèque (3x3)
    - R : Matrice de rotation (3x3)
    - t : Vecteur de translation (3x1)
    - point_3D : Coordonnées du point en 3D (x, y, z)

    Retourne :
    - Coordonnées projetées (u, v) en pixels
    """

    # S'assurer que t est bien (3,1)
    if t.shape != (3, 1):
        t = t.reshape(3, 1)

    # Matrice de projection P = K [R|t]
    Rt = np.hstack((R, t))  # Concaténer R et t (3x4)
    P = np.dot(K, Rt)  # P = K * [R|t] → (3x4)

    # Convertir le point 3D en coordonnées homogènes (ajouter 1)
    point_3D_h = np.array([point_3D[0], point_3D[1], point_3D[2], 1])

    # Projection 2D : x' = P * X
    projected_point = np.dot(P, point_3D_h)

    # Convertir en coordonnées cartésiennes (diviser par la coordonnée homogène)
    u = projected_point[0] / projected_point[2]
    v = projected_point[1] / projected_point[2]

    return (u, v)

def calculer_camera_params(points_2d, image_size):
    """
    Calcule les paramètres de la caméra.

    Arguments :
    - points_2d : Coordonnées 2D des points de référence.
    - image_size : Taille de l'image (largeur, hauteur).

    Retourne :
    - R : Matrice de rotation (3x3).
    - t : Vecteur de translation (3x1).
    - K : Matrice intrinsèque (3x3).
    """

    # Points 3D correspondant
    points_3d = np.array([
        [-152/2, 274/2, 76],
        [152/2, 274/2, 76],
        [152/2, -274/2, 76],
        [-152/2, -274/2, 76],
        [152/2, 0, 76],
        [-152/2, 0, 76]
    ], dtype=np.float32).reshape(-1, 1, 3)

    # Initialisation de la matrice intrinsèque K
    focal_length = max(image_size)  # Estimation de la focale
    K = np.array([
        [focal_length, 0, image_size[0] / 2],
        [0, focal_length, image_size[1] / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Initialisation des coefficients de distorsion (assumés nuls)
    dist_coeffs = np.zeros((5, 1))

    # Résolution de PnP pour obtenir la pose de la caméra
    success, rvec, tvec = cv2.solvePnP(
        points_3d, points_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise ValueError("Échec du solvePnP : impossible de calculer la pose de la caméra.")

    # Convertir rvec en matrice de rotation 3x3
    R, _ = cv2.Rodrigues(rvec)

    return R, tvec.reshape(3,1), K


def creation_gif_zone_image_proj(dossier_point,dossier_image_modele,table,table_proj_sol,pts_image2):
    """
        Fonction permettant de faire un gif/vidéo d'un point avec toutes les zones affichées sur la vidéo en utilisant les fonction:
            - supperposition()
            - fusionner_images()
    """

    if not os.path.isdir(os.path.join(dossier_point,"images_modele")):
        os.mkdir(os.path.join(dossier_point,"images_modele"))

    
    if not os.path.isdir(os.path.join(dossier_point,"images_video")):
        os.mkdir(os.path.join(dossier_point,"images_video"))

        
    if not os.path.isdir(os.path.join(dossier_point,"images_modele_fusion")):
        os.mkdir(os.path.join(dossier_point,"images_modele_fusion"))

        
    if not os.path.isdir(os.path.join(dossier_point,"soustraction")):
        os.mkdir(os.path.join(dossier_point,"soustraction"))

    chemin_video = os.path.join(dossier_point,os.path.split(dossier_point)[1]+".mp4")
    cap = cv2.VideoCapture(chemin_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    i = 0
    while True:
        ret, frame = cap.read()  # Lire une image de la vidéo
        if not ret:
            break  # Sortir si la vidéo est terminée
        cv2.imwrite(os.path.join(dossier_point,"images_video",str(i)+".jpg"), frame)
        i += 1
    
    
    compute_median_frame(chemin_video,save_path=os.path.join(dossier_point,"image_mediane.jpg"))

    for i in range(len(os.listdir(os.path.join(dossier_point,"heatmap")))):
        supperposition(os.path.join(dossier_point,"images_video",str(i)+".jpg"),
                       os.path.join(dossier_image_modele,"image_"+str(i)+".png"),
                       os.path.join(dossier_point,"enveloppe",str(i)+".png"),
               table,table_proj_sol,pts_image2,
               os.path.join(dossier_point,"images_modele",str(i)+".png"),False)
        subtract_images(os.path.join(dossier_point,"images_video",str(i)+".jpg")
                        ,os.path.join(dossier_point,"image_mediane.jpg")
                        ,save_path=os.path.join(dossier_point,"soustraction",str(i)+".jpg"))
        fusionner_images(os.path.join(dossier_point,"images_modele",str(i)+".png"), 
                         os.path.join(dossier_point,"soustraction",str(i)+".jpg"), 
                         os.path.join(dossier_point,"images_video",str(i)+".jpg"),
                         os.path.join(dossier_point,"images_modele_fusion",str(i)+".jpg"), 
                         seuil_noir=20)
    
    dossier_images = os.path.join(dossier_point,"images_modele_fusion")
    chemin_sortie = os.path.join(dossier_point,os.path.split(dossier_point)[1]+"_video_avec_model.mp4")
    creer_video_depuis_dossier_image(dossier_images, chemin_sortie, fps)


def creer_video_depuis_dossier_image(dossier_images, chemin_sortie, fps):
    # Lister et trier les images par nom
    images = os.listdir(dossier_images)
    images.sort()
    images.sort(key=len)

    # Lire la première image pour obtenir les dimensions
    image_test = cv2.imread(os.path.join(dossier_images, images[0]))
    hauteur, largeur, _ = image_test.shape

    # Initialiser l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec vidéo pour MP4
    video = cv2.VideoWriter(chemin_sortie, fourcc, fps, (largeur, hauteur))

    # Ajouter toutes les images à la vidéo
    for image_name in images:
        chemin_image = os.path.join(dossier_images, image_name)
        frame = cv2.imread(chemin_image)
        video.write(frame)

    # Finaliser la vidéo
    video.release()
        

def renomer_fichier(dossier):
    for fichier in os.listdir(dossier):
        if "heatmap_frame_" in fichier:  # Vérifier si le fichier contient "heatmap_frame_"
            nouveau_nom = fichier.replace("heatmap_frame_", "image_")  # Remplacement
            ancien_chemin = os.path.join(dossier, fichier)
            nouveau_chemin = os.path.join(dossier, nouveau_nom)
            
            os.rename(ancien_chemin, nouveau_chemin)  # Renommer le fichier

#renomer_fichier("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/heatmap")

def creer_video_to_image(dossier_point):
    """
        Fonction permettant d'extraire toutes les images d'une vidéo et de les mettre dans "images_video"
    """
    if not os.path.isdir(os.path.join(dossier_point,"images_video")):
        os.mkdir(os.path.join(dossier_point,"images_video"))

    chemin_video = os.path.join(dossier_point,os.path.split(dossier_point)[1]+".mp4")
    cap = cv2.VideoCapture(chemin_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    i = 0
    while True:
        ret, frame = cap.read()  # Lire une image de la vidéo
        if not ret:
            break  # Sortir si la vidéo est terminée
        cv2.imwrite(os.path.join(dossier_point,"images_video",str(i)+".jpg"), frame)
        i += 1


def cacul_zone_toutes_frame(dossier_point,nom_joueur):
    """
        Fonction permettant de calculer les intensités des couleurs pour la zone de la frappe pour toutes les frames du point donné
        Il faut donner le joueur que l'on veut analyser pour créer le fichier
        Entrée:
                - Le dossier du point
    """
    liste_images = os.listdir(os.path.join(dossier_point,"enveloppe"))
    df = pd.read_csv(os.path.join(dossier_point,os.path.split(dossier_point)[1]+"_annotation.csv"))
    num_point = int(os.path.split(dossier_point)[1].split("point_")[1])

    liste_debut_frappe_adverse = []
    liste_debut_frappe_joueur = []
    liste_num_coup = []
    liste_coor_frappe_x = []
    liste_coor_frappe_y = []
    liste_lateralite = []
    liste_effet_coup = []

    for index, row in df.iterrows():
        if row["nom"] != nom_joueur:
            liste_debut_frappe_adverse.append(row["time_frappe"])
        elif len(liste_debut_frappe_adverse) > len(liste_debut_frappe_joueur):
            liste_debut_frappe_joueur.append(row["time_frappe"])
            liste_coor_frappe_x.append(row["coor_frappe_x"])
            liste_coor_frappe_y.append(row["coor_frappe_y"])
            liste_lateralite.append(row["lateralite"])
            liste_effet_coup.append(row["effet_coup"])
            liste_num_coup.append(index+1)

    if len(liste_debut_frappe_adverse) != len(liste_debut_frappe_joueur):
        liste_debut_frappe_adverse.pop()
    
    df_intermediaire = pd.DataFrame({
        "debut": liste_debut_frappe_adverse,
        "fin": liste_debut_frappe_joueur,
        "num_coup": liste_num_coup,
        "coor_frappe_x": liste_coor_frappe_x,
        "coor_frappe_y": liste_coor_frappe_y,
        "lateralite": liste_lateralite,
        "effet_coup": liste_effet_coup
    })

    liste_valeurs_frappe = []
    liste_debut_frappe_adverse_frame = []
    liste_debut_frappe_joueur_frame = []
    liste_num_coup_frame = []
    liste_coor_frappe_x_frame = []
    liste_coor_frappe_y_frame = []
    liste_lateralite_frame = []
    liste_effet_coup_frame = []
    liste_num_frame = []
    liste_zone_modele_r = []
    liste_zone_modele_cd = []
    liste_num_point_frame = []

    for index, row in df_intermediaire.iterrows():
        for i in range(int(row["debut"]),int(row["fin"])+1):

            if os.path.isfile(os.path.join(dossier_point,"enveloppe",str(i)+".png")):
                chemin_image = os.path.join(dossier_point,"enveloppe",str(i)+".png")

                im = Image.open(chemin_image).convert('RGB')
                im_np = np.array(im)
                heatmap = im_np[:, :, 0].copy()
                heatmap_normalized_r = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
                heatmap = im_np[:, :, 1].copy()
                heatmap_normalized_g = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
                heatmap = im_np[:, :, 2].copy()
                heatmap_normalized_b = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255).astype(np.uint8)
                
                x = row['coor_frappe_x']#.values[0]
                y = row['coor_frappe_y']#.values[0]
                center = (int(x + (324 + 152 / 2)), int(y + (263 + 274 / 2)))
                val = '(' + str(heatmap_normalized_r[center[1], center[0]]) + ',' + str(heatmap_normalized_g[center[1], center[0]]) + ',' + str(heatmap_normalized_b[center[1], center[0]]) + ')'
                liste_valeurs_frappe.append(val)

                liste_debut_frappe_adverse_frame.append(int(row["debut"]))
                liste_debut_frappe_joueur_frame.append(int(row["fin"]))
                liste_num_coup_frame.append(int(row["num_coup"]))
                liste_num_frame.append(i)
                liste_coor_frappe_x_frame.append(row["coor_frappe_x"])
                liste_coor_frappe_y_frame.append(row["coor_frappe_y"])
                liste_zone_modele_r.append(int(heatmap_normalized_g[center[1], center[0]]))
                liste_zone_modele_cd.append(int(heatmap_normalized_b[center[1], center[0]]))
                liste_lateralite_frame.append(row["lateralite"])
                liste_effet_coup_frame.append(row["effet_coup"])
                liste_num_point_frame.append(num_point)
            else:
                break



    df_final = pd.DataFrame({
        "frame": liste_num_frame,
        "debut": liste_debut_frappe_adverse_frame,
        "fin": liste_debut_frappe_joueur_frame,
        "num_coup": liste_num_coup_frame,
        "num_point": liste_num_point_frame,
        "coor_frappe_x": liste_coor_frappe_x_frame,
        "coor_frappe_y": liste_coor_frappe_y_frame,
        "lateralite": liste_lateralite_frame,
        "effet_coup": liste_effet_coup_frame,
        "zone_modele": liste_valeurs_frappe,
        "zone_modele_r": liste_zone_modele_r,
        "zone_modele_cd": liste_zone_modele_cd
    })

    df_final.to_csv(os.path.join(dossier_point,os.path.split(dossier_point)[1]+"_zone_continue_"+nom_joueur+".csv"), index=False)  



def projeter_courbe(image_path, output_path, csv_path, num_frame, R, t, K):
    """
    Charge une image, projette une courbe 3D sur l'image et l'enregistre.

    Entrée:
            - Chemin de l'image d'entrée
            - Chemin de l'image de sortie avec la courbe dessinée
            - Chemin du csv contenant la trajectoire (_zone_joueur_avec_pos_balle_3D.csv)
            - Le numéro de la frame jusqu'où filtrer le csv
            - Matrice de rotation de la caméra
            - Matrice de translation de la caméra
            - Paramètres intrisecs de la caméra
    """

    dossier_point = os.path.split(os.path.split(csv_path)[0])[0]
    csv_annotation = os.path.join(dossier_point,os.path.split(dossier_point)[1]+"_annotation.csv")
    df_annotation = pd.read_csv(csv_annotation)

    df_filtered = df_annotation[df_annotation["time_frappe"] < num_frame]
    # Trouver la valeur avec le frame le plus proche
    valeur_num = 0
    valeur_num_prece = 0
    if not df_filtered.empty:
        closest_row = df_filtered.loc[df_filtered["time_frappe"].idxmax()]  # Prend la ligne avec le max de "frame"
        valeur_num = closest_row["time_frappe"] 

    
    """#Valeur frappe précédente
    df_filtered = df_annotation[df_annotation["time_frappe"] < num_frame].sort_values(by="time_frappe", ascending=False)
    if len(df_filtered) >= 2:
        deuxieme_plus_proche = df_filtered.iloc[1]  # Récupérer la deuxième ligne
        valeur_num_prece = deuxieme_plus_proche["time_frappe"]"""
    
    # 1. Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Erreur : Impossible de charger l'image.")
        return

    hauteur, largeur, _ = image.shape  # Récupérer la taille de l'image

    df = pd.read_csv(csv_path)  # Charger le fichier CSV
    df = df[(df['numero_pers'] == 4) & (df['frame'] <= num_frame+5)]
    #df['distance_x'] *= -1
    df['distance_y'] *= -1
    df['distance_z'] += 76
    points_3D = df[['distance_x', 'distance_y', 'distance_z']].values  # Extraire x, y, z

    # 3. Créer la matrice de projection P = K [R|t]
    Rt = np.hstack((R, t))  # Matrice 3x4 : [R|t]
    P = K @ Rt  # Matrice de projection 3x4

    # 4. Projeter les points 3D en 2D
    points_3D_homo = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))  # Ajouter 1 pour homogénéité
    points_2D_homo = P @ points_3D_homo.T  # Projection

    # Normaliser pour obtenir les coordonnées en pixels
    points_2D = (points_2D_homo[:2] / points_2D_homo[2]).T   


    color = (0,0,255)  
    thickness = 2

    # Paramètres pour le pointillé
    dash_length = 10  # Longueur de chaque segment
    gap_length = 10  # Espacement entre les segments

    # Calculer le vecteur direction
    """for i in range(valeur_num_prece,valeur_num-1):
        pt1 = tuple(points_2D[i].astype(int))
        pt2 = tuple(points_2D[i+1].astype(int))
        num_dashes = int(np.linalg.norm(np.array(pt2) - np.array(pt1)) // (dash_length + gap_length))
        if 0 <= pt1[0] < largeur and 0 <= pt1[1] < hauteur and 0 <= pt2[0] < largeur and 0 <= pt2[1] < hauteur:
            cv2.line(image, pt1, pt2, color  , 2)  
            for i in range(num_dashes):
                start = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * (i / num_dashes)),
                    int(pt1[1] + (pt2[1] - pt1[1]) * (i / num_dashes))
                )
                end = (
                    int(pt1[0] + (pt2[0] - pt1[0]) * ((i + 0.5) / num_dashes)),
                    int(pt1[1] + (pt2[1] - pt1[1]) * ((i + 0.5) / num_dashes))
                )
                cv2.line(image, start, end, color, thickness)"""


    # 6. Dessiner la courbe sur l’image
    """for i in range(valeur_num,len(points_2D) - 1-4):
        pt1 = tuple(points_2D[i].astype(int))
        pt2 = tuple(points_2D[i+1].astype(int))
        if 0 <= pt1[0] < largeur and 0 <= pt1[1] < hauteur and 0 <= pt2[0] < largeur and 0 <= pt2[1] < hauteur:
            cv2.line(image, pt1, pt2, (208,224,64)  , 2)  """

    
    """# 6. Dessiner la courbe sur l’image
    for i in range(len(points_2D) - 1-4,len(points_2D) - 1):
        pt1 = tuple(points_2D[i].astype(int))
        pt2 = tuple(points_2D[i+1].astype(int))
        if 0 <= pt1[0] < largeur and 0 <= pt1[1] < hauteur and 0 <= pt2[0] < largeur and 0 <= pt2[1] < hauteur:
            cv2.line(image, pt1, pt2, color  , 2)  """
    
    
    # 6. Dessiner la courbe sur l’image
    for i in range(valeur_num+1,len(points_2D) - 1-7):
        pt1 = tuple(points_2D[i].astype(int))
        pt2 = tuple(points_2D[i+1].astype(int))
        if 0 <= pt1[0] < largeur and 0 <= pt1[1] < hauteur and 0 <= pt2[0] < largeur and 0 <= pt2[1] < hauteur:
            cv2.line(image, pt1, pt2, color  , 2)

    # 7. Enregistrer l'image résultante
    cv2.imwrite(output_path, image)

def projeter_courbe_toutes_les_frames(chemin_point, R, t, K):
    """
        Fonction permettant d'appliquer projeter_courbe() sur toutes les frames d'un point et de créer la vidéo
        ATTENTION:
                    - Nécessite que le dossier images_modele_fusion soit créé
        Entrée:
                - Le dossier du point
                - Matrice de rotation de la caméra
                - Matrice de translation de la caméra
                - Paramètres intrisecs de la caméra
        Sortie:
                - Création du dossier des frames avec la trajectoire 
                - Création de la vidéo
    """


    if not os.path.isdir(os.path.join(chemin_point,"images_modele_fusion_traj")):
        os.mkdir(os.path.join(chemin_point,"images_modele_fusion_traj"))

    for i in range(len(os.listdir(os.path.join(chemin_point,"images_modele_fusion")))):
        image_path = os.path.join(chemin_point,"images_modele_fusion",str(i)+".jpg")
        output_path = os.path.join(chemin_point,"images_modele_fusion_traj",str(i)+".jpg")
        csv_path = os.path.join(chemin_point,"csv_json_openpose",os.path.split(chemin_point)[1]+"_zone_joueur_avec_pos_balle_3D.csv")
        projeter_courbe(image_path, output_path, csv_path, i, R, t, K)
    """
    for i in range(len(os.listdir(os.path.join(chemin_point,"images_video")))):
        image_path = os.path.join(chemin_point,"images_video",str(i)+".jpg")
        output_path = os.path.join(chemin_point,"images_modele_fusion_traj",str(i)+".jpg")
        csv_path = os.path.join(chemin_point,"csv_json_openpose",os.path.split(chemin_point)[1]+"_zone_joueur_avec_pos_balle_3D.csv")
        projeter_courbe(image_path, output_path, csv_path, i, R, t, K)"""


    chemin_video = os.path.join(chemin_point,os.path.split(chemin_point)[1]+".mp4")
    cap = cv2.VideoCapture(chemin_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    dossier_images = os.path.join(chemin_point,"images_modele_fusion_traj")
    chemin_sortie = os.path.join(chemin_point,os.path.split(chemin_point)[1]+"_video_avec_model_traj.mp4")
    creer_video_depuis_dossier_image(dossier_images, chemin_sortie, fps)


#mp4_to_gif("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/set_1_point_0_pose.mp4", 
#           "E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/set_1_point_0_pose.gif", 
#           fps=25)


def perspective_image(chemin_image1, pts_image2, table_proj_sol, output_path):
    """
        Fonction permettant de faire une perspective d'une image grace à des points connus
    """

    image1 = cv2.imread(chemin_image1)
    H, _ = cv2.findHomography(pts_image2, table_proj_sol, cv2.RANSAC)

    # 4️⃣ Appliquer la transformation sur `image2`
    image1_cv = cv2.cvtColor(np.array(image1), cv2.COLOR_RGBA2BGRA)  # Convertir en OpenCV
    h, w, _ = image1.shape
    image1_warped = cv2.warpPerspective(image1, H, (1280, 720))
    cv2.imwrite(output_path, image1_warped)



#cacul_zone_toutes_frame(os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/","set_1_point_5"),
#                            "ALEXIS-LEBRUN")



liste_points = os.listdir(os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips"))

#for point in liste_points:
#    cacul_zone_toutes_frame(os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/",point),
#                            "ALEXIS-LEBRUN")


#creer_video_to_image("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_2_point_20")



def supperposition2(chemin_image1,chemin_image2,chemin_enveloppe,pts_image1_table,pts_image1_table_proj,pts_image2,chemin_sauvegarde,faire_enveloppe):
    """
        Fonction qui fait supperposition entre deux image en plaçant en fonction de la position de la table
    """
    # 1️⃣ Charger les images
    image2 = cv2.imread(chemin_image2)  # Image à transformer
    image2_intermediaire = Image.open(chemin_image1).convert("RGBA")
    image1 = np.full_like(np.array(image2_intermediaire), [255, 255, 255, 255], dtype=np.uint8)# Image de base
    
    
    # Copier uniquement la région [263:537, 324:476] depuis l'image originale
    

    # Convertir en format PIL pour gérer la transparence
    image1_pil = Image.fromarray(image1)


    # 3️⃣ Calculer la matrice d’homographie
    H, _ = cv2.findHomography(pts_image2, pts_image1_table_proj, cv2.RANSAC)

    h, w, _ = image1.shape
    image2_warped = cv2.warpPerspective(image2, H, (w, h))

    """# Définir les 6 points du polygone
    points = np.array([[pts_image1_table[0][0], pts_image1_table[0][1]], [pts_image1_table[1][0], pts_image1_table[1][1]],  
                       [pts_image1_table[2][0], pts_image1_table[2][1]], [pts_image1_table[3][0], pts_image1_table[3][1]]], np.int32)
    # Créer un masque noir (même taille que l'image, mais en niveaux de gris)
    masque = np.zeros(image2_warped.shape[:2], dtype=np.uint8)
    # Dessiner un polygone blanc sur le masque
    cv2.fillPoly(masque, [points], 255)
    # Appliquer le masque : rendre les pixels blancs là où le masque est blanc
    image2_warped[masque == 255] = [255, 255, 255]
    
    points = np.array([[pts_image1_table_proj[0][0], pts_image1_table_proj[0][1]], [pts_image1_table_proj[1][0], pts_image1_table_proj[1][1]],  
                       [pts_image1_table_proj[2][0], pts_image1_table_proj[2][1]], [pts_image1_table_proj[3][0], pts_image1_table_proj[3][1]]], np.int32)
    # Créer un masque noir (même taille que l'image, mais en niveaux de gris)
    masque = np.zeros(image2_warped.shape[:2], dtype=np.uint8)
    # Dessiner un polygone blanc sur le masque
    cv2.fillPoly(masque, [points], 255)
    # Appliquer le masque : rendre les pixels blancs là où le masque est blanc
    image2_warped[masque == 255] = [255, 255, 255]    """    


    # Convertir en format PIL
    image2_warped_pil = Image.fromarray(cv2.cvtColor(image2_warped, cv2.COLOR_BGR2RGBA))

    

    # 5️⃣ Rendre le blanc transparent et appliquer la transparence
    image2_data = np.array(image2_warped_pil)
    r, g, b, a = image2_data[:, :, 0], image2_data[:, :, 1], image2_data[:, :, 2], image2_data[:, :, 3]
    mask = (r > 254) & (g > 254) & (b > 254)  # Détecte le blanc
    image2_data[mask] = [0, 0, 0, 0]  # Rendre transparent
    

    # 🔹 Appliquer une transparence de 50%
    image2_data[:, :, 3] = (image2_data[:, :, 3] * 1).astype(np.uint8)

    image2_transparent = Image.fromarray(image2_data)

    

    # 6️⃣ Superposition des images
    image1_pil.paste(image2_transparent, (0, 0), image2_transparent)






    # 7️⃣ Sauvegarder et afficher le résultat
    image1_pil.save(chemin_sauvegarde)
    #image1_pil.show()



le_point = "set_2_point_33"


dossier_point = "E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/"+le_point
dossier_image_modele = "E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/"+le_point+"/heatmap"
i = 75

table = np.array([[821.33, 363.55], [845.91, 459.84], [436.37, 456.65], [466.52, 362.25]], dtype=np.float32)  # Points sur image1
table_proj_sol = np.array([[807.0560241572812, 450.72322462289463], [850.8468941551321, 554.8325167532398], 
                           [454.57147485880085, 555.396836955502], [463.99038077237475, 448.45869007450403]], dtype=np.float32)
pts_image2 = np.array([[324, 263], [476, 263], [476, 537], [324, 537]], dtype=np.float32)  # Points sur image2
#supperposition2(os.path.join(dossier_point,"images_video",str(i)+".jpg"),
#                       os.path.join(dossier_image_modele,"image_"+str(i)+".png"),
#                       os.path.join(dossier_point,"enveloppe",str(i)+".png"),
#               table,table_proj_sol,pts_image2,
#               os.path.join(dossier_point,str(i)+"_proj.png"),False)









#creation_gif_zone_image_proj("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/"+le_point,
#                             "E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/"+le_point+"/heatmap",
#                             table,
#                             table_proj_sol,
#                             pts_image2)

#dezdez
#mp4_to_gif("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/set_1_point_0_video_avec_model.mp4", 
#           "E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/set_1_point_0_video_avec_model.gif", 
#           fps=25)

#edzdez


table = np.array([[821.33, 363.55], [845.91, 459.84], [436.37, 456.65], [466.52, 362.25]], dtype=np.float32)  # Points sur image1
table_proj_sol = np.array([[807.0560241572812, 450.72322462289463], [850.8468941551321, 554.8325167532398], 
                           [454.57147485880085, 555.396836955502], [463.99038077237475, 448.45869007450403]], dtype=np.float32)

pts_image2 = np.array([[324, 263], [476, 263], [476, 537], [324, 537]], dtype=np.float32) 
pts_image2 = np.array([[400-152/2, 400-274/2], [400+152/2, 400-274/2], [400+152/2, 400+274/2], [400-152/2, 400+274/2]], dtype=np.float32) 


#perspective_image("C:/Users/ReViVD/Downloads/frappe_rapport_table.jpg", pts_image2, table_proj_sol, "C:/Users/ReViVD/Downloads/frappe_rapport_joueur_pivote.jpg")
"""
# Paramètres de la caméra
focale_mm = 95.97
sensor_width_mm = 35
img_width_px = 1280
img_height_px = 720
cx, cy = img_width_px / 2, img_height_px / 2

# Points 2D correspondants dans l'image
points_2d = np.array([
    [547.24, 329.73],
    [739.09, 329.64],
    [747.41, 438.52],
    [536.63, 438.57],
    [741.98, 720-338.33],
    [543.03, 720-339.12]
], dtype=np.float32).reshape(-1, 1, 2)

image_size = (img_width_px, img_height_px)

# Calcul des paramètres de la caméra
R, t, K = calculer_camera_params(points_2d, image_size)
chemin_point = os.path.join("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2021_ChEuropeF_ClujNapoca/PRITHIKA-PAVADE_vs_SIBEL-ALTINKAYA/clips/set_1_point_12")
#creer_video_to_image(chemin_point)
projeter_courbe_toutes_les_frames(chemin_point, R, t, K)
"""


# Paramètres de la caméra
focale_mm = 38.45
sensor_width_mm = 35
img_width_px = 1280
img_height_px = 720
cx, cy = img_width_px / 2, img_height_px / 2

# Points 2D correspondants dans l'image
points_2d = np.array([
    [821.33, 363.55],
    [845.91, 459.84],
    [436.37, 456.65],
    [466.52, 362.25],
    [693.25, 720-263],
    [641.50, 720-357.50]
], dtype=np.float32).reshape(-1, 1, 2)

image_size = (img_width_px, img_height_px)

# Calcul des paramètres de la caméra
R, t, K = calculer_camera_params(points_2d, image_size)
chemin_point = os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/"+le_point)
#creer_video_to_image(chemin_point)
projeter_courbe_toutes_les_frames(chemin_point, R, t, K)

deze

"""image_path = os.path.join("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/ALEXIS-LEBRUN_vs_MA-LONG.jpg")
output_path = os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0/test_traj.jpg")
csv_path = os.path.join("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_1_point_0",
                        "csv_json_openpose/set_1_point_0_zone_joueur_avec_pos_balle_3D.csv")"""




liste_points = []

point_3D = np.array([-152/2, -274/2, 0])
point_2D = project_3d_to_2d(K, R, t, point_3D)
liste_points.append(point_2D)
point_3D = np.array([-152/2, 274/2, 0])
point_2D = project_3d_to_2d(K, R, t, point_3D)
liste_points.append(point_2D)
point_3D = np.array([152/2, 274/2, 0])
point_2D = project_3d_to_2d(K, R, t, point_3D)
liste_points.append(point_2D)
point_3D = np.array([152/2, -274/2, 0])
point_2D = project_3d_to_2d(K, R, t, point_3D)
liste_points.append(point_2D)
liste_points_triee = [liste_points[1],liste_points[2],liste_points[3],liste_points[0]]

print(liste_points_triee)


#liste_points_table_projetee = [(463.99038077237475, 448.45869007450403), (807.0560241572812, 450.72322462289463), 
#                               (850.8468941551321, 554.8325167532398), (454.57147485880085, 555.396836955502)]
#draw_points_on_image("set_2_point_31_image.jpg", liste_points_triee, "test_affichage_sol.jpg", point_color=(0, 255, 0), point_radius=5)
table = np.array([[821.33, 363.55], [845.91, 459.84], [436.37, 456.65], [466.52, 362.25]], dtype=np.float32)  # Points sur image1
table_proj_sol = np.array([[807.0560241572812, 450.72322462289463], [850.8468941551321, 554.8325167532398], 
                           [454.57147485880085, 555.396836955502], [463.99038077237475, 448.45869007450403]], dtype=np.float32)
pts_image2 = np.array([[324, 263], [476, 263], [476, 537], [324, 537]], dtype=np.float32)  # Points sur image2

#supperposition("C:/Users/ReViVD/Documents/GitHub/tt-espace/set_2_point_31_image.jpg","C:/Users/ReViVD/Documents/GitHub/tt-espace/image_21.png",
#               table,table_proj_sol,pts_image2,
#               "C:/Users/ReViVD/Documents/GitHub/tt-espace/image_superposee_proj_sol.png")
# Exemple d'utilisation

fusionner_images("image_superposee_proj_sol.png", "test2.jpg", "set_2_point_31_image.jpg", "image_fusionnee.jpg", seuil_noir=20)




#compute_average_frame("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_2_point_31/set_2_point_31.mp4", 
#                      save_path="C:/Users/ReViVD/Documents/GitHub/tt-espace/test.jpg")

#compute_median_frame("E:/sauvegarde/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_2_point_31/set_2_point_31.mp4", 
#                      save_path="C:/Users/ReViVD/Documents/GitHub/tt-espace/test.jpg")

#subtract_images("set_2_point_31_image.jpg","test.jpg",save_path="C:/Users/ReViVD/Documents/GitHub/tt-espace/test2.jpg")


