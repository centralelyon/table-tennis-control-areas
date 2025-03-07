from PIL import Image
import numpy as np

def remplacer_vert_par_rouge(image_pil):
    """
    Transforme tous les pixels verts en pixels rouges dans une image PIL.
    """
    # Convertir l'image en tableau NumPy
    image_np = np.array(image_pil)

    # Définition du seuil pour détecter le vert (RVB)
    seuil_vert = (image_np[:, :, 0] < 100) & (image_np[:, :, 1] > 150) & (image_np[:, :, 2] < 100)

    # Remplacer les pixels verts par du rouge (255, 0, 0)
    image_np[seuil_vert] = [255, 0, 0, 255]  # Avec canal alpha (RGBA)

    # Retourner l'image modifiée sous forme PIL
    return Image.fromarray(image_np)

# Charger une image
image = Image.open("C:/Users/ReViVD/Desktop/dataroom/pipeline-tt/2024_WttSmash_Singapour/ALEXIS-LEBRUN_vs_MA-LONG/clips/set_2_point_33/images_modele_fusion_traj/62.jpg").convert("RGBA")
#image = Image.open("C:/Users/ReViVD/Downloads/im.jpg").convert("RGBA")

# Appliquer la transformation
image_modifiee = remplacer_vert_par_rouge(image)

# Afficher l'image modifiée
image_modifiee.show()

# Sauvegarder l'image si besoin
# image_modifiee.save("image_modifiee.png")
