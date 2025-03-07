import numpy as np
import cv2
import os

# pour commencer sur une image blanche et vierge :

width, height = 800, 800
image=np.ones((height, width,3), np.uint8)*255


#demi terrain : 8m x 4m ; demi-table : 1,525m x 1,37m. on prend: 1px=1cm
# => zone totale : 8m x 8m

def quadrillage_sur_image(image, chemin_enregistrement, nom_enregistrement):
    """
        Fonction permettant de générer la table et les zones de l'espace de jeu
    """
    if not isinstance(image, np.ndarray) :
        image=cv2.imread(image)

    width=800
    height=800

    cv2.line(image, (324,537), (476,537), (100,100,100), 2) #bord inférieur
    cv2.line(image,(476,263), (476,537),(100,100,100),2) #bord droit
    cv2.line(image,(324,263),(324,537),(100,100,100),2) #bord gauche
    cv2.line(image,(324,263),(476,263),(100,100,100),2) #bord supérieur
    cv2.line(image,(324,400), (476,400),(100,100,100),2) #filet

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.4
    fontColor = (200, 200, 200)

    i=-15
    while i<15: # dessin des lignes verticales
        x=int(width/2 + (25.4+i*50.8))
        cv2.line(image, (x, 0), (x, height), (200,200,200), 1)
        i+=1

    j=-15
    while j<15: #dessin des lignes horizontales
        y=int(height/2 + j*45.7)
        cv2.line(image, (0,y),(width,y), (200,200,200),1)
        j+=1

    for i in range(1,9):
        x=int(width/2 + (25.4+(-1)*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "M{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "M{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+0*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "D{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "G{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+1*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "E{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "H{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+2*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "F{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "I{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+(-2)*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "G{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "D{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+(-3)*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "H{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "E{}".format(8+1-i), position, font, fontScale, fontColor)
    for i in range(1,9):
        x=int(width/2 + (25.4+(-4)*50.8))
        position = (x+13, int(height/2 + i*45.7)-15)
        cv2.putText(image, "I{}".format(i), position, font, fontScale, fontColor)
        position = (x+13, int(i*45.7)+15)
        cv2.putText(image, "F{}".format(8+1-i), position, font, fontScale, fontColor)

    cv2.imwrite(os.path.join(chemin_enregistrement,'{}.png'.format(nom_enregistrement)), image)



quadrillage_sur_image(image,'C:/Users/ReViVD/Downloads', 'environnement_table')