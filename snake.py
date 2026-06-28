import pygame
import random
import sys

# Initialisation
pygame.init()

# Constantes
LARGEUR = 800
HAUTEUR = 600
TAILLE_CASE = 20
FPS = 10

# Couleurs
NOIR = (0, 0, 0)
BLANC = (255, 255, 255)
VERT = (0, 200, 0)
ROUGE = (200, 0, 0)
GRIS = (40, 40, 40)

# Directions
HAUT = (0, -1)
BAS = (0, 1)
GAUCHE = (-1, 0)
DROITE = (1, 0)

class Snake:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.corps = [(LARGEUR // 4, HAUTEUR // 2)]
        self.direction = DROITE
        self.croissance = False
        self.score = 0
    
    def deplacer(self):
        tete_x, tete_y = self.corps[0]
        dx, dy = self.direction
        nouvelle_tete = (tete_x + dx * TAILLE_CASE, tete_y + dy * TAILLE_CASE)
        
        self.corps.insert(0, nouvelle_tete)
        
        if not self.croissance:
            self.corps.pop()
        else:
            self.croissance = False
    
    def changer_direction(self, nouvelle_direction):
        # Empêcher le demi-tour
        if (nouvelle_direction[0] * -1, nouvelle_direction[1] * -1) != self.direction:
            self.direction = nouvelle_direction
    
    def verifier_collision(self):
        tete = self.corps[0]
        
        # Collision avec les murs
        if (tete[0] < 0 or tete[0] >= LARGEUR or
            tete[1] < 0 or tete[1] >= HAUTEUR):
            return True
        
        # Collision avec soi-même
        if tete in self.corps[1:]:
            return True
        
        return False
    
    def manger(self):
        self.croissance = True
        self.score += 10
    
    def dessiner(self, surface):
        for i, segment in enumerate(self.corps):
            # La tête est plus claire
            if i == 0:
                couleur = (0, 255, 0)
            else:
                couleur = VERT
            
            rect = pygame.Rect(segment[0], segment[1], TAILLE_CASE - 1, TAILLE_CASE - 1)
            pygame.draw.rect(surface, couleur, rect)


class Nourriture:
    def __init__(self):
        self.position = (0, 0)
        self.generer()
    
    def generer(self):
        x = random.randint(0, (LARGEUR - TAILLE_CASE) // TAILLE_CASE) * TAILLE_CASE
        y = random.randint(0, (HAUTEUR - TAILLE_CASE) // TAILLE_CASE) * TAILLE_CASE
        self.position = (x, y)
    
    def dessiner(self, surface):
        rect = pygame.Rect(self.position[0], self.position[1], TAILLE_CASE - 1, TAILLE_CASE - 1)
        pygame.draw.rect(surface, ROUGE, rect)


def main():
    surface = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("Snake Game")
    horloge = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    
    snake = Snake()
    nourriture = Nourriture()
    
    en_cours = True
    en_attente_demarrage = True
    
    while en_cours:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                en_cours = False
            
            if event.type == pygame.KEYDOWN:
                if en_attente_demarrage:
                    en_attente_demarrage = False
                    continue
                
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    snake.changer_direction(HAUT)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    snake.changer_direction(BAS)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    snake.changer_direction(GAUCHE)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    snake.changer_direction(DROITE)
                elif event.key == pygame.K_r:
                    snake.reset()
                    nourriture.generer()
                elif event.key == pygame.K_ESCAPE:
                    en_cours = False
        
        if not en_attente_demarrage:
            snake.deplacer()
            
            # Vérifier collision avec la nourriture
            if snake.corps[0] == nourriture.position:
                snake.manger()
                nourriture.generer()
            
            # Vérifier collision
            if snake.verifier_collision():
                # Afficher écran de game over
                surface.fill(NOIR)
                game_over_text = font.render("GAME OVER! Score: " + str(snake.score), True, BLANC)
                restart_text = font.render("R pour recommencer | ESC pour quitter", True, GRIS)
                surface.blit(game_over_text, (LARGEUR // 2 - 150, HAUTEUR // 2))
                surface.blit(restart_text, (LARGEUR // 2 - 180, HAUTEUR // 2 + 40))
                pygame.display.flip()
                
                # Attendre input
                waiting = True
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            waiting = False
                            en_cours = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                snake.reset()
                                nourriture.generer()
                                waiting = False
                            elif event.key == pygame.K_ESCAPE:
                                waiting = False
                                en_cours = False
                continue
        
        # Dessin
        surface.fill(NOIR)
        
        if en_attente_demarrage:
            title = font.render("SNAKE GAME", True, VERT)
            start = font.render("Appuyez sur une touche pour commencer", True, BLANC)
            controls = font.render("ZQSD / Flèches pour bouger | R pour recommencer", True, GRIS)
            surface.blit(title, (LARGEUR // 2 - 80, HAUTEUR // 2 - 60))
            surface.blit(start, (LARGEUR // 2 - 160, HAUTEUR // 2))
            surface.blit(controls, (LARGEUR // 2 - 200, HAUTEUR // 2 + 40))
        else:
            snake.dessiner(surface)
            nourriture.dessiner(surface)
            
            # Afficher le score
            score_text = font.render("Score: " + str(snake.score), True, BLANC)
            surface.blit(score_text, (10, 10))
            
            # Lignes de grille
            for x in range(0, LARGEUR, TAILLE_CASE):
                pygame.draw.line(surface, GRIS, (x, 0), (x, HAUTEUR))
            for y in range(0, HAUTEUR, TAILLE_CASE):
                pygame.draw.line(surface, GRIS, (0, y), (LARGEUR, y))
        
        pygame.display.flip()
        horloge.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
