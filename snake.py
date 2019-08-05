import pygame
import sys
import time
import random
from pygame.locals import *
from utils import text_object, message_display, button
import threading

######################## Initialize colours ########################
# RED
redColour = pygame.Color(200,0,0)
brightRedColour = pygame.Color(255,0,0)
# GREEN
brightGreenColour = pygame.Color(0,255,0)
greenColour = pygame.Color(0,200,0)
brightGreenColour1 = (150, 255, 150)
darkGreenColour1 = (0, 155, 0)
# BLACK
blackColour = pygame.Color(0,0,0)
# WHITE
whiteColour = pygame.Color(255,255,255)
# GRAY
greyColour = pygame.Color(150,150,150)
LightGrey = pygame.Color(220,220,220)
####################################################################


class Manual:
    def __init__(self, display_width = 640, display_height = 480):
        self.fpsClock = pygame.time.Clock()

        self.display_width = display_width
        self.display_height = display_height
        self.playSurface = pygame.display.set_mode((self.display_width, self.display_height))
        self.init_game()

    def init_game(self):

        # Initialize initial position and object size
        self.snakePosition = [random.randint(4,5)*20, random.randint(4,5)*20] # Snake head
        self.snakeSegments = [[self.snakePosition[0], self.snakePosition[1]], 
                        [self.snakePosition[0]-20, self.snakePosition[1]], 
                        [self.snakePosition[0]-40, self.snakePosition[1]]]

        self.raspberryPosition = [random.randint(0, (self.display_width-20)//20)*20, random.randint(0, (self.display_height-20)//20)*20]
        self.raspberrySpawned = 1
        self.direction = 'right'
        self.changeDirection = self.direction
        self.score = 0
        self.done = False

    def gameOver(self, score):
        # Set fonts of caption
        gameOverFont = pygame.font.Font('arial.ttf', 72)
        gameOverSurf, gameOverRect = text_object('Game Over', gameOverFont, greyColour)
        gameOverRect.midtop = (320, 125)
        self.playSurface.blit(gameOverSurf, gameOverRect)
        # Display scores and set fonts
        scoreFont = pygame.font.Font('arial.ttf', 48)
        scoreSurf, scoreRect = text_object('SCORE:'+str(score), scoreFont, greyColour)
        scoreRect = scoreSurf.get_rect()
        scoreRect.midtop = (320, 225)
        self.playSurface.blit(scoreSurf, scoreRect)
        #pygame.display.update() # Refresh display

        button(self.playSurface, 'Again', self.display_width//4, self.display_height//8*7, self.display_width//2, self.display_height//8, greenColour, brightGreenColour, self.init_game)
        # https://stackoverflow.com/questions/55881619/sleep-doesnt-work-where-it-is-desired-to/55882173#55882173
        pygame.display.update()
        

    # Snake and raspberry
    def manual_play(self):

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN:
                if event.key == K_RIGHT or event.key == ord('d'):
                    self.changeDirection = 'right'
                    if not self.direction == 'left':
                        self.direction = self.changeDirection

                if event.key == K_LEFT or event.key == ord('a'):
                    self.changeDirection = 'left'
                    if not self.direction == 'right':
                        self.direction = self.changeDirection

                if event.key == K_UP or event.key == ord('w'):
                    self.changeDirection = 'up'
                    if not self.direction == 'down':
                        self.direction = self.changeDirection

                if event.key == K_DOWN or event.key == ord('s'):
                    self.changeDirection = 'down'
                    if not self.direction == 'up':
                        self.direction = self.changeDirection

                if event.key == K_ESCAPE:
                    #print('right')
                    pygame.event.post(pygame.event.Event(QUIT))

              
        if self.direction == 'right':
            self.snakePosition[0] += 20 
        if self.direction == 'left':
            self.snakePosition[0] -= 20 
        if self.direction == 'up':
            self.snakePosition[1] -= 20
        if self.direction == 'down':
            self.snakePosition[1] += 20

        # append snake head
        self.snakeSegments.insert(0, list(self.snakePosition))

        if self.snakePosition[0] == self.raspberryPosition[0] and self.snakePosition[1] == self.raspberryPosition[1]:
            self.raspberrySpawned = 0 
        else:
            self.snakeSegments.pop()

        if self.raspberrySpawned == 0:
            x = random.randrange(1, 32)
            y = random.randrange(1, 24)
            self.raspberryPosition = [int(x*20), int(y*20)]
            self.raspberrySpawned = 1
            self.score += 1
        
        # refresh frame
        self.playSurface.fill(blackColour)
        for position in self.snakeSegments[1:]:
            pygame.draw.rect(self.playSurface, whiteColour, Rect(position[0], position[1], 20, 20))
        pygame.draw.rect(self.playSurface, LightGrey, Rect(self.snakePosition[0], self.snakePosition[1], 20, 20))
        pygame.draw.rect(self.playSurface, redColour, Rect(self.raspberryPosition[0], self.raspberryPosition[1], 20, 20))

        pygame.display.flip()           
        
        if self.snakePosition[0]>self.display_width-20 or self.snakePosition[0]<0:
            self.done = True
            return self.score, self.done

        if self.snakePosition[1]>self.display_height-20 or self.snakePosition[1]<0:
            self.done = True
            return self.score, self.done

        for snakeBody in self.snakeSegments[1:]:
            if self.snakePosition[0] == snakeBody[0] and self.snakePosition[1] == snakeBody[1]:
                self.done = True
                return self.score, self.done

        if len(self.snakeSegments)<40:
            speed = 6 + len(self.snakeSegments)//4
        else:
            speed = 16
        self.fpsClock.tick(speed)
        #print('ok')
        return self.score, self.done

def run():
    pygame.display.set_caption('Manual!')
    # Load image and set icon
    image = pygame.image.load('joystick.ico')
    pygame.display.set_icon(image)

    pygame.init()
    game = Manual()
    while True:
        score, done = game.manual_play()
        if done:
            game.gameOver(score)

if __name__ == '__main__':
    pygame.display.set_caption('Deep Q Snake!')
    # Load image and set icon
    image = pygame.image.load('joystick.ico')
    pygame.display.set_icon(image)

    pygame.init()
    game = Manual()
    while True:
        score, done = game.manual_play()
        if done:
            game.gameOver(score)
