import pygame
import sys
import time
import random
from pygame.locals import *
from utils import text_object, message_display, button
import threading
#from Agent import train
from snake import run
from Agent import train, play

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

class intro:
    def __init__(self, playSurface, display_width, display_height):
        self.playSurface = playSurface
        self.display_width = display_width
        self.display_height = display_height

    def game_intro(self):
        intro = True
        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.playSurface.fill(blackColour)
            textFont = pygame.font.Font('arial.ttf', 73)
            textSurf, textRect = text_object('AI Snake', textFont, greyColour)
            textRect.center = (self.display_width//2, self.display_height//2)
            self.playSurface.blit(textSurf, textRect)
            button(self.playSurface,'GO!', self.display_width//4, self.display_height//8*7, self.display_width//4, self.display_height//8, greenColour, brightGreenColour, run)
            button(self.playSurface, 'AI', display_width//2, self.display_height//8*7, display_width//4, self.display_height//8, redColour, brightRedColour, self.AI_option)
            pygame.display.update()

    def AI_option(self):
        intro = True
        while intro:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.playSurface.fill(blackColour)
            textFont = pygame.font.Font('arial.ttf', 73)
            textSurf, textRect = text_object('AI Snake', textFont, greyColour)
            textRect.center = (self.display_width//2, self.display_height//2)
            self.playSurface.blit(textSurf, textRect)

            button(self.playSurface,'train', self.display_width//8, self.display_height//8*7, self.display_width//8, self.display_height//8, greenColour, brightGreenColour, train)
            button(self.playSurface,'play', self.display_width//4*3, self.display_height//8*7, self.display_width//8, self.display_height//8, redColour, brightRedColour, play)
           

            pygame.display.update()

if __name__ == '__main__':
    display_width = 640
    display_height = 480
    playSurface_intro = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Deep Q Snake!')
    # Load image and set icon
    image = pygame.image.load('joystick.ico')
    pygame.display.set_icon(image)
    pygame.init()
    intro = intro(playSurface_intro, display_width, display_height)
    intro.game_intro()
