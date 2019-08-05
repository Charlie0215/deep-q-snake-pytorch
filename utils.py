import pygame
import numpy as np
from PIL import Image, ImageOps


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

def button(playSurface, msg, x, y, w, h, inactive, active, action=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if (x+w > mouse[0] > x) and (y+h > mouse[1] > y):
        pygame.draw.rect(playSurface, active, (x, y, w, h))
        if click[0] == 1 and action != None:
            action()
    else:
        pygame.draw.rect(playSurface, inactive, (x, y, w, h))
    
    smallText = pygame.font.Font('arial.ttf', 20)
    textSurf, textRect = text_object(msg, smallText, blackColour)
    textRect.center = (x+w//2, y+h//2)
    playSurface.blit(textSurf, textRect)

def text_object(text, font, color):
	textSurface = font.render(text, True, color)
	return textSurface, textSurface.get_rect()

def message_display(display, text, display_width, display_height):
	largeText = pygame.font.Font('arial.ttf', 115)
	TextSurf, TextRect = text_object(text, largeText)
	TextRect.center = (display_width//2, display_height//2)
	display.blit(TextSurf, TextRect)
	pygame.display.flip()

def screenshot(playSurface, size, invert=False): #TODO - cv2 version
	data = pygame.image.tostring(playSurface, 'RGB')
	image = Image.frombytes('RGB', size, data)
	image = image.convert('L')
	image = image.resize(size)
	image = ImageOps.invert(image) if invert else image
	image = image.convert('1')
	matrix = np.asarray(image.getdata(), dtype=np.float64)
	return matrix.reshape(image.size[0], image.size[1])



