import pygame
import copy
from prog_settings import *
from classes import *

# Запуски
pygame.init()

pygame.font.init()
font = pygame.font.Font(None, 30)

file = open('save_score.txt', 'w')
text = 'Gen:' + '\t' + 'Score:\n'
file.write(text)
file.close()

for i in bacteria_list:
    i.neural_net.randomise(-2, 2)


# Настройка
screen = pygame.display.set_mode([game_pix_x, game_pix_y])
clock = pygame.time.Clock()
display = pygame.display

display.set_caption("Evolution")
# pygame.display.set_icon()


# Цикл игры
running = True
while running:
    clock.tick(fps)
    # Ввод процесса (события)
    for event in pygame.event.get():
        # проверить закрытие окна
        if event.type == pygame.QUIT:
            running = False

    # Обновление
    for i in bacteria_list:     # запускаем крипов
        i.live()
    if all(map(lambda x: x.alive is False, bacteria_list)):
        bacteria_list = genetic_algorithm(bacteria_list)

    if len(food_list) < food_count:    # проверяем есть ли еда
        if randint(0, 100) < chance_of_food_spawn:
            food_list.append(Food())

    # Визуализация (сборка)
    screen.fill(dark_blue)

    for i in food_list:
        screen.blit(i.sprite, [i.get_position[0] - i.sprite.get_width() / 2, i.get_position[1] - i.sprite.get_height() / 2])

    for i in bacteria_list:
        if i.alive is True:
            r_sprite = pygame.transform.rotate(i.sprite, i.rotation * 57.3 - 180)
            screen.blit(r_sprite, [i.get_position[0] - r_sprite.get_width() / 2, i.get_position[1] - r_sprite.get_height() / 2])

    count = font.render(str(len(list(filter(lambda x: x.alive is True, bacteria_list)))), True, green)
    screen.blit(count, (game_pix_x - 30, game_pix_y - 30))

    pygame.display.flip()

pygame.quit()
