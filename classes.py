import pygame
import copy
from prog_settings import *
from random import randint
from os import path
from math import sin, cos


class Game_Object:
    def __init__(self):
        self.game_time = 0
        self.sprite = None
        self.neural_net = None
        self.rotation = 0

        self.alive = True
        self.energy = 0
        self.max_energy = 0
        self.hunter_rate = 0
        self.food_eat = 0
        self.new_gen_created = 0

        self.visible_radius = 0
        self.eat_radius = 0
        self.move_speed = 0
        self.score = 0

        self.pos_x = randint(0, game_pix_x)
        self.pos_y = randint(0, game_pix_y)

    @property
    def get_position(self):
        return self.pos_x, self.pos_y

    def find_closest_by_range(self, obj_list: list):
        new_obj_list = [[] for _ in range(len(obj_list))]
        for i in range(0, len(obj_list)):
            new_obj_list[i].append(i)
            new_obj_list[i].append(obj_list[i])
            new_obj_list[i].append(((self.pos_x - obj_list[i].pos_x) ** 2 + (self.pos_y - obj_list[i].pos_y) ** 2) ** 0.5)
        new_obj_list.sort(key=lambda x: x[2])
        return new_obj_list

    def check_distance(self, list_of_obj: list):
        visible_objects = list(filter(lambda x: x[2] <= self.visible_radius, list_of_obj))
        return visible_objects

    def move(self, vec):
        speed = vec[0] * self.move_speed
        rad = vec[1] * 360 * 0.017
        if vec[2] > 0.5:
            if self.energy > new_gen_cost:
                self.create_new()

        self.rotation = rad
        self.pos_x = self.pos_x + speed * sin(self.rotation)
        self.pos_y = self.pos_y + speed * cos(self.rotation)

        # Голод
        pos_border_mod = 0
        if self.pos_x < border_range or self.pos_x > game_pix_x - border_range or self.pos_y < border_range or self.pos_y > game_pix_y - border_range:
            pos_border_mod = border_hit
        self.energy = self.hunter_rate * speed + b_base_hunter_val + pos_border_mod

        # Смерть и счет
        if self.energy <= 0:
            self.alive = False
            self.score = mod_score_for_food_eated * self.food_eat + mod_score_for_new_gen * self.new_gen_created

        if self.pos_x < 0:
            self.pos_x = 0
        elif self.pos_x > game_pix_x:
            self.pos_x = game_pix_x
        if self.pos_y < 0:
            self.pos_y = 0
        elif self.pos_y > game_pix_y:
            self.pos_y = game_pix_y

    def control(self):
        food = self.check_distance(self.find_closest_by_range(food_list))
        bacter = self.check_distance(self.find_closest_by_range(bacteria_list))
        if len(food) == 0:
            self.move(self.neural_net.feedforward(
                [self.pos_x / game_pix_x,
                 self.pos_y / game_pix_y,
                 self.rotation * 57.3 / 360,
                 self.energy / self.max_energy,
                 self.pos_y / game_pix_y,
                 self.pos_y / game_pix_y,
                 0,
                 len(food) / food_count,
                 len(bacter) / len(bacteria_list)]))
        else:
            if food[0][2] <= self.eat_radius:
                self.energy = self.clamp(self.energy + food[0][1].energy, 0, self.max_energy)
                self.food_eat += 1
                food_list.remove(food[0][1])
            self.move(self.neural_net.feedforward([self.pos_x / game_pix_x,
                                                   self.pos_y / game_pix_y,
                                                   self.rotation * 57.3 / 360,
                                                   self.energy / self.max_energy,
                                                   int(food[0][1].pos_x / game_pix_x),
                                                   int(food[0][1].pos_y / game_pix_y),
                                                   int(food[0][2]) / (game_pix_x ** 2 + game_pix_y ** 2) ** 0.5,
                                                   len(food) / food_count,
                                                   len(bacter) / len(bacteria_list)]))

    def live(self):
        if self.alive is True:
            self.game_time += 1
            self.control()

        if self.game_time > max_time_set:
            self.alive = False

    def create_new(self,):
        if len(bacteria_list) < gen_max_count:
            new_net = Bacteria()
            new_net.pos_x = self.pos_x + randint(-50, 50)
            new_net.pos_y = self.pos_y + randint(-50, 50)

            new_net.neural_net = copy.copy(self.neural_net)
            new_net.neural_net.mutation()
            bacteria_list.append(new_net)
            self.new_gen_created += 1

    @staticmethod
    def clamp(value, Min, Max):
        return max(min(value, Max), Min)


class Food(Game_Object):
    def __init__(self):
        super(Food, self).__init__()
        self.sprite = pygame.image.load(path.join('images', 'food.png'))
        self.energy = small_food_energy
        self.visible_radius = 100

    def randomise(self):
        self.pos_x = game_pix_x / food_count
        self.pos_y = game_pix_y / 2


class Bacteria(Game_Object):
    counter = 0

    def __init__(self):
        super(Bacteria, self).__init__()
        self.pos_x = game_pix_x / bacteria_count * self.counter
        self.pos_y = game_pix_y / 2
        self.counter_m()

        self.energy = b_start_energy
        self.max_energy = b_max_energy
        self.visible_radius = b_visible_radius
        self.eat_radius = 30
        self.hunter_rate = b_hunter_rate
        self.move_speed = b_move_speed

        self.neural_net = Neural_Network(10, 1, [10], 3, [1, 40, 10])

        self.sprite = pygame.image.load(path.join('images', 'bacteria.png'))

    @classmethod
    def counter_m(cls):
        cls.counter += 1
        if cls.counter > bacteria_count:
            cls.counter = 0


class Neuron:
    def __init__(self, weight: list = None, bias: int = 1, mutation_chance: int = 0, mutation_step: int = 0,
                 neuron=None):
        if neuron is None:
            if weight is None:
                weight = [i for i in range(2)]
            self.weight = copy.copy(weight)
            self.bias = copy.copy(bias)
            self.mutation_chance = copy.copy(mutation_chance)
            self.mutation_step = copy.copy(mutation_step)
        else:
            self.weight = copy.copy(neuron.weight)
            self.bias = copy.copy(neuron.bias)
            self.mutation_chance = copy.copy(neuron.mutation_chance)
            self.mutation_step = copy.copy(neuron.mutation_step)

    @staticmethod
    def sigmoid(val):
        return 1 / (1 + (2.718281828459045 ** - val))

    def feedforward(self, inputs: list):
        base_val = 0
        for i in range(len(inputs)):
            base_val += inputs[i] * self.weight[i]
        return self.sigmoid(base_val + self.bias)

    @property
    def tuning(self):
        return self.weight, self.bias, self.mutation_chance, self.mutation_step

    @property
    def console_log(self):
        return f'{self}\nWeight: {self.weight}\nBias: {self.bias}\nMutation chance: {self.mutation_chance} % \n' \
               f'Mutation step: {self.mutation_step}\n'

    def mutation(self):
        for i in range(len(self.weight)):
            if self.random_int(0, 100) <= self.mutation_chance:
                self.weight[i] += self.random_int(-self.mutation_step, self.mutation_step) * self.weight[i] / 100

    @staticmethod
    def random_int(a: int = 0, b: int = 1):
        val = randint(a, b)
        return val


# ? Нейроная сеть
class Neural_Network:
    def __init__(self,
                 number_of_inputs: int = 2,
                 hidden_layers_count: int = 0,
                 hidden_layer_sizes: list = None,
                 output_layer_size: int = 1,
                 tuning_for_Neuron: list = None,
                 neural_network=None):
        if neural_network is None:
            if tuning_for_Neuron is None:
                tuning_for_Neuron = [1, 0, 0]

            input_weights = [1 for _ in range(number_of_inputs + 1)]

            self.layers = []
            for i in range(hidden_layers_count):
                if i == 0:
                    layer = []
                    for j in range(hidden_layer_sizes[i]):
                        layer.append(Neuron(input_weights, tuning_for_Neuron[0], tuning_for_Neuron[1],
                                            tuning_for_Neuron[2]))
                    self.layers.append(layer)
                if i > 0:
                    layer = []
                    for j in range(hidden_layer_sizes[i]):
                        layer.append(Neuron([1 for i in range(len(self.layers[i - 1]) + 1)], tuning_for_Neuron[0],
                                            tuning_for_Neuron[1],
                                            tuning_for_Neuron[2]))
                    self.layers.append(layer)
            if hidden_layers_count == 0:
                layer = []
                for j in range(output_layer_size):
                    layer.append(
                        Neuron(input_weights, tuning_for_Neuron[0], tuning_for_Neuron[1],
                               tuning_for_Neuron[2]))
                self.layers.append(layer)
            else:
                layer = []
                for j in range(output_layer_size):
                    layer.append(
                        Neuron([1 for _ in range(len(self.layers[len(self.layers) - 1]) + 1)], tuning_for_Neuron[0],
                               tuning_for_Neuron[1],
                               tuning_for_Neuron[2]))
                self.layers.append(layer)
            self.inputs_cont = number_of_inputs
            self.hidden_layers_count = hidden_layers_count
            self.output_count = output_layer_size
            self.neuron_base_bias = tuning_for_Neuron[0]
            self.neuron_base_mut_c = tuning_for_Neuron[1]
            self.neuron_base_mut_s = tuning_for_Neuron[2]
        else:
            self.layers = []
            for i in neural_network.layers:
                s_layer = []
                for j in i:
                    s_layer.append(Neuron(neuron=j))
                self.layers.append(s_layer)
            self.inputs_cont = neural_network.inputs_cont
            self.hidden_layers_count = neural_network.hidden_layers_count
            self.output_count = neural_network.output_count
            self.neuron_base_bias = neural_network.neuron_base_bias
            self.neuron_base_mut_c = neural_network.neuron_base_mut_c
            self.neuron_base_mut_s = neural_network.neuron_base_mut_s

    @property
    def console_log(self):
        log = f'\t____Neural Network____\n{self}\nInputs len: {self.inputs_cont}\nHidden layers: {self.hidden_layers_count}\nOutput len: {self.output_count}\n\n'
        if self.hidden_layers_count != 0:
            for i in range(self.hidden_layers_count):
                log += f'\tHidden layer {i}:\n'
                for j in self.layers[i]:
                    log += j.console_log + '\n'
        log += '\tOutput layer:\n'
        for i in self.layers[len(self.layers) - 1]:
            log += i.console_log + '\n'
        return log

    def feedforward(self, inputs):
        FAQ = []
        for i in range(len(self.layers)):
            if i == 0:
                inp_ans = []
                for j in self.layers[i]:
                    inp_ans.append(j.feedforward(inputs))
                FAQ = inp_ans
            else:
                ans = []
                for j in self.layers[i]:
                    ans.append(j.feedforward(FAQ))
                FAQ = ans
        return FAQ

    def mutation(self):
        for layer in range(len(self.layers)):
            for n in range(len(self.layers[layer])):
                self.layers[layer][n].mutation()

    def save(self):
        file = open('Network.ini', 'w')
        try:
            save = str(self.inputs_cont) + '\n'
            save += str(self.hidden_layers_count) + '\n'
            save += str(self.output_count) + '\n'
            save += str(self.neuron_base_bias) + '\n'
            save += str(self.neuron_base_mut_c) + '\n'
            save += str(self.neuron_base_mut_s) + '\n'
            for i in self.layers:
                s_line = ''
                for j in i:
                    s_line += str(j.weight) + ' ' + str(j.bias)
                s_line += '\n'
                save += s_line
            file.write(save)
        finally:
            file.close()

    def load(self):
        file = open('Network.ini', 'r')
        try:
            text = file.readlines()
            for i in range(len(text)):
                text[i] = text[i].replace('\n', '')
            self.inputs_cont = int(text[0])
            self.hidden_layers_count = int(text[1])
            self.output_count = int(text[2])
            self.neuron_base_bias = int(text[3])
            self.neuron_base_mut_c = int(text[4])
            self.neuron_base_mut_s = int(text[5])
            self.layers = []
            for i in range(6, len(text)):
                layer = []
                text[i] = text[i].replace(',', '')
                text[i] = text[i].split('[')
                for j in range(len(text[i])):
                    text[i][j] = text[i][j].replace(']', '')
                    text[i][j] = text[i][j].split(' ')

                    if j != 0:
                        weights = []
                        bias_l = 0
                        bias_l = float(text[i][j][len(text[i][j]) - 1])
                        for n in range(len(text[i][j]) - 1):
                            weights.append(float(text[i][j][n]))
                        layer.append(Neuron(weights, bias_l, self.neuron_base_mut_c, self.neuron_base_mut_s))
                self.layers.append(layer)
        finally:
            file.close()

    def randomise(self, Min: int, Max: int):
        for layer in self.layers:
            for neuron in layer:
                for i in range(len(neuron.weight)):
                    a = randint(Min, Max)
                    neuron.weight[i] = a


# Списки
food_list = [Food() for _ in range(food_count)]
bacteria_list = [Bacteria() for _ in range(bacteria_count)]
gen = 0

best_of_10_gen = Bacteria()


def save_log(best_score):
    file = open('save_score.txt', 'a')
    text = str(gen) + '\t' + str(best_score) + '\n'
    file.write(text)
    file.close()


def genetic_algorithm(bacteria_list_in):
    global gen
    global best_of_10_gen

    bacteria_list_in.sort(key=lambda x: x.score, reverse=True)
    print(f'Current generation: {gen} Best score: {bacteria_list_in[0].score} Gen len: {len(bacteria_list_in)} Start gen: {bacteria_count}')
    save_log(bacteria_list_in[0].score)
    gen += 1
    if gen % 100 == 0:
        bacteria_list_in[0].neural_net.save()
    if gen % 10 == 0:
        if best_of_10_gen.score > bacteria_list_in[0].score:
            bacteria_list_in[0] = best_of_10_gen
        else:
            best_of_10_gen = bacteria_list_in[0]
    new_bacteria_list = [Bacteria() for _ in range(bacteria_count)]

    for i in range(bacteria_count):
        if i <= int(bacteria_count / 1.5):
            if i <= int(bacteria_count / 6):
                new_bacteria_list[i].neural_net = copy.copy(bacteria_list_in[i].neural_net)
            else:
                new_bacteria_list[i].neural_net = copy.copy(bacteria_list_in[0].neural_net)
            if i > 0:
                new_bacteria_list[i].neural_net.mutation()
        else:
            new_bacteria_list[i].neural_net.randomise(min_random, max_random)
    return new_bacteria_list
