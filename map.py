# Self Driving Car

# Importing the libraries
import torch
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture
np.random.seed(23)


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0



last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
first_update = True
max_steps = 2500
current_step = 0
done = True
on_road = 0
on_road_count = 0
off_road_count = 0
boundary_hit_count = 0
goal_hit_count = 0
train_episode_num = 0
eval_episode_num = 0
mode="Eval"
if os.path.exists("results") == False:
    os.makedirs("results")

if os.path.exists("sand_images") == False:
    os.makedirs("sand_images")	
	
	
full_eval_on_road_stats_file= open("results/full_eval_on_road_stats.csv","a")
eval_on_road_stats_file= open("results/eval_on_road_stats.csv","a")
train_on_road_stats_file= open("results/train_on_road_stats.csv","a")
full_eval_traversal_log= open("results/full_eval_traversal_log.txt","w")
eval_traversal_log= open("results/eval_traversal_log.txt","w")
train_traversal_log= open("results/train_traversal_log.txt","w")
traversal_log = None

img = None
arrow_img = None
global_counter = 0



def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global done
    global max_steps
    global current_step
    global img
    global arrow_img
    global global_counter
	
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    arrow_img = PILImage.open("./images/arrow.png")
    sand = np.asarray(img)/255	
    goal_x = 580
    goal_y = 350
    first_update = False
    done = False
    global swap
    swap = 0
    current_step = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0.0)
    #angle = float(0.0)
    #angle = ObjectProperty('float')
    rotation = BoundedNumericProperty(0.0)
    #rotation = ObjectProperty('float')
    #rotation = float(0.0)
    velocity_x = BoundedNumericProperty(0)
    velocity_y = BoundedNumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    event1 = None
    queue1_1 = None
    queue1_2 = None
    event2 = None
    queue2 = None
    event3 = None
    queue3 = None


    def move(self, rotation):
        #self.pos = Vector(*self.velocity) + self.pos
        self.x = int(self.velocity_x + self.x)
        self.y = int(self.velocity_y + self.y)
        self.pos = Vector(self.x, self.y)
        #self.pos = Vector(self.x, self.y)
        self.rotation = rotation
        self.angle = self.angle + self.rotation
 
        


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
	
    def serve_car(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.car.start_event = start_event
        self.car.reset_q = reset_q
        self.car.mode_q = mode_q
        self.car.state_q = state_q
        self.car.action_q = action_q
        self.car.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q

    
    def get_state(self, img, arrow_img, x, y,  car_angle, global_counter,longueur, largeur, traversal_log): 
        if x - 40 <0 or y-40 < 0 or x+40 > longueur-1 or y+40 > longueur-1:
            return np.ones((80,80))
        else:
            img_crop = img.crop((x -40, y-40, x+40, y +40))
            arrow_rotated = arrow_img.rotate(car_angle)
            arrow_size = (32,32)
            arrow_rotated = arrow_rotated.resize(arrow_size, PILImage.ANTIALIAS).convert("RGBA")
            img_crop.paste(arrow_rotated, (24, 48), arrow_rotated)
            if global_counter % 500 == 0:
                print("map: get_state: for image id: " + str(int(global_counter / 500) + 1) + " angle: " + str(car_angle))
                traversal_log.write("map: get_state: car angle: " + str(car_angle))
                img_crop.save("sand_images/sand_superimposed_car_" + str(int(global_counter / 500) + 1) + ".png", "PNG")
            state_value = np.asarray(img_crop)/255	
            return state_value
			

	
    def update(self, dt):

        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global first_update
        global done
        global current_step
        global max_steps
        global on_road
        global on_road_count
        global off_road_count
        global eval_on_road_stats_file
        global full_eval_on_road_stats_file
        global train_on_road_stats_file
        global mode
        global boundary_hit_count
        global goal_hit_count
        global train_episode_num
        global eval_episode_num
        global eval_traversal_log
        global full_eval_traversal_log
        global train_traversal_log
        global traversal_log
        global img
        global arrow_img
        global global_counter
        boundary_hit = 0
        distance_reduced = 0
        goal_hit = 0

        longueur = self.width
        largeur = self.height
        self.car.start_event.wait()
        if done == True:
            if traversal_log is not None:
                traversal_log.flush()
				
            if current_step > 0:
                if mode == "Train" :
                    train_on_road_stats_file.write(str(train_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n") 
                    train_on_road_stats_file.flush()
                elif mode == "Eval" :
                    eval_on_road_stats_file.write( str(train_episode_num) + "," +  str(eval_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n")
                    eval_on_road_stats_file.flush()	
                else:
                    full_eval_on_road_stats_file.write(str(eval_episode_num) + "," + str(on_road_count) + "," + str(off_road_count) + "," + str(boundary_hit_count) + "," + str(goal_hit_count) +  "\n")
                    full_eval_on_road_stats_file.flush()	
            reset = self.car.reset_q.get()

            if reset == True:
                print("first_update is set to True")
                first_update = True
				
            (mode, train_episode_num, eval_episode_num) = self.car.mode_q.get()
            print("mode: ", mode) 
            if mode == "Train":
                max_steps = 2500
                traversal_log = train_traversal_log
            elif mode == "Eval": 
                max_steps = 500
                traversal_log = eval_traversal_log
            else:
                max_steps = 2500
                traversal_log = full_eval_traversal_log				
			
        if first_update:
            init()
            on_road_count = 0
            off_road_count = 0
            boundary_hit_count = 0
            goal_hit_count = 0
            boundary_hit = 0
            distance_reduced = 0
            goal_hit = 0
            #self.car.pos = (100,100)
            self.car.pos = Vector(np.random.randint(60, longueur-60), np.random.randint(60, largeur-60))
            self.car.rotation = 0.0
            self.car.angle = 0.0
            print("After reset car position: ", self.car.pos)
            traversal_log.write("After reset car position: " + str(self.car.pos) + "\n")
            #state = sand[int(self.car.x)-40:int(self.car.x)+40, int(self.car.y)-40:int(self.car.y)+40]
            state = self.get_state( img, arrow_img, self.car.x, self.car.y, self.car.angle, global_counter,longueur, largeur, traversal_log)
			
			## TODO:
            #if state.shape[0] != 80 or state.shape[1] != 80:
            #    state = np.ones((80,80))
			
            xx = goal_x - self.car.x
            yy = goal_y - self.car.y
            if sand[int(self.car.x),int(self.car.y)] > 0:
                on_road = -1
                off_road_count += 1
            else :
                on_road = 1
                on_road_count += 1
            orientation = Vector(*self.car.velocity).angle((xx,yy))/360.			
            self.car.state_q.put((state, np.array([0, orientation, 0, on_road, goal_hit, boundary_hit, distance_reduced])))
            print("map.py self.car.state_q", self.car.state_q)
 		
   
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/360.
		
		
        action_array = self.car.action_q.get()
        #action = int(np.around(action_array[0]))
        rotation = action_array[0]
        rotation = 0.6 * rotation
        velocity = action_array[1]
        new_velocity = 0.4 + 1 + velocity*0.2
        print("map: Got rotation: ", rotation, " velocity: ", new_velocity)
        traversal_log.write("map: Got rotation: " + str(rotation) + " velocity: " + str(new_velocity) + "\n")
        #rotation = action2rotation[action]
        self.car.move(rotation)
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
		
        if self.car.x < 40:
            #self.car.x = 20
            last_reward = -50
            self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))
            self.car.rotation = 0.0
            self.car.angle = 0.0
            print("Hit Boundary: new car postion: ",self.car.pos, " rotation: ", self.car.rotation, " angle: ", self.car.angle)
            traversal_log.write("Hit Boundary: new car postion: " + str(self.car.pos) +  " rotation: " + str(self.car.rotation) + " angle: " + str(self.car.angle) +  "\n")
            boundary_hit_count += 1
        if self.car.x > self.width - 40:
            #self.car.x = self.width - 20
            last_reward = -50
            self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))
            self.car.rotation = 0.0
            self.car.angle = 0.0
            print("Hit Boundary: new car postion: ",self.car.pos, " rotation: ", self.car.rotation, " angle: ", self.car.angle)
            traversal_log.write("Hit Boundary: new car postion: " + str(self.car.pos) +  " rotation: " + str(self.car.rotation) + " angle: " + str(self.car.angle) +  "\n")
            boundary_hit_count += 1
        if self.car.y < 40:
            #self.car.y = 20
            last_reward = -50
            self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))
            self.car.rotation = 0.0
            self.car.angle = 0.0
            print("Hit Boundary: new car postion: ",self.car.pos, " rotation: ", self.car.rotation, " angle: ", self.car.angle)
            traversal_log.write("Hit Boundary: new car postion: " + str(self.car.pos) +  " rotation: " + str(self.car.rotation) + " angle: " + str(self.car.angle) +  "\n")
            boundary_hit_count += 1
        if self.car.y > self.height - 40:
            #self.car.y = self.height - 20
            last_reward = -50
            self.car.pos = Vector(np.random.randint(100, longueur-100), np.random.randint(100, largeur-100))
            self.car.rotation = 0.0
            self.car.angle = 0.0
            print("Hit Boundary: new car postion: ",self.car.pos, " rotation: ", self.car.rotation, " angle: ", self.car.angle)
            traversal_log.write("Hit Boundary: new car postion: " + str(self.car.pos) +  " rotation: " + str(self.car.rotation) + " angle: " + str(self.car.angle) +  "\n")
            boundary_hit_count += 1

        if sand[int(self.car.x),int(self.car.y)] > 0:
            #vel = 0.4 + np.random.uniform(0, 2)
            #self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle)            
            last_reward = -3
            on_road = 0
            off_road_count += 1
            print(1,  current_step + 1, int(self.car.x), int(self.car.y), goal_x, goal_y, float(distance - last_distance),  im.read_pixel(int(self.car.x),int(self.car.y)), last_reward)
            traversal_log.write("1" + " " +  str(current_step + 1) + " " +  str (int(self.car.x)) + " " +  str(int(self.car.y)) + " " + str(goal_x) + " " +  str(goal_y) + " " +  str(float(distance - last_distance)) + " " + str(im.read_pixel(int(self.car.x),int(self.car.y))) + " " + str(last_reward) + "\n")
        else: # otherwise
            #self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            self.car.velocity = Vector(new_velocity, 0).rotate(self.car.angle)
            on_road = 1
            on_road_count += 1
            last_reward = -1
            
            if distance < last_distance:
                last_reward = last_reward + 7
                
            else:
                last_reward = last_reward + 4.5
                on_road = 1			
            print(0, current_step + 1, int(self.car.x), int(self.car.y), goal_x, goal_y, float(distance - last_distance),  im.read_pixel(int(self.car.x),int(self.car.y)), last_reward)
            traversal_log.write("0" + " " +  str(current_step + 1) + " " +  str (int(self.car.x)) + " " +  str(int(self.car.y)) + " " + str(goal_x) + " " +  str(goal_y) + " " +  str(float(distance - last_distance)) + " " + str(im.read_pixel(int(self.car.x),int(self.car.y))) + " " + str(last_reward)+ "\n")
        

        if distance < 25:
            last_reward = 150
            goal_hit_count += 1
            goal_hit=1
			
			
            if swap == 1:
                print("Hit the Goal 2: (" + str(goal_x) + ", " + str(goal_y) + ")")
                traversal_log.write("Hit the Goal 2: (" + str(goal_x) + ", " + str(goal_y) + ")\n")

                goal_x = 580
                goal_y = 350
                swap = 0
            else:
                print("Hit the Goal 1: (" + str(goal_x) + ", " + str(goal_y) + ")")
                traversal_log.write("Hit the Goal 1: (" + str(goal_x) + ", " + str(goal_y) + ")\n")
                #goal_x = 212
                #goal_y = 150
                #goal_x = 975
                #goal_y = 110
                goal_x = 983
                goal_y = 585
                swap = 1

        last_distance = distance
		
        #next_state = sand[int(self.car.x)-40:int(self.car.x)+40, int(self.car.y)-40:int(self.car.y)+40]
        next_state = self.get_state(img, arrow_img, self.car.x, self.car.y, self.car.angle, global_counter,longueur, largeur, traversal_log)
		## TODO:
        #if next_state.shape[0] != 80 or next_state.shape[1] != 80:
        #    next_state = np.ones((80,80))
		
        reward = last_reward
        current_step += 1
        global_counter += 1
        car_angle = self.car.angle
        if car_angle > 360:
            car_angle = car_angle % 360
        elif car_angle < -360:
            car_angle = car_angle % (-360)
			
        if current_step >= max_steps:
            done = True
        distance_diff = (distance - last_distance)/4
        self.car.next_state_reward_done_tuple_q.put(((next_state, np.array([car_angle/ 360, orientation,  distance_diff, on_road, goal_hit, boundary_hit, distance_reduced])), reward, done, current_step))
        #print("map.py self.car.next_state_reward_done_tuple_q", self.car.next_state_reward_done_tuple_q)
		#self.car.event2.set()
		

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self, start_event, reset_q, mode_q, state_q, action_q, next_state_reward_done_tuple_q):
        super(CarApp, self).__init__()
        self.start_event = start_event
        self.reset_q = reset_q
        self.mode_q = mode_q
        self.state_q = state_q
        self.action_q = action_q
        self.next_state_reward_done_tuple_q = next_state_reward_done_tuple_q

    def build(self):
        parent = Game()
        parent.serve_car(self.start_event, self.reset_q, self.mode_q, self.state_q, self.action_q, self.next_state_reward_done_tuple_q)
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
 

    def load(self, obj):
        print("loading last saved brain...")


# Running the whole thing
if __name__ == '__main__':
    #car_app_instance = CarApp(1,2,3)
    #car_app_instance.run()
    #print("CarApp state_dim=",car_app_instance.state_dim)
    print("Hi")
