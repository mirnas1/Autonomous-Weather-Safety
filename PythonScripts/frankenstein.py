
#part 1 (initalizes everything)
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


import torch
from torchvision import transforms
import time
from ultralytics import YOLO
import math
start = time.time()
import carla
import weakref
import random
import cv2

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_UP
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_m
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90
MODEL_PATH = '/home/boyang/carla/ameer-stuff/train15/weights/best.pt'
MODEL = YOLO(MODEL_PATH)
#CURRENT_WEATHER = 'not available'

#part 1 (end)

#imports from manual_control 
from carla import ColorConverter as cc



class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
    
    def tick(self, car, clock, weather, max_speed):
        # Retrieve relevant data
        transform = car.get_transform()
        velocity = car.get_velocity()
        #print('deez velocities: ', velocity)
        # Convert velocity to km/h
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 0.62137

        # Gather info text
        self._info_text = [
            'Speed:   % 15.0f mp/h' % speed,
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'Current Weather: %s' % weather, 
            'Max Safe Speed: % 5.0f mp/h' % max_speed, 
        ]

    def render(self, display):
        info_surface = pygame.Surface((220, self.dim[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset = 4

        for item in self._info_text:
            surface = self._font_mono.render(item, True, (255, 255, 255))
            display.blit(surface, (8, v_offset))
            v_offset += 18



#part 2 
class SimInfo(object):
    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.display = None
        self.image = None
        self.hud = None  # Initialize HUD here
        self.weather = 'not available'
        self.speed_limit = 65
        self.max_speed = self.speed_limit
        """max speed is a constant based off the speed limit of four-lane divided highways in the US, in a full fledged version,
        depending on the local speed limit, the max speed would be updated accordingly"""
        self.capture = True
        self.counter = 0
        self.pose = []
        self.log = False

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controlled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.seat.leon')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        #makes it so i can save images to disk every 120 frames / 4 seconds
        def camera_callback(image):
            self = weak_self()
            self.set_image(weak_self, image)
            if image.frame % 60 == 0:
                #image.save_to_disk('/home/boyang/carla/ameer-stuff/sim-images/wet-carla/%.6d.png' % image.frame)
                array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (image.height, image.width, 4))
                array = array[:, :, :3]  # Remove alpha channel
                array = array[:, :, ::-1]  # Convert to BGR

                # Convert to RGB for YOLO model
                image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

                # Perform classification
                results = MODEL.predict(image_rgb)
                for result in results:
                    # Assuming single image input
                    probs = result.probs  # Get probabilities for the first image
                    classifaction = probs.top1
                    if classifaction == 0:
                        self.weather = 'dry'
                        if self.max_speed != self.speed_limit:
                            self.max_speed = self.speed_limit
                    elif classifaction == 1:
                        self.weather = 'rain'
                        #using constant of recommended PSI for tesla model 3 (current vehicle in simulation, obviously the car using this alg would update its recommended PSI accordingly)
                        self.max_speed = 9 * math.sqrt(44) - 1
                    elif classifaction == 2:
                        self.weather = 'snow'
                        self.max_speed = self.speed_limit / 2
                        
                        #weather = 'snow'
                    else:
                        weather = 'not available'
                    print('juicy and accurate classification: ', self.weather)

                print('AAAAAAAAAAAAAAAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')

    
        self.camera.listen(camera_callback)

        #self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        current_velocity = car.get_velocity()
        current_speed = 3.6 * math.sqrt(current_velocity.x**2 + current_velocity.y**2 + current_velocity.z**2) * 0.62137

        if keys[K_w] or keys[K_UP]:
            if self.weather == 'dry':
                if current_speed >= (self.speed_limit - 2):
                    control.throttle = 0
                    control.reverse = False
                else: 
                    control.throttle = 1
                    control.reverse = False
            elif self.weather == 'rain':
                if current_speed >= (self.max_speed - 2):
                    control.throttle = 0
                    control.reverse = False
                else: 
                    control.throttle = 1
                    control.reverse = False
            elif self.weather == 'snow':
                if current_speed >= (self.max_speed - 4):
                    control.throttle = 0
                    control.reverse = False
                else:
                    control.throttle = 1
                    control.reverse = False
        elif keys[K_s] or keys[K_DOWN]:
            if self.weather == 'dry':
                control.throttle = 1
                control.reverse = True
            elif self.weather == 'rain':
                if current_speed >= (self.max_speed - 4):
                    control.throttle = 0
                    control.reverse = True
                else: 
                    control.throttle = 1
                    control.reverse = True
            elif self.weather == 'snow':
                if current_speed >= (self.max_speed - 6):
                    control.throttle = 0
                    control.reverse = True
                else:
                    control.throttle = 1
                    control.reverse = True
        if keys[K_a] or keys[K_LEFT]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d] or keys[K_RIGHT]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]
        if keys[K_m]:
            if self.log:
                self.log = False
                np.savetxt('log/pose.txt', self.pose)
            else:
                self.log = True
            pass

        car.apply_control(control)
        return False
    
    #can comment out for initial testing purposes
    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False


    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    #also can comment out for initial testing purposes
    def log_data(self):
        global start
        freq = 1 / (time.time() - start)

        #		sys.stdout.write("\rFrequency:{}Hz		Logging:{}".format(int(freq),self.log))
        sys.stdout.write("\r{}".format(self.car.get_transform().rotation))

        sys.stdout.flush()
        if self.log:
            name = 'log/' + str(self.counter) + '.png'
            position = self.car.get_transform()
            pos = None
            pos = (
            int(self.counter), position.location.x, position.location.y, position.location.z, position.rotation.roll,
            position.rotation.pitch, position.rotation.yaw)
            self.pose.append(pos)
            self.counter += 1
        start = time.time()

    def game_loop(self):
        """
        Main program loop.
        """

        try:
            pygame.init()
            self.hud = HUD(VIEW_WIDTH, VIEW_HEIGHT)  # Initialize HUD after pygame.init()

            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(2.0)
            #self.world = carla.client.load_world('Town01_Opt')
            #line below lets us load in seperate carla map 
            #self.world = self.client.load_world('Town05')
            self.world = self.client.get_world()
            self.setup_car()
            self.setup_camera()
            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            vehicles = self.world.get_actors().filter('vehicle.*')

            while True:
                self.world.tick()
                self.hud.tick(self.car, pygame_clock, self.weather, self.max_speed)
                self.capture = True
                pygame_clock.tick_busy_loop(30)
                self.render(self.display)
                self.hud.render(self.display)
                pygame.display.flip()
                pygame.event.pump()
                self.log_data()
                cv2.waitKey(1)
                if self.control(self.car):
                    return

        # except Exception as e: print(e)
        finally:

            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.car.destroy()
            pygame.quit()
            cv2.destroyAllWindows()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """
    Initializes the client-side bounding box demo.
    """

    try:
        client = SimInfo()
        client.game_loop()
    finally:
        print('EXIT')

if __name__ == '__main__':
    main()
