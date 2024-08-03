import carla

def change_weather():
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set weather conditions
    weather = carla.WeatherParameters(
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=30.0,
        sun_azimuth_angle=180.0,
        sun_altitude_angle=70.0,
        fog_density=0.1,
        fog_distance=0.1,

    world.set_weather(weather)

if __name__ == '__main__':
    change_weather()
