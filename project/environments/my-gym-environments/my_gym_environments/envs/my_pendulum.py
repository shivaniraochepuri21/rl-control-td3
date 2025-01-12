__credits__ = ["Carlos Luis"]

from os import path
from typing import Optional

import numpy as np

# import gym
# from gym import spaces
import gymnasium as gym
from gymnasium import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

class MyPendulumEnv(gym.Env):
    """
       ### Description

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ### Action Space

    The action is a `ndarray` with shape `(1,)` representing the torque applied to free end of the pendulum.

    | Num | Action | Min  | Max |
    |-----|--------|------|-----|
    | 0   | Torque | -2.0 | 2.0 |

    ### Observation Space

    The observation is a `ndarray` with shape `(3,)` representing the x-y coordinates of the pendulum's free
    end and its angular velocity.

    | Num | Observation      | Min  | Max |
    |-----|------------------|------|-----|
    | 0   | x = cos(theta)   | -1.0 | 1.0 |
    | 1   | y = sin(theta)   | -1.0 | 1.0 |
    | 2   | Angular Velocity | -8.0 | 8.0 |

    ### Rewards

    The reward function is defined as:

    *r = -(theta<sup>2</sup> + 0.1 * theta_dt<sup>2</sup> + 0.001 * torque<sup>2</sup>)*

    where `$\theta$` is the pendulum's angle normalized between *[-pi, pi]* (with 0 being in the upright position).
    Based on the above equation, the minimum reward that can be obtained is
    *-(pi<sup>2</sup> + 0.1 * 8<sup>2</sup> + 0.001 * 2<sup>2</sup>) = -16.2736044*,
    while the maximum reward is zero (pendulum is upright with zero velocity and no torque applied).

    ### Starting State

    The starting state is a random angle in *[-pi, pi]* and a random angular velocity in *[-1,1]*.

    ### Episode Truncation

    The episode truncates at 200 time steps.

    ### Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```
    gym.make('Pendulum-v1', g=9.81)
    
    ```

    ### Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.K = 0.01
        self.lam = 0.0
        self.lam2 = 0.0
        self.w1 = 1.0
        self.w2 = 0.001

        self.targ_m = 0.0
        self.max_speed = 8
        self.max_action = 2.0
        self.dt = 0.05
        self.action_add = [0.0]
        self.action_scale = [1.0]

        self.min_state0 = -np.pi
        self.max_state0 = np.pi
        self.min_state1 = -self.max_speed
        self.max_state1 = self.max_speed
        self.min_state = np.array([self.min_state0, self.min_state0])
        self.max_state = np.array([self.max_state1, self.max_state1])
        self.min_K  = 0.0        
        self.max_K  = 1000.0
        self.min_e = self.min_state - self.max_state
        self.max_e = self.max_state - self.min_state
        self.min_edot = (self.min_e - self.max_e)/self.dt 
        self.max_edot = (self.max_e - self.min_e)/self.dt 

        self.ref_model_min = np.linalg.norm(self.min_edot + self.K*self.min_e) # K >= 0
        self.ref_model_max = np.linalg.norm(self.max_edot + self.K*self.max_e)

        self.init_state = np.array([0.1*np.pi/180, 0.0])
        self.targ_state = np.array([179.0*np.pi/180, 0.0])
        self.test_init_state = np.array([0.1*np.pi/180, 0.0])
        self.test_targ_state = np.array([179.0*np.pi/180, 0.0])

        self.min_obs = np.array([-1, -1, -1, -1, self.min_state1, self.min_state1, self.ref_model_min, self.ref_model_min, self.min_K])
        self.max_obs = np.array([1, 1, 1, 1, self.max_state1, self.max_state1, self.ref_model_max, self.ref_model_max, self.max_K])

        # high_state = np.array([np.pi, self.max_speed], dtype=np.float32)
        # low_state = np.array([0, -self.max_speed], dtype=np.float32)
        # self.observation_space = spaces.Box(low=-high_state, high=high_state, shape=(2,), dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_action == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        # high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)

        self.observation_space = spaces.Box(low=self.min_obs, high=self.max_obs, shape=(9,), dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, shape=(1,), dtype=np.float32
        )

    def step(self, u):
        th, thdot = self.state
        done = False
        done_ = False    

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt
        u = u[0]
        self.last_u = u

        # u = np.clip(u, -self.max_action, self.max_action)[0]

        # costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)
        # costs = (angle_normalize(self.targ_state[0]) - angle_normalize(th)) ** 2 + 0.1 * (self.targ_state[1] - thdot)**2 + 0.001 * (u**2)
        # rewards = -costs

        # rewards = - (self.K**2 * e**2) - e_dot**2 - 0.001*u**2 - 2*abs(self.K * e * e_dot)
        
        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt
        # newth = newth % (2*self.max_state0)
        # if newth == 2*np.pi:
            # newth = 0.0
        new_state = np.array([newth, newthdot])

        # e = angle_normalize(self.targ_state[0]) - angle_normalize(th)
        # self.e = angle_normalize(self.targ_state[0]) - angle_normalize(th)
        # e_new = angle_normalize(self.targ_state[0]) - angle_normalize(new_state[0])
        self.e = self.targ_state[0] - th
        e_new = self.targ_state[0] - new_state[0]

        self.e_dot = (e_new - self.e)/self.dt
        self.m = np.linalg.norm(self.e_dot + self.K * self.e)

        rewards = -(self.w1*abs(self.m - self.targ_m) + self.lam*abs(angle_normalize(self.targ_state[0]) - angle_normalize(new_state[0]))**2 + self.lam2*abs(self.e_dot)**2 + self.w2*u**2)

        # newth = newth % (2*np.pi)
        # newth = np.clip(newth, 0, 2*np.pi)
        # if np.round(newth,3) - np.round(2*np.pi, 3) == 0.000:
            # newth = 0.0

        if self.render_mode == "human":
            self.render()
        # print('env state', np.rad2deg(self.state[0]), self.state[1])
        # if np.linalg.norm(abs(self.state[0] - self.targ_state[0])) <= (np.pi/180):
            # done = True
            # done_ = True
        self.state = new_state
        info = {'error': e_new, 'error_dot':self.e_dot, 'theta':self.state[0], 'theta_dot':self.state[1]}
        return self._get_obs(), rewards, done, done_, info
    
    def reset(self, *, init_state = None, targ_state = None, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is None:
            high = np.array([DEFAULT_X, DEFAULT_Y])
            # high = np.array([1.0, 1.0, DEFAULT_Y])
            # high = np.array([2*np.pi, self.max_speed])
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            high = np.array([x, y])

        low = -high  # We enforce symmetric limits.
        self.last_u = None
        # low = np.array([0, -self.max_speed])

        if init_state is None:
            self.state = self.np_random.uniform(low=low, high=high)
        else:
            self.state = init_state
         
        # if np.round(self.state[0],3) - np.round(2*np.pi, 3) == 0.000:
            # self.state[0] = 0.0
        if targ_state is None:
            self.targ_state = self.targ_state
        else:
            self.targ_state = targ_state

        if self.render_mode == "human":
            self.render()
        
        # info = {'init_theta': self.state[0], 'init_theta_dot': self.state[1], 'targ_theta': self.targ_state[0], 'targ_theta_dot': self.targ_state[1]}

        # self.e = angle_normalize(self.targ_state[0]) - angle_normalize(self.state[0])
        self.e = self.targ_state[0] - self.state[0]
        self.e_dot = (self.e - self.e)/self.dt
        self.m = np.linalg.norm(self.e_dot + self.K * self.e)
        info = {'error': self.e, 'error_dot': self.e_dot, 'theta':self.state[0], 'theta_dot':self.state[1], 'K': self.K, 'model': self.m}

        return self._get_obs(), info

    def _get_obs(self):
        theta, thetadot = self.state
        theta_targ, thetadot_targ = self.targ_state
        return np.array([np.cos(theta), np.sin(theta), np.cos(theta_targ), np.sin(theta_targ), thetadot, thetadot_targ, self.m, self.targ_m, self.K], dtype=np.float32)
        # return self.state
    
    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
    # return x