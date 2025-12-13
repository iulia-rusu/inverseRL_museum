# Gymnasium & Minigrid imports
import gymnasium as gym  # Correct way to import Gymnasium
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.constants import DIR_TO_VEC
from minigrid.core.grid import Grid
from minigrid.core.actions import Actions
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv



from gymnasium.utils.play import play


import numpy as np

import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from scipy.ndimage import binary_dilation
import pandas as pd
import pickle

""" This code runs a museum maze training environment on a remote server.
Maze and reward_struct are two required files that should be present in the same directory as the
script.
"""


with open("/home/irusu/mask.pkl", "rb") as f:
    mask = pickle.load(f)

with open("/home/irusu/reward_struct.pkl", "rb") as f:
    reward_struct = pickle.load(f)

#environment




class SimpleEnv(MiniGridEnv):
    def __init__(
            self, 
            agent_start_pos=(1, 30), # bottom left entrance ( 1, 30) top right corner (63, 5)
            agent_start_dir=0, 
            max_steps=700,
            mask = None, 
            reward_mask = None,
            
            **kwargs,
    ):
        
    
        height, width = mask.shape
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.agent_dir = 0 #facing up always
        # self.goal_pos = (8, 1)
        self.mask = mask
        self.reward_mask = reward_mask
        self.visited_reward_0 = set()  # holds (x, y) tuples

        
        
        
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            # grid_size=max(width, height),
            width=width,
            height=height,
            max_steps=max_steps,
            **kwargs,
        )

        # Restore correct values
        self.width = width
        self.height = height

        self.action_space = gym.spaces.Discrete(5)
    @staticmethod
    def _gen_mission():
        return "Museum"

    def _gen_grid(self, width, height):

        self.grid = Grid(self.width, self.height)

        for y in range(self.mask.shape[0]):
            for x in range(self.mask.shape[1]):
                if not self.mask[y, x]:
                    self.grid.set(x, y, Wall())



        #place goal
        # self.put_obj(Goal(), 8, 1)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos #check this
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Museum"

        # print(f"agent_start = {self.agent_start_pos}")
        print(f"[DEBUG] Initialized grid {self.width}x{self.height}, agent at {self.agent_pos}")

    def step(self, action):
        x, y = map(int, self.agent_pos)

        # Define absolute movement
        if action == 0:      # stay
            dx, dy = 0, 0
        elif action == 1:    # up
            dx, dy = 0, -1
        elif action == 2:    # down
            dx, dy = 0, 1
        elif action == 3:    # left
            dx, dy = -1, 0
        elif action == 4:    # right
            dx, dy = 1, 0
        else:
            raise ValueError(f"Invalid action: {action}")

        new_x, new_y = int(x + dx), int(y + dy)

        # Stay in bounds
        if not (0 <= new_x < self.width and 0 <= new_y < self.height):
            new_x, new_y = x, y

        # Check for wall
        target_cell = self.grid.get(new_x, new_y)
        if target_cell is not None and not target_cell.can_overlap():
            new_x, new_y = x, y  # can't move into wall

        self.agent_pos = (new_x, new_y)

        # --- REWARD LOGIC ---
        if self.reward_mask[new_y, new_x] == 0:
            if (new_x, new_y) in self.visited_reward_0:
                reward = -1  # revisited exhibit
            else:
                reward = 0   # first time visit
                self.visited_reward_0.add((new_x, new_y))
        else:
            reward = self.reward_mask[new_y, new_x]

        obs = self.gen_obs()
        done = False
        info = {}

        return obs, reward, done, info
    
    def count_states(self):
        free_cells = sum(1 for x in range(self.grid.width)
                      for y in range(self.grid.height)
                      if not self.grid.get(x, y)) * 4
        return free_cells 

        
@dataclass

class ModelState:
    #tables and arrays
    Q_table: np.ndarray = field(default_factory=lambda: np.array([]))
    Pi_a_s: np.ndarray = field(default_factory=lambda: np.array([]))
    P_s_given_s_a: np.ndarray = field(default_factory=lambda: np.array([]))
    P_s_by_s: np.ndarray = field(default_factory=lambda: np.array([]))
    
    allowed_state_idx: np.ndarray = field(default_factory=lambda: np.array([]))
    
    #scalars
    beta: Optional[float] = None
    num_actions: Optional[int] = None
    num_states: Optional[int] = None

    #lists
    info_to_go_term_training: List[float] = field(default_factory=list)
    pi_analysis_term_training: List[float] = field(default_factory=list)

    #list for P_s
    positions_directions_list_ps: List[float] = field(default_factory=list)
    positions_directions_list_neglogps: List[float] = field(default_factory=list)

    #dictionaries
    transition_counts: Dict = field(default_factory=dict)


    # utility functions

class GridMixin:
    def position_to_state_index(self, pos=None):
        """Map (x, y) to state index assuming row-major order."""
        if pos is None:
            x, y = self.env.agent_pos
        else:
            x, y = pos
        return y * self.env.width + x

    def state_index_to_position(self, idx):
        """Map scalar state index to (x, y) position."""
        x = idx % self.env.width
        y = idx // self.env.width
        return x, y

    def find_state_indexes(self):
        """Return all non-wall, walkable state indices."""
        state_indices = []
        for y in range(self.env.height):
            for x in range(self.env.width):
                if self.env.grid.get(x, y) is None:  # Empty/walkable
                    state_indices.append(self.position_to_state_index((x, y)))
        return state_indices

    def next_state_index(self, current_state_idx, action):
        """Given a state index and action, return the resulting state index."""
        x, y = self.state_index_to_position(current_state_idx)

        # Movement logic
        if action == 0:  # stay
            pass
        elif action == 1:  # up
            y -= 1
        elif action == 2:  # down
            y += 1
        elif action == 3:  # left
            x -= 1
        elif action == 4:  # right
            x += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        
        # Convert the new (x, y, direction) back to a state index.
        next_state = self.position_to_state_index((x, y))
        if next_state not in self.state.allowed_state_idx:
            # If the next state is not allowed, return the current state
            return current_state_idx
        return next_state

    def find_all_next_states(self):
        """Find all possible next states for each state and action."""
        for state in self.state.allowed_state_idx:
            for action in range(self.state.num_actions):
                next_state = self.next_state_index(state, action)
                self.state.P_s_given_s_a[state, action, next_state] = 1 


    def find_connected_states(self):
        """
        Build and return a connectivity matrix P_s_by_s where the element at [state, next_state]
        is given by the probability from self.Pi_a_s for the action that leads from state to next_state.
        """
        # Reset the connectivity matrix at the beginning.
        self.state.P_s_by_s = np.zeros((self.state.num_states, self.state.num_states))
        # Loop over all states.
        for state in self.state.allowed_state_idx:
            # Loop over all actions for the state.
            for action in range(self.state.num_actions):
                # Find the next state: assume a deterministic transition where exactly one entry is 1.
                next_state = np.argmax(self.state.P_s_given_s_a[state, action, :])
                # Set the connectivity matrix: you might choose to sum if multiple actions lead to the same state.
                self.state.P_s_by_s[state, next_state] += self.state.Pi_a_s[state, action]


#Training

    
##################working code! ######################
class FreeEnergy(GridMixin):
    def __init__(self, env, state: ModelState, epochs = 20, beta = 5):
        self.state = state
        self.env = env
        state: ModelState
         # self.max_steps = 100
        self.beta = beta
        self.epochs = epochs
        self.state = state
        self.state.allowed_state_idx = self.find_state_indexes()
        self.state.num_states = env.width * env.height
        self.state.num_actions = 5
        #shapes defined for first time
        self.state.P_s_given_s_a = np.zeros((self.state.num_states, self.state.num_actions, self.state.num_states)) # P(s'|s,a) matrix
        self.state.P_s_by_s = np.zeros((self.state.num_states, self.state.num_states)) # P(s'|s) matrix
        self.state.Pi_a_s = np.full((self.state.num_states, self.state.num_actions), 1/self.state.num_actions) # pi(s,a) matrix
      
        self.Free_energy_table = np.full((self.state.num_states, self.state.num_actions), 0.0) # Free energy table


        self.reward_mask = reward_struct
        self.state.beta = beta

        
        self.Pi_a = np.full((self.state.num_actions), 1/self.state.num_actions)
        self.P_s = np.zeros(self.state.num_states)
        self.visited_reward_0 = set()  # holds (x, y) tuples

    def train_BA(self):
        max_ba_iters = 10
        tol = 1e-2
        self.find_all_next_states()
        
        for epoch in range(self.epochs):
            print(f"Epoch {epoch }/{self.epochs} starting...")
            self.steps = 0
            current_state = np.random.choice(self.state.allowed_state_idx) #2071 if want to start at entrance to train
            #print
            # print(f"current state: {current_state}")
            x, y = self.state_index_to_position(current_state)
            # print(f"current position: {x}, {y}")
            self.env.agent_pos = (x, y)
            # print(f"current agent position: {self.env.agent_pos}")
          

            initial_s = np.zeros(self.state.num_states)
            initial_s[current_state] = 1

            while self.steps < 150:  # ← Now limits to 100 steps per epoch
                action = np.random.choice(np.arange(self.state.num_actions), p=self.state.Pi_a_s[current_state])
                # print(f"Step {self.steps}: action={action}, current_state={current_state}")
                _, _, _, _ = self.env.step(action)
                self.env.render()
                next_state = self.position_to_state_index()
                # print(f"next state: {next_state}")

                # reward = self.env.reward_mask[self.env.agent_pos[1], self.env.agent_pos[0]] \
                #         if self.env.reward_mask is not None else 0

                if self.reward_mask[self.env.agent_pos[1], self.env.agent_pos[0]] == 0:
                    if (self.env.agent_pos[0], self.env.agent_pos[1]) in self.visited_reward_0:
                        reward = -1  # revisited exhibit
                    else:
                        reward = 0   # first time visit
                        self.visited_reward_0.add((self.env.agent_pos[0], self.env.agent_pos[1]))
                else:
                    reward = self.reward_mask[self.env.agent_pos[1], self.env.agent_pos[0]]

                self.find_connected_states()
                Ps_s_matrix = np.linalg.matrix_power(self.state.P_s_by_s, self.steps)
                P_s = np.dot(initial_s.T, Ps_s_matrix)

                assert np.isclose(np.sum(P_s), 1, atol=1e-5), f"P(s) sum error: {np.sum(P_s)}"

                F_0 = -np.log(P_s[next_state] + 1e-15) - self.beta * reward
                G_0 = sum(
                    (np.log(self.state.Pi_a_s[next_state, a] + 1e-15) - np.log(self.Pi_a[a] + 1e-15)
                    + self.Free_energy_table[next_state, a]) * self.state.Pi_a_s[next_state, a]
                    for a in range(self.state.num_actions)
                )
                F_0 += G_0

                self.Free_energy_table[current_state, action] = F_0

                #######loop of calculations#####
                for iteration in range(max_ba_iters):
                  

                    self.Pi_a = P_s@self.state.Pi_a_s 
                
                  
                    
                    assert np.isclose(np.sum(self.Pi_a), 1), f"Sum of Pi_a is not 1: {np.sum(self.Pi_a)}"

                    
                    # 1) log π(a)  — shape (A,)
                    log_pi_a = np.log(self.Pi_a + 1e-15)

                    # 2) log-joint   log π(a) − β F(s,a)  — broadcast to shape (S,A)
                    log_joint = log_pi_a  -self.Free_energy_table

                    # 3) row-shift: subtract max in each state to keep exp() ≤ 1
                    log_joint -= log_joint.max(axis=1, keepdims=True)

                    # 4) exponentiate and normalise each row
                    new_Pi_a_s = np.exp(log_joint)
                    new_Pi_a_s /= new_Pi_a_s.sum(axis=1, keepdims=True)   # Σ_a π(a|s)=1
                                       
                    #calculate Z, works only for small beta
                    # element_wise_a_by_q_table = self.Pi_a * np.exp(-  self.Free_energy_table)
                    
                    # zeta = np.sum((element_wise_a_by_q_table), axis = 1)
                   
                    
                    # #calculate for policy using partition fuction

                    # new_Pi_a_s = (self.Pi_a * np.exp(-  self.Free_energy_table))/ zeta.reshape(-1,1) 
                    diff = np.linalg.norm(new_Pi_a_s - self.state.Pi_a_s, ord='fro')
                    # print(f"Epoch {epoch}: Frobenius norm difference: {diff:.4f}") # with percent
                    
                    self.state.Pi_a_s = new_Pi_a_s
                    iteration += 1
                    if diff < tol:
                        # print(f"Convergence reached at epoch {epoch} "f"after {iteration} BA iteration(s); "f"Δ = {diff:.4e}")
                        break

                      # Frobenius norm for matrix difference
                    
                # assert(False)
                current_state = next_state
                # print(f"Epoch {epoch}: Step {self.steps}, Current state: {current_state}")
                self.steps += 1
                # print(f"Epoch {epoch}: Step {self.steps}, Current state: {current_state}, Action: {action}, Reward: {reward}")


class QRunner(GridMixin):
    def __init__(self, env, state: ModelState):
        self.state = state
        self.env = env
        
    
    def run_policy_2(self):
        """Run the environment using the learned policy from a Q-table."""

        self.env.reset()[0]  # Reset environment
        current_state = self.position_to_state_index()  # Convert starting position to index
        done = False
        self.step_count = 0  # Track steps to prevent infinite loops

        while not done and self.step_count < 1000:  # Prevent infinite loops
  
            action = np.random.choice(np.arange(self.state.num_actions), p=self.state.Pi_a_s[current_state])  # Choose best action
            next_obs, _, done, _ = self.env.step(action)  # Take action
            next_state = self.position_to_state_index()  # Convert new state
            self.env.render()  # Visualize movement
            # Calculate info to go term for the current step
            current_state = next_state  # Update current state
            self.step_count += 1


"""
Workflow
"""
#instantiate state instance, can also import from another file if needed
state = ModelState()

env = SimpleEnv(render_mode=None, mask = mask, reward_mask= reward_struct)
#env.reset needed to bypass the step counter function which is otherwise needed but doesn't exist in SimpleEnv
env.reset();

free = FreeEnergy(env, state, beta = 5)


free.train_BA()




# take out the policy from the free.train_BA() run
np.save("pi_a_s.npy", free.state.Pi_a_s)











