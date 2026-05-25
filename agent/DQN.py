from typing import Optional, Any
import copy
from collections import deque
import random
import numpy as np
from scipy.special import stdtrit
import torch
import torch.nn as nn
from .robust import hq_opt

class DQN:
    '''
    Implementation of DQN
    clone_steps controls # of steps before target network weights are updated to most recent q-network weights
    train_steps controls # of steps between each training iteration
    n_epochs controls # of epochs for each training iteration
    n_batches controls # of batches to train in each epoch
    clip gradients controls whether to clip gradient to [-1,1]
    epsilon controls the epsilon greedy param
    training_mode controls whether to switch off update buffer, target network and train q-network
    '''

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        discount: float,
        qfunc: Optional[torch.nn.Module] = None,
        epsilon: float = 0.1,
        buffer_max_length: int = int(1e6),
        clone_steps: int = 256,
        train_steps: int = 8,
        batch_size: int = 128,
        n_batches: int = 1,
        n_epochs: int = 1,
        lr: float=0.001,
        clip_gradients: bool = False,
        loss_fn: nn.Module = nn.MSELoss(),
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
    ):
        '''Initialize the DQN agent with the given parameters.'''

        # Environment and policy related
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.epsilon = epsilon  # Epsilon greedy
        self.discount = discount

        # Q-function neural network related
        if qfunc is None:
            self.q = nn.Sequential(
                nn.Linear(obs_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        else:
            self.q = qfunc
        self.lr = lr
        self.buffer_max_length = buffer_max_length  # In terms of steps
        self.batch_size = batch_size
        self.clone_steps = clone_steps
        self.train_steps = train_steps
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.clip_gradients = clip_gradients
        self.device = device
        self.q.to(self.device)
        self.target_q = copy.deepcopy(self.q)
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=self.lr)

        self.training_mode = True
        self.steps = 0
        self.buffer = deque(maxlen=self.buffer_max_length)
        self.loss_fn = loss_fn
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if seed is not None:
            random.seed(seed)

    def agent_start(self, observation):
        observation = self.check_obs(observation)
        action = self.get_action(observation).cpu()
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_step(self, reward, observation, info=None):
        observation = self.check_obs(observation)
        reward = self.check_reward(reward)
        action = self.get_action(observation).cpu()

        if self.training_mode: self.train_mode_actions(reward, observation, False, info) #NOTE: must be before new action/obs replaces self.prev_action/self.prev_state
        self.prev_state = observation
        self.prev_action = action
        return action

    def agent_end(self, reward, observation, info=None):
        observation = self.check_obs(observation)
        reward = self.check_reward(reward)
        if self.training_mode: self.train_mode_actions(reward, observation, True, info)

    def get_action(self, observation, greedy=False):
        with torch.no_grad():
            q_values = self.q(observation.to(self.device))
        actions = torch.argmax(q_values, dim=-1, keepdim=True)

        if (self.epsilon > 0) and (not greedy) and self.training_mode:
            epsilon_greedy = torch.rand(actions.shape[0]) < self.epsilon
            if epsilon_greedy.sum() > 0:
                actions[epsilon_greedy] = torch.randint(0, self.num_actions, (epsilon_greedy.sum(), 1), device=self.device)

        return actions.cpu()

    def train_mode_actions(self, reward, observation, terminal, info=None):
        ''' actions to take when agent in training mode i.e. adding to replay buffer, cloning target q network and training q network'''
        self.steps += 1
        self.add_to_replay_buffer(reward, observation, terminal, info)
        if self.clone_q_net_condition(): self.clone_q()
        if self.training_condition(): self.update_q()

    def add_to_replay_buffer(self, reward, observation, terminal, info=None):
        ''' add step sequence to buffer '''
        terminal_state = torch.tensor([terminal], dtype=torch.bool).repeat(self.prev_state.shape[0], 1)
        if info is None:
            values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state))
        else:
            if isinstance(info, tuple):
                values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state, *info))
            else:
                values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state, info))
        self.buffer.extend(values)
        while len(self.buffer) > self.buffer_max_length:
            self.buffer.popleft()

    def training_condition(self):
        bool_step_multiple = (self.steps % self.train_steps == 0)
        return bool_step_multiple and len(self.buffer) >= self.batch_size * self.n_batches

    def clone_q_net_condition(self):
        bool_step_multiple = (self.steps % self.clone_steps == 0)
        return bool_step_multiple and len(self.buffer) >= self.batch_size * self.n_batches

    def clone_q(self):
        self.target_q.load_state_dict(self.q.state_dict())

    def update_q(self):
        ''' train the self.q neural network by drawing from replay buffer '''
        for _ in range(self.n_epochs):
            batches = random.sample(self.buffer, self.batch_size*self.n_batches)
            for j in range(self.n_batches):
                batch = batches[j*self.batch_size:(j+1)*self.batch_size]
                samples = list(zip(*batch))
                current_states = torch.stack(samples[0])
                actions = torch.stack(samples[1])
                rewards = torch.stack(samples[2])
                next_states = torch.stack(samples[3])
                terminal_state = torch.stack(samples[4])
                self.train_batch(current_states, actions, rewards, next_states, terminal_state)

    def train_batch(self, current_states, actions, rewards, next_states, terminal_state):
        ''' train self.q neural network given a batch '''
        rewards = rewards.unsqueeze(-1)
        current_states, actions, rewards, next_states, terminal_state = self.to_device([current_states, actions, rewards, next_states, terminal_state])
        next_actions = self.get_action(next_states, greedy=True)
        row_indices = np.arange(next_actions.shape[0])
        with torch.no_grad():
            next_state_q = self.target_q(next_states)
            next_state_q = next_state_q[row_indices, next_actions.squeeze().to(torch.int32)].unsqueeze(-1)
        not_terminal = torch.logical_not(terminal_state)
        targets = (rewards + self.discount * next_state_q * not_terminal).squeeze()
        row_indices = np.arange(actions.shape[0])
        current_state_q = self.q(current_states)[row_indices, actions.squeeze().to(torch.int32)]
        loss = self.loss_fn(current_state_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients: torch.nn.utils.clip_grad_value_(self.q.parameters(), 1.)
        self.optimizer.step()

    def to_device(self, list):
        device_var = []
        for var in list:
            var = var.to(self.device)
            device_var.append(var)
        return device_var

    def check_obs(self, obs:torch.Tensor|np.ndarray):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, torch.Tensor):
            pass
        else:
            raise ValueError(f'Invalid observation type: {type(obs)}')
        if obs.ndim > 2:
            obs = obs.squeeze()
        if obs.ndim == 1:
            obs = obs.unsqueeze(-1)
        return obs

    def check_reward(self, reward:torch.Tensor|np.ndarray|float):
        if isinstance(reward, float):
            reward = torch.tensor([reward], dtype=torch.float32)
        elif isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32)
        elif isinstance(reward, torch.Tensor):
            pass
        else:
            raise ValueError(f'Invalid reward type: {type(reward)}')
        if reward.ndim > 1:
            reward = reward.squeeze()
        if reward.ndim == 0:
            reward = reward.unsqueeze(-1)
        if reward.ndim != 1:
            raise ValueError(f'Reward cannot be reshaped to 1D: {reward.shape}')
        return reward

class RDQN(DQN):

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        discount: float,
        N_nu: int,
        sinkhorn_dist: float,
        delta: float,
        lamda_init: float,
        lamda_lr: float,
        lamda_max_iter: int,
        lamda_step_size: int,
        lamda_gamma: float,
        norm_ord: int,
        action_space: torch.Tensor,
        writer: torch.utils.tensorboard.SummaryWriter = None,
        qfunc: Optional[torch.nn.Module] = None,
        epsilon: float = 0.1, # Epsilon greedy
        buffer_max_length: int = int(1e6),
        clone_steps: int = 256,
        train_steps: int = 8,
        batch_size: int = 128,
        n_batches: int = 1,
        n_epochs: int = 1,
        lr: float = 0.001,
        clip_gradients: bool = False,
        loss_fn: nn.Module = nn.MSELoss(),
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
    ):
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            epsilon=epsilon,
            discount=discount,
            qfunc=qfunc,
            lr=lr,
            buffer_max_length=buffer_max_length,
            batch_size=batch_size,
            clone_steps=clone_steps,
            train_steps=train_steps,
            n_batches=n_batches,
            n_epochs=n_epochs,
            clip_gradients=clip_gradients,
            loss_fn=loss_fn,
            device=device,
            seed=seed,
        )
        self.N_nu = N_nu  # Number of samples from nu
        self.sinkhorn_dist = sinkhorn_dist  # Sinkhorn distance function
        self.delta = delta  # Sinkhorn entropy regularization coefficient
        self.lamda_init = lamda_init  # Initial value of lambda in robust optimization
        self.lamda_lr = lamda_lr  # Learning rate for lambda
        self.lamda_max_iter = lamda_max_iter
        self.lamda_step_size = lamda_step_size  # Step size for learning rate scheduler
        self.lamda_gamma = lamda_gamma  # Gamma for learning rate scheduler
        self.norm_ord = norm_ord  # Order of norm for Wasserstein distance
        self.action_space = action_space
        self.writer = writer
        self.q_updates = 0

    def add_to_replay_buffer(self, reward, observation, terminal, info=None):
        ''' add step sequence to buffer '''
        terminal_state = torch.tensor([terminal], dtype=torch.bool).repeat(self.prev_state.shape[0], 1)
        lamda_init = torch.tensor([self.lamda_init], dtype=torch.float32).repeat(self.prev_state.shape[0], 1)
        if info is None:
            values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state, lamda_init))
        else:
            if isinstance(info, tuple):
                values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state, lamda_init, *info))
            else:
                values = list(zip(self.prev_state, self.prev_action, reward, observation, terminal_state, lamda_init, info))
        self.buffer.extend(values)
        while len(self.buffer) > self.buffer_max_length:
            self.buffer.popleft()

    def update_q(self):
        ''' train the self.q neural network by drawing from replay buffer '''
        for _ in range(self.n_epochs):
            idx_list = random.sample(range(len(self.buffer)), self.batch_size*self.n_batches)
            batches = [self.buffer[i] for i in idx_list]
            # batches = random.sample(self.buffer, self.batch_size*self.n_batches)
            for j in range(self.n_batches):
                batch = batches[j*self.batch_size:(j+1)*self.batch_size]
                samples = list(zip(*batch))
                current_states = torch.stack(samples[0])
                actions = torch.stack(samples[1])
                rewards = torch.stack(samples[2])
                next_states = torch.stack(samples[3])
                terminal_state = torch.stack(samples[4])
                lamda_inits = torch.stack(samples[5])
                if len(samples) > 6:
                    info = []
                    for i in range(6, len(samples)):
                        info.append(torch.stack(samples[i]))
                else:
                    info = None
                self.train_batch(current_states, actions, rewards, next_states, terminal_state, lamda_inits, info, idx_list[j*self.batch_size:(j+1)*self.batch_size])

    def process_batch(self, current_states, actions, rewards, next_states, terminal_state, lamda_inits):
        batch_size = current_states.shape[0]
        act_values = self.action_space[actions].to(self.device)
        act_values = act_values.unsqueeze(-1) if actions.ndim == 1 else act_values
        current_states, actions, rewards, next_states, lamda_inits = self.to_device([current_states, actions, rewards, next_states, lamda_inits])
        not_terminal = torch.logical_not(terminal_state).to(self.device)
        return current_states, actions, rewards, next_states, not_terminal, lamda_inits, batch_size, act_values

    def compute_loss_and_update(self, current_states, actions, targets, mask):
        row_indices = np.arange(actions.shape[0])
        current_state_q = self.q(current_states.to(self.device))[row_indices, actions.squeeze().to(torch.int32)]

        if (~mask).any():
            loss = self.loss_fn(current_state_q[mask], targets[mask].to(self.device))
        else:
            loss = self.loss_fn(current_state_q, targets.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_value_(self.q.parameters(), 1.)
        self.optimizer.step()
        return loss

    def cache_lambdas(self, lambdas, batch_size, buffer_indices, mask):
        # cache the lambda value in the buffer
        lambdas_cpu = lambdas.cpu()
        for i in range(batch_size):
            if mask[i]:  # Only cache successful optimizations
                sample = list(self.buffer[buffer_indices[i]])
                sample[5] = lambdas_cpu[i:i+1]  # Update lambda value
                self.buffer[buffer_indices[i]] = tuple(sample)

    def log_indicators(self, rewards, next_states, not_terminal, targets, lambda_iters, lambdas, mask):
        with torch.no_grad():
            standard_q_targets = rewards + self.discount * self.target_q(next_states).max(dim=-1).values * not_terminal.squeeze()
        q_hq_diff = standard_q_targets - targets

        if self.writer is not None:
            self.writer.add_scalar('Lambda/iterations', lambda_iters, self.q_updates)
            self.writer.add_scalar('Lambda/max lambda', lambdas.max(), self.q_updates)
            self.writer.add_scalar('Lambda/min lambda', lambdas.min(), self.q_updates)
            self.writer.add_scalar('Lambda/median lambda', torch.median(lambdas), self.q_updates)
            self.writer.add_scalar('Lambda/ebar_neg', (~mask).sum(), self.q_updates)
            self.writer.add_scalar('HQ/Min_HQ_delta', q_hq_diff.min(), self.q_updates)
            self.writer.add_scalar('HQ/Max_HQ_delta', q_hq_diff.max(), self.q_updates)
            self.writer.add_scalar('HQ/Mean_HQ_delta', q_hq_diff.mean(), self.q_updates)
            self.writer.add_scalar('HQ/mean_HQ_values', targets.mean(), self.q_updates)

class PORDQN(RDQN):
    '''
    Robust DQN class tailored for the MMD simulator environment
    Uses a student-t distribution for sampling distribution nu
    '''

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        discount: float,
        nu_scale: float,
        nu_df: float,
        action_space: torch.Tensor,
        sinkhorn_dist: float = 0.1,
        delta: float = 0.1,
        N_nu: int = 200,
        lamda_init: float = 1.0,
        lamda_lr: float = 0.1,
        lamda_max_iter: int = 250,
        lamda_step_size: int = 100,
        lamda_gamma: float = 0.1,
        norm_ord: int = 1,
        qfunc: Optional[torch.nn.Module] = None,
        epsilon: float = 0.1,
        buffer_max_length: int = int(1e6),
        clone_steps: int = 256,
        train_steps: int = 8,
        batch_size: int = 128,
        n_batches: int = 1,
        n_epochs: int = 1,
        lr: float = 0.001,
        clip_gradients: bool = False,
        loss_fn: nn.Module = nn.MSELoss(),
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
        writer: Optional[Any] = None,
        calibrator: Optional[Any] = None,
    ):
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            epsilon=epsilon,
            discount=discount,
            N_nu=N_nu,
            sinkhorn_dist=sinkhorn_dist,
            delta=delta,
            lamda_init=lamda_init,
            lamda_lr=lamda_lr,
            lamda_max_iter=lamda_max_iter,
            lamda_step_size=lamda_step_size,
            lamda_gamma=lamda_gamma,
            norm_ord=norm_ord,
            action_space=action_space,
            writer=writer,
            qfunc=qfunc,
            lr=lr,
            buffer_max_length=buffer_max_length,
            batch_size=batch_size,
            clone_steps=clone_steps,
            train_steps=train_steps,
            n_batches=n_batches,
            n_epochs=n_epochs,
            clip_gradients=clip_gradients,
            loss_fn=loss_fn,
            device=device,
            seed=seed,
        )

        # Robustness related
        self.nu_scale = nu_scale  # Scale of the student-t distribution
        self.nu_df = nu_df  # Degrees of freedom of the student-t distribution
        self.calibrator = calibrator

        x = np.linspace(0, 1, N_nu + 2)
        self.nu_dist_samples = stdtrit(self.nu_df, x[1:-1])*self.nu_scale
        self.nu_dist_samples = torch.tensor(self.nu_dist_samples, dtype=torch.float32, requires_grad=False)

    def sample_from_nu(self, states, action_values, n_outer, n_inner):
        z = self.nu_dist_samples
        next_states = z.repeat(states.shape[0],1).unsqueeze(1).unsqueeze(-1) # (batch_size, n_outer, n_inner, 1)
        return next_states

    def get_reward_fn(self, interest, cost):
        '''
        interest: exp(r*dt) - 1 (batch_size)
        cost: |change in action| * cost in % (batch_size)
        '''
        def reward_fn(states, action, next_states, interest=interest, cost=cost):
            action = action.unsqueeze(1) # (batch_size, 1, 1)
            n_mc1 = next_states.shape[1]
            n_mc2 = next_states.shape[2]
            next_states = next_states.reshape(next_states.shape[0], n_mc1*n_mc2, next_states.shape[-1])

            non_cash_return = next_states.exp() - 1.
            cash_weight = 1. - action.sum(dim=1, keepdim=True) # (batch_size, 1)
            interest = interest.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1)
            cost = cost.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1)
            portfolio_simple_return = 1. + cash_weight * interest + action * non_cash_return - cost # (batch_size, n_mc1*n_mc2, 1)
            reward = torch.log(portfolio_simple_return)
            reward = reward.reshape(reward.shape[0], n_mc1, n_mc2, 1)
            return reward
        return reward_fn

    def modify_states(self, states, act_values, next_states, nu_states, nu_rewards):
        expanded_states = states.clone()
        expanded_states[:, :59] = states[:, 1:60] # shift history
        expanded_states[:, 61:62] = act_values # update portfolio weight
        expanded_states[:, 62:63] = next_states[:,0,62:63] # update dt
        expanded_states = expanded_states.repeat(self.N_nu, 1, 1).permute(1, 0, 2).unsqueeze(1) # (batch_size, 1, n_inner, 1)
        expanded_states[:,:,:,59:60] = nu_states # update latest log return
        expanded_states[:,:,:,60] = expanded_states[:,:,:,60] + nu_rewards # update reward
        next_states = next_states[:,:,59:60].clone() # reduce to 1D
        return expanded_states, next_states

    def train_batch(self, current_states, actions, rewards, next_states, terminal_state, lamda_inits, info, buffer_indices):
        current_states, actions, rewards, next_states, not_terminal, lamda_inits, batch_size, act_values = self.process_batch(current_states, actions, rewards, next_states, terminal_state, lamda_inits)
        interest = info[0].to(self.device)
        cost = info[1].to(self.device)
        reward_fn = self.get_reward_fn(interest, cost)
        if hasattr(self, "calibrator") and self.calibrator is not None:
            epsilon_batch = self.calibrator.predict_epsilon(current_states, act_values)
        else:
            epsilon_batch = self.sinkhorn_dist

        targets, lambdas, lambda_iters, mask = hq_opt(self.target_q, current_states, act_values, next_states.unsqueeze(1), self.sample_from_nu, reward_fn, epsilon_batch, self.delta, 1, self.N_nu, self.discount, lamda_inits, self.lamda_max_iter, self.lamda_lr, self.lamda_step_size, self.lamda_gamma, self.norm_ord, not_terminal, self.modify_states, self.device)

        loss = self.compute_loss_and_update(current_states, actions, targets, mask)
        self.cache_lambdas(lambdas, batch_size, buffer_indices, mask)
        if self.writer is not None:
            self.log_indicators(rewards, next_states, not_terminal, targets, lambda_iters, lambdas, mask)

        self.q_updates += 1

        return loss

class GUSRDQN(RDQN):
    '''
    Robust DQN class tailored for the gambling environment
    Uses a uniform distribution for sampling distribution nu
    '''

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        discount: float,
        action_space: torch.Tensor,
        neg_reward_factor: float,
        sinkhorn_dist: float = 0.1,
        delta: float = 0.1,
        N_nu: int = 200,
        lamda_init: float = 1.0,
        lamda_lr: float = 0.1,
        lamda_max_iter: int = 250,
        lamda_step_size: int = 100,
        lamda_gamma: float = 0.1,
        norm_ord: int = 1,
        qfunc: Optional[torch.nn.Module] = None,
        epsilon: float = 0.1,
        buffer_max_length: int = int(1e6),
        clone_steps: int = 256,
        train_steps: int = 8,
        batch_size: int = 128,
        n_batches: int = 1,
        n_epochs: int = 1,
        lr: float = 0.001,
        clip_gradients: bool = False,
        loss_fn: nn.Module = nn.MSELoss(),
        device: torch.device = torch.device('cpu'),
        seed: Optional[int] = None,
        writer: Optional[Any] = None,
    ):
        super().__init__(
            obs_dim=obs_dim,
            num_actions=num_actions,
            epsilon=epsilon,
            discount=discount,
            N_nu=N_nu,
            sinkhorn_dist=sinkhorn_dist,
            delta=delta,
            lamda_init=lamda_init,
            lamda_lr=lamda_lr,
            lamda_max_iter=lamda_max_iter,
            lamda_step_size=lamda_step_size,
            lamda_gamma=lamda_gamma,
            norm_ord=norm_ord,
            action_space=action_space,
            writer=writer,
            qfunc=qfunc,
            lr=lr,
            buffer_max_length=buffer_max_length,
            batch_size=batch_size,
            clone_steps=clone_steps,
            train_steps=train_steps,
            n_batches=n_batches,
            n_epochs=n_epochs,
            clip_gradients=clip_gradients,
            loss_fn=loss_fn,
            device=device,
            seed=seed,
        )

        # Environment and policy related
        self.neg_reward_factor = neg_reward_factor
        self.nu_dist_samples = torch.linspace(0, 1, N_nu + 2)[1:-1]

    def sample_from_nu(self, states, action_values, n_outer, n_inner):
        batch_size = states.shape[0]
        next_states = self.nu_dist_samples.repeat(batch_size, n_outer, 1).unsqueeze(-1) # (batch_size, n_outer, n_inner, 1)
        return next_states

    def reward_fn(self, states, action, next_states):
        reward = action.unsqueeze(1).unsqueeze(1) * (next_states - states.unsqueeze(1).unsqueeze(1))
        reward[reward < 0.] = reward[reward < 0.] * self.neg_reward_factor
        return reward

    def train_batch(self, current_states, actions, rewards, next_states, terminal_state, lamda_inits, info, buffer_indices):
        current_states, actions, rewards, next_states, not_terminal, lamda_inits, batch_size, act_values = self.process_batch(current_states, actions, rewards, next_states, terminal_state, lamda_inits)

        targets, lambdas, lambda_iters, mask = hq_opt(self.target_q, current_states, act_values, next_states.unsqueeze(1), self.sample_from_nu, self.reward_fn, self.sinkhorn_dist, self.delta, 1, self.N_nu, self.discount, lamda_inits, self.lamda_max_iter, self.lamda_lr, self.lamda_step_size, self.lamda_gamma, self.norm_ord, not_terminal, device=self.device)

        loss = self.compute_loss_and_update(current_states, actions, targets, mask)
        self.cache_lambdas(lambdas, batch_size, buffer_indices, mask)
        if self.writer is not None:
            self.log_indicators(rewards, next_states, not_terminal, targets, lambda_iters, lambdas, mask)

        self.q_updates += 1

        return loss