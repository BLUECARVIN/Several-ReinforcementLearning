import torch
from torch import nn
from torch.distributions import Categorical, MultivariateNormal


class ActorCritic(nn.Module):
    def __init__(self, 
                 state_dim, 
                 action_dim, 
                 n_latent_var, 
                 device='cuda:0', 
                 env_type='discrete',
                 action_std=0.5,
                 **kwargs):
        super(ActorCritic, self).__init__()
        self.device = device
        self.env_type = env_type
        # Actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var[0]),
            nn.Tanh(),
            nn.Linear(n_latent_var[0], n_latent_var[1]),
            nn.Tanh(),
            nn.Linear(n_latent_var[1], action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var[0]),
            nn.Tanh(),
            nn.Linear(n_latent_var[0], n_latent_var[1]),
            nn.Tanh(),
            nn.Linear(n_latent_var[1], 1)
        )
        self.action_layer = self.action_layer.to(device)
        self.value_layer = self.value_layer.to(device)

        if self.env_type == 'continuous':
            self.action_var = torch.full((action_dim, ), action_std*action_std).to(self.device)

    def forward(self):
        raise NotImplemented

    def act(self, state, memory):
        if self.env_type == 'discrete':
            state = torch.from_numpy(state).float().to(self.device)
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)

            action = dist.sample()

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

            return action.item()
        elif self.env_type == 'continuous':
            # state = torch.from_numpy(state).float().to(self.device)
            action_mean = self.action_layer(state)
            conv_mat = torch.diag(self.action_var).to(self.device)

            dist = MultivariateNormal(action_mean, conv_mat)
            action = dist.sample()
            action_logprobs = dist.log_prob(action)

            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))
            
            return action.detach()

    def evaluate(self, state, action):
        # state = torch.from_numpy(state).float().to(self.device)
        if self.env_type == 'discrete':
            action_probs = self.action_layer(state)
            dist = Categorical(action_probs)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()

            state_value = self.value_layer(state)
        
        elif self.env_type == 'continuous':
            action_mean = self.action_layer(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)

            dist = MultivariateNormal(action_mean, cov_mat)

            action_logprobs = dist.log_prob(action)
            dist_entropy = dist.entropy()
            state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy
        

        

