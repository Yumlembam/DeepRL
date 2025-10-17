import gym, time, numpy as np
env = gym.make("CartPole-v0")
obs = env.reset()
if isinstance(obs, tuple): obs = obs[0]   # in case of wrappers

t = time.perf_counter()
steps = 10000
for _ in range(steps):
    obs, r, done, info = env.step(env.action_space.sample())
    if done:
        obs = env.reset()
        if isinstance(obs, tuple): obs = obs[0]
dt = time.perf_counter() - t
print(f"{steps} steps in {dt:.3f}s -> {steps/dt:.0f} steps/s")
