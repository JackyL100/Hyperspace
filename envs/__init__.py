from gym.envs.registration import register

register(
    id='Hyperspace-v0',
    entry_point='envs.Hyperspace:HyperSpaceEnv',
    max_episode_steps=2000,
    kwargs={
        'size' : (1000,1000),
        'locations' : [(40, 200), (40, 500), (40, 700)],
        'num_bad_guys' : 3
    }
)