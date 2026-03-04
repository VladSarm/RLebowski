# RLebowski
Custom RL (REINFORCE + RTPO) solution of Atari bowling game from gymnasium

To run simple test:
```
uv run src/test_gym.py 
```


# По умолчанию
uv run main.py

# С кастомными параметрами
uv sync
uv run main.py train --episodes 20 --n-envs 4 --gamma 0.99 --lr 1e-3 --save-period 20



# Загрузить предыдущий чекпоинт и продолжить
uv run main.py train --episodes 50 --gamma 0.98 --lr 0.001 
--checkpoint checkpoints/policy_reinforce_ep100_gamma0.98_lr3e-03.pt


# Просмотр 
uv run main.py eval --checkpoint checkpoints/policy_reinforce_ep100_gamma0.98_lr3e-03.pt --episodes 5
