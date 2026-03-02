import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt

from lebowski_character import draw_lebowski_game, replace_bowler

gym.register_envs(ale_py)

env = gym.make("ALE/Bowling-v5", render_mode="rgb_array", mode=2)
env.reset()

# Маленький спрайт Dude для игры
dude = draw_lebowski_game()

fig, ax = plt.subplots(figsize=(8, 7))
frame = replace_bowler(env.render(), dude)
img = ax.imshow(frame)
ax.axis("off")
ax.set_title("The Dude abides\n↑↓ — move/steer, SPACE — throw, SHIFT+↑↓ — throw+steer, R — reset, Q — quit")


def step(action):
    _, _, terminated, truncated, _ = env.step(action)
    frame = replace_bowler(env.render(), dude)
    img.set_data(frame)
    fig.canvas.draw_idle()
    if terminated or truncated:
        env.reset()


def on_key(event):
    if event.key == "up":
        step(2)       # UP — move or steer mid-roll
    elif event.key == "down":
        step(3)       # DOWN — move or steer mid-roll
    elif event.key == " ":
        step(1)       # FIRE — throw
    elif event.key == "shift+up":
        step(4)       # UPFIRE — throw + steer up
    elif event.key == "shift+down":
        step(5)       # DOWNFIRE — throw + steer down
    elif event.key == "r":
        env.reset()
        frame = replace_bowler(env.render(), dude)
        img.set_data(frame)
        fig.canvas.draw_idle()
    elif event.key == "q":
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)
plt.tight_layout()
plt.show()

env.close()
