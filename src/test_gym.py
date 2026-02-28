import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt


gym.register_envs(ale_py)


def run_visual_test(steps: int = 500) -> None:
    env = gym.make("ALE/Bowling-v5", render_mode="rgb_array")
    obs, info = env.reset()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    frame = env.render()
    image = ax.imshow(frame)
    ax.set_title("ALE/Bowling-v5 random policy")
    ax.axis("off")

    for step in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print("===ACTION===")
        print(f"Step: {step + 1}, Action: {action}")
        print("===REWARD===")
        print(f"Reward: {reward:.2f}")
        print(" ")
        
        frame = env.render()
        image.set_data(frame)
        ax.set_xlabel(f"step={step + 1} reward={reward:.2f}")
        plt.pause(0.01)

        if terminated or truncated:
            obs, info = env.reset()

    env.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual_test()
