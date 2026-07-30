# ============================================================
# Vision2Drive Evaluation Pipeline
# ============================================================

from metadrive import MetaDriveEnv

from evaluation.configs import EVAL_CONFIG
from evaluation.inference import Vision2DriveInference
from evaluation.recorder import EpisodeRecorder
from evaluation.metrics import summarize_metrics
from evaluation.visualize import EvaluationVisualizer
from evaluation.utils import (
    get_device,
    make_directory,
    print_header,
    print_metrics,
    save_json,
    save_csv,
    seed_everything,
)

from training.rl.observation import process_observation
from training.rl.environment import environment_step


# ============================================================
# MetaDrive Environment
# ============================================================

ENV_CONFIG = {

    "use_render": EVAL_CONFIG["render"],

    "traffic_density": 0.1,

    "num_scenarios": 100,

    "start_seed": EVAL_CONFIG["seed"],

}


# ============================================================
# Evaluation
# ============================================================

def evaluate():

    seed_everything(EVAL_CONFIG["seed"])

    make_directory(EVAL_CONFIG["output_dir"])

    device = get_device()

    print_header("Vision2Drive Evaluation")

    model = Vision2DriveInference(
        checkpoint_path=EVAL_CONFIG["checkpoint_path"],
        device=device,
    )

    model.summary()

    env = MetaDriveEnv(ENV_CONFIG)

    visualizer = EvaluationVisualizer(
        output_dir=EVAL_CONFIG["output_dir"]
    )

    episodes = []

    # --------------------------------------------------------
    # Run Episodes
    # --------------------------------------------------------

    for episode in range(EVAL_CONFIG["episodes"]):

        recorder = EpisodeRecorder()

        observation, info = env.reset()

        observation = process_observation(observation)

        done = False

        step = 0

        while not done:

            action = model.predict(observation)

            observation, reward, terminated, truncated, info = environment_step(
                env,
                action,
            )

            recorder.record(
                reward=reward,
                action=action,
                info=info,
                timestamp=step,
            )

            observation = process_observation(observation)

            step += 1

            done = (
                terminated
                or truncated
                or step >= EVAL_CONFIG["max_steps"]
            )

        episode_data = recorder.get_episode()

        episodes.append(episode_data)

        print(
            f"Episode {episode+1:03d}/{EVAL_CONFIG['episodes']} | "
            f"Reward: {episode_data['reward']:.2f} | "
            f"Success: {episode_data['success']}"
        )

    env.close()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = summarize_metrics(episodes)

    print_metrics(metrics)

    # --------------------------------------------------------
    # Save Results
    # --------------------------------------------------------

    if EVAL_CONFIG["save_json"]:

        save_json(
            metrics,
            f"{EVAL_CONFIG['output_dir']}/metrics.json",
        )

    if EVAL_CONFIG["save_csv"]:

        save_csv(
            metrics,
            f"{EVAL_CONFIG['output_dir']}/metrics.csv",
        )

    # --------------------------------------------------------
    # Generate Figures
    # --------------------------------------------------------

    if EVAL_CONFIG["save_plots"]:

        visualizer.generate(
            episodes,
            metrics,
        )

    print_header("Evaluation Complete")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    evaluate()