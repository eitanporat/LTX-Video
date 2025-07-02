import subprocess


def generate_video_full_pipeline():
    prompt = "A turquoise river flows through a rocky canyon, cascading over a waterfall."

    cmd = [
        "python", "inference.py",
        "--prompt", prompt,
        "--height", "640",
        "--width", "480",
        "--num_frames", "24",
        "--seed", "42",
        "--pipeline_config", "configs/ltxv-2b-0.9.6-dev.yaml",
    ]

    print("Running inference...")
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    generate_video_full_pipeline()