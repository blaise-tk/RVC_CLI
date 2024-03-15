import os
import subprocess
import json

from .configs.config import Config
from .train.extract.preparing_files import generate_config, generate_filelist
from .lib.tools.pretrained_selector import pretrained_selector

from .train.process.model_blender import model_blender
from .train.process.model_information import model_information
config = Config()
current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

# Get TTS Voices
with open(os.path.join("lib", "tools", "tts_voices.json"), "r") as f:
    voices_data = json.load(f)

locales = list({voice["Locale"] for voice in voices_data})


# Infer
def run_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    input_path,
    output_path,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
):
    infer_script_path = os.path.join("rvc", "infer", "infer.py")
    command = [
        "python",
        *map(
            str,
            [
                infer_script_path,
                f0up_key,
                filter_radius,
                index_rate,
                hop_length,
                f0method,
                input_path,
                output_path,
                pth_path,
                index_path,
                split_audio,
                f0autotune,
                rms_mix_rate,
                protect,
                clean_audio,
                clean_strength,
                export_format,
            ],
        ),
    ]
    subprocess.run(command)
    return f"File {input_path} inferred successfully.", output_path


# Batch infer
def run_batch_infer_script(
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    input_folder,
    output_folder,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
):
    infer_script_path = os.path.join("rvc", "infer", "infer.py")

    audio_files = [
        f for f in os.listdir(input_folder) if f.endswith((".mp3", ".wav", ".flac"))
    ]
    print(f"Detected {len(audio_files)} audio files for inference.")

    for audio_file in audio_files:
        if "_output" in audio_file:
            pass
        else:
            input_path = os.path.join(input_folder, audio_file)
            output_file_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_path = os.path.join(
                output_folder,
                f"{output_file_name}_output{os.path.splitext(audio_file)[1]}",
            )
            print(f"Inferring {input_path}...")

        command = [
            "python",
            *map(
                str,
                [
                    infer_script_path,
                    f0up_key,
                    filter_radius,
                    index_rate,
                    hop_length,
                    f0method,
                    input_path,
                    output_path,
                    pth_path,
                    index_path,
                    split_audio,
                    f0autotune,
                    rms_mix_rate,
                    protect,
                    clean_audio,
                    clean_strength,
                    export_format,
                ],
            ),
        ]
        subprocess.run(command)

    return f"Files from {input_folder} inferred successfully."


# TTS
def run_tts_script(
    tts_text,
    tts_voice,
    f0up_key,
    filter_radius,
    index_rate,
    rms_mix_rate,
    protect,
    hop_length,
    f0method,
    output_tts_path,
    output_rvc_path,
    pth_path,
    index_path,
    split_audio,
    f0autotune,
    clean_audio,
    clean_strength,
    export_format,
):
    tts_script_path = os.path.join("rvc", "lib", "tools", "tts.py")
    infer_script_path = os.path.join("rvc", "infer", "infer.py")

    if os.path.exists(output_tts_path):
        os.remove(output_tts_path)

    command_tts = [
        "python",
        tts_script_path,
        tts_text,
        tts_voice,
        output_tts_path,
    ]

    command_infer = [
        "python",
        *map(
            str,
            [
                infer_script_path,
                f0up_key,
                filter_radius,
                index_rate,
                hop_length,
                f0method,
                output_tts_path,
                output_rvc_path,
                pth_path,
                index_path,
                split_audio,
                f0autotune,
                rms_mix_rate,
                protect,
                clean_audio,
                clean_strength,
                export_format,
            ],
        ),
    ]
    subprocess.run(command_tts)
    subprocess.run(command_infer)
    return f"Text {tts_text} synthesized successfully.", output_rvc_path


# Preprocess
def run_preprocess_script(model_name, dataset_path, sampling_rate):
    per = 3.0 if config.is_half else 3.7
    preprocess_script_path = os.path.join("rvc", "train", "preprocess", "preprocess.py")
    command = [
        "python",
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(logs_path, model_name),
                dataset_path,
                sampling_rate,
                per,
            ],
        ),
    ]

    os.makedirs(os.path.join(logs_path, model_name), exist_ok=True)
    subprocess.run(command)
    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(model_name, rvc_version, f0method, hop_length, sampling_rate):
    model_path = os.path.join(logs_path, model_name)
    extract_f0_script_path = os.path.join(
        "rvc", "train", "extract", "extract_f0_print.py"
    )
    extract_feature_script_path = os.path.join(
        "rvc", "train", "extract", "extract_feature_print.py"
    )

    command_1 = [
        "python",
        extract_f0_script_path,
        *map(
            str,
            [
                model_path,
                f0method,
                hop_length,
            ],
        ),
    ]
    command_2 = [
        "python",
        extract_feature_script_path,
        *map(
            str,
            [
                config.device,
                "1",
                "0",
                "0",
                model_path,
                rvc_version,
                "True",
            ],
        ),
    ]
    subprocess.run(command_1)
    subprocess.run(command_2)

    generate_config(rvc_version, sampling_rate, model_path)
    generate_filelist(f0method, model_path, rvc_version, sampling_rate)
    return f"Model {model_name} extracted successfully."


# Train
def run_train_script(
    model_name,
    rvc_version,
    save_every_epoch,
    save_only_latest,
    save_every_weights,
    total_epoch,
    sampling_rate,
    batch_size,
    gpu,
    pitch_guidance,
    pretrained,
    custom_pretrained,
    g_pretrained_path=None,
    d_pretrained_path=None,
):
    f0 = 1 if str(pitch_guidance) == "True" else 0
    latest = 1 if str(save_only_latest) == "True" else 0
    save_every = 1 if str(save_every_weights) == "True" else 0

    if str(pretrained) == "True":
        if str(custom_pretrained) == "False":
            pg, pd = pretrained_selector(f0)[rvc_version][sampling_rate]
        else:
            if g_pretrained_path is None or d_pretrained_path is None:
                raise ValueError(
                    "Please provide the path to the pretrained G and D models."
                )
            pg, pd = g_pretrained_path, d_pretrained_path
    else:
        pg, pd = "", ""

    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [
        "python",
        train_script_path,
        *map(
            str,
            [
                "-se",
                save_every_epoch,
                "-te",
                total_epoch,
                "-pg",
                pg,
                "-pd",
                pd,
                "-sr",
                sampling_rate,
                "-bs",
                batch_size,
                "-g",
                gpu,
                "-e",
                os.path.join(logs_path, model_name),
                "-v",
                rvc_version,
                "-l",
                latest,
                "-c",
                "0",
                "-sw",
                save_every,
                "-f0",
                f0,
            ],
        ),
    ]

    subprocess.run(command)
    run_index_script(model_name, rvc_version)
    return f"Model {model_name} trained successfully."


# Index
def run_index_script(model_name, rvc_version):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        "python",
        index_script_path,
        os.path.join(logs_path, model_name),
        rvc_version,
    ]

    subprocess.run(command)
    return f"Index file for {model_name} generated successfully."


# Model extract
def run_model_extract_script(
    pth_path, model_name, sampling_rate, pitch_guidance, rvc_version, epoch, step
):
    f0 = 1 if str(pitch_guidance) == "True" else 0
    model_extract_script_path = os.path.join(
        "rvc", "train", "process", "extract_small_model.py"
    )
    command = [
        "python",
        model_extract_script_path,
        pth_path,
        model_name,
        sampling_rate,
        f0,
        rvc_version,
        epoch,
        step,
    ]

    subprocess.run(command)
    return f"Model {model_name} extracted successfully."


# Model information
def run_model_information_script(pth_path):
    print(model_information(pth_path))


# Model blender
def run_model_blender_script(model_name, pth_path_1, pth_path_2, ratio):
    message, model_blended = model_blender(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Tensorboard
def run_tensorboard_script():
    tensorboard_script_path = os.path.join(
        "rvc", "lib", "tools", "launch_tensorboard.py"
    )
    command = [
        "python",
        tensorboard_script_path,
    ]
    subprocess.run(command)


# Download
def run_download_script(model_link):
    download_script_path = os.path.join("rvc", "lib", "tools", "model_download.py")
    command = [
        "python",
        download_script_path,
        model_link,
    ]
    subprocess.run(command)
    return f"Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(pretraineds_v1, pretraineds_v2, models, exe):
    prerequisites_script_path = os.path.join(
        "rvc", "lib", "tools", "prerequisites_download.py"
    )
    command = [
        "python",
        prerequisites_script_path,
        *map(
            str,
            [
                "--pretraineds_v1",
                pretraineds_v1,
                "--pretraineds_v2",
                pretraineds_v2,
                "--models",
                models,
                "--exe",
                exe,
            ],
        ),
    ]
    subprocess.run(command)
    return "Prerequisites installed successfully."


# API
def run_api_script(ip, port):
    command = [
        "env/Scripts/uvicorn.exe" if os.name == "nt" else "uvicorn",
        "api:app",
        "--host",
        ip,
        "--port",
        port,
    ]
    subprocess.run(command)