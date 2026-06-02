import torchaudio

ROOT = "."

def download_google_speech_commands(root=ROOT):
    dataset = torchaudio.datasets.SPEECHCOMMANDS(
        root=root,
        url="speech_commands_v0.02",
        folder_in_archive="SpeechCommands",
        download=True,
    )

    print("Download complete.")
    print(f"Number of samples: {len(dataset)}")
    print(f"Dataset saved under: {root}/SpeechCommands/speech_commands_v0.02")

if __name__ == "__main__":
    download_google_speech_commands()