from GPT_SoVITS.module.data_utils import TextAudioSpeakerLoader

loader = TextAudioSpeakerLoader(
    {"exp_dir": "logs/mona_voice", "cleaned_text": True},
    version="v2Pro"
)
print(len(loader))