import os
from argparse import ArgumentParser

# Argument parser.
parser = ArgumentParser()
# data related args.
parser.add_argument('--video_id', type=int, default=0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--audio_file', type=str, default='')


if __name__ == "__main__":
    # Parse argument
    args = parser.parse_args()
    n = args.video_id
    gpu = args.gpu_id
    audio_file = args.audio_file
    _, name = os.path.split(audio_file)
    audio_name, ext = os.path.splitext(name)

    # convert to 25fps
    cmd0 = f'ffmpeg -i ./Data/{n}.mp4 -r 25 -strict 2 ./Data/{n}.mp4'
    os.system(cmd0)

    # extract frames
    cmd1 = f'cd Data/; python extract_frame1.py {n}.mp4'
    os.system(cmd1)

    # 3d recon
    cmd2 = f'cd Deep3DFaceReconstruction/; CUDA_VISIBLE_DEVICES={gpu} python demo_19news.py ../Data/{n}'
    os.system(cmd2)

    # fine-tune audio
    cmd3 = f'cd Audio/code; python train_19news_1.py {n} {gpu}' 
    os.system(cmd3)

    # fine-tune video
    cmd4 = f'cd render-to-video; python train_19news_1.py {n} {gpu}'
    os.system(cmd4)

    # copy audio to corrsponded folder
    cmd5 = f'cp {audio_file} ./Audio/audio'
    os.system(cmd5)

    # synthesis new results
    cmd6 = f'cd Audio/code; python test_personalized.py {audio_name} {n} {gpu}'
    os.system(cmd6)