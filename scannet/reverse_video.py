import imageio

def main(fpath, save_path):
    video = imageio.mimread(fpath, memtest=False)
    reversed_video = []
    for frame in video:
        reversed_video.append(frame[::-1, ::-1, :])

    imageio.mimsave(save_path, reversed_video)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_path, args.save_path)
