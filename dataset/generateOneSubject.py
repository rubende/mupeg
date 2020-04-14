import os
import numpy as np
import importlib
MuPeG = importlib.import_module('MuPeG')


np.random.seed(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate artificial videos with one subject')
    parser.add_argument('--dataset', type=str, required=True,
                        default="casiab", choices=['casiab', 'tumgaid', 'other'],
                        help="Dataset name. Used tho select metadata and default folder. "
                             "Try 'casiab', 'tumgaid' or 'other'.")
    parser.add_argument('--inputtype', type=str, required=True,
                        choices=['video', 'image'],
                        help="Input type."
                             "Try 'video' or 'image'.")
    parser.add_argument('--datasetdir', type=str, required=False,
                        help='Full path to dataset directory')
    parser.add_argument('--siltdir', type=str, required=False,
                        help='Full path to silhouettes directory')
    parser.add_argument('--idsdir', type=str, requiered=False,
                        help="Id file")
    parser.add_argument('--outputdir', type=str, required=False,
                        help='Full path to output directory')
    parser.add_argument('--background', type=str, required=False,
                        help='Full path to background image')
    parser.add_argument('--videotypes', type=str, nargs='+', required=False,
                        help='Types of videos for augmentation')
    parser.add_argument('--height', type=int, required=False,
                        help='Video height.')
    parser.add_argument('--width', type=int, required=False,
                        help='Video width.')
    parser.add_argument('--framerate', type=int, required=False,
                        help='Video frame rate.')

    script_path = os.path.dirname(os.path.abspath(__file__))

    args = parser.parse_args()
    dataset = args.dataset
    inputtype = args.inputtype
    datasetdir = args.datasetdir
    siltdir = args.siltdir
    idsdir = args.idsdir
    outputdir = args.outputdir
    background = args.background
    videotypes = args.videotypes
    height = args.height
    width = args.width
    framerate = args.framerate

    if dataset == 'casiab':
        datasetdir = script_path + "/casiab/" if datasetdir is None else datasetdir
        siltdir = script_path + "/casiab_silhouettes/" if siltdir is None else siltdir
        idsdir = script_path + "casiab_ids.txt" if idsdir is None else idsdir
        outputdir = script_path + "/mupeg_one_person/" if outputdir is None else outputdir
        background = script_path + "/casiab_background.png" if background is None else background
        videotypes = ["nm-05-090", "nm-06-090"] if videotypes is None else videotypes
        height = 240 if height is None else height
        width = 320 if width is None else width
        framerate = 25 if framerate is None else framerate
    elif dataset == 'tumgaid':
        datasetdir = script_path + "/tumgaid/" if datasetdir is None else datasetdir
        siltdir = script_path + "/tumgaid_silhouettes/" if siltdir is None else siltdir
        idsdir = script_path + "tumgaid_ids.txt" if idsdir is None else idsdir
        outputdir = script_path + "/mupeg_one_person/" if outputdir is None else outputdir
        background = script_path + "/tumgaid_background.png" if background is None else background
        videotypes = ["n05", "n06"] if videotypes is None else videotypes
        height = 480 if height is None else height
        width = 640 if width is None else width
        framerate = 30 if framerate is None else framerate

    else:
        if not all(v is not None for v in [datasetdir, siltdir, outputdir, background, videotypes, height, width, framerate]):
            raise argparse.ArgumentTypeError('If you select "others" in dataset, you need to complete all the input arguments.')


    if inputtype == 'video':
        MuPeG.MuPeGGenerator.generate_one_subject_from_videos(datasetdir, siltdir, idsdir, outputdir, background,
                                                              videotypes, height, width, framerate)
    else:
        MuPeG.MuPeGGenerator.generate_one_subject_from_images(datasetdir, siltdir, idsdir, outputdir, background,
                                                              videotypes, height, width, framerate)






