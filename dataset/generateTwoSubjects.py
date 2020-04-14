import os
import numpy as np
import importlib
MuPeG = importlib.import_module('MuPeG')


np.random.seed(0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate artificial videos with two subjects')
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
    parser.add_argument('--idsdir', type=str, required=False,
                        help="Id file")
    parser.add_argument('--outputdir', type=str, required=False,
                        help='Full path to output directory')
    parser.add_argument('--videotypes_background', type=str, nargs='+', required=False,
                        help='Types of videos for augmentation background')
    parser.add_argument('--videotypes_foreground', type=str, nargs='+', required=False,
                        help='Types of videos for augmentation foregorund')
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
    videotypes_background = args.videotypes_background
    videotypes_foreground = args.videotypes_foreground
    height = args.height
    width = args.width
    framerate = args.framerate

    if dataset == 'casiab':
        datasetdir = script_path + "/casiab/" if datasetdir is None else datasetdir
        siltdir = script_path + "/casiab_silhouettes/" if siltdir is None else siltdir
        idsdir = script_path + "casiab_ids.txt" if idsdir is None else idsdir
        outputdir = script_path + "/mupeg_one_person/" if outputdir is None else outputdir
        videotypes_background = ["nm-05-090", "nm-06-090", "bg-01-090", "bg-02-090", "cl-01-090", "cl-02-090"]  \
            if videotypes_background is None else videotypes_background
        videotypes_foreground = ["nm-05-090", "nm-06-090"] if videotypes_foreground is None else videotypes_foreground
        height = 240 if height is None else height
        width = 320 if width is None else width
        framerate = 25 if framerate is None else framerate
    elif dataset == 'tumgaid':
        datasetdir = script_path + "/tumgaid/" if datasetdir is None else datasetdir
        siltdir = script_path + "/tumgaid_silhouettes/" if siltdir is None else siltdir
        idsdir = script_path + "tumgaid_ids.txt" if idsdir is None else idsdir
        outputdir = script_path + "/mupeg_one_person/" if outputdir is None else outputdir
        videotypes_background = ["b01", "b02", "n05", "n06", "s01", "s02"] if videotypes_background is None else \
            videotypes_background
        videotypes_foreground = ["n05", "n06"] if videotypes_foreground is None else videotypes_foreground
        height = 480 if height is None else height
        width = 640 if width is None else width
        framerate = 30 if framerate is None else framerate

    else:
        if not all(v is not None for v in [datasetdir, siltdir, outputdir, videotypes_background, videotypes_foreground,
                                           height, width, framerate]):
            raise argparse.ArgumentTypeError('If you select "others" in dataset, you need to complete all the input arguments.')


    if inputtype == 'video':
        MuPeG.MuPeGGenerator.generate_two_subjects_from_videos(datasetdir, siltdir, idsdir, outputdir,
                                                               videotypes_background, videotypes_foreground, height,
                                                               width, framerate)
    else:
        MuPeG.MuPeGGenerator.generate_two_subjects_from_images(datasetdir, siltdir, idsdir, outputdir,
                                                               videotypes_background, videotypes_foreground, height,
                                                               width, framerate)






