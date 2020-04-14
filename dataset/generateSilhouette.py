import os
import numpy as np
import importlib
SilhouetteDetector = importlib.import_module('SilhouetteDetector')


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
    parser.add_argument('--outputdir', type=str, required=False,
                        help='Full path to output directory')

    script_path = os.path.dirname(os.path.abspath(__file__))

    args = parser.parse_args()
    dataset = args.dataset
    inputtype = args.inputtype
    datasetdir = args.datasetdir
    outputdir = args.outputdir


    if dataset == 'casiab':
        datasetdir = script_path + "/casiab/" if datasetdir is None else datasetdir
        outputdir = script_path + "/casiab_silhouette/" if outputdir is None else outputdir
    elif dataset == 'tumgaid':
        datasetdir = script_path + "/tumgaid/" if datasetdir is None else datasetdir
        outputdir = script_path + "/tumgaid_silhouettes/" if outputdir is None else outputdir

    else:
        if not all(v is not None for v in [datasetdir, outputdir]):
            raise argparse.ArgumentTypeError('If you select "others" in dataset, you need to complete all the input arguments.')


    if inputtype == 'video':
        SilhouetteDetector.MuPeGGenerator.silhouettes_from_videos(datasetdir, outputdir)
    else:
        SilhouetteDetector.silhouettes_from_images(datasetdir, outputdir)






