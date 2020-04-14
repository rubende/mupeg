import cv2
import sys
import numpy as np
import os
import glob
import random

def generate_image(image_base, image_silhouettes, mask_silhouettes):
    _, mask_silhouettes = cv2.threshold(mask_silhouettes, 127, 1, cv2.THRESH_BINARY)
    image_silhouettes[:, :, 0] *= mask_silhouettes
    image_silhouettes[:, :, 1] *= mask_silhouettes
    image_silhouettes[:, :, 2] *= mask_silhouettes

    image_base[:, :, 0] *= 1 - mask_silhouettes
    image_base[:, :, 1] *= 1 - mask_silhouettes
    image_base[:, :, 2] *= 1 - mask_silhouettes

    return image_base + image_silhouettes


def generate_one_subject_from_images(datasetdir, siltdir, idsdir, outputdir, background, videotypes, height,
                                     width, framerate):
    ids = np.loadtxt(idsdir).astype(int)  # Id of users used to generate videos

    perm_ids_1 = np.random.permutation(ids)

    folders = sorted([f for f in glob.glob(datasetdir + "*", recursive=True)])

    for i in range(len(perm_ids_1)):
        for j in videotypes:

            matching = [s for s in folders if str(perm_ids_1[i]).zfill(3) in s]

            paths_subject_1 = sorted(
                [f.replace(datasetdir, "") for f in glob.glob(matching[0] + "/" + j + "/*.jpg",
                                                                      recursive=True)])

            out = cv2.VideoWriter(
                outputdir + paths_subject_1[0][:4] + paths_subject_1[0][5:8] + ".mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            for k in range(len(paths_subject_1)):
                image_base1 = cv2.imread(background, 1)  # We store a background image for each subject
                image_silhouettes = cv2.imread(datasetdir + paths_subject_1[k], 1)
                mask_silhouettes = cv2.imread(siltdir + paths_subject_1[k], 0)
                mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)

                out.write(added_image)

            out.release()


def generate_two_subjects_from_images(datasetdir, siltdir, idsdir, outputdir, videotypes_background,
                                    videotypes_foreground, height, width, framerate):

    ids = np.loadtxt(idsdir).astype(int)

    perm_ids_1 = np.random.permutation(ids)

    folders = sorted([f for f in glob.glob(datasetdir + "*", recursive=True)])

    for i in range(len(perm_ids_1)):
        for j in videotypes_background:
            perm_ids_2 = np.random.permutation(ids)
            perm_ids_3 = np.random.permutation(ids)

            offset_ids_2 = 0
            offset_ids_3 = 0

            if perm_ids_1[i] == perm_ids_2[0]:
                offset_ids_2 = offset_ids_2 + 1

            if perm_ids_1[i] == perm_ids_3[0]:
                offset_ids_3 = offset_ids_3 + 1

            if perm_ids_2[0 + offset_ids_2] == perm_ids_3[0 + offset_ids_3]:
                offset_ids_2 = offset_ids_2 + 1


            matching = [s for s in folders if str(perm_ids_1[i]).zfill(3) in s]

            paths_subject_1 = sorted(
                [f.replace(datasetdir, "") for f in glob.glob(matching[0] + "/" + j + "/*.jpg",
                                                                recursive=True)])

            matching = [s for s in folders if str(perm_ids_2[0 + offset_ids_2]).zfill(3) in s]

            paths_subject_2 = sorted(
                [f.replace(datasetdir, "") for f in glob.glob(matching[0] + "/" + videotypes_foreground[0] + "/*.jpg",
                                                              recursive=True)])

            matching = [s for s in folders if str(perm_ids_3[0 + offset_ids_2]).zfill(3) in s]

            paths_subject_3 = sorted(
                [f.replace(datasetdir, "") for f in glob.glob(matching[0] + "/" + videotypes_foreground[1] + "/*.jpg",
                                                              recursive=True)])

            out2 = cv2.VideoWriter(
                outputdir + paths_subject_1[0][:4] + paths_subject_1[0][5:8] + "_" + paths_subject_2[0][:4] +
                paths_subject_2[0][5:8] + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            out3 = cv2.VideoWriter(
                outputdir + paths_subject_1[0][:4] + paths_subject_1[0][5:8] + "_" + paths_subject_3[0][:4] +
                paths_subject_3[0][5:8] + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            for k in range(len(paths_subject_1)):

                image_base1 = cv2.imread(datasetdir + paths_subject_1[k], 1)
                image_base2 = image_base1.copy()

                # With subject 2
                if k < len(paths_subject_2):
                    image_silhouettes = cv2.imread(datasetdir + paths_subject_2[k], 1)
                    mask_silhouettes = cv2.imread(siltdir + paths_subject_2[k], 0)
                    mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                    added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)
                else:
                    added_image = image_base1

                out2.write(added_image)

                # With subject 3
                if k < len(paths_subject_3):
                    image_silhouettes = cv2.imread(datasetdir + paths_subject_3[k], 1)
                    mask_silhouettes = cv2.imread(siltdir + paths_subject_3[k], 0)
                    mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                    added_image = generate_image(image_base2, image_silhouettes, mask_silhouettes)
                else:
                    added_image = image_base2

                out3.write(added_image)

            out2.release()
            out3.release()


def generate_one_subject_from_videos(datasetdir, siltdir, idsdir, outputdir, background, videotypes, height,
                                     width, framerate):

    if not all(
            v is not None for v in
            [datasetdir, siltdir, outputdir, background, videotypes, height, width, framerate]):
        sys.exit("Some variable is none.")

    ids = np.loadtxt(idsdir).astype(int)

    perm_ids_1 = np.random.permutation(ids)

    videos = sorted([f for f in glob.glob(datasetdir + "*", recursive=True)])

    for i in range(len(perm_ids_1)):
        for j in videotypes:

            matching = [s for s in videos if str(perm_ids_1[i]).zfill(3) in s]

            matching = [s for s in matching if j in s]

            vidcap = cv2.VideoCapture(matching)

            out = cv2.VideoWriter(
                outputdir + str(perm_ids_1[i]).zfill(3) + "-" + j + ".mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            k = 0
            success, image = vidcap.read()
            while success:

                if not os.path.isfile(
                        siltdir + str(perm_ids_1[i]).zfill(3) + "-" + j + "/" + str(k).zfill(3) + ".jpg"):
                    added_image = cv2.imread(background, 1)
                else:
                    image_base1 = cv2.imread(background, 1)
                    image_silhouettes = image
                    mask_silhouettes = cv2.imread(
                        siltdir + str(perm_ids_1[i]).zfill(3) + "-" + j + "/" + str(k).zfill(3) + ".jpg", 0)
                    mask_silhouettes = cv2.resize(mask_silhouettes, (width, height), interpolation=cv2.INTER_AREA)
                    added_image = generate_image(image_base1, image_silhouettes, mask_silhouettes)

                out.write(added_image)
                success, image = vidcap.read()
                k = k + 1

            out.release()

def generate_two_subjects_from_videos(datasetdir, siltdir, idsdir, outputdir, videotypes_background,
                                    videotypes_foreground, height, width, framerate):

    if not all(
            v is not None for v in
            [datasetdir, siltdir, outputdir, videotypes_background,
                                    videotypes_foreground, height, width, framerate]):
        sys.exit("Some variable is none.")

    videos_per_subject = len(videotypes_background)

    videos_to_merge = 0

    ids = np.loadtxt(idsdir).astype(int)

    perm_ids_1 = np.random.permutation(ids)

    videos = sorted([f for f in glob.glob(datasetdir + "*", recursive=True)])

    for i in range(len(perm_ids_1)):
        for j in videotypes_background:

            matching = [s for s in videos if str(perm_ids_1[i]).zfill(3) in s]

            paths_subject_1 = [s for s in matching if j in s]


        uniques_paths_subject_1_2 = []
        uniques_paths_subject_1_3 = []
        for l in range(videos_per_subject):
            uniques_paths_subject_1_2.append(random.choice(paths_subject_1[l]))
            uniques_paths_subject_1_3.append(random.choice(paths_subject_1[l]))

        for j in range(videos_per_subject):
            perm_ids_2 = np.random.permutation(ids)
            perm_ids_3 = np.random.permutation(ids)

            offset_ids_2 = 0
            offset_ids_3 = 0

            if (perm_ids_1[i] == perm_ids_2[0:videos_to_merge]).any():
                offset_ids_2 = offset_ids_2 + videos_to_merge
                if (perm_ids_2[offset_ids_2:offset_ids_2 + videos_to_merge] == perm_ids_3[
                                                                               offset_ids_3:offset_ids_3 + videos_to_merge]).any():
                    offset_ids_2 = offset_ids_2 + videos_to_merge

            if (perm_ids_1[i] == perm_ids_3[0:videos_to_merge]).any():
                offset_ids_3 = offset_ids_3 + videos_to_merge
                if (perm_ids_2[offset_ids_2:offset_ids_2 + videos_to_merge] == perm_ids_3[
                                                                               offset_ids_3:offset_ids_3 + videos_to_merge]).any():
                    offset_ids_2 = offset_ids_2 + videos_to_merge

            if (perm_ids_2[offset_ids_2:offset_ids_2 + videos_to_merge] == perm_ids_3[
                                                                           offset_ids_3:offset_ids_3 + videos_to_merge]).any():
                offset_ids_2 = offset_ids_2 + videos_to_merge

            name_2 = ""
            name_3 = ""

            paths_temp_2 = []
            paths_temp_3 = []

            for l in range(videos_to_merge):
                art_vid = videotypes_foreground  # Foreground videos
                random.shuffle(art_vid)

                matching = [s for s in videos if str(perm_ids_2[offset_ids_2 + l]).zfill(3) in s]
                paths_subject_2 = [s for s in matching if art_vid[0] in s]

                matching = [s for s in videos if str(perm_ids_2[offset_ids_3 + l]).zfill(3) in s]
                paths_subject_3 = [s for s in matching if art_vid[1] in s]


                paths_temp_2.append(random.choice(paths_subject_2))
                paths_temp_3.append(random.choice(paths_subject_3))

                name_2 = name_2 + "_" + paths_temp_2[-1][:-4]
                name_3 = name_3 + "_" + paths_temp_3[-1][:-4]

            out2 = cv2.VideoWriter(
                outputdir + uniques_paths_subject_1_2[j][:-4] + name_2 +
                ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            out3 = cv2.VideoWriter(
                outputdir + uniques_paths_subject_1_3[j][:-4] + name_3 +
                "_M.mp4", cv2.VideoWriter_fourcc(*'mp4v'), framerate, (width, height))

            images1_2 = []
            images1_3 = []

            vidcap = cv2.VideoCapture(datasetdir + uniques_paths_subject_1_2[j])
            success, image = vidcap.read()
            while success:
                images1_2.append(image)
                success, image = vidcap.read()

            vidcap = cv2.VideoCapture(datasetdir + uniques_paths_subject_1_3[j])
            success, image = vidcap.read()
            while success:
                images1_3.append(image)
                success, image = vidcap.read()

            for l in range(videos_to_merge):

                images2 = []
                images3 = []

                vidcap = cv2.VideoCapture(datasetdir + paths_temp_2[l])
                success, image = vidcap.read()
                while success:
                    images2.append(image)
                    success, image = vidcap.read()

                vidcap = cv2.VideoCapture(datasetdir + paths_temp_3[l])
                success, image = vidcap.read()
                while success:
                    images3.append(image)
                    success, image = vidcap.read()

                for k in range(len(images1_2)):

                    image_base2 = images1_2[k].copy()
                    # With subject 2
                    if k < len(images2):
                        image_silhouettes = images2[k]
                        mask_silhouettes = cv2.imread(
                            siltdir + paths_temp_2[l][:-4] + "/" + str(k).zfill(3) + ".jpg", 0)
                        if np.all(mask_silhouettes != None):
                            mask_silhouettes = cv2.resize(mask_silhouettes, (width, height),
                                                          interpolation=cv2.INTER_AREA)
                            added_image = generate_image(image_base2, image_silhouettes, mask_silhouettes)
                        else:
                            added_image = image_base2
                    else:
                        added_image = image_base2
                    images1_2[k] = added_image

                for k in range(len(images1_3)):
                    image_base3 = images1_3[k].copy()
                    # With subject 3
                    if k < len(images3):
                        image_silhouettes = images3[k]
                        mask_silhouettes = cv2.imread(
                            siltdir + paths_temp_3[l][:-4] + "/" + str(k).zfill(3) + ".jpg", 0)
                        if np.all(mask_silhouettes != None):
                            mask_silhouettes = cv2.resize(mask_silhouettes, (width, height),
                                                          interpolation=cv2.INTER_AREA)
                            added_image = generate_image(image_base3, cv2.flip(image_silhouettes, 1),
                                                         cv2.flip(mask_silhouettes, 1))
                        else:
                            added_image = image_base3
                    else:
                        added_image = image_base3
                    images1_3[k] = added_image

            for k in range(len(images1_2)):
                out2.write(images1_2[k])

            for k in range(len(images1_3)):
                out3.write(images1_3[k])

            out2.release()
            out3.release()