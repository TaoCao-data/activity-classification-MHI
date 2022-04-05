import cv2
import numpy as np
import os
import matplotlib.pyplot

VID_DIR = "input_videos"
OUT_DIR = "output"

# MHI helper functions
def create_binary_img(video_name, start_frame, end_frame, gaussian_size, sigma, morph_kernal_size, tau_h):
    # video
    video = os.path.join(VID_DIR, video_name)
    # generate images from videos
    img_gen = video_frame_generator(video)

    # morph_kernal_size
    morph_kernal_size = np.ones(morph_kernal_size, dtype = np.uint8)

    # starting img
    img_start = img_gen.__next__()
    # starting img, pre-process
    img_start = cv2.cvtColor(img_start, cv2.COLOR_BGR2GRAY)
    img_start = cv2.GaussianBlur(img_start, gaussian_size, sigma)

    # counter to check starting frame
    count = 0

    while img_start is not None:
        if count == start_frame:

            # next image
            img_next = img_gen.__next__()
            # next image, preprocess
            img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
            img_next = cv2.GaussianBlur(img_next, gaussian_size, sigma)

            # initiate binary img storage
            diffs = []
            next = []

            for i in range(start_frame, end_frame):
                # calculate image difference
                img_diff = np.abs(cv2.subtract(img_next, img_start)) >= tau_h
                img_diff = img_diff.astype(np.uint8)
                img_diff = cv2.morphologyEx(img_diff, cv2.MORPH_OPEN, morph_kernal_size)

                # store images
                next.append(img_next)
                diffs.append(img_diff)

                # update images and counter
                img_start = img_next

                img_next = img_gen.__next__()
                img_next = cv2.cvtColor(img_next, cv2.COLOR_BGR2GRAY)
                img_next = cv2.GaussianBlur(img_next, gaussian_size, sigma)

                count += 1
            break

        # update starting image to the starting frame
        img_start = img_gen.__next__()
        img_start = cv2.cvtColor(img_start, cv2.COLOR_BGR2GRAY)
        img_start = cv2.GaussianBlur(img_start, gaussian_size, sigma)

        count += 1

    return diffs, next

def create_MHI(binary_imgs, tau):
    # initiate MHI matrix
    MHI = np.zeros(binary_imgs[0].shape, dtype=np.float)
    length_binary_imgs = len(binary_imgs)

    # calculate MHI
    for i, binary_img in enumerate(binary_imgs):

        if i == length_binary_imgs:
            break

        mask1 = binary_img == 1
        mask0 = binary_img == 0

        MHI = tau * mask1 + np.clip(np.subtract(MHI, np.ones(MHI.shape)), 0, 255) * mask0

    MHI = MHI.astype(np.uint8)

    return MHI

def create_hu(image):
    # moment pairs
    moment_pairs = [(2,0), (0,2), (1,1), (1,2), (2,1), (2,2), (3,0), (0,3)]

    u_00 = image.sum()
    h, w = image.shape

    x_mean = np.sum(np.arange(w) * image) / u_00
    y_mean = np.sum(np.arange(h).reshape((-1,1)) * image) / u_00

    # initiate empty hu matrix
    central_moments = np.zeros(len(moment_pairs))
    scale_inv_moments = np.zeros(len(moment_pairs))

    for i, (p, q) in enumerate(moment_pairs):
        x_diff = np.arange(w) - x_mean
        y_diff = np.arange(h) - y_mean

        u_pq = np.sum(((y_diff ** q).reshape((-1,1))) * (x_diff ** p) * image)
        v_pq = u_pq / u_00 ** (1+(p+q)/2)

        central_moments[i] = u_pq
        scale_inv_moments[i] = v_pq

    return central_moments, scale_inv_moments

# plot confusion matrix, reference https://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
def plot_confusion_matrix(confusion_matrix, action_list):

    # set up the figure
    matplotlib.pyplot.figure(figsize=(10, 7))
    matplotlib.pyplot.imshow(confusion_matrix, interpolation='nearest', cmap=matplotlib.pyplot.cm.Oranges)
    matplotlib.pyplot.title('Confusion Matrix')
    matplotlib.pyplot.ylabel('True')
    matplotlib.pyplot.xlabel('Predicted')
    matplotlib.pyplot.xticks(np.arange(len(action_list)), action_list)
    matplotlib.pyplot.yticks(np.arange(len(action_list)), action_list)
    matplotlib.pyplot.colorbar()
    # matplotlib.pyplot.tight_layout()

    # plot the figure
    confusion_matrix = (confusion_matrix * 100 / confusion_matrix.sum()).astype(np.uint) / 100.0
    threshold = confusion_matrix.max() / 2.

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            matplotlib.pyplot.text(j, i, confusion_matrix[i, j], horizontalalignment="center",
                    color="white" if confusion_matrix[i, j] > threshold else "black")

    # save the figure
    filename = 'confusion_matrix.png'
    matplotlib.pyplot.savefig(os.path.join(OUT_DIR, filename))

#video helper functions, reference from assignment3
def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)

def video_frame_generator(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)

def mark_location(image, result):

    actions = {'boxing':1, 'handclapping':2, 'handwaving':3, 'jogging':4, 'running':5, 'walking':6}

    color = (205, 0, 0)
    h, w, d = image.shape
    p1 = [int(w/7), int(h/5)]
    p2 = [p1[0]+350, p1[1]+80]

    for key, val in actions.items():
        if val == result:
            txt = key

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, "(predicated:{})".format(txt), (p1[0]-5, p1[1]+20), font, 2.5, color, 1)
    # cv2.imshow('show',image)

def video_create(video_name, res, frame_starts, frame_ends):

    frame_ids = [20, 125, 179, 270, 467, 535]

    fps = 40

    video = os.path.join(VID_DIR, video_name)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = "output/mhi_{}".format(video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    counter_init = 1
    output_counter = counter_init
    save_image_counter = 1

    frame_num = 1

    while image is not None:

        # the result is updated upon the output_counter
        result = res[(output_counter - 1) % len(res)]
        frame_start = frame_starts[(output_counter - 1) % len(res)]
        frame_end = frame_ends[(output_counter - 1) % len(res)]

        if frame_num == frame_end:
            output_counter += 1

        if frame_num >= frame_start and frame_num <= frame_end:
            mark_location(image, result)

        current_id = frame_ids[(save_image_counter - 1) % len(frame_ids)]

        if current_id == frame_num:
            out_str = "Test_frame_output" + "-{}.png".format(current_id)
            save_image(out_str, image)
            save_image_counter += 1

        video_out.write(image)

        image = image_gen.__next__()

        frame_num += 1

    video_out.release()

# classifier class
class Classifier():

    def __init__(self):
        self.classifier = cv2.ml.KNearest_create()

    def train(self, X, y):
        # train the classifier
        self.classifier.train(X, cv2.ml.ROW_SAMPLE, y)

    def cross_validation(self, X, y):
        # initiate confusion matrix
        confusion_matrix = np.zeros((6, 6))

        # leave one out cross validation
        for i in range(X.shape[0]):
            # training set
            train_set = list(range(0, i)) + list(range(i+1, X.shape[0]))
            # train
            self.classifier.train(X[train_set], cv2.ml.ROW_SAMPLE, y[train_set])

            # validation set
            x_validation = np.array([X[i]])
            y_validation = np.array([y[i]])
            # validate
            val, y_pred, neighbors, dists = self.classifier.findNearest(x_validation, 1)

            # confusion matrix
            confusion_matrix[y_validation-1, int(y_pred[0]) - 1] += 1

        # plot confusion matrix
        plot_confusion_matrix(confusion_matrix = confusion_matrix,
            action_list = ['boxing','handclapping','handwaving','jogging','running','walking'])

    def predict(self, X):
        try:
            val, y_pred, neighbors, dists = self.classifier.findNearest(np.array([X]), 1)
        except Exception:
            print ('Error encounted during predictions!')

        y = y_pred[0, 0]
        return y

# main class
class Main():

    def __init__(self):
        # initiate test video
        self.testvideo = "test_video.mp4"

    def process_and_train(self):
        # action names and labels
        act_name_label = [('boxing',1), ('handclapping',2), ('handwaving',3), ('jogging',4), ('running',5), ('walking',6)]

        # starting and ending frames of each action in the training videos, three actions per type of action
        frames = {'boxing':[(0, 36),(36, 72),(72, 108)],
                  'handclapping':[(0, 27),(27, 54),(54, 81)],
                  'handwaving':[(0, 48),(48, 96),(96, 144)],
                  'jogging':[(15, 70),(145, 200),(245, 300)],
                  'running':[(15, 37),(114, 137),(192, 216)],
                  'walking':[(18, 88),(242, 320),(441, 511)]}

        # parameters, to be tuned for each action
        tau_h = [8, 8, 10, 40, 5, 5]
        tau = [50, 30, 50, 50, 30, 10]
        saved_frames = [5, 12, 20]

        # initiate motion history images, motion energy images and  labels
        MHIS = []
        MEIS = []
        y_true = []

        # create MHIS and MEIS
        for i, (act_name, act_label) in enumerate(act_name_label):
            for j, (start_frame, end_frame) in enumerate(frames[act_name]):

                # video file
                video = 'person15_' + str(act_name) + '_d1_uncomp.avi'

                # create binary images
                diffs, next = create_binary_img(video_name = video, start_frame=start_frame, end_frame=end_frame,
                                gaussian_size=(5, 5), sigma=0, morph_kernal_size=(3, 3), tau_h=tau_h[i])

                # create motion history images
                MHI = create_MHI(binary_imgs=diffs, tau=tau[i]).astype(np.float)
                # normalize motion history images
                cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)

                # create motion energy images
                MEI = (255 * MHI > 0).astype(np.uint8)

                # store MHI, MEI, and labels
                MHIS.append(MHI)
                MEIS.append(MEI)
                y_true.append(act_label)

            # save binary images
            for j in saved_frames:
                out_str = "binary_diff" + "-{}-{}.png".format(act_name, j)
                cv2.normalize(diffs[j], diffs[j], 0, 255, cv2.NORM_MINMAX)
                save_image(out_str, diffs[j])

            for j in saved_frames:
                out_str = "binary_img" + "-{}-{}.png".format(act_name, j)
                save_image(out_str, next[j])

            # save MHI and MEI
            out_str = "MHI" + "-{}.png".format(act_name)
            save_image(out_str, MHI)

            out_str = "MEI" + "-{}.png".format(act_name)
            save_image(out_str, MEI)

        # calculate hu moments
        central_moments = []
        scale_inv_moments = []

        for MHI, MEI in zip(MHIS, MEIS):
            c_MHI, s_MHI = create_hu(MHI)
            c_MEI, s_MEI = create_hu(MEI)

            central_moments.append(np.append(c_MHI, c_MEI))
            scale_inv_moments.append(np.append(s_MHI, s_MEI))

        # train the classifier
        x_train = np.array(scale_inv_moments).astype(np.float32)
        y_train = np.array(y_true).astype(np.int)

        self.classifier = Classifier()
        self.classifier.cross_validation(x_train, y_train)
        self.classifier.train(x_train, y_train)

    def process_and_predict(self):

        # # determine starting and ending frame for each action
        # # video
        # video = os.path.join(VID_DIR, self.testvideo)
        # # generate images from videos
        # img_gen = video_frame_generator(video)
        #
        # # starting img
        # img_start = img_gen.__next__()
        #
        # # counter to check starting frame
        # count = 0
        # # selected frame
        # start_frame = 525
        # end_frame = 585
        #
        # while count <= end_frame:
        #     if count >= start_frame:
        #         cv2.imshow('frame', cv2.resize(img_start, (600, 400)))
        #         cv2.waitKey(200)
        #
        #     # update starting image to the selected frame
        #     img_start = img_gen.__next__()
        #     count += 1

        # starting and ending frames of each action
        start_frame = [10, 115, 169, 260, 457, 525]
        length_frame = [100, 53, 73, 120, 58, 60]
        end_frame = [start + length for start, length in zip(start_frame, length_frame)]

        # parameters, to be tuned for each action
        tau_h = [50, 30, 30, 20, 20, 15]
        tau = [100, 53, 73, 60, 60, 60]
        saved_frames = [10,20,30]

        # initiate motion history images, motion energy images and  labels
        MHIS = []
        MEIS = []

        # create MHIS and MEIS
        for i in range(len(start_frame)):

            # create binary images
            diffs, next = create_binary_img(video_name = self.testvideo, start_frame=start_frame[i], end_frame=start_frame[i] + length_frame[i],
                            gaussian_size=(15, 15), sigma=10, morph_kernal_size=(9, 9), tau_h=tau_h[i])

            # create motion history images
            MHI = create_MHI(binary_imgs=diffs, tau=tau[i]).astype(np.float)
            # normalize motion history images
            cv2.normalize(MHI, MHI, 0.0, 255.0, cv2.NORM_MINMAX)

            # create motion energy images
            MEI = (255 * MHI > 0).astype(np.uint8)

            # store MHI, MEI, and labels
            MHIS.append(MHI)
            MEIS.append(MEI)

            # save binary images
            for j in saved_frames:
                out_str = "test_binary_diff" + "-{}-{}.png".format(i, j)
                cv2.normalize(diffs[j], diffs[j], 0, 255, cv2.NORM_MINMAX)
                save_image(out_str, diffs[j])

            for j in saved_frames:
                out_str = "test_binary_img" + "-{}-{}.png".format(i, j)
                save_image(out_str, next[j])

            # save MHI and MEI
            out_str = "test_MHI" + "-{}.png".format(i)
            save_image(out_str, MHI)

            out_str = "test_MEI" + "-{}.png".format(i)
            save_image(out_str, MEI)

        # calculate hu moments
        central_moments = []
        scale_inv_moments = []

        for MHI, MEI in zip(MHIS, MEIS):
            c_MHI, s_MHI = create_hu(MHI)
            c_MEI, s_MEI = create_hu(MEI)

            central_moments.append(np.append(c_MHI, c_MEI))
            scale_inv_moments.append(np.append(s_MHI, s_MEI))

        # predict
        x_test = np.array(scale_inv_moments).astype(np.float32)
        y_test_predict = [self.classifier.predict(x_test_single) for x_test_single in x_test]

        # true = [1, 2, 3, 4, 5, 6]
        # print('true:', true)
        # print('predicted:', y_test_predict)

        # create videos
        video_create(video_name=self.testvideo, res=y_test_predict, frame_starts=start_frame, frame_ends=end_frame)

if __name__=='__main__':
    # initiate classifier
    cls = Main()

    # build and train classifier
    cls.process_and_train()

    # predict with classifier
    cls.process_and_predict()
