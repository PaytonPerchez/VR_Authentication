import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from tsai.all import *
import sklearn.metrics as skm
from sklearn.model_selection import KFold

num_of_users = 10
stim_per_user = 20

def load_dataset_tsai(users=np.arange(1, num_of_users + 1, 1), samples=np.arange(0, stim_per_user, 1)):
    is_initialized = False
    dataset = []
    users = list(users)

    for user_id in users:
        data_path = r'User Data\User' + str(user_id) + r'\PupilData'
        for s in samples:
            df = load_valid_pupil_data(data_path + str(s) + '.txt')
            process_data(df, interpolate=True, filter_outliers=True, smooth_data=True, normalize=True)

            # Allocate space for dataset if appropriate
            if is_initialized == False:
                dataset = np.empty((len(users) * len(samples), df.shape[1], df.shape[0]), dtype=float)
                is_initialized = True

            sample_index = (users.index(user_id) * len(samples)) + s
            dataset[sample_index, 0] = df['left sizes'].to_numpy()
            dataset[sample_index, 1] = df['right sizes'].to_numpy()
            dataset[sample_index, 2] = df['R'].to_numpy()
            dataset[sample_index, 3] = df['G'].to_numpy()
            dataset[sample_index, 4] = df['B'].to_numpy()
    return dataset

def kfold_tests():

    X = load_dataset_tsai()
    y = create_labels(randomize_labels(100))
    users = np.arange(num_of_users)

    start = time.time()
    accuracies, FARs, FRRs = leave_one_out(X, y, users, 100, LSTM, valid_size=0.2)
    end = time.time()

    with open(('Model Evaluation\\LSTM kfold.txt'), 'a', encoding="utf-8") as f:
        f.write('Accuracies: ' + str(accuracies) + '\n')
        f.write('FARs: ' + str(FARs) + '\n')
        f.write('FRRs: ' + str(FRRs) + '\n')
        f.write('Average Accuracy = ' + str(sum(accuracies) / len(accuracies)) + '\n')
        f.write('Average FAR      = ' + str(sum(FARs) / len(FARs)) + ' secs\n')
        f.write('Average FRR      = ' + str(sum(FRRs) / len(FRRs)) + ' secs\n')
        f.write('Total time: ' + str(end - start) + '\n')

    start = time.time()
    accuracies, FARs, FRRs = leave_one_out(X, y, users, 100, GRU, valid_size=0.2)
    end = time.time()

    with open(('Model Evaluation\\GRU kfold.txt'), 'a', encoding="utf-8") as f:
        f.write('Accuracies: ' + str(accuracies) + '\n')
        f.write('FARs: ' + str(FARs) + '\n')
        f.write('FRRs: ' + str(FRRs) + '\n')
        f.write('Average Accuracy = ' + str(sum(accuracies) / len(accuracies)) + '\n')
        f.write('Average FAR      = ' + str(sum(FARs) / len(FARs)) + ' secs\n')
        f.write('Average FRR      = ' + str(sum(FRRs) / len(FRRs)) + ' secs\n')
        f.write('Total time: ' + str(end - start) + '\n')

    start = time.time()
    accuracies, FARs, FRRs = leave_one_out(X, y, users, 100, TCN, valid_size=0.2)
    end = time.time()

    with open(('Model Evaluation\\TCN kfold.txt'), 'a', encoding="utf-8") as f:
        f.write('Accuracies: ' + str(accuracies) + '\n')
        f.write('FARs: ' + str(FARs) + '\n')
        f.write('FRRs: ' + str(FRRs) + '\n')
        f.write('Average Accuracy = ' + str(sum(accuracies) / len(accuracies)) + '\n')
        f.write('Average FAR      = ' + str(sum(FARs) / len(FARs)) + ' secs\n')
        f.write('Average FRR      = ' + str(sum(FRRs) / len(FRRs)) + ' secs\n')
        f.write('Total time: ' + str(end - start) + '\n')

    start = time.time()
    accuracies, FARs, FRRs = leave_one_out(X, y, users, 75, ResNet, valid_size=0.2)
    end = time.time()

    with open(('Model Evaluation\\ResNet kfold.txt'), 'a', encoding="utf-8") as f:
        f.write('Accuracies: ' + str(accuracies) + '\n')
        f.write('FARs: ' + str(FARs) + '\n')
        f.write('FRRs: ' + str(FRRs) + '\n')
        f.write('Average Accuracy = ' + str(sum(accuracies) / len(accuracies)) + '\n')
        f.write('Average FAR      = ' + str(sum(FARs) / len(FARs)) + ' secs\n')
        f.write('Average FRR      = ' + str(sum(FRRs) / len(FRRs)) + ' secs\n')
        f.write('Total time: ' + str(end - start) + '\n')

def leave_one_out(X, y, users, epochs, architecture, valid_size=0, kwargs={}):

    tfms  = [None, [Categorize()]]
    batch_size = 64
    kfold = KFold(n_splits=len(users), shuffle=True, random_state=200)
    accuracies = []
    FARs = []
    FRRs = []

    for train, test in kfold.split(users):
        random.seed(300)
        train_indices = generate_sample_indices(train)
        random.shuffle(train_indices)
        random.seed(400)
        test_indices = generate_sample_indices(test)
        random.shuffle(test_indices)
        random.seed(None)
        #train_x = X.take(train_indices, axis=0)
        print('Test User: ' + str(test[0]))

        splits = []
        if (valid_size == 0):
            splits = (train_indices, train_indices)
        else:
            train_y = y.take(train_indices)
            splits = get_splits(train_y, valid_size=valid_size, stratify=True, show_plot=False, random_state=500)
        dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
        dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[batch_size, batch_size * 2], num_workers=0)

        model = create_model(architecture, dls=dls, **kwargs)
        learn = Learner(dls, model, metrics=accuracy)

        start = time.time()
        rate = float(str(learn.lr_find())[len('SuggestedLRs(valley='):][:-1])
        rate_time = time.time() - start
        print('Learning rate time: ' + str(rate_time))
        print("Rate: " + str(rate))

        start = time.time()
        learn.fit_one_cycle(epochs, lr_max=rate)
        train_time = time.time() - start
        print('Training time: ' + str(train_time))

        test_x = X.take(test_indices, axis=0)
        test_y = y.take(test_indices)
        valid_dl = dls.valid
        test_ds = valid_dl.dataset.add_test(test_x, test_y)
        test_dl = valid_dl.new(test_ds)

        test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl, with_decoded=True, save_preds=None, save_targs=None)
        acc = skm.accuracy_score(test_targets, test_preds)
        print(f'Accuracy: {acc:1.6f}')

        N, TP, TN, FP, FN = metrics_multi_common(test_preds, test_targets)

        if (test_y[0] == 0):
            FAR = FP / (TN + FP)
            print('FAR: ' + str(FAR))
            FARs.append(FAR)
        else:
            FRR = FN / (TP + FN)
            print('FRR: ' + str(FRR))
            FRRs.append(FRR)

        accuracies.append(acc)

        with open(('Model Evaluation\\' + str(learn.model).split('\n')[0][:-1] + ' kfold.txt'), 'a', encoding="utf-8") as f:
            f.write(str(train) + ', ' + str(test) + '\n')
            f.write('Learning rate    = ' + str(rate) + '\n')
            f.write('Learning time    = ' + str(rate_time) + '\n')
            f.write('Number of epochs = ' + str(epochs) + '\n')
            f.write('Training time    = ' + str(train_time) + ' secs\n')
            f.write(f'Accuracy         = {acc:1.6f}, Test size = ' + str(N) + '\n')
            f.write('Test predictions = ' + str(test_preds) + '\n')
            f.write('Test targets     = ' + str(test_targets) + '\n')
            f.write('True positive    = ' + str(TP) + ', True negative  = ' + str(TN) + '\n')
            f.write('False positive   = ' + str(FP) + ', False negative = ' + str(FN) + '\n')
            if (test_y[0] == 0):
                f.write('FAR              = ' + str(FAR) + '\n')
            else:
                f.write('FRR              = ' + str(FRR) + '\n')
            f.write('\n')

    print('Accuracies: ' + str(accuracies))
    print('FARs: ' + str(FARs))
    print('FRRs: ' + str(FRRs))

    print('Average Accuracy = ' + str(sum(accuracies) / len(accuracies)))
    print('Average FAR      = ' + str(sum(FARs) / len(FARs)))
    print('Average FRR      = ' + str(sum(FRRs) / len(FRRs)))

    return accuracies, FARs, FRRs

def generate_sample_indices(users=[1]):
    sample_indices = np.empty(len(users) * stim_per_user, dtype=int)
    for i in range(len(users)):
        sample_indices[(i*stim_per_user):((i*stim_per_user)+stim_per_user)] = np.arange(users[i]*stim_per_user,(users[i]*stim_per_user)+stim_per_user, 1)

    return sample_indices

def randomize_labels(seed=None):
    user_labels = np.empty(num_of_users, dtype=int)

    random.seed(seed)
    for i in range(num_of_users):
        user_labels[i] = random.randint(0, 1)
    random.seed(None)

    return user_labels

def create_labels(labels=np.arange(0, num_of_users, 1)):
    sample_labels = np.empty(len(labels) * stim_per_user, dtype=int)

    for i in range(len(labels)):
        sample_labels[(i * stim_per_user):((i + 1) * stim_per_user)] = labels[i]

    return sample_labels

def examine_data(user_id, samples=np.arange(0, stim_per_user, 1), durations=['00:00:24.000'], **kwargs):
    for s in samples:
        #print('Loading...')
        data_path = r'User Data\User' + str(user_id) + r'\PupilData' + str(s) + '.txt'
        data = load_valid_pupil_data(data_path)

        #print('Processing:', end='')
        process_data(data, **kwargs)
        plot_path=r'User Data\User' + str(user_id) + r'\Graphs\PupilData' + str(s)

        #print('\nPlotting:', end='')
        for duration in durations:
            #print(' ' + duration, end='')
            current_path = plot_path + '_' + duration.replace(':', ';') + '.png'
            plot_data(data.loc[:duration, ['left sizes', 'right sizes']], plot_path=current_path, **kwargs)
            #print('[*]', end='')
        print('\nProcessed ' + str(s) + '...')

def process_data(data, interpolate=False, resample=False, filter_outliers=False, smooth_data=False, normalize=False, **kwargs):
    if interpolate:
        #print(' Interpolation', end='')
        # Fill in missing values (interpolate between valid values and forward/back fill missing starting and ending values)
        data['left sizes'] = data['left sizes'].interpolate().ffill().bfill()
        data['right sizes'] = data['right sizes'].interpolate().ffill().bfill()
        #print('[*]', end='')

    if resample:
        #print(' Resample', end='')
        #downsample
        data = data.resample(str(np.float32(100000/120)) + 'ms').mean().interpolate()
        #print('[*]', end='')

    if filter_outliers:
        #print(' FilterOutliers', end='')
        start = '00:00:02.000'
        left_outliers = extract_outliers(data.loc[start:, 'left sizes'])
        right_outliers = extract_outliers(data.loc[start:, 'right sizes'])
        data.loc[start:, 'left sizes'] = data.loc[start:, 'left sizes'][~left_outliers]
        data.loc[start:, 'right sizes'] = data.loc[start:, 'right sizes'][~right_outliers]
        if interpolate:
            data['left sizes'] = data['left sizes'].interpolate().ffill().bfill()
            data['right sizes'] = data['right sizes'].interpolate().ffill().bfill()
        #print('[*]', end='')

    if smooth_data:
        #print(' SmoothData', end='')
        # Use min period to avoid NaN values, small window to preseve detail, and center for accuracy
        data['left sizes'] = data['left sizes'].rolling(window=5, center=True, min_periods=1).mean()
        data['right sizes'] = data['right sizes'].rolling(window=5, center=True, min_periods=1).mean()
        #print('[*]', end='')

    if normalize:
        #print(' Normalize', end='')
        data['left sizes'] = normalize_data(data['left sizes'])
        data['right sizes'] = normalize_data(data['right sizes'])
        #print('[*]', end='')

    return data

def extract_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

def normalize_data(data, min_val=None, max_val=None):
    if min_val is None:
        min_val = data.min()
    if max_val is None:
        max_val = data.max()
    scaling_val = max_val - min_val
    normalized_vals = data.to_numpy() # much faster to replace values in numpy array than data frame
    for i in range(len(normalized_vals)):
        normalized_vals[i] = (normalized_vals[i] - min_val) / scaling_val
    return normalized_vals

def plot_data(data, include_points=False, save_plots=False, display_plots=True, plot_path=None, **kwargs):
    if (display_plots or (save_plots and (plot_path is not None))):
        xs = data.index
        fig, ax = plt.subplots(figsize=(6.5, 4))

        if include_points:
            ax.plot(xs, data['left sizes'], 'o', label=('left sizes'), alpha=0.7)
            ax.plot(xs, data['right sizes'], 'o', label=('right sizes'), alpha=0.7)
        if data['left sizes'].isna().any():
            ax.plot(xs, data['left sizes'].interpolate().ffill().bfill(), label=('left interpolated'), alpha=0.7)
            ax.plot(xs, data['right sizes'].interpolate().ffill().bfill(), label=('left interpolated'), alpha=0.7)
        else:
            ax.plot(xs, data['left sizes'], label=('left interpolated'), alpha=0.7)
            ax.plot(xs, data['right sizes'], label=('right interpolated'), alpha=0.7)

        #ax.set_xlim(0, (data.shape[0] * (1000 / 120)))
        ax.legend(loc='upper right', ncol=2)

        if (save_plots and (plot_path is not None)):
            plt.savefig(plot_path)
        if display_plots:
            plt.show()
        plt.close('all')

def load_valid_pupil_data(path):

    left_sizes = []
    right_sizes = []
    colors = []
    total_timesteps = 0
    
    with open(path, 'r', encoding="utf-8") as f:
        # Remove unnecessary characters
        current_line = f.readline()[len('RGBA('):][:-len(', 1.000)\n')]

        # Extract rgb values
        colors = current_line.split(', ')

        # Get total number of samples
        current_line = f.readline()[len('('):][:-len(')\n')]

        # Get total timestamps, total missing left, and total missing right
        counts = current_line.split(', ')
        total_timesteps = -1 * int(counts[0])

        left_sizes = ma.masked_all(total_timesteps, dtype=float)
        right_sizes = ma.masked_all(total_timesteps, dtype=float)

        prefix_len = len('(')
        suffix_len = len(')\n')

        # Preprocessing step 1: Load data and remove duplicates
        for i in range(total_timesteps):
            # Get timestamp, left pupil size, and right pupil size
            current_line = f.readline()[prefix_len:][:-suffix_len]
            data = current_line.split(', ')

            left_sizes[i] = float(data[1])
            right_sizes[i] = float(data[2])

    prev_left_size = 0
    prev_right_size = 0
    skip_counter = 5
    wait_for_min = False

    # Preprocessing step 1: Load data and remove duplicates
    for i in range(total_timesteps):
        # Get left pupil size, and right pupil size
        left_size = left_sizes[i]
        right_size = right_sizes[i]

        # Only add valid pupil sizes to dataset
        if ((left_size != -1) and (right_size != -1) and
            (prev_left_size != -1) and (prev_right_size != -1) and
            (prev_left_size != left_size) and (prev_right_size != right_size)):

            if (wait_for_min and (left_sizes[i] > prev_left_size)):
                wait_for_min = False

            # Skip n values after blinks to avoid invalid data
            if ((skip_counter < 5) or wait_for_min):
                left_sizes[i] = ma.masked
                right_sizes[i] = ma.masked
                skip_counter += 1
        # Prepare to skip invalid blink values
        elif ((skip_counter >= 5) and ((prev_left_size == -1) or (prev_right_size == -1))):
            skip_counter = 0
            wait_for_min = True
            left_sizes[i] = ma.masked
            right_sizes[i] = ma.masked
        else:
            left_sizes[i] = ma.masked
            right_sizes[i] = ma.masked

        prev_left_size = left_size
        prev_right_size = right_size

    return pd.DataFrame(data={
        'left sizes': left_sizes,
        'right sizes': right_sizes,
        'R': [float(colors[0])] * total_timesteps,
        'G': [float(colors[1])] * total_timesteps,
        'B': [float(colors[2])] * total_timesteps},
        index=pd.to_timedelta(np.arange(0, total_timesteps/120, 1/120), unit='s'))
