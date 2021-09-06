import glob, nrrd, torch, os, scipy.ndimage, torch.utils.data
import numpy as np
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import ndimage
from torch.utils.data.sampler import Sampler

FULL_TRAINING_ROOT = "../training_set"
EDGES_TRAINING_ROOT = "../training_edges"
EDGES_FULL_IMPLANT_TRAINING_ROOT = "../training_edges_full_implant"

TEST_ROOT = "../test_set_for_participants"
TEST_EDGES_ROOT = "../TEST_edges_full_implant"
ADDITIONAL_TEST_ROOT = "../additional_test_set_for_participants"

MAIN_TRAIN_ROOT = EDGES_FULL_IMPLANT_TRAINING_ROOT

ERODE = 1
CUT_FACE = 95
TAKE_UPPER = 125 + ERODE


def get_largest_cc(volume):
    CC_STRUCT_26 = [[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                    [[1, 1, 1], [1, 1, 1], [1, 1, 1]]]
    labels, num = scipy.ndimage.measurements.label(volume, structure=CC_STRUCT_26)
    numels = ndimage.sum(volume, labels, range(num + 1))
    return np.array(numels == max(numels))[labels]


def filter_implant(prediction, defect):
    prediction[prediction != 0] = 1
    prediction[prediction == defect] = 0
    prediction = ndimage.morphology.binary_opening(prediction)
    return get_largest_cc(prediction)


def plot_loss(training, validation):
    plt.plot(training, 'g', label='Training Loss')
    plt.plot(validation, 'b', label='Validation Loss')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross Entropy')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(MAIN_TRAIN_ROOT, "Plots/LOSS.png"))
    plt.close()


def plot_dice_metrics(ds, bds):
    plt.plot(ds, 'r', label='Dice Score')
    plt.plot(bds, 'b', label='Border Dice Score')
    plt.title('Dice Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(MAIN_TRAIN_ROOT, "Plots/Dice.png"))
    plt.close()


def plot_hd_metrics(hd, hd_95):
    plt.plot(hd, 'r', label='Hausdorff distance')
    plt.plot(hd_95, 'b', label='95th HD percentile')
    plt.title('Hausdorff distance metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Hausdorff distance')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(MAIN_TRAIN_ROOT, "Plots/HD.png"))
    plt.close()


def plot_slice_in_out_truth(data_in, in_pts, data_out, out_pts, data_truth, truth_pts, n_slice, it, loss=0.5, validation=False):
    sin = data_in[:, :, n_slice]
    sout = data_out[:, :, n_slice]
    target = data_truth[:, :, n_slice]
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05)
    fig.set_figheight(6)
    fig.set_figwidth(20)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax1.set_title(f'Input #{in_pts}pts')
    ax2.set_title(f'Output #{out_pts}pts')
    ax3.set_title(f'Target #{truth_pts}pts')

    ax1.imshow(sin)
    ax2.imshow(sout)
    ax3.imshow(target)

    plt.text(550, 3, f"Dice Loss: {loss}", fontsize=16)
    if validation:
        plt.savefig(os.path.join(MAIN_TRAIN_ROOT, "Plots/train_skull" + "VALIDATION" + str(it) + ".png"))
    else:
        plt.savefig(os.path.join(MAIN_TRAIN_ROOT, "Plots/train_skull" + str(it) + ".png"))
    plt.close()


def plot_slice(data, n_slice, axis='z'):
    if axis == 'x':
        slice_array = data[n_slice, :, :]
    elif axis == 'y':
        slice_array = data[:, n_slice, :]
    elif axis == 'z':
        slice_array = data[:, :, n_slice]
    else:
        print("Wrong axis! [x,y,z] supported")
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(f'Skull slice #{n_slice} along {axis}-axis')
    plt.imshow(slice_array)
    ax.set_aspect('equal')
    plt.show()


class InfSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


class CollationAndTransformation:
    def __init__(self, pepper, pepper_value=0.):
        self.pepper = pepper
        self.pepper_value = pepper_value

    def random_pt_sample(self, coords_list):
        crop_coords_list = []
        for coords in coords_list:
            np.random.shuffle(coords)
            sel = coords[: int(len(coords)*(1 - self.pepper_value))]
            crop_coords_list.append(sel)
        return crop_coords_list

    def __call__(self, list_data):
        defective, complete, shape = list(zip(*list_data))
        if self.pepper:
            defective = self.random_pt_sample(defective)
            complete = [np.concatenate((com, defective[idx])) for idx, com in enumerate(complete)]
        return {
            "defective": ME.utils.batched_coordinates(defective),
            "complete": [torch.from_numpy(comp).float() for comp in complete],
            "shape": shape
        }


def get_train_valid_loader(batch_size, shuffle, num_workers, repeat):
    train, valid = get_train_valid_dset(MAIN_TRAIN_ROOT, 4)

    args = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": CollationAndTransformation(False),
            "pin_memory": True,
            "drop_last": False,
            }

    if repeat:
        args["sampler"] = InfSampler(train, shuffle)
    else:
        args["shuffle"] = shuffle
    train_loader = torch.utils.data.DataLoader(train, **args)

    if repeat:
        args["sampler"] = InfSampler(valid, shuffle)
    valid_loader = torch.utils.data.DataLoader(valid, **args)

    return train_loader, valid_loader


def cut_skull(skull, face_by=CUT_FACE, take_upper=TAKE_UPPER):
    take_upper += int(skull.shape[2] * 0.05)
    skull_coords = np.where(skull != 0)
    skull_coords = np.c_[skull_coords[0], skull_coords[1], skull_coords[2]]
    first_nonzeros = np.amin(skull_coords, axis=0)
    last_zeros = np.amax(skull_coords, axis=0)

    # Note that one can also simply zero the region, as done for the face, however it introduces some computational
    # overhead and it was decided to cut only along z-axis to keep plots consistent, the trade off of this is that now
    # the "difference" has to be memorized for padding when you want to convert back to the original volume dimensions.
    c_skull = skull[:, :, :last_zeros[2] + 1]
    c_skull = c_skull[:, :, -take_upper:]
    c_skull[:, :first_nonzeros[1] + face_by, :] = 0

    return c_skull, skull.shape[2] - (last_zeros[2] + 1)


def get_edges(skull, iters=ERODE):
    eroded = ndimage.morphology.binary_erosion(skull, iterations=iters)
    edges = np.logical_xor(skull, eroded)
    edges[:, :, :ERODE] = 0
    return edges


def get_train_valid_dset(train_root, n_valid):
    defective = glob.glob(train_root + '/defective_skull/*.nrrd')
    complete = glob.glob(train_root + '/complete_skull/*.nrrd')
    implant = glob.glob(train_root + '/implant/*.nrrd')

    defective, complete, implant = sorted(defective), sorted(complete), sorted(implant)
    list_train = list(zip(defective[n_valid:], complete[n_valid:], implant[n_valid:]))
    list_valid = list(zip(defective[:n_valid], complete[:n_valid], implant[:n_valid]))
    return CranialDataset(list_train), CranialDataset(list_valid)


def get_testing_dataloader():
    test_data = sorted(glob.glob(TEST_EDGES_ROOT + '/defective_skull/*.nrrd'))
    additional_test = sorted(glob.glob(ADDITIONAL_TEST_ROOT + '*.nrrd'))
    args = {
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": True,
            "drop_last": False,
            }

    test_loader = torch.utils.data.DataLoader(CranialTestDataset(test_data), **args)
    additional = torch.utils.data.DataLoader(CranialTestDataset(additional_test), **args)
    return test_loader, additional


class CranialDataset(torch.utils.data.Dataset):
    def __init__(self, skull_path_triplet):
        self.skull_files = np.array(skull_path_triplet)
        self.cache = {}

    def __len__(self):
        return len(self.skull_files)

    def __getitem__(self, i):
        mirror_x = np.random.randint(2)
        idx = i + len(self.skull_files) if mirror_x else i
        if idx in self.cache:
            return self.cache[idx]
        else:
            def_path = self.skull_files[i][0]
            compl_path = self.skull_files[i][1]
            print(f"M[{mirror_x}] Loading defective {i} from {def_path}")
            defective, _ = nrrd.read(def_path)
            defective, _ = cut_skull(defective)
            if mirror_x:
                defective = defective[::-1, :, :]
            defective_ones = np.where(defective != 0)
            defective_coords = np.c_[defective_ones[0], defective_ones[1], defective_ones[2]]

            print(f"M[{mirror_x}] Loading complete {i} from {compl_path}")
            complete, _ = nrrd.read(compl_path)
            complete, _ = cut_skull(complete)
            if mirror_x:
                complete = complete[::-1, :, :]
            complete_ones = np.where(complete != 0)
            complete_coords = np.c_[complete_ones[0], complete_ones[1], complete_ones[2]]

            print("Loaded Def#", len(defective_coords), " -> Compl#", len(complete_coords))
            res = torch.Size([1, 1, defective.shape[0], defective.shape[1], defective.shape[2]])

            ret = (defective_coords, complete_coords, res)
            self.cache[idx] = ret
            return ret


class CranialTestDataset(torch.utils.data.Dataset):
    def __init__(self, test):
        self.skull_files = np.array(test)
        self.cache = {}

    def __len__(self):
        return len(self.skull_files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            def_path = self.skull_files[idx]
            defective, _ = nrrd.read(def_path)

            orig_res = torch.Size([1, 1, defective.shape[0], defective.shape[1], defective.shape[2] + 1])
            print(f"M[] Loaded defective {idx} from {def_path}")
            defective, dif_from_top = cut_skull(defective)
            defective_ones = np.where(defective > 0)
            defective_coords = np.c_[defective_ones[0], defective_ones[1], defective_ones[2]]

            print("Loaded Def#", len(defective_coords))
            res = torch.Size([1, 1, defective.shape[0], defective.shape[1], defective.shape[2]])
            ret = (ME.utils.batched_coordinates([defective_coords]), res, orig_res, dif_from_top)
            self.cache[idx] = ret
            return ret


def dataset_to_edges(default_root=FULL_TRAINING_ROOT, edges_root=EDGES_FULL_IMPLANT_TRAINING_ROOT, dense_implant=True):
    if os.path.exists(edges_root):
        print("This path is already present")
    else:
        os.mkdir(edges_root)
        os.mkdir(edges_root + '/defective_skull')
        os.mkdir(edges_root + '/complete_skull')
        os.mkdir(edges_root + '/implant')
        os.mkdir(edges_root + '/Plots')

    defective = glob.glob(default_root + '/defective_skull/*.nrrd')
    complete = glob.glob(default_root + '/complete_skull/*.nrrd')
    implant = glob.glob(default_root + '/implant/*.nrrd')

    defective, complete, implant = sorted(defective), sorted(complete), sorted(implant)
    skull_files = list(zip(defective, complete, implant))

    for idx, triplet in enumerate(skull_files):
        filename = str(idx).zfill(3)
        print("Processing ", idx)
        defective, h = nrrd.read(triplet[0])
        defective = get_edges(defective)
        nrrd.write(edges_root + f'/defective_skull/{filename}.nrrd', defective.astype('int32'), h)

        implant, h = nrrd.read(triplet[2])
        if not dense_implant:
            implant = get_edges(implant)
        nrrd.write(edges_root + f'/implant/{filename}.nrrd', implant.astype('int32'), h)
        
        if dense_implant:
            complete = defective + implant
        else:
            complete, h = nrrd.read(triplet[1])
            complete = get_edges(complete)

        nrrd.write(edges_root + f'/complete_skull/{filename}.nrrd', complete.astype('int32'), h)


def test_set_to_edges(test_dir=TEST_ROOT,
                      test_dir_edges=TEST_EDGES_ROOT, additional=True):

    additional_dir = ""
    if additional:
        additional_dir = test_dir_edges + "/Additional"

    test_data = sorted(glob.glob(test_dir + '/*.nrrd'))

    if os.path.exists(test_dir_edges):
        print("Already present")
        return
    else:
        os.mkdir(test_dir_edges)
        os.mkdir(test_dir_edges + '/defective_skull')
        if additional:
            os.mkdir(additional_dir)

    for idx, skull in enumerate(test_data):
        print("Processing ", idx)
        string = str(idx).zfill(3)
        defective, h = nrrd.read(skull)
        def_edges = get_edges(defective).astype('int32')
        nrrd.write(test_dir_edges + f'/defective_skull/{string}.nrrd', def_edges, header=h)

    if additional:
        test_data = sorted(glob.glob(ADDITIONAL_TEST_ROOT + '/*.nrrd'))
        for idx, skull in enumerate(test_data):
            print("Processing ", idx)
            defective, h = nrrd.read(skull)
            def_edges = get_edges(defective).astype('int32')
            nrrd.write(additional_dir + f'/additional{idx}.nrrd', def_edges, header=h)
