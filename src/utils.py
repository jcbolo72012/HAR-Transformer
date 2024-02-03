from src.library import *
from src.data_handling.data_handler import DataHandler
from src.characterDefinitions import getHandwritingCharacterDefinitions


def load_from_mat_file():
    topDirs = ['Datasets']
    dataDirs = ['t5.2019.05.08','t5.2019.11.25','t5.2019.12.09','t5.2019.12.11','t5.2019.12.18',
                't5.2019.12.20','t5.2020.01.06','t5.2020.01.08','t5.2020.01.13','t5.2020.01.15']
    charDef = getHandwritingCharacterDefinitions()
    all_tensors = []
    all_labels = []
    for directory in dataDirs:
        
        mat = f'./{topDirs[0]}/{directory}/singleLetters.mat'
        data = sio.loadmat(mat)
        ctr = 0
        for letter in charDef['charList']:
            t = torch.Tensor(data[f'neuralActivityCube_{letter}'])
            qty = t.shape[0]
            labels = torch.Tensor([ctr]*qty)
            ctr += 1
    #         if t.shape[0] == 27:
            all_tensors.append(t)
            all_labels.append(labels)

    tensor_data = torch.cat(all_tensors, dim=0).double()# .transpose(-1,-2)
    print(tensor_data.shape)
    # tensor_data = np.repeat(tensor_data[..., np.newaxis], 3, -1).transpose(-1,-2).transpose(-2,-3)

    # tensor_data = tensor_data.transpose(-1,0).transpose(-1,-2)
    tensor_labels = torch.cat(all_labels).long()

    return tensor_data, tensor_labels

def get_neural_data():
    torch.cuda.empty_cache()
    data_x, data_y = load_from_mat_file()
    data_x = torch.tensor(data_x)
    data_y_orig, data_y = np.unique(data_y, return_inverse=True)
    n_values = np.max(data_y) + 1
    data_y = np.eye(n_values)[data_y]

    
    data_y = torch.tensor(data_y)

    dh = DataHandler()
    # data_x = torch.concat((train_x, test_x), dim=0).permute(0, 2, 1)
    # data_y = torch.concat((train_y, test_y), dim=0)
    dh.dataset_x = data_x
    dh.dataset_y = data_y
    return dh

def get_data(train_path, test_path):
    torch.cuda.empty_cache()
    train_x, train_y = load_from_tsfile(
        train_path, return_data_type='numpy3d')
    train_x = torch.tensor(train_x)
    train_y_orig, train_y = np.unique(train_y, return_inverse=True)
    n_values = np.max(train_y) + 1
    train_y = np.eye(n_values)[train_y]

    test_x, test_y = load_from_tsfile(
        test_path, return_data_type='numpy3d')
    test_x = torch.tensor(test_x)
    test_y_orig, test_y = np.unique(test_y, return_inverse=True)
    n_values = np.max(test_y) + 1
    test_y = np.eye(n_values)[test_y]
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    dh = DataHandler()
    data_x = torch.concat((train_x, test_x), dim=0).permute(0, 2, 1)
    data_y = torch.concat((train_y, test_y), dim=0)
    dh.dataset_x = data_x
    dh.dataset_y = data_y
    return dh

def get_activation_fn(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu
    raise ValueError(f"Activation should be relu/gelu, not {activation}.")
