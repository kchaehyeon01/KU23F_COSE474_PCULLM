import os
import time
import argparse
import torch
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
# from models.autoencoder import *
from models.pclip_PEncoder import * ## 수정 : autoencoder가 아닌 CLIP을 위한 Point Cloud Encoder 사용
from evaluation import EMD_CD

# 수정 : CLIP 사용
import clip
from PIL import Image
device = "cuda" if torch.cuda.is_available() else "cpu"
model_clip, preprocess = clip.load("ViT-B/32", device=device)
from sklearn import preprocessing

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./logs_PCLIP/pclip_PEncoder/ckpt_0.000000_5000.pt')
parser.add_argument('--categories', type=str_list, default=['all'])
parser.add_argument('--save_dir', type=str, default='./results')
parser.add_argument('--device', type=str, default='cuda')
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='./data/shapenet.hdf5')
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()

synsetid_to_cate = {
    '02691156': 'airplane', '02773838': 'bag', '02801938': 'basket',
    '02808440': 'bathtub', '02818832': 'bed', '02828884': 'bench',
    '02876657': 'bottle', '02880940': 'bowl', '02924116': 'bus',
    '02933112': 'cabinet', '02747177': 'can', '02942699': 'camera',
    '02954340': 'cap', '02958343': 'car', '03001627': 'chair',
    '03046257': 'clock', '03207941': 'dishwasher', '03211117': 'monitor',
    '04379243': 'table', '04401088': 'telephone', '02946921': 'tin_can',
    '04460130': 'tower', '04468005': 'train', '03085013': 'keyboard',
    '03261776': 'earphone', '03325088': 'faucet', '03337140': 'file',
    '03467517': 'guitar', '03513137': 'helmet', '03593526': 'jar',
    '03624134': 'knife', '03636649': 'lamp', '03642806': 'laptop',
    '03691459': 'speaker', '03710193': 'mailbox', '03759954': 'microphone',
    '03761084': 'microwave', '03790512': 'motorcycle', '03797390': 'mug',
    '03928116': 'piano', '03938244': 'pillow', '03948459': 'pistol',
    '03991062': 'pot', '04004475': 'printer', '04074963': 'remote_control',
    '04090263': 'rifle', '04099429': 'rocket', '04225987': 'skateboard',
    '04256520': 'sofa', '04330267': 'stove', '04530566': 'vessel',
    '04554684': 'washer', '02992529': 'cellphone',
    '02843684': 'birdhouse', '02871439': 'bookshelf',
    # '02858304': 'boat', no boat in our dataset, merged into vessels
    # '02834778': 'bicycle', not in our taxonomy
}
label_list = [v for _, v in synsetid_to_cate.items()]

# Logging
save_dir = os.path.join(args.save_dir, 'pclip_PEncoder_%s_%d' % ('_'.join(args.categories), int(time.time())) )
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logger = get_logger('test', save_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Checkpoint
ckpt = torch.load(args.ckpt)
seed_all(ckpt['args'].seed)

# Datasets and loaders
logger.info('Loading datasets...')
test_dset = ShapeNetCore(
    path=args.dataset_path,
    cates=args.categories,
    split='test',
    scale_mode=ckpt['args'].scale_mode
)
test_loader = DataLoader(test_dset, batch_size=args.batch_size, num_workers=0)

# Model
logger.info('Loading model...')
model = PCLIP_PEncoder(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'])

test_loss = []
# all_recons = []
for i, batch in enumerate(tqdm(test_loader)):
    point = batch['pointcloud'].to(args.device) # shape : torch.Size([32, 2048, 3])
    # ref = batch['pointcloud'].to(args.device)
    # shift = batch['shift'].to(args.device)
    # scale = batch['scale'].to(args.device)

    cate = batch['cate'] ## 수정 : category 정보를 함께 이용

    model.eval()
    with torch.no_grad():
        # code = model.encode(ref)
        # recons = model.decode(code, ref.size(1), flexibility=ckpt['args'].flexibility).detach()
        code_point = model.encode(point)
        text = clip.tokenize(cate).to(device)
        code_text = model_clip.encode_text(text)
        lossfn = torch.nn.MSELoss()
        loss = lossfn(code_point, code_text)  # input, target
        test_loss.append(loss.detach().cpu())
    print("test loss :", test_loss)

    # ref = ref * scale + shift
    # recons = recons * scale + shift

    # all_ref.append(ref.detach().cpu())
    # all_recons.append(recons.detach().cpu())
test_loss = np.array(test_loss)
print(test_loss)
print(test_loss.shape)

# all_ref = torch.cat(all_ref, dim=0)
# all_recons = torch.cat(all_recons, dim=0)

# logger.info('Saving point clouds...')
# np.save(os.path.join(save_dir, 'ref.npy'), all_ref.numpy())
# np.save(os.path.join(save_dir, 'out.npy'), all_recons.numpy())

# logger.info('Start computing metrics...')
# metrics = EMD_CD(all_recons.to(args.device), all_ref.to(args.device), batch_size=args.batch_size)
# cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
# logger.info('CD:  %.12f' % cd)
# logger.info('EMD: %.12f' % emd)
