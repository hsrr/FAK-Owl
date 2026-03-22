from header import *
from datasets import *
from model import *
from config import *
import yaml


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--config', default='./fake_config/train_guardian.yaml')
    parser.add_argument('--model', default='openllama_peft', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', default='./ckpt/train_DGM4/', type=str)
    parser.add_argument('--log_path', default='./ckpt/train_DGM4/log_rest/', type=str)
    parser.add_argument('--seed', default=777, type=int)
    parser.add_argument('--imagebind_ckpt_path', default='/data1/yaxiong/UniMMFakeDet/FAK-Owl/pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth',
                        type=str)
    parser.add_argument('--vicuna_ckpt_path', default='/data1/yaxiong/ckpt/vicuna_ckpt/7b_v0/',
                        type=str)
    parser.add_argument('--delta_ckpt_path', default='/data1/yaxiong/UniMMFakeDet/FAK-Owl/pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt',
                        type=str)
    parser.add_argument('--max_tgt_len', default=1024, type=int)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--patience', default=3, type=int)

    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    os.makedirs(path, exist_ok=True)


def main(**args):
    config_env(args)
    args['ds_config_path'] = f'dsconfig/{args["model"]}_stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])
    best_model_path = os.path.join(args['save_path'], 'best')
    build_directory(best_model_path)

    seed = args['seed'] + torch.distributed.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )

    config = yaml.load(open(args['config'], 'r'), Loader=yaml.Loader)
    train_dataset = create_dataset(config, is_train=True)
    val_dataset = create_dataset(config, is_train=False)

    train_iter, sampler = load_DGM4_dataset(train_dataset, args)
    val_iter = load_DGM4_val_dataset(val_dataset, args)

    args['epochs'] = 12

    length = args['epochs'] * len(train_dataset) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_dataset) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # early stopping state
    patience = args['patience']
    best_val_loss = float('inf')
    epochs_no_improve = 0

    pbar = tqdm(total=length)
    current_step = 0

    for epoch_i in range(args['epochs']):
        if args['local_rank'] == 0:
            print(f'\n=== Epoch {epoch_i + 1}/{args["epochs"]} ===')

        for batch in train_iter:
            agent.train_model(
                batch,
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1

        # validation
        val_loss, val_acc = agent.validate(val_iter)

        if args['local_rank'] == 0:
            print(f'Epoch {epoch_i + 1}: val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}%')
            logging.info(f'Epoch {epoch_i + 1}: val_loss={val_loss:.4f}, val_acc={val_acc * 100:.2f}%')

        agent.save_model(args['save_path'], epoch_i)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            agent.save_model(best_model_path, epoch_i)
            if args['local_rank'] == 0:
                print(f'  -> Best model saved (val_loss={val_loss:.4f})')
        else:
            epochs_no_improve += 1
            if args['local_rank'] == 0:
                print(f'  -> No improvement ({epochs_no_improve}/{patience})')

        if epochs_no_improve >= patience:
            if args['local_rank'] == 0:
                print(f'Early stopping at epoch {epoch_i + 1}')
            break

        torch.distributed.barrier()

    torch.distributed.barrier()
    if args['local_rank'] == 0:
        print(f'Training finished. Best val_loss={best_val_loss:.4f}')


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    args['layers'] = [7, 15, 23, 31]
    main(**args)
