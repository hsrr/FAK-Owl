from header import *

class DeepSpeedAgent:
    
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model
        self.load_stage_1_parameters(args["delta_ckpt_path"])



        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.Cross_Modal_Reason.named_parameters():
            param.requires_grad = True

        for name, param in self.model.Multi_level.named_parameters():
            param.requires_grad = True

        for name, param in self.model.Segmentation_Verification.named_parameters():
            param.requires_grad = True

        for name, param in self.model.Bbox_Verification.named_parameters():
            param.requires_grad = True

        # load config parameters of deepspeed
        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _ , _ = deepspeed.initialize(
            model=self.model, 
            model_parameters=self.model.parameters(),
            config_params=ds_params, 
            dist_init_required=True,
            args=types.SimpleNamespace(**args)
        )

    @torch.no_grad()
    def predict(self, batch):
        self.model.eval()
        string = self.model.generate_one_sample(batch)
        return string

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        if current_step % 100 == 0:
            pbar.set_description(f'[!] step: {current_step}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args['log_path'] and current_step % self.args['logging_step'] == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}')
            
        mle_acc *= 100
        return mle_acc

    @torch.no_grad()
    def validate(self, val_iter):
        self.ds_engine.module.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_count = 0
        for batch in val_iter:
            loss, acc = self.ds_engine.module(batch)
            total_loss += loss.item()
            total_acc += acc
            total_count += 1
        avg_loss = total_loss / max(1, total_count)
        avg_acc = total_acc / max(1, total_count)

        loss_tensor = torch.tensor([avg_loss, avg_acc, total_count],
                                   device=torch.cuda.current_device())
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
        avg_loss = loss_tensor[0].item() / world_size
        avg_acc = loss_tensor[1].item() / world_size

        self.ds_engine.module.train()
        return avg_loss, avg_acc

    def save_model(self, path, current_step):
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v.data.cpu()
        torch.save(checkpoint, f'{path}/pytorch_model.pt')
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')

    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(delta_ckpt, strict=False)
