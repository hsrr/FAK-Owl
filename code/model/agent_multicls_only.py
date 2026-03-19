from header import *


class DeepSpeedAgentMultiClsOnly:
    def __init__(self, model, args):
        super(DeepSpeedAgentMultiClsOnly, self).__init__()
        self.args = args
        self.model = model
        self.load_stage_1_parameters(args["delta_ckpt_path"])

        # Freeze everything first, then only train LoRA + image projection.
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        for _, param in self.model.llama_proj.named_parameters():
            param.requires_grad = True

        for name, param in self.model.llama_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters configured for multiclass-only training.")

        ds_params = json.load(open(self.args["ds_config_path"]))
        # Compatibility shim for newer deepspeed+pydantic validation:
        # - fp16.opt_level is no longer accepted
        # - bf16 may require `enabled` instead of legacy `enable`
        fp16_cfg = ds_params.get("fp16", {})
        if isinstance(fp16_cfg, dict):
            fp16_cfg.pop("opt_level", None)
        bf16_cfg = ds_params.get("bf16", {})
        if isinstance(bf16_cfg, dict) and "enable" in bf16_cfg and "enabled" not in bf16_cfg:
            bf16_cfg["enabled"] = bf16_cfg.pop("enable")
        ds_params["scheduler"]["params"]["total_num_steps"] = self.args["total_steps"]
        ds_params["scheduler"]["params"]["warmup_num_steps"] = max(
            10, int(self.args["total_steps"] * self.args["warmup_rate"])
        )
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=trainable_params,
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args),
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
        pbar.set_description(f"[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}")
        pbar.update(1)
        if self.args["local_rank"] == 0 and self.args["log_path"] and current_step % self.args["logging_step"] == 0:
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(
                f"[!] progress: {round(pbar.n/pbar.total, 5)}; remaining time: {remaining}; "
                f"loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc*100, 2)}"
            )

        mle_acc *= 100
        return mle_acc

    def save_model(self, path, current_step):
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                print(k)
                checkpoint[k] = v
        torch.save(checkpoint, f"{path}/pytorch_model.pt")
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        print(f"[!] save model into {path}")

    def load_stage_1_parameters(self, path):
        delta_ckpt = torch.load(path, map_location=torch.device("cpu"))
        self.model.load_state_dict(delta_ckpt, strict=False)
