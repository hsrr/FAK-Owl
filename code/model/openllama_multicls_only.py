from header import *
import torch
from transformers import StoppingCriteriaList

from .openllama import (
    OpenLLAMAPEFTModel,
    PROMPT_START,
    StoppingCriteriaSub,
    process_batch_instance,
)


class OpenLLAMAPEFTModelMultiClsOnly(OpenLLAMAPEFTModel):
    """
    Multiclass-only variant:
    - keeps the base image->LLM path
    - removes bbox/segmentation auxiliary objectives from training
    - does not inject forgery auxiliary prompts during generation
    """

    def forward(self, inputs):
        image_paths = inputs["images"]
        img_embeds, _, _, _ = self.encode_image_from_tensor(image_paths)

        output_texts = inputs["texts"]
        input_ids, target_ids, attention_mask = process_batch_instance(
            self.llama_tokenizer, output_texts, self.max_tgt_len
        )
        inputs_embeds, targets, attention_mask = self.prompt_wrap(
            img_embeds, input_ids, target_ids, attention_mask, forgery_embedding=None
        )

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        return loss, gen_acc

    def prepare_generation_embedding(self, inputs, web_demo=False):
        prompt = inputs["prompt"]
        if not inputs["image_paths"]:
            raise ValueError("Multiclass-only generation currently requires image_paths.")

        feature_embeds, _, _, _ = self.encode_image_from_tensor(inputs["image_paths"])
        inputs["modality_embeds"].append(feature_embeds)

        batch_size = feature_embeds.shape[0]
        p_before = PROMPT_START
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_before_embeds = self.llama_model.model.model.embed_tokens(
            p_before_tokens.input_ids
        ).expand(batch_size, -1, -1)

        p_middle = "</Img> "
        p_middle_tokens = self.llama_tokenizer(
            p_middle, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        p_middle_embeds = self.llama_model.model.model.embed_tokens(
            p_middle_tokens.input_ids
        ).expand(batch_size, -1, -1)

        text = prompt + "\n### Assistant:"
        p_after_tokens = self.llama_tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        p_after_embeds = self.llama_model.model.model.embed_tokens(
            p_after_tokens.input_ids
        ).expand(batch_size, -1, -1)

        bos = (
            torch.ones(
                [batch_size, 1],
                dtype=p_before_tokens.input_ids.dtype,
                device=p_before_tokens.input_ids.device,
            )
            * self.llama_tokenizer.bos_token_id
        )
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)
        inputs_embeds = torch.cat(
            [bos_embeds, p_before_embeds, feature_embeds, p_middle_embeds, p_after_embeds],
            dim=1,
        )
        return inputs_embeds

    def generate(self, inputs, web_demo=False):
        input_embeds = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text

    def generate_logits(self, inputs, generation_config, web_demo=False):
        input_embeds = self.prepare_generation_embedding(inputs, web_demo)
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[2277], encounters=1)])

        outputs_all = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs["max_tgt_len"],
            top_p=inputs["top_p"],
            temperature=inputs["temperature"],
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            generation_config=generation_config,
        )
        outputs = outputs_all.sequences
        scores = outputs_all.scores
        output_text = self.llama_tokenizer.decode(outputs[0][:-2], skip_special_tokens=True)
        return output_text, scores
