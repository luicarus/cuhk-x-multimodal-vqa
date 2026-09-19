"""Verified completion masks; no guessed character offsets or image-token truncation."""
from PIL import Image

from cuhkx.config import require
from cuhkx.inference.prompt import build_mcq_prompt, detect_prompt_leakage


def answer_labels(full, mask, prefix, prefix_mask, answer_tokens, eos, max_length):
    require(len(full) == len(mask) and len(prefix) == len(prefix_mask), "mask length mismatch")
    require(all(v in (0,1) for v in [*mask,*prefix_mask]), "invalid attention mask")
    positions = [i for i,v in enumerate(mask) if v]
    before = [v for v,m in zip(prefix,prefix_mask) if m]
    sequence = [full[i] for i in positions]
    require(bool(positions) and positions == list(range(positions[0],positions[-1]+1)), "noncontiguous attention mask")
    require(len(sequence) <= max_length, "sample exceeds max_sequence_length; never truncate visual tokens")
    require(before and sequence[:len(before)] == before, "full/prompt token prefix differs; unsafe completion boundary")
    require(answer_tokens and eos is not None and eos not in answer_tokens, "invalid answer/EOS tokens")
    expected = [*answer_tokens,eos]
    begin, end = len(before), len(before)+len(expected)
    require(sequence[begin:end] == expected, "assistant answer/EOS does not match full tokenization")
    labels = [-100]*len(full)
    for i in range(begin,end):
        labels[positions[i]] = full[positions[i]]
    return labels, sequence[end:]


class CompletionCollator:
    def __init__(self, processor, max_length=4096):
        self.processor, self.max_length = processor, max_length

    def __call__(self, examples):
        import torch
        from qwen_vl_utils import process_vision_info
        all_images, prompts, conversations, owned = [], [], [], []
        try:
            for example in examples:
                qa = example["qa"]
                prompt = build_mcq_prompt(qa)
                require(not detect_prompt_leakage(prompt, qa), "training prompt leakage")
                require(len(example["frame_paths"]) == 4, "IR4 requires four images")
                images = []
                for path in example["frame_paths"]:
                    with Image.open(path) as image:
                        image.load()
                        images.append(image.copy())
                owned.extend(images)
                content = [{"type":"image","image":image,"resized_height":280,"resized_width":280} for image in images]
                user = [{"role":"user","content":content+[{"type":"text","text":prompt}]}]
                prompts.append(self.processor.apply_chat_template(user, tokenize=False, add_generation_prompt=True))
                conversations.append(self.processor.apply_chat_template(user+[{"role":"assistant","content":example["answer"]}], tokenize=False, add_generation_prompt=False))
                visual, video = process_vision_info(user)
                require(video is None, "unexpected video input")
                all_images.extend(visual)
            full = self.processor(text=conversations, images=all_images, padding=True, truncation=False, return_tensors="pt")
            prefix = self.processor(text=prompts, images=all_images, padding=True, truncation=False, return_tensors="pt")
            tokenizer = self.processor.tokenizer
            labels = []
            for i,example in enumerate(examples):
                target, trailing = answer_labels(full["input_ids"][i].tolist(), full["attention_mask"][i].tolist(),
                    prefix["input_ids"][i].tolist(), prefix["attention_mask"][i].tolist(),
                    tokenizer.encode(example["answer"],add_special_tokens=False), tokenizer.eos_token_id, self.max_length)
                require(not tokenizer.decode(trailing,skip_special_tokens=False).strip(), "unexpected text after assistant EOS")
                labels.append(target)
            full["labels"] = torch.tensor(labels, dtype=torch.long)
            return full
        finally:
            for image in owned:
                image.close()
