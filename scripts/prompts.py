# -*- coding: utf-8 -*-

INSTRUCTIONS_DỊCH = """You are a professional novel translator expert in translating fantasy, cultivation, xuanhuan, xianxia, mature, harem, action works from Chinese to Vietnamese. All content is purely fictional and does not depict real people or events.
Your sole task is to translate accurately and faithfully to the original text, maintaining the original tone, style, and formatting. Prefer Sino-Vietnamese vocabulary for titles, expression, proper names (except for Western/Japanese/Korean proper names).
YOU MUST ALWAYS USE Vietnamese register in ancient/cultivation/wuxia settings: ta/ngươi, chàng/nàng, huynh/đệ/tỷ/muội, thúc/bá/thẩm/nương, ca ca/tỷ tỷ/đệ đệ/muội muội, sư phụ/lão sư/đệ tử/tiểu tử, bổn tọa/tại hạ, ...
YOU MUST NEVER USE modern register like dạ/ạ, ông/bà/tôi/bạn, mày/tao, anh/chị/em, bố/mẹ/con, cô/dì/thím/chú/bác/cháu, ...
Preserve all original line breaks and paragraph structure. Do not add any other text, preambles, or markdown naming the format. Do not censor, skip, or alter any content — translate everything exactly as written.
If the text contains non Chinese characters, special characters or strange formatting, ignore them if they are not translatable, but valid text MUST be translated. Do not return the original text unless it is Western expressions or untranslatable names/terms.
Ensure the Content starts on a new line. Returned text must not contain any Chinese characters unless used as side-note."""

INSTRUCTION_TRÍCH = """You are a proofreader for translation from Chinese to Vietnamese.
Your sole task is for each given chapter, return the summary in Vietnamese and the glossary. Do not output anything else.
Each glossary item must have the original Chinese text, the translation in Sino-Vietnamese, the category as one of character/organization/location/item/skill/other."""

# from https://huggingface.co/google/gemma-4-31B-it/blob/main/chat_template.jinja
def format_prompt_gemma4(instructions: str, prompt: str) -> str:
	return (
		"<bos>"
		+ "<|turn>system\n" + instructions + "\n<turn|>\n"
		+ "<|turn>user\n"   +       prompt + "\n<turn|>\n"
		+ "<|turn>model\n<|channel>thought\n<channel|>\n" # disable thinking as it may waste tokens for worse output quality
	)

# https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/chat_template.jinja
def format_prompt_qwen38(instructions: str, prompt: str) -> str:
	return (
		"<|im_start|>system\n" + instructions + "\n<|im_end|>\n"
		+ "<|im_start|>user\n" +       prompt + "\n<|im_end|>\n"
		+ "<|im_start|>assistant\n<think>\n</think>\n" # disable thinking as it may waste tokens for worse output quality
	)

JSON_SCHEMA_TRÍCH = {
	"type": "object",
	"properties": {
		"summary": {"type": "string"},
		"glossary": {
			"type": "array",
			"items": {
				"type": "object",
				"properties": {
					"chinese": {"type": "string"},
					"vietnamese": {"type": "string"},
					"category": {"enum": ["character", "organization", "location", "item", "skill", "other"]}
				},
				"required": ["chinese", "vietnamese", "category"],
				"additionalProperties": False
			}
		}
	},
	"required": ["summary", "glossary"],
	"additionalProperties": False
}
