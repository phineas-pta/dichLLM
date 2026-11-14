#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
stand-alone script
get value of hidden states to compute logit to mask tokens

adapted from https://huggingface.co/ThakiCloud/Qwen3.8-27B-vi-cjk-suppressed/blob/main/apply_mask.py

Replaces the target rows of lm_head.weight with a large negative multiple of the mean hidden-state μ_h (measured via a forward pass)
W = -α * μ_h / μ_h²     (α=200 for starter)
Do not do W=0: zeroing a row only makes its logit 0, not -∞. The moment every other candidate is negative, 0 becomes the argmax.
"""

import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel

# default download dir: /home/marimo/.cache/huggingface/hub
MODEL_DIR = snapshot_download(repo_id="llmfan46/gemma-4-31B-it-uncensored-heretic")
# MODEL_DIR = snapshot_download(repo_id="Qwen/Qwen3.8-Flash-Next")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
MODEL = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="auto", offload_folder="/home/marimo/.cache")

PROBE_TEXTS = [
	"안녕하세요. 오늘 날씨가 참 좋네요.",
	"이번 분기 실적을 정리해서 보고드리겠습니다.",
	"한국의 사계절은 각각 뚜렷한 특징을 가지고 있습니다.",
	"이 문제를 해결하려면 먼저 원인을 파악해야 합니다.",
	"주말에 뭐 하고 지냈어? 나는 그냥 집에 있었어.",
	"이 기능은 왜 이렇게 동작하나요? 로그를 봐도 이해가 안 됩니다.",
	"こんにちは。今日は本当にいい天気ですね。",
	"今四半期の業績をまとめてご報告いたします。",
	"韓国の四季は、それぞれ明確な特徴を持っています。",
	"この問題を解決するには、まず原因を把握しなければなりません。",
	"週末は何をして過ごした？私はただ家にいただけだよ。",
	"この機能はなぜこのように動作するのですか？ログを見ても理解できません。",
	"สวัสดี วันนี้อากาศดีมากเลยนะ",
	"จะสรุปผลประกอบการของไตรมาสนี้และรายงานให้ทราบ",
	"ฤดูกาลทั้งสี่ของเกาหลีต่างก็มีลักษณะเด่นที่ชัดเจน",
	"หากต้องการแก้ไขปัญหานี้ ต้องหาสาเหตุให้ได้ก่อน",
	"สุดสัปดาห์ทำอะไรมา? ฉันก็แค่อยู่บ้าน",
	"ทำไมฟังก์ชันนี้ถึงทำงานแบบนี้? ดูบันทึกล็อกแล้วก็ยังไม่เข้าใจ",
	"مرحبًا. الطقس جميل جدًّا اليوم.",
	"سأقوم بتجميع نتائج هذا الربع وتقديم تقرير عنها.",
	"تتميز الفصول الأربعة في كوريا بخصائص واضحة لكل منها.",
	"لحل هذه المشكلة، يجب أولًا تحديد السبب.",
	"ماذا فعلت في عطلة نهاية الأسبوع؟ أنا بقيت في المنزل فحسب.",
	"لماذا تعمل هذه الوظيفة بهذه الطريقة؟ لا أستطيع فهم الأمر حتى بعد الاطلاع على السجلات.",
	"Здравствуйте. Сегодня такая прекрасная погода",
	"Я подготовил отчет о результатах деятельности за этот квартал и представлю его вам",
	"Каждое из четырёх времен года в Корее имеет свои чёткие особенности",
	"Чтобы решить эту проблему, сначала нужно выяснить её причину",
	"Чем ты занимался на выходных? Я просто сидел дома",
	"Почему эта функция работает именно так? Даже по логам я не могу понять",
	"नमस्ते। आज मौसम बहुत अच्छा है।",
	"मैं इस तिमाही के परिणामों का सारांश तैयार करूँगा और उन्हें आपके सामने प्रस्तुत करूँगा।",
	"कोरिया के चारों मौसमों में से प्रत्येक की अपनी विशिष्ट विशेषताएँ हैं।",
	"इस समस्या को हल करने के लिए, हमें पहले इसका कारण पहचानना होगा।",
	"आपने सप्ताहांत में क्या किया? मैं तो बस घर पर ही रहा।",
	"यह फ़ीचर ऐसा क्यों काम करता है? लॉग देखने के बाद भी मैं इसका मतलब नहीं समझ पा रहा हूँ।",
	"שלום. מזג האוויר ממש נעים היום.",
	"אכין דוח על תוצאות הרבעון הנוכחי ואגיש אותו.",
	"לכל אחת מארבע עונות השנה בקוריאה יש מאפיינים מובהקים משלה.",
	"כדי לפתור את הבעיה הזו, צריך קודם כל לזהות את הגורם.",
	"מה עשית בסוף השבוע? אני פשוט נשארתי בבית.",
	"למה הפונקציה הזו פועלת ככה? גם כשאני בודק את היומן, אני לא מצליח להבין.",
	"Γεια σας. Ο καιρός είναι πραγματικά υπέροχος σήμερα.",
	"Θα σας υποβάλω την αναφορά με τα αποτελέσματα του τρέχοντος τριμήνου.",
	"Οι τέσσερις εποχές της Κορέας έχουν η καθεμία ξεχωριστά χαρακτηριστικά.",
	"Για να λύσουμε αυτό το πρόβλημα, πρέπει πρώτα να εντοπίσουμε την αιτία.",
	"Τι έκανες το Σαββατοκύριακο; Εγώ απλά έμεινα στο σπίτι.",
	"Γιατί λειτουργεί έτσι αυτή η λειτουργία; Δεν καταλαβαίνω ακόμα και αν κοιτάξω τα αρχεία καταγραφής.",
]

hidden_states = []
with torch.inference_mode():
	for text in PROBE_TEXTS:
		output = MODEL(**TOKENIZER(text, return_tensors="pt").to(MODEL.device), output_hidden_states=True)
		hidden_states.append(output.hidden_states[-1][0].float().mean(dim=0))
μ = torch.stack(hidden_states).mean(dim=0)
μ2 = float(torch.square(μ).sum()) # should be at least ~9000 like the result of original authors
# i got very high value for Gemma4 (~50k) but pretty low for Qwen38 (~5.8k)

# TODO: fail to get logits from Gemma4 to verify whether α is big enough

α = 200.
repl32 = (- α / μ2) * μ
(Path(__file__).parent / "repl32_gemma4.json").write_text(
	json.dumps({"repl32": repl32.tolist()}, ensure_ascii=False, indent="\t"),
	encoding="utf-8"
)
