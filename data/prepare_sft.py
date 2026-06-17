"""
Giai đoạn 2 — Chuẩn bị dữ liệu SFT (Supervised Fine-Tuning).

Tạo dữ liệu instruction-following tương thích với tokenizer của model đã pretrain.

Usage:
    python data/prepare_sft.py                            # dùng dataset có sẵn
    python data/prepare_sft.py --input data/my_sft.jsonl  # custom dataset

Custom JSONL format (mỗi dòng một example):
    {"instruction": "câu hỏi...", "response": "câu trả lời..."}
"""

import argparse
import json
import os
import pickle
import numpy as np


PROMPT_PREFIX = "Human: {instruction}\n\nAssistant: "
PROMPT_FULL   = "Human: {instruction}\n\nAssistant: {response}"


# ─── Built-in SFT dataset (50 cặp instruction-response, dùng ký tự ASCII cơ bản) ───

BUILTIN_SFT_DATA = [
    {"instruction": "What is love?",
     "response": "Love is a deep feeling of affection and care for another person. It brings joy, warmth, and meaning to our lives."},
    {"instruction": "How do you write a good story?",
     "response": "A good story needs compelling characters, a clear conflict, and a satisfying resolution. Start with an interesting premise, develop your characters, and build tension toward a climax."},
    {"instruction": "What makes a great leader?",
     "response": "A great leader inspires others through vision, integrity, and compassion. They listen carefully, make wise decisions, and help their people grow."},
    {"instruction": "Explain what courage means.",
     "response": "Courage is the strength to face fear and act rightly despite danger or uncertainty. It is not the absence of fear, but the decision that something else is more important."},
    {"instruction": "Write a short poem about the sea.",
     "response": "The great sea stretches wide and far,\nBeneath the moon and evening star.\nIts waves roll in with ancient song,\nOf tides and time and tales grown long."},
    {"instruction": "What is wisdom?",
     "response": "Wisdom is the ability to use knowledge and experience to make good judgments. It comes through reflection, learning from mistakes, and caring for what truly matters."},
    {"instruction": "How should one treat others?",
     "response": "Treat others with kindness, respect, and honesty. Consider their feelings as you would your own, and act with fairness and generosity."},
    {"instruction": "What is the purpose of education?",
     "response": "Education opens the mind to new ideas, develops skills, and prepares us to contribute to society. It teaches us to think clearly, question wisely, and grow as people."},
    {"instruction": "Describe a beautiful sunset.",
     "response": "The sky transforms into a canvas of gold and crimson as the sun descends. The horizon glows with warm colors that fade into the deep blue of night while the first stars appear."},
    {"instruction": "What is friendship?",
     "response": "Friendship is a bond of trust, loyalty, and mutual care between people. True friends support each other through difficulty and share joy in times of happiness."},
    {"instruction": "How does one become wise?",
     "response": "Wisdom comes through experience, reflection, and a willingness to learn. Read widely, listen carefully to others, admit your mistakes, and always seek to understand rather than merely to know."},
    {"instruction": "Write a greeting for a new day.",
     "response": "Good morning! May this new day bring you fresh opportunities, clear thinking, and moments of joy. Rise with purpose and meet the world with an open heart."},
    {"instruction": "What is the value of patience?",
     "response": "Patience allows us to endure difficulty without despair and to wait for the right moment. It is the foundation of persistence and brings rewards that haste cannot."},
    {"instruction": "Explain what justice means.",
     "response": "Justice is the fair and equal treatment of all people according to what they deserve. It means upholding truth, punishing wrongdoing, and protecting the rights of every person."},
    {"instruction": "How do you calm a troubled mind?",
     "response": "Take slow breaths and release your worries one by one. Remind yourself that many difficulties pass with time. Seek calm thoughts, a quiet place, and the company of those who care for you."},
    {"instruction": "What does it mean to be honest?",
     "response": "To be honest means to speak the truth and act with integrity, even when it is difficult. It means not deceiving others and being true to your own values and character."},
    {"instruction": "Describe the feeling of hope.",
     "response": "Hope is a warm light within us that shines even in dark times. It is the belief that things can improve and that our efforts matter. Hope gives us strength to persist."},
    {"instruction": "What is the importance of kindness?",
     "response": "Kindness costs little but means everything. A kind word or act can lift a spirit, mend a wound, and build lasting bonds. In a world of hardship, kindness makes life bearable and beautiful."},
    {"instruction": "Write about the beauty of nature.",
     "response": "Nature is full of wonder, from the delicate flower to the mighty mountain. The forest breathes with life, the river sings its ancient song, and the open sky invites us to think beyond ourselves."},
    {"instruction": "What is the meaning of sorrow?",
     "response": "Sorrow is a deep feeling of sadness, often following loss. Though painful, sorrow is a sign of love and connection. It teaches us what matters most and helps us grow in compassion."},
    {"instruction": "How do you find peace of mind?",
     "response": "Peace of mind comes from accepting what you cannot change and working on what you can. Let go of worry, focus on the present moment, and find meaning in simple things."},
    {"instruction": "What is beauty?",
     "response": "Beauty is found in harmony, grace, and the quality that stirs deep feeling within us. It exists in art, in nature, in music, and in the kindness between people."},
    {"instruction": "How should we remember those who have passed?",
     "response": "Remember them with love and gratitude for what they gave us. Speak their names, honor their values, and carry forward the good they brought into the world."},
    {"instruction": "What is the role of music in life?",
     "response": "Music touches what words cannot reach. It expresses joy and sorrow, binds communities, and stirs the deepest feelings in the human heart."},
    {"instruction": "Write a short farewell.",
     "response": "Farewell, dear friend. Though we part now, the bond we share remains. May your journey be safe, your heart be light, and our paths cross again in happier times."},
    {"instruction": "What is the value of hard work?",
     "response": "Hard work builds character, develops skill, and creates lasting achievement. The rewards it brings are earned and therefore precious. Through honest effort we become capable of more than we thought."},
    {"instruction": "Describe the feeling of joy.",
     "response": "Joy is a bright warmth that fills the heart and lifts the spirit. It is the delight of a child at play, the satisfaction of work well done, and the happiness of being loved and understood."},
    {"instruction": "What makes life meaningful?",
     "response": "Life is made meaningful through love, purpose, and connection. When we care for others, pursue what we believe in, and contribute to something greater than ourselves, we find deep fulfillment."},
    {"instruction": "How do you apologize sincerely?",
     "response": "A sincere apology acknowledges the harm done, takes responsibility without excuse, and shows a genuine intention to do better. Say the words honestly and then prove them through your actions."},
    {"instruction": "What is the power of words?",
     "response": "Words carry great power. They can heal or harm, inspire or discourage, create peace or stir conflict. Choose your words with care, for they shape the world around you."},
    {"instruction": "Write about the changing seasons.",
     "response": "Spring brings new life after winter. Summer burns with energy and growth. Autumn turns the world to gold as things prepare to rest. Winter covers all in silence, holding the promise of renewal."},
    {"instruction": "What does it mean to forgive?",
     "response": "To forgive is to release the burden of anger and hurt, not because the wrong was acceptable, but because holding it harms you more. Forgiveness is an act of strength and self-care."},
    {"instruction": "Describe a peaceful morning.",
     "response": "The morning air is still and cool. Birds begin to sing softly as light spreads across the sky. The world is quiet and unhurried, offering a moment of clarity before the day begins."},
    {"instruction": "What is loyalty?",
     "response": "Loyalty is steadfast commitment to a person, cause, or principle. A loyal friend stands with you in difficulty, a loyal person acts with consistency and faithfulness."},
    {"instruction": "How do you overcome fear?",
     "response": "Face your fear gradually and with purpose. Understand it, prepare for what you can control, and take one step forward even when uncertain. Each small act of courage makes the next one easier."},
    {"instruction": "What is dignity?",
     "response": "Dignity is the quality of being worthy of honor and respect. It comes from within and remains even in suffering. To treat others with dignity is to recognize their inherent worth."},
    {"instruction": "Write a simple expression of gratitude.",
     "response": "Thank you for your kindness and care. Your generosity has meant more to me than words can fully express. I am grateful for your presence in my life."},
    {"instruction": "What is the gift of time?",
     "response": "Time is the most precious thing we can give another person. When you offer your time and full attention, you say that they matter more than anything else."},
    {"instruction": "Describe the feeling of wonder.",
     "response": "Wonder is that catch of breath when something beautiful reveals itself. It opens the eyes and quiets the mind, filling us with curiosity and awe at the richness of the world."},
    {"instruction": "What is strength of character?",
     "response": "Strength of character is the ability to act according to your values even under pressure. It is shown in how you treat others when no one is watching and how you keep your word."},
    {"instruction": "How does one earn trust?",
     "response": "Trust is earned through consistent honesty, reliability, and care over time. Keep your promises, speak truthfully, act with integrity, and show that you have the other person's wellbeing at heart."},
    {"instruction": "Write about the sound of rain.",
     "response": "Rain falls with a steady gentle drumming on rooftops and leaves. It fills the air with the fresh scent of earth and washes the world clean. Its rhythm is ancient and calming."},
    {"instruction": "What does home mean to you?",
     "response": "Home is wherever we feel safe, loved, and ourselves. It is not merely a place but a feeling created by the people we cherish and the warmth of belonging."},
    {"instruction": "Explain the importance of honesty.",
     "response": "Honesty is the foundation of all trust. Without it, relationships crumble and societies fail. When we speak and act truthfully, we build something real and lasting among the people around us."},
    {"instruction": "What is silence?",
     "response": "Silence is the space in which we can truly hear, think, and feel. It is not emptiness but a fullness of its own kind, offering rest from noise and a chance to be present with ourselves."},
    {"instruction": "How do you show respect?",
     "response": "Show respect by listening carefully, speaking kindly, honoring differences, and treating others as worthy of consideration. Respect is given freely and does not require that others earn it first."},
    {"instruction": "Write a comforting message for someone in pain.",
     "response": "I see your pain and I am here with you. You are not alone in this. Though things feel dark now, you have strength within you and there are those who care for you deeply."},
    {"instruction": "What is ambition?",
     "response": "Ambition is a strong desire to achieve something meaningful. When guided by good values, it drives us to develop our abilities and contribute to the world."},
    {"instruction": "Describe the stars at night.",
     "response": "The stars scatter across the dark sky like thousands of distant fires. Each one burns with its own light, indifferent to time. Looking up at them fills us with awe at how vast the world truly is."},
    {"instruction": "What is the meaning of sacrifice?",
     "response": "Sacrifice is the willingness to give up something valued for the sake of something more important. It is an act of love and commitment that reveals the depth of a person's character."},
    {"instruction": "How do you keep hope alive in difficult times?",
     "response": "Keep hope alive by focusing on what you can do, not on what you cannot control. Surround yourself with those who uplift you, find small victories to celebrate, and remember that all seasons change."},
]


# ─── Core functions ───────────────────────────────────────────────────────────

def load_vocab(data_dir: str) -> tuple[dict, dict]:
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Vocab metadata not found at {meta_path}.\n"
            "Run: python data/prepare.py  first."
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"]


def encode(text: str, stoi: dict) -> list[int]:
    """Encode text, replacing unknown chars with the most common fallback."""
    fallback = stoi.get(" ", 0)
    return [stoi.get(c, fallback) for c in text]


def tokenize_example(
    instruction: str, response: str, stoi: dict, block_size: int
) -> dict | None:
    """
    Returns {"input_ids": [...], "labels": [...]} where labels is -1 for
    prompt positions and the target token id for response positions.
    Returns None if the example is too long.
    """
    prefix = PROMPT_PREFIX.format(instruction=instruction)
    full = PROMPT_FULL.format(instruction=instruction, response=response)

    prefix_ids = encode(prefix, stoi)
    full_ids = encode(full, stoi)

    if len(full_ids) + 1 > block_size:
        return None   # skip examples that exceed context length

    labels = [-1] * len(prefix_ids) + full_ids[len(prefix_ids):]

    return {"input_ids": full_ids, "labels": labels}


def prepare_sft(
    input_path: str | None = None,
    out_dir: str = "data",
    train_split: float = 0.9,
    block_size: int = 256,
):
    stoi, _ = load_vocab(out_dir)

    # Load raw data
    if input_path is not None:
        print(f"Loading custom SFT data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw = [json.loads(line) for line in f if line.strip()]
    else:
        print("Using built-in SFT dataset")
        raw = BUILTIN_SFT_DATA

    print(f"  {len(raw)} raw examples")

    # Tokenize
    examples = []
    skipped = 0
    for item in raw:
        result = tokenize_example(item["instruction"], item["response"], stoi, block_size)
        if result is None:
            skipped += 1
        else:
            examples.append(result)

    print(f"  {len(examples)} tokenized  |  {skipped} skipped (too long)")

    # Train / val split
    n = int(train_split * len(examples))
    train_examples = examples[:n]
    val_examples = examples[n:]

    train_path = os.path.join(out_dir, "sft_train.pkl")
    val_path = os.path.join(out_dir, "sft_val.pkl")

    with open(train_path, "wb") as f:
        pickle.dump(train_examples, f)
    with open(val_path, "wb") as f:
        pickle.dump(val_examples, f)

    print(f"  Saved {len(train_examples)} train → {train_path}")
    print(f"  Saved {len(val_examples)} val   → {val_path}")
    print("\nDone. Run `python finetune_sft.py` to start SFT.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--block_size", type=int, default=256)
    args = parser.parse_args()
    prepare_sft(args.input, args.out_dir, args.train_split, args.block_size)
