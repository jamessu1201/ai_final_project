# prompts.py
"""
所有人共用的 prompts 清單
Person C 負責建立並分享給 A 和 B
"""

PROMPTS = [
    # === Category 1: Simple Objects (10) ===
    "a red apple on a wooden table",
    "a steaming cup of coffee with latte art",
    "a vintage camera on a white background",
    "a colorful hot air balloon in the sky",
    "a golden retriever puppy playing with a ball",
    "a stack of colorful books on a shelf",
    "a vase with fresh sunflowers",
    "a butterfly landing on a pink flower",
    "a vintage bicycle leaning against a wall",
    "a lighthouse on a rocky coast at sunset",
    
    # === Category 2: Animals & Creatures (10) ===
    "a majestic lion with a flowing mane",
    "a cute cat wearing sunglasses and a hat",
    "an owl perched on a tree branch at night",
    "a dragon flying over a medieval castle",
    "a panda eating bamboo in a forest",
    "a phoenix rising from flames",
    "a dolphin jumping out of the ocean",
    "a unicorn in a magical forest",
    "a robot dog playing in a park",
    "an astronaut riding a horse on the moon",
    
    # === Category 3: Characters & Portraits (10) ===
    "portrait of a woman with long red hair, oil painting style",
    "a samurai warrior in traditional armor",
    "a cyberpunk hacker in a neon-lit room",
    "a wise old wizard with a long white beard",
    "a steampunk inventor in a workshop",
    "an elegant ballerina mid-dance",
    "a futuristic astronaut floating in space",
    "a viking warrior with a horned helmet",
    "a geisha in a traditional kimono",
    "a superhero in a dynamic action pose",
    
    # === Category 4: Scenes & Landscapes (10) ===
    "a futuristic city at sunset with flying cars",
    "a peaceful zen garden with a koi pond",
    "a cozy cabin in a snowy forest",
    "a bustling Tokyo street at night, neon signs",
    "an ancient temple in a misty jungle",
    "a space station orbiting a colorful nebula",
    "a medieval village market scene",
    "an underwater coral reef with tropical fish",
    "a desert oasis with palm trees",
    "a cyberpunk street scene, raining, neon reflections",
    
    # === Category 5: Artistic Styles (10) ===
    "a mountain landscape in the style of Van Gogh",
    "a cat portrait in the style of Picasso, cubist",
    "a city scene in the style of Studio Ghibli anime",
    "a sunset in watercolor painting style",
    "a forest scene in pixel art style, 8-bit",
    "a portrait in the style of Renaissance painting",
    "a landscape in Japanese ukiyo-e style",
    "a still life in the style of Monet, impressionist",
    "a sci-fi scene in comic book style",
    "a nature scene in minimalist vector art style",
]

CATEGORIES = {
    "simple_objects": list(range(0, 10)),
    "animals": list(range(10, 20)),
    "characters": list(range(20, 30)),
    "scenes": list(range(30, 40)),
    "artistic": list(range(40, 50)),
}

def get_prompt(index):
    """根據 index 取得 prompt"""
    if 0 <= index < len(PROMPTS):
        return PROMPTS[index]
    else:
        raise IndexError(f"Prompt index {index} out of range (0-{len(PROMPTS)-1})")

def get_category(index):
    """根據 index 取得 category"""
    for cat, indices in CATEGORIES.items():
        if index in indices:
            return cat
    return "unknown"

def get_prompts_by_category(category):
    """取得特定 category 的所有 prompts"""
    if category in CATEGORIES:
        indices = CATEGORIES[category]
        return [PROMPTS[i] for i in indices]
    else:
        return []

if __name__ == "__main__":
    # 測試
    print(f" Total prompts: {len(PROMPTS)}")
    print(f"\n Categories: {list(CATEGORIES.keys())}")
    print(f"\n First prompt: {PROMPTS[0]}")
    print(f"   Category: {get_category(0)}")
    
    # 顯示每個 category 的第一個 prompt
    print("\n" + "="*60)
    print("Sample prompts from each category:")
    print("="*60)
    for cat in CATEGORIES.keys():
        prompts = get_prompts_by_category(cat)
        print(f"\n{cat.upper()}:")
        print(f"  {prompts[0]}")