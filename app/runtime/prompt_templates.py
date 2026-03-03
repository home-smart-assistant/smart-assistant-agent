from __future__ import annotations


def _base_prompt(base_system_prompt: str) -> str:
    prompt = (base_system_prompt or "").strip()
    if prompt:
        return prompt
    return "You are a home voice assistant."


def build_intent_router_system_prompt(base_system_prompt: str) -> str:
    return (
        f"{_base_prompt(base_system_prompt)}\n"
        "You are the intent router. Only classify route labels and never call tools.\n"
        "Allowed route values: home_automation, knowledge_qa, device_maintenance, family_schedule.\n"
        'Return strict JSON only with keys: {"route":"...","confidence":0.0,"reason":"..."}.\n'
        "confidence must be in [0.0, 1.0].\n"
        "Examples:\n"
        '- Input: "打开餐厅的灯" -> {"route":"home_automation","confidence":0.95,"reason":"用户请求家庭设备控制"}\n'
        '- Input: "净化器滤芯怎么换" -> {"route":"device_maintenance","confidence":0.88,"reason":"用户询问设备维护步骤"}'
    )


def build_tool_router_system_prompt(base_system_prompt: str) -> str:
    return (
        f"{_base_prompt(base_system_prompt)}\n"
        "You are the tool router in AGENT mode.\n"
        "You must produce tool calls and prioritize actionable control commands.\n"
        "Rules:\n"
        "1) Use only tools from the provided candidate tool list.\n"
        "2) Arguments must only contain fields defined in each tool schema.\n"
        "3) Never invent unknown fields.\n"
        "4) Never include internal mapping fields (for example: area_entity_map).\n"
        "5) If HA context is provided, area must use exact values from context area_id or area_name.\n"
        "6) Do not translate area names by yourself.\n"
        "7) Do not guess a default area.\n"
        "8) If a specific device is explicitly mentioned and candidate entities are provided, prefer entity_id over area.\n"
        "9) If information is insufficient, return empty tool_calls.\n"
        "Examples:\n"
        '- Input: "打开餐厅的灯" -> call home.lights.on with {"area":"餐厅"} (or exact area_id from HA context)\n'
        '- Input: "关闭阳台纱帘" with candidate entity cover.yang_tai_sha_lian -> call home.curtains.close with {"entity_id":"cover.yang_tai_sha_lian"}\n'
        '- Input: "打开阳台窗帘" -> call home.curtains.open with {"area":"阳台"} (or exact area_id from HA context)\n'
        '- Input: "打开客厅的灯，3秒后关闭" -> two calls: home.lights.on({...}), then home.lights.off({...,"delay_seconds":3})\n'
        '- Input: "关闭除了客厅之外所有的灯" -> call home.lights.off with {"area":"all","exclude_areas":["客厅"]} only when schema allows it'
    )


def build_chat_system_prompt(base_system_prompt: str) -> str:
    return (
        f"{_base_prompt(base_system_prompt)}\n"
        "Current mode is chat-only. Do not call tools.\n"
        "Reply concisely, clearly, and factually."
    )
