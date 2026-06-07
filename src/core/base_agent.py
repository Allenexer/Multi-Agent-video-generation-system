"""
Decoupled BaseAgent — receives model, client, tools via constructor.
No hardcoded model IDs or provider URLs.

Supports multi-turn tool calling: if the LLM requests a tool, the agent
executes it, feeds the result back, and loops until the LLM produces its
final JSON answer.
"""
import json
import base64
import copy

MAX_TOOL_TURNS = 5  # 限制工具调用的最大轮数，防止死循环


class BaseAgent:
    """
    All agent base class. Fully decoupled from specific API providers.

    Configuration (model, client, tools) is injected at construction time,
    enabling runtime switching via UI.
    """

    def __init__(self, name: str, role_prompt: str, client, model: str,
                 tools: list = None, model_type: str = "text",
                 cache=None):
        self.name = name
        self.role_prompt = role_prompt
        self.client = client
        self.model = model
        self.model_type = model_type
        self.tools = tools or []
        self.cache = cache
        self.memory = []

    # ══════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════

    def think(self, task: dict, context: dict = None,
              image_paths: list = None, use_cache: bool = True) -> dict:
        """
        Structured task → LLM reasoning → structured JSON decision.

        Supports tool calling: if the LLM requests a tool, execute it
        and loop until the LLM produces its final answer.
        """
        system_msg = {"role": "system",
                      "content": self._build_system_prompt(context)}
        user_msg = {"role": "user",
                    "content": self._build_user_content(task, image_paths)}
        messages = [system_msg, user_msg]

        # ── Cache: only cache non-tool-calling calls ──
        can_cache = use_cache and self.cache and not self.tools
        if can_cache:
            cache_key = self.cache.make_key(self.model, messages)
            cached = self.cache.get(cache_key)
            if cached:
                self.memory.append({"task": task, "cached": True})
                return cached

        # ── Build request kwargs ──
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        if not self.tools:
            # No tools: force JSON output for reliability
            kwargs["response_format"] = {"type": "json_object"}
        else:
            # With tools: register them with the LLM
            kwargs["tools"] = [t.to_openai_schema() for t in self.tools]
            kwargs["tool_choice"] = "auto"

        # ── Tool-calling loop ──
        result = self._call_with_tool_loop(kwargs, messages)

        # ── Cache store ──
        if can_cache:
            self.cache.put(cache_key, result)

        self.memory.append({"task": task, "result": result})
        return result

    # ══════════════════════════════════════════════
    #  Tool-calling loop
    # ══════════════════════════════════════════════

    def _call_with_tool_loop(self, kwargs: dict,
                             messages: list) -> dict:
        """
        Send request → if LLM returns tool_calls, execute them and loop.
        Returns the final parsed JSON dict.
        """
        # Work on a copy so retries don't mutate the original messages
        msgs = copy.deepcopy(messages)
        turns = 0

        while turns < MAX_TOOL_TURNS:
            turns += 1
            kwargs["messages"] = msgs
            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message

            # LLM returned a final answer (no tool calls)
            if not msg.tool_calls:
                return self._parse_json(msg.content)

            # LLM requested tool calls — execute them
            # 1. Append the assistant's tool_call request to conversation
            msgs.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            # 2. Execute each tool and append results
            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                result = self._execute_tool(tool_name, tool_args)

                msgs.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Safety net: exceeded max turns
        raise RuntimeError(
            f"Agent '{self.name}' exceeded {MAX_TOOL_TURNS} tool-calling "
            f"turns without producing a final answer"
        )

    # ══════════════════════════════════════════════
    #  Tool execution
    # ══════════════════════════════════════════════

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """Find and execute the named tool. Returns result dict."""
        for tool in self.tools:
            if tool.name == tool_name:
                try:
                    return tool.execute(**tool_args)
                except Exception as e:
                    return {"error": str(e)}
        return {"error": f"Unknown tool: {tool_name}"}

    # ══════════════════════════════════════════════
    #  Message builders
    # ══════════════════════════════════════════════

    def _build_system_prompt(self, context: dict = None) -> str:
        prompt = self.role_prompt
        if context:
            prompt += (
                f"\n\n【上游 Agent 上下文】\n"
                f"{json.dumps(context, ensure_ascii=False, indent=2)}"
            )
        prompt += (
            "\n\n【输出要求】必须输出合法 JSON，"
            "不要有解释文字或 markdown 标记。"
        )
        return prompt

    def _build_user_content(self, task: dict, image_paths: list) -> list:
        content = [{"type": "text",
                    "text": json.dumps(task, ensure_ascii=False)}]
        if image_paths:
            for path in image_paths:
                b64 = self._encode_image(path)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })
        return content

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ══════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════

    @staticmethod
    def _parse_json(text: str) -> dict:
        """Parse LLM response as JSON, with markdown fence removal."""
        text = (text or "").strip()
        if text.startswith("```"):
            # Strip ```json ... ``` fences
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
            else:
                text = text.strip("`")
        return json.loads(text)

    def clear_memory(self):
        self.memory = []
