"""VisionForge router: System 1 (fast, deterministic + 1-token classifier) and
System 2 (deep reasoning) with an automatic LiteLLM provider cascade."""
import os
import re
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import litellm

log = logging.getLogger("visionforge.router")
litellm.drop_params = True
litellm.suppress_debug_info = True

# --------------------------------------------------------------------------- #
# Provider cascade                                                            #
# --------------------------------------------------------------------------- #
FAST_CASCADE = [
    {"tier": "primary", "model": os.getenv("VF_PRIMARY_MODEL", "anthropic/claude-3-5-haiku-20241022"),
     "key_env": os.getenv("VF_PRIMARY_KEY_ENV", "ANTHROPIC_API_KEY")},
    {"tier": "gemini_flash", "model": os.getenv("VF_GEMINI_FLASH", "gemini/gemini-2.0-flash"), "key_env": "GEMINI_API_KEY"},
    {"tier": "groq_llama", "model": os.getenv("VF_GROQ_LLAMA", "groq/llama-3.3-70b-versatile"), "key_env": "GROQ_API_KEY"},
    {"tier": "groq_deepseek", "model": os.getenv("VF_GROQ_DEEPSEEK", "groq/deepseek-r1-distill-llama-70b"), "key_env": "GROQ_API_KEY"},
    {"tier": "openrouter_free", "model": os.getenv("VF_OPENROUTER_FREE", "openrouter/meta-llama/llama-3.3-70b-instruct:free"),
     "key_env": "OPENROUTER_API_KEY"},
]

REASONING_CASCADE = [
    {"tier": "primary_reasoning", "model": os.getenv("VF_REASONING_MODEL", "anthropic/claude-sonnet-4-20250514"),
     "key_env": os.getenv("VF_PRIMARY_KEY_ENV", "ANTHROPIC_API_KEY")},
    {"tier": "gemini_pro", "model": os.getenv("VF_GEMINI_PRO", "gemini/gemini-2.5-pro"), "key_env": "GEMINI_API_KEY"},
    {"tier": "groq_deepseek", "model": os.getenv("VF_GROQ_DEEPSEEK", "groq/deepseek-r1-distill-llama-70b"), "key_env": "GROQ_API_KEY"},
    {"tier": "openrouter_r1", "model": os.getenv("VF_OPENROUTER_R1", "openrouter/deepseek/deepseek-r1:free"), "key_env": "OPENROUTER_API_KEY"},
]

RATE_LIMIT_PATTERN = re.compile(r"429|rate.?limit|quota|resource.?exhausted|too many requests|overloaded", re.I)
COOLDOWN_SECONDS = float(os.getenv("VF_PROVIDER_COOLDOWN", "90"))
STATE_PATH = os.getenv("VF_ROUTER_STATE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_state.json"))

# --------------------------------------------------------------------------- #
# Token helpers                                                               #
# --------------------------------------------------------------------------- #
def compress_json(obj: Any) -> str:
    """Minified JSON payload (no whitespace) to save prompt tokens."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str)


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        return int(litellm.token_counter(model=model, text=text))
    except Exception:
        return max(1, len(text) // 4)


def prune_messages(messages: List[Dict[str, str]], max_tokens: int = 6000) -> List[Dict[str, str]]:
    """Sliding-window pruning: keep system messages + most recent turns within budget."""
    system = [m for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    budget = max_tokens - sum(estimate_tokens(m.get("content", "")) for m in system)
    kept: List[Dict[str, str]] = []
    for m in reversed(others):
        cost = estimate_tokens(m.get("content", ""))
        if cost > budget and kept:
            break
        if cost > budget:
            content = m.get("content", "")
            m = {"role": m["role"], "content": content[-(budget * 4):]}
            cost = budget
        kept.insert(0, m)
        budget -= cost
        if budget <= 0:
            break
    return system + kept


# --------------------------------------------------------------------------- #
# Router state (scratchpad preserved across provider switches)                #
# --------------------------------------------------------------------------- #
@dataclass
class RouterState:
    active_provider: str = "none"
    active_model: str = "none"
    scratchpad: List[Dict[str, Any]] = field(default_factory=list)
    cooldowns: Dict[str, float] = field(default_factory=dict)
    dead_tiers: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "calls": 0, "fast_calls": 0, "reasoning_calls": 0, "failovers": 0,
        "rule_decisions": 0, "per_provider": {}})

    def note(self, kind: str, payload: Any) -> None:
        self.scratchpad.append({"t": time.time(), "kind": kind, "data": payload})
        if len(self.scratchpad) > 200:
            self.scratchpad = self.scratchpad[-200:]

    def save(self, path: str = STATE_PATH) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self), f, indent=1)
        except OSError as exc:
            log.warning("state save failed: %s", exc)

    @classmethod
    def load(cls, path: str = STATE_PATH) -> "RouterState":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st = cls()
                for k, v in data.items():
                    if hasattr(st, k):
                        setattr(st, k, v)
                return st
            except (OSError, json.JSONDecodeError):
                pass
        return cls()


class ProvidersExhausted(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Router                                                                      #
# --------------------------------------------------------------------------- #
SEARCH_TRIGGERS = re.compile(
    r"\b(real|actual|exact|official|blueprint|schematic|spec|specs|specification|dimension|dimensions|"
    r"height|width|length|scale|size|measurement|cm|mm|inch|inches|feet|meters?|brand|model number|"
    r"true to life|accurate|reference)\b", re.I)
AMBIGUITY_TRIGGERS = re.compile(r"\b(or|maybe|either|unsure|not sure|could be|something like|ish)\b|\?", re.I)
RIG_TRIGGERS = re.compile(r"\b(rig|rigged|animate|animation|animated|character|creature|humanoid|deform|skeleton|armature)\b", re.I)

BACKENDS = {
    "trellis": {"min_vram_gb": 16, "multi_image": True, "quality": 3},
    "instantmesh": {"min_vram_gb": 10, "multi_image": True, "quality": 2},
    "triposr": {"min_vram_gb": 4, "multi_image": False, "quality": 1},
}


class Router:
    def __init__(self) -> None:
        self.state = RouterState.load()

    # ---------------- deterministic rules (0 tokens) -----------------------
    def needs_web_search(self, prompt: str, object_class: str = "") -> bool:
        text = f"{prompt} {object_class}"
        hit = bool(SEARCH_TRIGGERS.search(text))
        self.state.metrics["rule_decisions"] += 1
        self.state.note("rule.web_search", {"hit": hit})
        return hit

    def is_rig_request(self, prompt: str) -> bool:
        return bool(RIG_TRIGGERS.search(prompt or ""))

    def pick_backend(self, n_images: int, has_gpu: bool, vram_gb: float, available: List[str]) -> str:
        self.state.metrics["rule_decisions"] += 1
        ranked = sorted([b for b in available if b in BACKENDS], key=lambda b: -BACKENDS[b]["quality"])
        for b in ranked:
            spec = BACKENDS[b]
            if n_images > 1 and not spec["multi_image"]:
                continue
            if has_gpu and vram_gb >= spec["min_vram_gb"]:
                return b
        for b in ranked:
            if n_images == 1 or BACKENDS[b]["multi_image"]:
                return b
        return ranked[-1] if ranked else "triposr"

    def qa_verdict(self, score: float, pass_threshold: float = 0.72, gray_zone: float = 0.05,
                   context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deterministic pass/fail; the 1-token classifier only breaks ties in the gray zone."""
        self.state.metrics["rule_decisions"] += 1
        if score >= pass_threshold + gray_zone:
            return {"pass": True, "method": "rule", "score": score}
        if score < pass_threshold - gray_zone:
            return {"pass": False, "method": "rule", "score": score}
        ctx = compress_json(context or {})
        answer = self.fast_decide(
            f"3D asset QA. similarity={score:.3f} threshold={pass_threshold}. ctx={ctx}. "
            f"Is the reconstruction acceptable for delivery? Answer P (pass) or F (fail).",
            options=["P", "F"], default="F")
        return {"pass": answer == "P", "method": "fast_llm", "score": score}

    # ---------------- System 1: 1-token classifier -------------------------
    def fast_decide(self, question: str, options: List[str], default: str) -> str:
        messages = [
            {"role": "system", "content": "You are a classifier. Reply with exactly one character from the allowed set. No words."},
            {"role": "user", "content": f"{question}\nAllowed: {''.join(options)}"},
        ]
        try:
            res = self.complete(messages, cascade=FAST_CASCADE, max_tokens=1, temperature=0.0, purpose="fast")
            token = (res["content"] or "").strip().upper()[:1]
            return token if token in options else default
        except ProvidersExhausted:
            return default

    def classify_object(self, prompt: str, caption: str = "", few_shot: str = "") -> Dict[str, Any]:
        """Returns object_class, ambiguity flag and remesh hints. Escalates to System 2 only if ambiguous."""
        text = f"{prompt} {caption}".strip()
        ambiguous_rule = bool(AMBIGUITY_TRIGGERS.search(text)) or len(text.split()) < 2
        if not ambiguous_rule:
            words = re.findall(r"[a-zA-Z]+", prompt.lower())
            noun = words[-1] if words else "object"
            verdict = self.fast_decide(
                f"Is '{text[:300]}' a clear, single, unambiguous physical object description? Y or N.",
                options=["Y", "N"], default="N")
            if verdict == "Y":
                out = {"object_class": noun, "ambiguous": False, "system": 1,
                       "animation_candidate": self.is_rig_request(prompt), "notes": "fast path"}
                self.state.note("classify", out)
                return out
        reasoning = self.reason(
            "Classify the object for 3D reconstruction. Return compact JSON with keys: "
            "object_class (snake_case noun), ambiguous (bool), symmetric (bool), animation_candidate (bool), "
            "suggested_target_faces (int 2000-60000), notes (short string).",
            f"description={compress_json(text)} few_shot_examples={few_shot or 'none'}")
        parsed = self._extract_json(reasoning.get("content", "")) or {}
        out = {
            "object_class": str(parsed.get("object_class", "object")).lower().replace(" ", "_"),
            "ambiguous": bool(parsed.get("ambiguous", True)),
            "symmetric": bool(parsed.get("symmetric", True)),
            "animation_candidate": bool(parsed.get("animation_candidate", self.is_rig_request(prompt))),
            "suggested_target_faces": int(parsed.get("suggested_target_faces", 12000)),
            "notes": str(parsed.get("notes", "")),
            "system": 2,
        }
        self.state.note("classify", out)
        return out

    # ---------------- System 2: reasoning ----------------------------------
    def reason(self, instruction: str, payload: str, max_tokens: int = 600) -> Dict[str, Any]:
        messages = prune_messages([
            {"role": "system", "content": "You are a senior technical 3D director. Think carefully, answer concisely in JSON."},
            {"role": "user", "content": f"{instruction}\n{payload}"},
        ])
        return self.complete(messages, cascade=REASONING_CASCADE, max_tokens=max_tokens, temperature=0.2, purpose="reasoning")

    # ---------------- cascade engine ---------------------------------------
    def _tier_available(self, tier: Dict[str, str]) -> bool:
        if tier["tier"] in self.state.dead_tiers:
            return False
        if not os.getenv(tier["key_env"]):
            return False
        until = self.state.cooldowns.get(tier["tier"], 0.0)
        return time.time() >= until

    def complete(self, messages: List[Dict[str, str]], cascade: List[Dict[str, str]],
                 max_tokens: int = 256, temperature: float = 0.0, purpose: str = "fast") -> Dict[str, Any]:
        errors: List[str] = []
        for tier in cascade:
            if not self._tier_available(tier):
                continue
            try:
                t0 = time.time()
                resp = litellm.completion(model=tier["model"], messages=messages, max_tokens=max_tokens,
                                          temperature=temperature, timeout=60)
                content = resp.choices[0].message.content or ""
                usage = getattr(resp, "usage", None)
                p = int(getattr(usage, "prompt_tokens", 0) or 0)
                c = int(getattr(usage, "completion_tokens", 0) or 0)
                self._record_usage(tier, p, c, purpose, time.time() - t0)
                return {"content": content, "model": tier["model"], "tier": tier["tier"],
                        "usage": {"prompt_tokens": p, "completion_tokens": c}}
            except Exception as exc:  # noqa: BLE001 - we must classify every provider error
                msg = f"{tier['tier']}: {type(exc).__name__}: {str(exc)[:160]}"
                errors.append(msg)
                if isinstance(exc, litellm.exceptions.AuthenticationError):
                    self.state.dead_tiers.append(tier["tier"])
                    log.error("auth failed, tier disabled: %s", msg)
                elif isinstance(exc, litellm.exceptions.RateLimitError) or RATE_LIMIT_PATTERN.search(str(exc)):
                    self.state.cooldowns[tier["tier"]] = time.time() + COOLDOWN_SECONDS
                    self.state.metrics["failovers"] += 1
                    log.warning("rate limit intercepted, failing over: %s", msg)
                else:
                    self.state.cooldowns[tier["tier"]] = time.time() + 20
                    self.state.metrics["failovers"] += 1
                    log.warning("provider error, failing over: %s", msg)
                self.state.note("failover", {"from": tier["tier"], "error": msg})
                self.state.save()
        raise ProvidersExhausted("All providers failed or cooling down: " + " | ".join(errors))

    def _record_usage(self, tier: Dict[str, str], p: int, c: int, purpose: str, latency: float) -> None:
        m = self.state.metrics
        m["prompt_tokens"] += p
        m["completion_tokens"] += c
        m["total_tokens"] += p + c
        m["calls"] += 1
        m["fast_calls" if purpose == "fast" else "reasoning_calls"] += 1
        pp = m["per_provider"].setdefault(tier["tier"], {"calls": 0, "tokens": 0, "latency_s": 0.0})
        pp["calls"] += 1
        pp["tokens"] += p + c
        pp["latency_s"] = round(pp["latency_s"] + latency, 3)
        self.state.active_provider = tier["tier"]
        self.state.active_model = tier["model"]
        self.state.save()

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def telemetry(self) -> Dict[str, Any]:
        return {"provider": self.state.active_provider, "model": self.state.active_model, **self.state.metrics}


ROUTER = Router()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = ROUTER
    print("web search:", r.needs_web_search("a 1:1 scale Leica M6 camera with exact dimensions"))
    print("backend:", r.pick_backend(1, True, 24, ["trellis", "triposr", "instantmesh"]))
    print("qa:", r.qa_verdict(0.9))
    print(compress_json(r.telemetry()))
