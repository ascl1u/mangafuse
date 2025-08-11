from __future__ import annotations

from typing import List


class GeminiTranslator:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for translation via Gemini")
        try:
            from google import genai  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "google-genai is required. Install AI deps from backend/requirements-ai.txt"
            ) from exc
        self._client = genai.Client(api_key=api_key)

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        # Keep batching small to be gentle on limits; simple and deterministic
        batch_size = 8
        out: List[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            prompt = (
                "Translate the following Japanese manga dialogue lines into concise, natural English suitable for speech bubbles.\n"
                "Constraints:\n"
                "- Output exactly one English line per input line, in the same order.\n"
                "- Keep the English length roughly similar to the Japanese length.\n"
                "- If a name appears, do not attempt a translation and output the English romaji'.\n"
                "- If an explicit word appears, replace it with the neutral placeholder 'banana'.\n"
                "- Output only the translations; no numbering, bullets, or extra commentary.\n\n"
                + "\n".join(f"- {t}" for t in chunk)
            )
            resp = self._client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
            lines = [line_text.strip("- ").strip() for line_text in (resp.text or "").splitlines() if line_text.strip()]
            if len(lines) < len(chunk):
                lines += [""] * (len(chunk) - len(lines))
            out.extend(lines[: len(chunk)])
        return out


