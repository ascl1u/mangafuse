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
        batch_size = 8
        out: List[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            prompt = (
                "Act as a professional manga translator. Translate Japanese text to English. Follow these rules meticulously:\n"
                "Rules:\n"
                "- Keep the translated English length roughly similar to the Japanese length.\n"
                "- Proper nouns (e.g. names of people and places) and cultural concepts (e.g. sensei, katana, ryokan) should not be translated. Directly output the English romaji.\n"
                "- Retain `-san`, `-kun`, `-chan`, `-sama`, and other honorifics. Directly attach them to the English romaji.\n"
                "- Preserve the formality level (`desu/masu` vs. plain/casual). \n"
                "- Add implied subjects/pronouns where essential for English clarity.\n"
                "- Adjust verb tense/aspect for natural English flow, even if the Japanese is ambiguous.\n"
                "- Output exactly one English line per input line, in the same order.\n"
                "- Output only the translations; no numbering, bullets, or extra commentary.\n\n"
                + "\n".join(f"- {t}" for t in chunk)
            )
            resp = self._client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
            lines = [line_text.strip("- ").strip() for line_text in (resp.text or "").splitlines() if line_text.strip()]
            if len(lines) < len(chunk):
                lines += [""] * (len(chunk) - len(lines))
            out.extend(lines[: len(chunk)])
        return out


