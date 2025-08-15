from __future__ import annotations

from typing import List
from pydantic import BaseModel

try:
    from google import genai
except ImportError as exc:
    raise RuntimeError(
        "google-genai is required. Install AI deps via: pip install 'google-generativeai[pydantic]'"
    ) from exc

class TranslationPair(BaseModel):
    """A simple schema for a Japanese to English translation pair."""
    original_japanese: str
    english_translation: str

# --- FIX: Create a wrapper model to hold the list ---
# This works around the SDK bug by not using List[...] as the top-level schema.
class TranslationList(BaseModel):
    """A container for a list of translation pairs."""
    translations: List[TranslationPair]


class GeminiTranslator:
    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for translation via Gemini")

        # 1. Use genai.Client() and store the client instance.
        self._client = genai.Client(api_key=api_key)

        # 2. Define the configuration components.
        self._model_name = "gemini-2.0-flash"

        self._generation_config = {
            "response_mime_type": "application/json",
            "response_schema": TranslationList,
        }

    def translate_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        # Combine all instructions and examples into a single, robust prompt.
        prompt_header = (
            "You are a professional manga translator. Your task is to translate a list of unordered Japanese text snippets into natural, fluent English. "
            "You must respond with a JSON object containing a 'translations' key, which holds an array of translation pair objects. Each object must have an 'original_japanese' and 'english_translation' key.\n\n"
            "--- Example 1 ---\n"
            "Japanese: もう大丈夫！ なぜって？ 私が来た！\n"
            "English: It's fine now. Why? Because I am here!\n\n"
            "--- Example 2 ---\n"
            "Japanese: エレン、あなたがいれば、私は何でもできる\n"
            "English: Eren... If you're with me... I can do anything.\n\n"
            "--- Example 3 ---\n"
            "Japanese: 死んで勝つと死んでも勝つは全然違うんだよ\n"
            "English: “Winning by dying” and “winning even if you die” are two completely different things.\n\n"
            "--- End of Examples ---\n\n"
            "Now, translate the following list of Japanese text snippets:\n"
        )

        full_prompt_text = prompt_header + "\n".join(f"- {text}" for text in texts)

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=full_prompt_text,
                config=self._generation_config
            )
            
            # The .parsed attribute will now be an instance of TranslationList.
            parsed_response: TranslationList = response.parsed
            
            if not parsed_response or not parsed_response.translations:
                 print("Warning: Parsed response was empty or contained no translations.")
                 return [""] * len(texts)

            # Access the list of translations from the wrapper object.
            parsed_items = parsed_response.translations
            translation_map = {item.original_japanese: item.english_translation for item in parsed_items}
            
            ordered_translations = [
                translation_map.get(text, f"Translation not found for: {text}") for text in texts
            ]
            
            return ordered_translations

        except Exception as e:
            print(f"An error occurred during Gemini API call: {e}")
            return [""] * len(texts)