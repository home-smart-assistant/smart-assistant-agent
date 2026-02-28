from __future__ import annotations

import unittest

from app.core.text_codec import EncodingNormalizationError, normalize_text


class TestTextCodec(unittest.TestCase):
    def test_keep_utf8_text(self) -> None:
        self.assertEqual("玄关", normalize_text("玄关", field_path="text", strict=True))

    def test_repair_mojibake(self) -> None:
        self.assertEqual("玄关", normalize_text("çå³", field_path="text", strict=True))
        self.assertEqual("书房灯", normalize_text("ä¹¦æ¿ç¯", field_path="text", strict=True))

    def test_raise_on_unrecoverable(self) -> None:
        with self.assertRaises(EncodingNormalizationError):
            normalize_text("ÃÃÃÃ", field_path="text", strict=True)


if __name__ == "__main__":
    unittest.main()
