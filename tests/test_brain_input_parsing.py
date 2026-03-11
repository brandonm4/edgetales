import unittest

from engine import _narrator_player_input_block, _parse_player_input_segments


class BrainInputParsingTests(unittest.TestCase):
    def test_plain_text_is_action(self):
        parsed = _parse_player_input_segments("I run a diagnostic and reroute power.")
        self.assertEqual(parsed["action"], "I run a diagnostic and reroute power.")
        self.assertEqual(parsed["spoken"], [])
        self.assertEqual(parsed["system"], [])

    def test_quotes_and_pipes_are_separated(self):
        parsed = _parse_player_input_segments('I raise my hands. "Stand down." |Do I have line of sight?|')
        self.assertEqual(parsed["action"], "I raise my hands.")
        self.assertEqual(parsed["spoken"], ["Stand down."])
        self.assertEqual(parsed["system"], ["Do I have line of sight?"])

    def test_conditional_plain_text_stays_in_action(self):
        parsed = _parse_player_input_segments("I run a diagnostic. If I have engine power, I aim the thrusters at the tug.")
        self.assertIn("If I have engine power", parsed["action"])

    def test_narrator_input_block_uses_structured_tags(self):
        block = _narrator_player_input_block('I run a diagnostic. "Kole, back off." |Do I have engine power?|')
        self.assertIn("<player_action>I run a diagnostic.</player_action>", block)
        self.assertIn("<player_dialog>Kole, back off.</player_dialog>", block)
        self.assertIn("<system_query>Do I have engine power?</system_query>", block)
        self.assertNotIn("<player_words>", block)


if __name__ == "__main__":
    unittest.main()
