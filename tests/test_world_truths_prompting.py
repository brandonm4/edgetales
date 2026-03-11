import unittest

from engine import EngineConfig, GameState, _world_truths_block, get_narrator_system


class WorldTruthsPromptingTests(unittest.TestCase):
    def test_world_truths_block_present(self):
        game = GameState(world_truths="FTL only works through gate pairs.")
        block = _world_truths_block(game)
        self.assertIn("<world_truths>", block)
        self.assertIn("FTL only works through gate pairs.", block)

    def test_narrator_system_mentions_world_truths(self):
        game = GameState(world_truths="No real-time faster-than-light communication.")
        system = get_narrator_system(EngineConfig(narration_lang="English"), game)
        self.assertIn("WORLD TRUTHS", system)
        self.assertIn("No real-time faster-than-light communication.", system)


if __name__ == "__main__":
    unittest.main()
