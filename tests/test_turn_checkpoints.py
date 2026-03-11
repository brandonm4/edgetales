import unittest

from engine import GameState, capture_turn_checkpoint, restore_game_state


class TurnCheckpointTests(unittest.TestCase):
    def test_capture_and_restore_checkpoint(self):
        game = GameState(player_name="BRND", scene_count=1, health=5, supply=5, momentum=2)
        messages = [
            {"scene_marker": "Scene 1"},
            {"role": "assistant", "content": "Opening."},
        ]
        checkpoint = capture_turn_checkpoint(game, messages)
        game.health = 2
        game.supply = 1
        restore_game_state(game, checkpoint["game_state"])
        self.assertEqual(game.health, 5)
        self.assertEqual(game.supply, 5)
        self.assertEqual(game.scene_count, 1)


if __name__ == "__main__":
    unittest.main()
