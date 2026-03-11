import unittest

from engine import GameState, RollResult, build_action_prompt


class ActionPromptResolutionTests(unittest.TestCase):
    def test_strong_hit_prompt_forbids_undercutting_win(self):
        game = GameState(
            player_name="BRND",
            character_concept="AI warship core",
            setting_genre="science_fiction",
            setting_tone="dark_gritty",
            setting_description="A dying warship hangs in orbit.",
            scene_count=3,
            current_location="Core conduit bay",
            current_scene_context="Leaking fusion and hostile salvage cutters.",
        )
        brain = {
            "player_intent": "Destroy the tug with aimed thrusters.",
            "approach": "overload the remaining thrusters",
            "position": "risky",
            "effect": "great",
            "target_npc": None,
            "world_addition": None,
            "dramatic_question": "Can BRND cripple the tug before it counters?",
        }
        roll = RollResult(1, 1, 2, 3, "iron", 1, 3, "STRONG_HIT", "strike", False)
        prompt = build_action_prompt(game, brain, roll, [], [], [], player_words="I fire the thrusters at the tug.")
        self.assertIn("Clean success. The player achieves the intended objective decisively", prompt)
        self.assertIn("Do not undercut the win", prompt)
        self.assertIn("<resolution_contract>", prompt)


if __name__ == "__main__":
    unittest.main()
