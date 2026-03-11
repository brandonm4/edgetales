import unittest

from engine import (
    EngineConfig,
    GameState,
    _format_state_answer_text,
    _relevant_established_facts,
    _sanitize_established_facts,
    _upsert_established_facts,
    _is_pure_system_query_turn,
    _move_narration_contract,
    _normalize_strict_scene_resolution,
    RollResult,
    _sanitize_metadata_for_strict_mode,
    _should_generate_state_answer,
    build_no_roll_action_prompt,
    build_dialog_prompt,
    get_narrator_system,
)


class StrictModeGuardrailTests(unittest.TestCase):
    def test_strict_mode_generates_state_answer_for_system_query(self):
        config = EngineConfig(strict_mode=True)
        brain = {"move": "dialog"}
        self.assertTrue(_should_generate_state_answer(
            config,
            'I scan the panel. |Do I have engine power?|',
            brain,
        ))

    def test_strict_mode_generates_state_answer_for_gather_information(self):
        config = EngineConfig(strict_mode=True)
        brain = {"move": "gather_information"}
        self.assertTrue(_should_generate_state_answer(
            config,
            "I run a diagnostic on the engines.",
            brain,
        ))

    def test_non_strict_mode_skips_state_answer(self):
        config = EngineConfig(strict_mode=False)
        brain = {"move": "gather_information"}
        self.assertFalse(_should_generate_state_answer(
            config,
            "I run a diagnostic on the engines.",
            brain,
        ))

    def test_metadata_sanitizer_drops_invented_entities(self):
        game = GameState(npcs=[{"id": "npc_1", "name": "Kole"}])
        metadata = {
            "scene_context": "The tug scrapes past the hull.",
            "location_update": "Engine bay",
            "time_update": None,
            "memory_updates": [
                {"npc_id": "npc_1", "event": "Saw the thrusters flare.", "emotional_weight": "wary"},
                {"npc_id": "npc_new", "event": "Appeared from nowhere.", "emotional_weight": "neutral"},
            ],
            "new_npcs": [{"name": "Random Stranger"}],
            "npc_renames": [{"old_name": "Kole", "new_name": "Admiral Kole"}],
            "npc_details": [{"npc_id": "npc_1", "description": "Secret emperor"}],
            "deceased_npcs": [{"npc_id": "npc_1"}],
        }
        sanitized = _sanitize_metadata_for_strict_mode(game, metadata)
        self.assertEqual(sanitized["scene_context"], "The tug scrapes past the hull.")
        self.assertEqual(sanitized["location_update"], "Engine bay")
        self.assertEqual(len(sanitized["memory_updates"]), 1)
        self.assertEqual(sanitized["memory_updates"][0]["npc_id"], "npc_1")
        self.assertEqual(sanitized["new_npcs"], [])
        self.assertEqual(sanitized["npc_renames"], [])
        self.assertEqual(sanitized["npc_details"], [])
        self.assertEqual(sanitized["deceased_npcs"], [])

    def test_dialog_prompt_includes_state_answer_in_strict_mode(self):
        game = GameState(
            player_name="BRND",
            character_concept="AI warship core",
            setting_genre="science_fiction",
            setting_tone="dark_gritty",
            setting_description="A dying warship hangs in orbit.",
            current_location="Core conduit bay",
            current_scene_context="Leaking fusion and hostile salvage cutters.",
            health=5,
            spirit=5,
            supply=5,
        )
        brain = {"player_intent": "Run a diagnostic", "dramatic_question": ""}
        prompt = build_dialog_prompt(
            game,
            brain,
            player_words='I run a diagnostic. |Do I have engine power?|',
            state_answer={
                "answer_summary": "Partial engine power is available.",
                "usable_power": "yes",
                "targeting_possible": "uncertain",
                "player_status": "You are undamaged.",
                "ship_status": "Thrusters respond intermittently.",
                "facts": ["Port thrusters still answer low-output commands."],
            },
            config=EngineConfig(strict_mode=True),
        )
        self.assertIn("<state_answer>", prompt)
        self.assertIn("Partial engine power is available.", prompt)
        self.assertIn("Do NOT describe the player character as physically injured", prompt)

    def test_narrator_system_strict_mode_adds_coherence_rules(self):
        prompt = get_narrator_system(EngineConfig(strict_mode=True))
        self.assertIn("STRICT MODE: factual coherence outranks flourish.", prompt)
        self.assertIn("Do NOT reinterpret a diagnostic", prompt)

    def test_move_contract_for_gather_information_answers_first(self):
        contract = _move_narration_contract({"move": "gather_information"})
        self.assertIn("answer the player's question", contract)
        self.assertIn("brief atmosphere", contract)

    def test_strict_scene_resolution_normalizes_optional_fields(self):
        normalized = _normalize_strict_scene_resolution({
            "narration": "You read the panel output.",
            "scene_context": "The engine room hums weakly.",
            "location_update": None,
            "time_update": None,
            "memory_updates": [],
            "new_npcs": [
                {"name": "A", "description": "first npc entry", "disposition": "neutral"},
                {"name": "B", "description": "second npc entry", "disposition": "neutral"},
            ],
        })
        self.assertEqual(normalized["npc_renames"], [])
        self.assertEqual(normalized["npc_details"], [])
        self.assertEqual(normalized["deceased_npcs"], [])
        self.assertEqual(len(normalized["new_npcs"]), 1)

    def test_no_roll_prompt_marks_routine_resolution(self):
        game = GameState(
            player_name="BRND",
            character_description="An AI core aboard a warship.",
            setting_genre="science_fiction",
            setting_tone="dark_gritty",
            setting_description="A battered warship drifts near a nebula.",
            current_location="Sensor pit",
            current_scene_context="The tug is holding position at the edge of passive scans.",
        )
        brain = {
            "player_intent": "Check the sensor feed and pan across the tug silhouette.",
            "approach": "careful observation",
            "move": "gather_information",
        }
        prompt = build_no_roll_action_prompt(game, brain, player_words="I pan the sensors across the tug.")
        self.assertIn("NO ROLL", prompt)
        self.assertIn("routine, safe, or purely expressive", prompt)

    def test_pure_system_query_detection(self):
        self.assertTrue(_is_pure_system_query_turn(
            "|System Analysis - Do I have any drones capable of space flight?|"
        ))
        self.assertFalse(_is_pure_system_query_turn(
            'I check the panel. |Do I have any drones capable of space flight?|'
        ))

    def test_state_answer_text_formats_as_direct_answer(self):
        text = _format_state_answer_text({
            "answer_summary": "You have no active external repair drones on standby.",
            "facts": [
                "Two maintenance drones remain inside the hull.",
                "No salvage or EVA-capable drones are currently powered for flight.",
            ],
        })
        self.assertIn("You have no active external repair drones on standby.", text)
        self.assertIn("- Two maintenance drones remain inside the hull.", text)

    def test_established_facts_upsert_and_relevance(self):
        game = GameState(established_facts=[
            {
                "subject": "ship_drones",
                "category": "maintenance_drones_active",
                "value": "two internal maintenance drones confirmed",
                "confidence": "high",
                "source": "opening_scene",
                "scene_established": 1,
            }
        ])
        _upsert_established_facts(game, [
            {
                "subject": "ship_drones",
                "category": "external_repair_drones",
                "value": "none confirmed spaceworthy",
                "confidence": "medium",
                "source": "system_query_inference",
                "scene_established": 6,
            }
        ])
        relevant = _relevant_established_facts(game, "Do I have repair drones capable of space flight?")
        self.assertEqual(len(game.established_facts), 2)
        self.assertTrue(any(f["category"] == "external_repair_drones" for f in relevant))

    def test_sanitize_established_facts_discards_incomplete_entries(self):
        facts = _sanitize_established_facts([
            {
                "subject": "ship_repairs",
                "category": "repair_activity",
                "value": "internal drones are actively patching and scavenging",
                "confidence": "high",
                "source": "scene_resolution",
                "scene_established": 7,
            },
            {
                "subject": "",
                "category": "sensor_status",
                "value": "improved",
            },
        ], scene_count=7, source="scene_resolution")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["category"], "repair_activity")


if __name__ == "__main__":
    unittest.main()
