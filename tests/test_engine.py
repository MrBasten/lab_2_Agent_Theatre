"""Offline integration tests; no model download, credentials or paid requests."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from theatre.__main__ import build_providers
from theatre.engine import (
    DemoProvider,
    GenerationError,
    _actor_turn,
    _ask_json,
    _compact,
    _validate_action,
    demo_providers,
    run_play,
    save_result,
    select_speaker,
)
from theatre.prompts import CHARACTERS, NARRATOR_DIRECTION_PROMPT, actor_prompt


class FakeTools:
    def schemas(self):
        return [{"name": "search_files"}, {"name": "read_file"}, {"name": "estimate_preservation_cost"}]

    def execute(self, name, arguments):
        if name == "search_files":
            return {
                "query": arguments["query"],
                "hits": [
                    {"id": f"D{i}", "path": f"D{i}_test.md", "text": f"Document {i}", "score": 1.0}
                    for i in range(1, 6)
                ],
            }
        if name == "read_file":
            return {
                "id": arguments["id"],
                "path": arguments["id"] + "_test.md",
                "text": "Голоса разрешены только внутри здания.",
            }
        if name == "estimate_preservation_cost":
            return {"total_cost": 1800, "volunteer_pages": 480, "paid_pages": 720}
        raise ValueError("Unknown tool")


class FixedProvider:
    name = "test-real-provider"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses)


class EngineTests(unittest.TestCase):
    def test_spoken_tool_identifiers_are_repaired(self):
        action = {"action": "speak", "text": "Открой D4 через read_file.", "stage_direction": "кивает"}
        replacement = {**action, "text": "Подвинь журнал. Вот эта строка."}
        provider = FixedProvider([json.dumps(action), json.dumps(replacement)])
        result = _actor_turn(provider, "vera", {}, None, [], set(), [])
        self.assertEqual(result["text"], replacement["text"])
        self.assertEqual(provider.calls, 2)

    def test_real_mode_adds_model_generated_curtain_without_extra_actor_turn(self):
        class ModelDouble:
            def __init__(self, role, fixture):
                self.name = f"test:{role}"
                self.fixture = fixture

            def complete(self, messages, **kwargs):
                if "безмолвный финал" in messages[0]["content"]:
                    self.closing_calls = getattr(self, "closing_calls", 0) + 1
                    if self.closing_calls == 1:
                        return json.dumps({
                            "voice_records": "stay",
                            "decision": "Каталог сохраняют.",
                            "signatures": "Вера и Марк подписывают акт.",
                            "transfer": "Герои уносят каталог и записи голосов.",
                            "terminal": "Экран гаснет.",
                        }, ensure_ascii=False)
                    return json.dumps({
                        "voice_records": "stay",
                        "decision": "Голосовые записи остаются в архиве.",
                        "signatures": "Вера и Марк подписывают акт.",
                        "transfer": "Вера выносит каталог.",
                        "terminal": "Экран гаснет.",
                    }, ensure_ascii=False)
                return self.fixture.complete(messages, **kwargs)

        providers = {role: ModelDouble(role, fixture) for role, fixture in demo_providers().items()}
        result = run_play(providers, tools=FakeTools(), mode="cloud")
        self.assertEqual(result["stats"]["actor_turns"], 18)
        self.assertEqual(result["turns"][-1]["speaker"], "narrator")
        self.assertIn("Вера и Марк подписывают акт.", result["turns"][-1]["text"])
        self.assertEqual(providers["narrator"].closing_calls, 2)

    def test_complete_fixture_is_deterministic_and_inspectable(self):
        first = run_play(demo_providers(), tools=FakeTools(), mode="demo")
        second = run_play(demo_providers(), tools=FakeTools(), mode="demo")
        self.assertEqual(first, second)
        self.assertEqual(first["stats"]["actor_turns"], 18)
        self.assertEqual(first["stats"]["turns_by_character"], {"vera": 6, "mark": 6, "echo": 6})
        actor_turns = [t for t in first["turns"] if t["speaker"] != "narrator"]
        self.assertEqual(len({t["text"] for t in actor_turns}), 18)
        self.assertEqual(len(first["plan"]["acts"]), 3)
        self.assertTrue(first["metadata"]["fixture"])
        self.assertIn("не сгенерированы LLM", first["metadata"]["warning"])
        self.assertIsNone(first["metadata"]["elapsed_seconds"])
        self.assertEqual(first["stats"]["tool_calls"], 13)
        self.assertEqual(first["stats"]["successful_tool_calls"], 13)
        self.assertIn(
            "estimate_preservation_cost", {e["name"] for e in first["events"] if e["type"] == "tool"}
        )
        self.assertEqual(
            sum(first["stats"]["generation_calls"].values()),
            len([e for e in first["events"] if e["type"] == "generation"]),
        )

    def test_director_cannot_starve_or_immediately_repeat(self):
        turns = []
        for i in range(40):
            speaker, reason = select_speaker(turns, preferred="vera")
            self.assertTrue(reason)
            if turns:
                self.assertNotEqual(speaker, turns[-1]["speaker"])
            turns.append({"speaker": speaker})
            if i >= 3:
                self.assertEqual({t["speaker"] for t in turns[-4:]}, set(CHARACTERS))

    def test_minimum_act_has_all_characters(self):
        providers = demo_providers()
        providers["director"] = DemoProvider("director")
        result = run_play(providers, turns_per_act=3, acts=5, tools=FakeTools(), mode="demo")
        for act in range(1, 6):
            self.assertEqual(
                {t["speaker"] for t in result["turns"] if t["act"] == act and t["speaker"] != "narrator"},
                set(CHARACTERS),
            )

    def test_fixture_cannot_masquerade_as_real_generation(self):
        for mode in ("cloud", "live", "local"):
            with self.assertRaisesRegex(ValueError, "Fixture"):
                run_play(demo_providers(), mode=mode)

    def test_cloud_mode_routes_all_roles_without_creating_local_provider(self):
        environment = {"folder_id": "test-folder", "api_key": "test-key", "LOCAL_DEVICE": "invalid"}
        for mark_model in ("", "another-model"):
            with self.subTest(mark_model=mark_model), patch.dict(
                "os.environ", {**environment, "MARK_MODEL": mark_model}, clear=True
            ):
                providers = build_providers("cloud")
                main = providers["vera"]
                self.assertEqual(set(providers), {"planner", "director", "narrator", "vera", "mark", "echo"})
                for role in ("planner", "director", "narrator", "echo"):
                    self.assertIs(providers[role], main)
                if mark_model:
                    self.assertEqual(providers["mark"].model, "gpt://test-folder/another-model")
                else:
                    self.assertIs(providers["mark"], main)

    def test_self_address_is_repaired_without_replacing_actor_speech(self):
        bad = {
            "action": "speak",
            "text": "Марк, послушай. Я проверю новую смету.",
            "stage_direction": "смотрит на Веру",
            "addressed_to": "mark",
        }
        good = {**bad, "text": "Вера, я проверю новую смету перед отключением.", "addressed_to": "vera"}
        provider = FixedProvider([json.dumps(bad), json.dumps(good)])
        action = _actor_turn(provider, "mark", {"act_number": 1}, None, [], set(), [])
        self.assertEqual(action["text"], good["text"])
        self.assertEqual(provider.calls, 2)

    def test_role_prompt_uses_other_character_for_direct_address_example(self):
        mark_prompt = actor_prompt("mark", [])
        self.assertIn("Вера, послушай", mark_prompt)
        self.assertNotIn("Марк, послушай", mark_prompt)
        self.assertIn('"addressed_to": "vera"', mark_prompt)
        self.assertNotIn('"vera|mark|echo|null"', mark_prompt)

    def test_coverage_preserves_valid_director_choice_and_labels_overrides(self):
        class InsistentDirector(DemoProvider):
            def complete(self, messages, **kwargs):
                return json.dumps({"speaker": "vera", "reason": "Вера задаёт главный вопрос."})

        providers = demo_providers()
        providers["director"] = InsistentDirector("director")
        result = run_play(providers, mode="demo", tools=FakeTools(), acts=1, turns_per_act=3)
        selections = [event for event in result["events"] if event["type"] == "selection"]
        self.assertEqual(selections[0]["speaker"], "vera")
        self.assertEqual(selections[0]["guard"], "выбор режиссёра")
        self.assertEqual({event["speaker"] for event in selections}, set(CHARACTERS))
        for event in selections[1:]:
            self.assertNotEqual(event["speaker"], event["nomination"]["speaker"])
            self.assertEqual(event["guard"], "каждый персонаж участвует в каждом действии")

    def test_demo_never_calls_a_real_provider(self):
        providers = demo_providers()
        real = FixedProvider([])
        providers["planner"] = real
        with self.assertRaisesRegex(ValueError, "mix fixture and real"):
            run_play(providers, mode="demo")
        self.assertEqual(real.calls, 0)

    def test_bad_json_gets_one_repair_then_fails(self):
        provider = FixedProvider(["not json", "[]"])
        with self.assertRaisesRegex(GenerationError, "after repair"):
            _ask_json(provider, [], lambda value: None)
        self.assertEqual(provider.calls, 2)

    def test_json_markdown_fence_is_accepted(self):
        provider = FixedProvider(['```json\n{"a": 1}\n```'])
        self.assertEqual(_ask_json(provider, [], lambda value: None), {"a": 1})

    def test_backend_error_propagates_without_fixture_fallback(self):
        class Broken:
            name = "broken"

            def complete(self, *args, **kwargs):
                raise ConnectionError("network unavailable")

        with self.assertRaisesRegex(ConnectionError, "network unavailable"):
            _ask_json(Broken(), [], lambda value: None)

    def test_fabricated_source_is_rejected(self):
        invalid = json.dumps(
            {"action": "speak", "text": "Проверено.", "stage_direction": "тихо", "citations": ["D99"]}
        )
        provider = FixedProvider([invalid, invalid])
        with self.assertRaisesRegex(GenerationError, "unseen citations"):
            _actor_turn(
                provider, "vera", {"act_number": 1, "retrieval_completed": True}, FakeTools(), [], set(), []
            )
        self.assertEqual(provider.calls, 2)

    def test_unseen_source_can_be_repaired_without_silently_deleting_it(self):
        invalid = {
            "action": "speak",
            "text": "Думаю, ещё не всё потеряно.",
            "stage_direction": "тихо",
            "citations": ["D99"],
        }
        repaired = {**invalid, "citations": []}
        provider = FixedProvider([json.dumps(invalid), json.dumps(repaired)])
        action = _actor_turn(
            provider, "vera", {"act_number": 1, "retrieval_completed": True}, FakeTools(), [], set(), []
        )
        self.assertEqual(action["citations"], [])
        self.assertEqual(provider.calls, 2)

    def test_unseen_source_can_be_repaired_to_request_its_document(self):
        speech = {
            "action": "speak",
            "text": "Проверим новый документ.",
            "stage_direction": "читает",
            "citations": ["D2"],
        }
        request = {"action": "tool", "name": "read_file", "arguments": {"id": "D2"}}
        provider = FixedProvider([json.dumps(speech), json.dumps(request), json.dumps(speech)])
        events, sources = [], {"D1"}
        action = _actor_turn(provider, "mark", {"act_number": 1}, FakeTools(), events, sources, [])
        self.assertEqual(action["citations"], ["D2"])
        self.assertEqual(provider.calls, 3)
        self.assertEqual(events[0]["name"], "read_file")
        self.assertIn("D2", sources)

    def test_first_speech_is_repaired_to_a_real_retrieval(self):
        speech = {
            "action": "speak",
            "text": "Сначала проверим свидетельства.",
            "stage_direction": "открывает папку",
            "citations": [],
        }
        search = {"action": "tool", "name": "search_files", "arguments": {"query": "архив"}}
        provider = FixedProvider([json.dumps(speech), json.dumps(search), json.dumps(speech)])
        events, sources, memory = [], set(), []
        action = _actor_turn(provider, "vera", {"act_number": 1}, FakeTools(), events, sources, memory)
        self.assertTrue(action["_retrieval_completed"])
        self.assertEqual(provider.calls, 3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["name"], "search_files")
        self.assertIn("D1", sources)

    def test_empty_retrieval_allows_honest_uncertainty_and_is_remembered(self):
        class EmptyTools(FakeTools):
            def execute(self, name, arguments):
                return {"query": arguments["query"], "hits": []}

        search = {"action": "tool", "name": "search_files", "arguments": {"query": "несуществующее"}}
        speech = {
            "action": "speak",
            "text": "Подтверждения в архиве не нашлось. Я не знаю ответа.",
            "stage_direction": "закрывает папку",
            "citations": [],
        }
        memory = []
        first = FixedProvider([json.dumps(search), json.dumps(speech)])
        action = _actor_turn(first, "echo", {"act_number": 1}, EmptyTools(), [], set(), memory)
        self.assertTrue(action["_retrieval_completed"])
        self.assertEqual(memory[0]["result"]["hits"], [])
        second = FixedProvider([json.dumps(speech)])
        _actor_turn(second, "echo", {"act_number": 1}, EmptyTools(), [], set(), memory)
        self.assertEqual(second.calls, 1)

    def test_failed_tools_do_not_satisfy_retrieval_requirement(self):
        class BrokenTools(FakeTools):
            def execute(self, name, arguments):
                return {"error": "archive unavailable"}

        search = json.dumps({"action": "tool", "name": "search_files", "arguments": {"query": "архив"}})
        provider = FixedProvider([search, search])
        with self.assertRaisesRegex(GenerationError, "retrieval not completed"):
            _actor_turn(provider, "mark", {"act_number": 1}, BrokenTools(), [], set(), [])
        self.assertEqual(provider.calls, 2)

    def test_dialogue_without_tools_still_allows_first_speech(self):
        speech = {"action": "speak", "text": "Давайте поговорим.", "stage_direction": "тихо", "citations": []}
        provider = FixedProvider([json.dumps(speech)])
        action = _actor_turn(provider, "vera", {"act_number": 1}, None, [], set(), [])
        self.assertEqual(action["text"], speech["text"])
        self.assertEqual(provider.calls, 1)

    def test_long_history_is_bounded_without_changing_saved_dialogue(self):
        original = {
            "speaker": "echo",
            "text": "Нет. " * 420,
            "stage_direction": "долго молчит " * 35,
            "act": 1,
            "addressed_to": "vera",
        }
        compact = _compact([original] * 8)
        self.assertEqual(len(compact), 6)
        self.assertLessEqual(len(compact[0]["text"]), 600)
        self.assertLessEqual(len(compact[0]["stage_direction"]), 200)
        self.assertIn("сокращено", compact[0]["text"])
        self.assertEqual(original["text"], "Нет. " * 420)
        self.assertEqual(original["stage_direction"], "долго молчит " * 35)

    def test_missing_actor_direction_is_generated_by_narrator_without_replacing_speech(self):
        class NoGestureActor(DemoProvider):
            spoken = []

            def complete(self, messages, **kwargs):
                value = json.loads(super().complete(messages, **kwargs))
                if value.get("action") == "speak":
                    self.spoken.append(value["text"])
                    value.pop("stage_direction")
                return json.dumps(value, ensure_ascii=False)

        class GestureNarrator(DemoProvider):
            direction_requests = []

            def complete(self, messages, **kwargs):
                if messages[0]["content"] == NARRATOR_DIRECTION_PROMPT:
                    self.direction_requests.append(json.loads(messages[1]["content"]))
                    return json.dumps({"stage_direction": "Марк складывает схему и отступает от стола."})
                return super().complete(messages, **kwargs)

        providers = demo_providers()
        actor, narrator = NoGestureActor("mark"), GestureNarrator("narrator")
        providers.update(mark=actor, narrator=narrator)
        result = run_play(providers, tools=FakeTools(), acts=1, turns_per_act=3, mode="demo")
        turn = next(t for t in result["turns"] if t["speaker"] == "mark")
        self.assertEqual(turn["text"], actor.spoken[0])
        self.assertEqual(narrator.direction_requests[0]["text"], actor.spoken[0])
        self.assertEqual(turn["stage_direction"], "Марк складывает схему и отступает от стола.")
        self.assertEqual(turn["stage_direction_source"], "narrator")
        self.assertEqual(result["stats"]["stage_completions"], 1)
        self.assertEqual(len([e for e in result["events"] if e["type"] == "stage_completion"]), 1)

    def test_narrator_failure_cannot_create_a_placeholder_direction(self):
        class NoGestureActor(DemoProvider):
            def complete(self, messages, **kwargs):
                value = json.loads(super().complete(messages, **kwargs))
                if value.get("action") == "speak":
                    value["stage_direction"] = ""
                return json.dumps(value, ensure_ascii=False)

        class BrokenNarrator(DemoProvider):
            def complete(self, messages, **kwargs):
                if messages[0]["content"] == NARRATOR_DIRECTION_PROMPT:
                    return '{"stage_direction":""}'
                return super().complete(messages, **kwargs)

        providers = demo_providers()
        providers.update(mark=NoGestureActor("mark"), narrator=BrokenNarrator("narrator"))
        with self.assertRaisesRegex(GenerationError, "stage_direction"):
            run_play(providers, tools=FakeTools(), acts=1, turns_per_act=3, mode="demo")

    def test_optional_actor_gesture_still_rejects_malformed_supplied_values(self):
        speech = {"action": "speak", "text": "Я проверю архив."}
        _validate_action(speech)
        _validate_action({**speech, "stage_direction": ""})
        for invalid in (None, 42, [], "x" * 501):
            with self.subTest(direction=invalid), self.assertRaises(ValueError):
                _validate_action({**speech, "stage_direction": invalid})

    def test_repeated_actor_text_is_repaired_across_characters_and_whitespace(self):
        previous = "Вера, я проверю каждую запись, прежде чем решать судьбу целого архива."
        repeated = {"action": "speak", "text": "  " + previous.upper().replace(" ", "\n ") + "  "}
        fresh = {
            "action": "speak",
            "text": "Марк, проверка важна, но сначала спросим, кто разрешил нам копировать голоса.",
        }
        provider = FixedProvider([json.dumps(repeated), json.dumps(fresh)])
        action = _actor_turn(
            provider, "echo", {"act_number": 1}, None, [], set(), [], previous_actor_texts=[previous]
        )
        self.assertEqual(action["text"], fresh["text"])
        self.assertEqual(provider.calls, 2)

    def test_repeated_actor_text_after_repair_fails_without_substitution(self):
        previous = "Вера, я проверю каждую запись, прежде чем решать судьбу целого архива."
        repeated = json.dumps({"action": "speak", "text": previous})
        provider = FixedProvider([repeated, repeated])
        with self.assertRaisesRegex(GenerationError, "duplicate actor text"):
            _actor_turn(
                provider, "mark", {"act_number": 1}, None, [], set(), [], previous_actor_texts=[previous]
            )
        self.assertEqual(provider.calls, 2)

    def test_distinct_actor_text_needs_no_quality_repair(self):
        previous = "Я сохраню голоса, пока их владельцы сами не попросят об обратном."
        new = "А я спрошу разрешение до переноса: после отключения спрашивать будет поздно."
        provider = FixedProvider([json.dumps({"action": "speak", "text": new})])
        action = _actor_turn(
            provider, "mark", {"act_number": 1}, None, [], set(), [], previous_actor_texts=[previous]
        )
        self.assertEqual(action["text"], new)
        self.assertEqual(provider.calls, 1)

    def test_small_rewording_is_repaired_but_changed_fact_is_kept(self):
        previous = "Вера, давай сначала проверим голос, а потом подпишем акт и вынесем каталог."
        reworded = "Марк, давай сначала проверим голос, а потом вместе подпишем акт и вынесем каталог."
        changed = "Давай не будем проверять голос, а сразу подпишем акт и вынесем каталог."
        provider = FixedProvider([json.dumps({"action": "speak", "text": text}) for text in (reworded, changed)])
        action = _actor_turn(provider, "echo", {}, None, [], set(), [], previous_actor_texts=[previous])
        self.assertEqual(action["text"], changed)
        self.assertEqual(provider.calls, 2)

    def test_duplicate_scene_is_repaired_by_the_real_narrator_protocol(self):
        class RepeatingOnceNarrator(DemoProvider):
            first_scene = None
            repeated = False

            def complete(self, messages, **kwargs):
                context = json.loads(messages[1]["content"])
                if context["act_number"] == 1:
                    self.first_scene = json.loads(super().complete(messages, **kwargs))
                    return json.dumps(self.first_scene)
                if not self.repeated:
                    self.repeated = True
                    return json.dumps({**self.first_scene, "text": self.first_scene["text"].upper()})
                return super().complete(messages, **kwargs)

        providers = demo_providers()
        providers["narrator"] = RepeatingOnceNarrator("narrator")
        result = run_play(providers, tools=FakeTools(), acts=2, turns_per_act=3, mode="demo")
        scenes = [turn["text"] for turn in result["turns"] if turn["speaker"] == "narrator"]
        self.assertNotEqual(scenes[0], scenes[1])
        self.assertEqual(result["stats"]["generation_calls"]["narrator"], 3)
        self.assertEqual(result["stats"]["quality"]["duplicate_scene_texts"], 0)

    def test_persistently_repeated_scene_fails_instead_of_exporting_a_stale_scene(self):
        class RepeatingNarrator(DemoProvider):
            first_scene = None

            def complete(self, messages, **kwargs):
                if self.first_scene is None:
                    self.first_scene = super().complete(messages, **kwargs)
                return self.first_scene

        providers = demo_providers()
        providers["narrator"] = RepeatingNarrator("narrator")
        with self.assertRaisesRegex(GenerationError, "duplicate scene text"):
            run_play(providers, tools=FakeTools(), acts=2, turns_per_act=3, mode="demo")

    def test_old_artifact_export_counts_repeats_and_prominently_warns(self):
        result = run_play(demo_providers(), tools=FakeTools(), acts=2, turns_per_act=3, mode="demo")
        actor_turns = [turn for turn in result["turns"] if turn["speaker"] in CHARACTERS]
        actor_turns[1]["text"] = actor_turns[0]["text"].upper().replace(" ", "  ")
        scene_turns = [turn for turn in result["turns"] if turn["speaker"] == "narrator"]
        scene_turns[1]["text"] = scene_turns[0]["text"]
        result["stats"].pop("quality")  # Simulate an artifact saved before this check existed.
        with tempfile.TemporaryDirectory() as directory:
            paths = save_result(result, directory)
            exported = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
            quality = exported["stats"]["quality"]
            self.assertEqual(quality["unique_actor_texts"], 5)
            self.assertEqual(quality["duplicate_actor_turns"], 1)
            self.assertEqual(quality["duplicate_scene_texts"], 1)
            self.assertEqual(len(quality["warnings"]), 2)
            self.assertIn("ПРЕДУПРЕЖДЕНИЕ О ПОВТОРАХ", exported["metadata"]["warning"])
            self.assertIn("ПРЕДУПРЕЖДЕНИЕ О ПОВТОРАХ", Path(paths["markdown"]).read_text(encoding="utf-8"))
            self.assertIn("ПРЕДУПРЕЖДЕНИЕ О ПОВТОРАХ", Path(paths["html"]).read_text(encoding="utf-8"))
            # Re-export must not multiply the warning, and the caller isn't mutated.
            self.assertNotIn("quality", result["stats"])
            again = save_result(exported, directory)
            reexported = json.loads(Path(again["json"]).read_text(encoding="utf-8"))
            self.assertEqual(reexported["metadata"]["warning"].count("ПРЕДУПРЕЖДЕНИЕ О ПОВТОРАХ"), 1)

    def test_actor_tool_loop_is_bounded(self):
        action = json.dumps({"action": "tool", "name": "search_files", "arguments": {"query": "архив"}})
        provider = FixedProvider([action] * 3)
        events = []
        with self.assertRaisesRegex(GenerationError, "tool budget"):
            _actor_turn(provider, "echo", {"act_number": 1}, FakeTools(), events, set(), [])
        self.assertEqual(provider.calls, 3)
        self.assertEqual(len(events), 2)

    def test_private_character_motives_do_not_leak_into_other_prompts(self):
        vera = actor_prompt("vera", [])
        self.assertIn(CHARACTERS["vera"]["secret"], vera)
        self.assertNotIn(CHARACTERS["echo"]["secret"], vera)
        self.assertNotIn(CHARACTERS["mark"]["secret"], vera)

    def test_evidence_is_retained_and_callback_sees_events(self):
        seen = []
        result = run_play(
            demo_providers(), tools=FakeTools(), turns_per_act=3, acts=2, mode="demo", on_event=seen.append
        )
        self.assertEqual(seen, result["events"])
        sources = {s: set() for s in CHARACTERS}
        # A tool event records real output; citations must be actor-local.
        for event in seen:
            if event["type"] == "tool":
                self.assertIn(event["speaker"], CHARACTERS)
                self.assertNotIn("error", event["result"])
                sources[event["speaker"]].update(h["id"] for h in event["result"].get("hits", []))
        for turn in result["turns"]:
            if turn["speaker"] in sources:
                self.assertLessEqual(set(turn["citations"]), sources[turn["speaker"]])

    def test_export_escapes_untrusted_model_html(self):
        result = run_play(demo_providers(), tools=FakeTools(), turns_per_act=3, acts=1, mode="demo")
        result["title"] = '<script>alert("x")</script>'
        result["turns"][0]["text"] = '<img src=x onerror="alert(1)">'
        with tempfile.TemporaryDirectory() as directory:
            paths = save_result(result, directory)
            exported = Path(paths["html"]).read_text(encoding="utf-8")
            self.assertNotIn("<script>", exported)
            self.assertNotIn("<img", exported)
            self.assertIn("&lt;script&gt;", exported)
            self.assertEqual(
                json.loads(Path(paths["json"]).read_text(encoding="utf-8"))["title"], result["title"]
            )

    def test_input_bounds_and_action_validation(self):
        for kwargs in ({"acts": 0}, {"acts": True}, {"turns_per_act": 2}, {"turns_per_act": 31}):
            with self.assertRaises(ValueError):
                run_play(demo_providers(), mode="demo", **kwargs)
        with self.assertRaises(ValueError):
            _validate_action(
                {"action": "speak", "text": "hi", "stage_direction": "nod", "addressed_to": "unknown"}
            )


if __name__ == "__main__":
    unittest.main()
