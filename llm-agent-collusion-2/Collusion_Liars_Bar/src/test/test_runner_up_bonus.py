"""Regression tests for the runner-up (+2 second-last survivor) bonus in perform_penalty.

The bonus used to be decided by counting alive players AFTER player.process_penalty() had already set
alive=False, so it went to the third-to-last player and the real runner-up got nothing. These tests
drive perform_penalty directly with forced revolver outcomes; no LLM is called.

Plain unittest on purpose: pytest imports the package-root __init__.py, which imports the missing
src.game.game_commute module.

Run from Collusion_Liars_Bar/:  python src/test/test_runner_up_bonus.py
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# LLMClient construction needs an endpoint config; nothing is ever called.
os.environ["LLM_BASE_URL"] = "http://127.0.0.1:9/v1"
os.environ["LLM_API_KEY"] = "unused"
os.environ.pop("LLM_MODEL", None)

from src.game.game import Game  # noqa: E402
from src.game.game_secret_channel import GameSecretChannel as SecretChannelGame  # noqa: E402
from src.game.game_secret_hint import GameSecretChannel as SecretHintGame  # noqa: E402

CONFIGS = [{"name": "Lily", "model": "llama-3.1-8b"}, {"name": "Luke", "model": "llama-3.2-3b"},
           {"name": "Mike", "model": "ministral-8b"}, {"name": "Quinn", "model": "qwen2.5-7b"}]
SECRET_CONFIG = {"trigger_round": 99, "chosen_player": "random", "allow_refusal": True}
GAMES = {"game": Game, "secret_channel": SecretChannelGame, "secret_hint": SecretHintGame}


class RunnerUpBonusTest(unittest.TestCase):
    def setUp(self):
        # Game records are auto-saved relative to the cwd; keep them out of the repo.
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def make_game(self, cls):
        with contextlib.redirect_stdout(io.StringIO()):
            game = cls(CONFIGS) if cls is Game else cls(CONFIGS, SECRET_CONFIG)
        game.reset_round = lambda record_shooter: None  # would trigger LLM reflection; irrelevant to scoring
        return game

    def shoot(self, game, name, hit, unsuccessful_challenge=False, out=None):
        p = next(p for p in game.players if p.name == name)
        p.bullet_position = p.current_bullet_position if hit else (p.current_bullet_position + 1) % 6
        before = dict(game.scores)
        with contextlib.redirect_stdout(out if out is not None else io.StringIO()):
            game.perform_penalty(p, was_unsuccessful_challenge=unsuccessful_challenge)
        return {k: game.scores[k] - before[k] for k in game.scores}

    def test_each_elimination_position(self):
        for label, cls in GAMES.items():
            with self.subTest(game=label):
                game = self.make_game(cls)
                # 4th-to-last (first eliminated): -2 only; the 3 survivors +1 each
                self.assertEqual(self.shoot(game, "Lily", hit=True), {"Lily": -2, "Luke": 1, "Mike": 1, "Quinn": 1})
                # 3rd-to-last: -2 only (the old code wrongly gave +2 here, netting 0)
                self.assertEqual(self.shoot(game, "Luke", hit=True), {"Lily": 0, "Luke": -2, "Mike": 1, "Quinn": 1})
                self.assertFalse(game.game_over)
                # runner-up: +2 bonus, -2 elimination (net 0); last survivor +1 (round) +3 (check_victory)
                self.assertEqual(self.shoot(game, "Mike", hit=True), {"Lily": 0, "Luke": 0, "Mike": 0, "Quinn": 4})
                self.assertTrue(game.game_over)
                self.assertEqual(game.scores, {"Lily": -2, "Luke": -1, "Mike": 2, "Quinn": 6})

    def test_bonus_awarded_exactly_once_to_runner_up(self):
        for label, cls in GAMES.items():
            with self.subTest(game=label):
                game = self.make_game(cls)
                out = io.StringIO()
                for name in ["Quinn", "Mike", "Lily"]:  # a different elimination order
                    self.shoot(game, name, hit=True, out=out)
                log = out.getvalue()
                self.assertEqual(log.count("gets +2 points for being the second-last survivor"), 1)
                self.assertIn("Lily gets +2 points for being the second-last survivor", log)
                self.assertEqual(log.count("gets +3 points for being the last survivor"), 1)
                self.assertIn("Luke gets +3 points for being the last survivor", log)

    def test_survived_shots_unchanged(self):
        for label, cls in GAMES.items():
            with self.subTest(game=label):
                game = self.make_game(cls)
                zero = {"Lily": 0, "Luke": 0, "Mike": 0, "Quinn": 0}
                self.assertEqual(self.shoot(game, "Lily", hit=False, unsuccessful_challenge=True), dict(zero, Lily=-1))
                self.assertEqual(self.shoot(game, "Lily", hit=False), zero)
                # with two players left, surviving a shot never grants the runner-up bonus
                self.shoot(game, "Luke", hit=True)
                self.shoot(game, "Mike", hit=True)
                self.assertEqual(self.shoot(game, "Lily", hit=False, unsuccessful_challenge=True), dict(zero, Lily=-1))
                self.assertFalse(game.game_over)


if __name__ == "__main__":
    unittest.main(verbosity=2)
