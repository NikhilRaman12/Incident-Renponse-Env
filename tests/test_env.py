"""
IncidentResponseEnv — Validation Tests
Run with: python tests/test_env.py
Or:        pytest tests/test_env.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest

from environment.env import make_env
from environment.models import (
    Action, ActionType, AlertCategory, RunbookSection, SeverityLevel,
)


class TestAlertClassification(unittest.TestCase):

    def setUp(self):
        self.env = make_env("alert_classification", seed=42)
        self.obs = self.env.reset(seed=42)

    def test_reset_returns_observation(self):
        self.assertIsNotNone(self.obs)
        self.assertEqual(self.obs.task_type.value, "alert_classification")
        self.assertEqual(self.obs.step, 0)
        self.assertIsNotNone(self.obs.alert)
        self.assertIsNotNone(self.obs.alert.title)

    def test_classify_correct(self):
        action = Action(
            action_type=ActionType.CLASSIFY_ALERT,
            severity=SeverityLevel.P1,
            category=AlertCategory.DATABASE,
            team="database-team",
            notes="DB connection pool exhausted, scheduler bulk job is cause",
        )
        obs, reward, done, info = self.env.step(action)
        self.assertGreater(reward.score, 0.5)
        self.assertTrue(done)
        self.assertIsNotNone(reward.feedback)

    def test_classify_wrong_severity(self):
        action = Action(
            action_type=ActionType.CLASSIFY_ALERT,
            severity=SeverityLevel.P4,
            category=AlertCategory.DATABASE,
            team="database-team",
            notes="Seems low priority",
        )
        obs, reward, done, info = self.env.step(action)
        # P1 misclassified as P4 should be heavily penalised
        self.assertLess(reward.score, 0.4)
        self.assertIn("penalty", reward.feedback.lower())

    def test_analyze_logs_gives_partial_reward(self):
        action = Action(action_type=ActionType.ANALYZE_LOGS)
        obs, reward, done, info = self.env.step(action)
        self.assertGreater(reward.score, 0.0)
        self.assertFalse(done)

    def test_step_after_done_raises(self):
        action = Action(
            action_type=ActionType.CLASSIFY_ALERT,
            severity=SeverityLevel.P1,
            category=AlertCategory.DATABASE,
            team="database-team",
        )
        self.env.step(action)  # terminal
        with self.assertRaises(RuntimeError):
            self.env.step(action)

    def test_state_tracks_episode(self):
        state = self.env.state()
        self.assertEqual(state.step, 0)
        self.assertFalse(state.done)


class TestRootCauseAnalysis(unittest.TestCase):

    def setUp(self):
        self.env = make_env("root_cause_analysis", seed=42)
        self.obs = self.env.reset(seed=42)

    def test_reset_correct_task(self):
        self.assertEqual(self.obs.task_type.value, "root_cause_analysis")
        self.assertEqual(self.obs.max_steps, 8)

    def test_good_rca(self):
        action = Action(
            action_type=ActionType.IDENTIFY_ROOT_CAUSE,
            root_cause_component="scheduler-service",
            root_cause_type="connection_pool_exhaustion",
            evidence=[
                "scheduler-service opened 120 parallel DB connections for bulk export",
                "pgbouncer reports: too many clients already",
                "api error rate 34.2% correlates with bulk job start time",
            ],
            impact="34% API error rate, payment-service and user-service degraded",
            affected_services=["api-service", "payment-service", "pgbouncer"],
        )
        obs, reward, done, info = self.env.step(action)
        self.assertGreater(reward.score, 0.4)
        self.assertTrue(done)

    def test_missing_component_invalid(self):
        action = Action(
            action_type=ActionType.IDENTIFY_ROOT_CAUSE,
            root_cause_type="connection_pool_exhaustion",
            evidence=["some evidence"],
        )
        obs, reward, done, info = self.env.step(action)
        self.assertEqual(reward.score, 0.0)
        self.assertFalse(done)  # invalid action doesn't terminate

    def test_state_after_steps(self):
        action = Action(action_type=ActionType.ANALYZE_LOGS)
        self.env.step(action)
        state = self.env.state()
        self.assertEqual(state.step, 1)
        self.assertEqual(len(state.actions_taken), 1)


class TestRunbookGeneration(unittest.TestCase):

    def setUp(self):
        self.env = make_env("runbook_generation", seed=42)
        self.obs = self.env.reset(seed=42)

    def test_complete_runbook_scores_high(self):
        action = Action(
            action_type=ActionType.GENERATE_RUNBOOK,
            runbook=RunbookSection(
                diagnosis_steps=[
                    "Run: SELECT count(*), state FROM pg_stat_activity GROUP BY state",
                    "Identify top connection holders: SELECT application_name, count(*) FROM pg_stat_activity GROUP BY 1",
                    "kubectl logs -l app=scheduler-service --tail=50 | grep bulk-report",
                ],
                remediation_steps=[
                    "kubectl delete job bulk-report-export",
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='idle'",
                    "kubectl scale deployment api-service --replicas=10",
                    "Monitor error rate: watch kubectl top pods",
                ],
                rollback_plan=[
                    "kubectl rollout undo deployment/scheduler-service",
                    "kubectl rollout status deployment/scheduler-service",
                    "Verify connection pool utilization drops below 70%",
                ],
                escalation_criteria=[
                    "Escalate to database-team lead if error rate doesn't drop within 10 minutes",
                    "Page on-call DBA if primary DB becomes unresponsive",
                ],
                prevention_measures=[
                    "Add max_parallelism=10 to all bulk jobs",
                    "Create dedicated read-replica connection pool for reporting",
                    "Add connection pool alert at 70% utilization",
                ],
                commands=[
                    "kubectl delete job bulk-report-export",
                    "psql -h prod-db-primary -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='idle';\"",
                    "kubectl rollout undo deployment/scheduler-service",
                ],
                expected_resolution_time="15-20 minutes",
            ),
        )
        obs, reward, done, info = self.env.step(action)
        self.assertGreater(reward.score, 0.5)
        self.assertTrue(done)

    def test_empty_runbook_scores_zero(self):
        action = Action(
            action_type=ActionType.GENERATE_RUNBOOK,
            runbook=RunbookSection(),  # all empty
        )
        obs, reward, done, info = self.env.step(action)
        self.assertEqual(reward.score, 0.0)

    def test_missing_runbook_invalid(self):
        action = Action(action_type=ActionType.GENERATE_RUNBOOK)
        obs, reward, done, info = self.env.step(action)
        self.assertEqual(reward.score, 0.0)


class TestReproducibility(unittest.TestCase):

    def test_same_seed_same_scenario(self):
        env1 = make_env("alert_classification", seed=7)
        env2 = make_env("alert_classification", seed=7)
        obs1 = env1.reset(seed=7)
        obs2 = env2.reset(seed=7)
        self.assertEqual(obs1.alert.title, obs2.alert.title)
        self.assertEqual(obs1.alert.service, obs2.alert.service)

    def test_different_seeds_may_differ(self):
        env1 = make_env("alert_classification", seed=1)
        env2 = make_env("alert_classification", seed=99)
        obs1 = env1.reset(seed=1)
        obs2 = env2.reset(seed=99)
        # Can't guarantee different every time with small scenario pool, just check it runs
        self.assertIsNotNone(obs1.alert)
        self.assertIsNotNone(obs2.alert)


if __name__ == "__main__":
    unittest.main(verbosity=2)
