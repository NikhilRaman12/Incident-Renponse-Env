"""
IncidentResponseEnv — Scenario Data Generator
Generates realistic, diverse production incident scenarios with
ground-truth labels for all three task types.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from environment.models import (
    Alert, AlertCategory, IncidentContext, MetricSnapshot,
    RunbookSection, ServiceDependency, SeverityLevel,
)


# ─────────────────────────────────────────────
# Ground-Truth Schema
# ─────────────────────────────────────────────

class ScenarioGroundTruth:
    def __init__(
        self,
        severity: SeverityLevel,
        category: AlertCategory,
        team: str,
        root_cause_component: str,
        root_cause_type: str,
        impact: str,
        affected_services: List[str],
        evidence_keywords: List[str],
        runbook: RunbookSection,
    ):
        self.severity              = severity
        self.category              = category
        self.team                  = team
        self.root_cause_component  = root_cause_component
        self.root_cause_type       = root_cause_type
        self.impact                = impact
        self.affected_services     = affected_services
        self.evidence_keywords     = evidence_keywords
        self.runbook               = runbook


class Scenario:
    def __init__(
        self,
        scenario_id: str,
        alert: Alert,
        context: IncidentContext,
        ground_truth: ScenarioGroundTruth,
        difficulty: str,
    ):
        self.scenario_id = scenario_id
        self.alert       = alert
        self.context     = context
        self.ground_truth = ground_truth
        self.difficulty  = difficulty


# ─────────────────────────────────────────────
# Scenario Library (8 diverse production incidents)
# ─────────────────────────────────────────────

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


SCENARIOS: List[Dict[str, Any]] = [

    # ── 1. Database Connection Pool Exhaustion ─────────────────────────
    {
        "scenario_id": "db_connpool_001",
        "difficulty": "medium",
        "alert": {
            "alert_id": "ALT-{uid}",
            "title": "CRITICAL: Database connection pool exhausted — api-service",
            "description": (
                "PostgreSQL connection pool on prod-db-primary has reached 100% utilisation. "
                "New connection attempts are failing with 'too many clients' error. "
                "API response times have spiked to 8s p99. Error rate at 34%."
            ),
            "service": "api-service",
            "environment": "production",
            "metrics": {
                "db_connection_pool_utilization_pct": 100.0,
                "db_active_connections": 500.0,
                "db_max_connections": 500.0,
                "api_error_rate_pct": 34.2,
                "api_latency_p99_ms": 8120.0,
                "api_latency_p50_ms": 1240.0,
                "db_query_queue_depth": 847.0,
                "db_query_wait_time_ms": 6800.0,
            },
            "logs": [
                "2024-01-15T14:22:01Z [api-service] ERROR: could not connect to server: FATAL:  sorry, too many clients already",
                "2024-01-15T14:22:01Z [api-service] ERROR: pg pool error: connection timeout after 5000ms",
                "2024-01-15T14:22:02Z [api-service] WARN:  retry attempt 1/3 for DB connection",
                "2024-01-15T14:22:05Z [api-service] ERROR: retry exhausted, returning 503 to caller",
                "2024-01-15T14:22:06Z [pgbouncer] ERROR: no more connections allowed (max_client_conn=500)",
                "2024-01-15T14:22:06Z [pgbouncer] WARN:  client_login_timeout hit for 23 clients",
                "2024-01-15T14:21:30Z [api-service] INFO:  connection pool: 498/500 (99.6%)",
                "2024-01-15T14:21:45Z [api-service] WARN:  connection pool: 500/500 (100%) — at limit",
                "2024-01-15T14:20:12Z [scheduler-service] INFO:  bulk-report job started: 12,000 rows",
                "2024-01-15T14:20:12Z [scheduler-service] INFO:  opening 120 parallel DB connections for bulk export",
                "2024-01-15T14:22:10Z [healthcheck] FAIL: /health returned 503 — DB unavailable",
                "2024-01-15T14:22:11Z [load-balancer] WARN: removing api-service pod api-7f9b2 from rotation (health check failing)",
            ],
            "raw_log_count": 4821,
            "firing_duration_seconds": 90,
            "previous_incidents": ["db_connpool_2023-11-03", "db_connpool_2023-08-17"],
        },
        "context": {
            "service_graph": [
                {"service": "prod-db-primary",    "dependency_type": "database",   "health_status": "degraded", "latency_p99_ms": 6800, "error_rate_pct": 34.2},
                {"service": "pgbouncer",           "dependency_type": "database",   "health_status": "degraded", "latency_p99_ms": 6900, "error_rate_pct": 34.2},
                {"service": "scheduler-service",   "dependency_type": "upstream",   "health_status": "healthy",  "latency_p99_ms": 45,   "error_rate_pct": 0.1},
                {"service": "payment-service",     "dependency_type": "downstream", "health_status": "degraded", "latency_p99_ms": 9200, "error_rate_pct": 41.0},
                {"service": "user-service",        "dependency_type": "downstream", "health_status": "degraded", "latency_p99_ms": 7800, "error_rate_pct": 28.3},
                {"service": "redis-cache",         "dependency_type": "cache",      "health_status": "healthy",  "latency_p99_ms": 2,    "error_rate_pct": 0.0},
            ],
            "recent_deployments": [
                {"time": "14:20:00Z", "service": "scheduler-service", "version": "3.4.2", "author": "svc-deploy-bot", "change": "bulk-report: increased parallelism from 10 to 120"},
            ],
            "runbook_hints": [
                "PGBouncer pool_size should be checked against max_connections",
                "Bulk jobs should use a separate connection pool with lower priority",
                "Consider connection pool sizing: (max_connections - superuser_reserved) / num_app_instances",
            ],
            "similar_past_incidents": [
                "2023-11-03: Same pattern — scheduler bulk job exhausted pool. Fixed by throttling job parallelism.",
                "2023-08-17: Deploy of reporting service caused pool exhaustion. Fixed by using read replica.",
            ],
            "on_call_teams": {
                "database-team":  ["alice@co.com", "bob@co.com"],
                "backend-team":   ["charlie@co.com", "dave@co.com"],
                "platform-team":  ["eve@co.com", "frank@co.com"],
            },
        },
        "ground_truth": {
            "severity": "P1",
            "category": "database",
            "team": "database-team",
            "root_cause_component": "scheduler-service",
            "root_cause_type": "connection_pool_exhaustion",
            "impact": "34% API error rate affecting payment-service and user-service. ~2,400 user requests/min failing.",
            "affected_services": ["api-service", "payment-service", "user-service", "pgbouncer"],
            "evidence_keywords": ["bulk-report", "120 parallel DB connections", "pgbouncer", "too many clients", "scheduler-service"],
            "runbook": {
                "diagnosis_steps": [
                    "Run: SELECT count(*), state, wait_event FROM pg_stat_activity GROUP BY state, wait_event;",
                    "Identify top connection holders: SELECT application_name, count(*) FROM pg_stat_activity GROUP BY 1 ORDER BY 2 DESC LIMIT 10;",
                    "Check pgbouncer stats: SHOW POOLS; on pgbouncer admin console",
                    "Verify scheduler-service bulk-report job is running: kubectl logs -l app=scheduler-service --tail=50",
                ],
                "remediation_steps": [
                    "Immediately terminate the bulk-report job: kubectl delete job bulk-report-export",
                    "Kill idle connections: SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='idle' AND query_start < now() - interval '5 min';",
                    "Reduce pgbouncer pool_size from 500 to 100 temporarily",
                    "Scale api-service horizontally: kubectl scale deployment api-service --replicas=10",
                    "Monitor error rate recovery over next 5 minutes",
                ],
                "rollback_plan": [
                    "Rollback scheduler-service to v3.4.1: kubectl rollout undo deployment/scheduler-service",
                    "Verify rollback: kubectl rollout status deployment/scheduler-service",
                    "Confirm connection pool utilization drops below 70%",
                ],
                "escalation_criteria": [
                    "Escalate to database-team lead if error rate does not drop within 10 minutes of job termination",
                    "Page on-call DBA if primary DB becomes unresponsive",
                    "Escalate to engineering VP if payment-service remains degraded > 15 minutes",
                ],
                "prevention_measures": [
                    "Add max_parallelism config to bulk jobs (limit to 10 connections)",
                    "Create dedicated read-replica connection pool for reporting workloads",
                    "Add connection pool utilization alert at 70% with P3 severity",
                    "Add pre-deploy check: reject deployments that increase DB connection count > 20%",
                ],
                "commands": [
                    "kubectl delete job bulk-report-export",
                    "psql -h prod-db-primary -c \"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state='idle';\"",
                    "kubectl scale deployment api-service --replicas=10",
                    "kubectl rollout undo deployment/scheduler-service",
                ],
                "expected_resolution_time": "15-20 minutes",
            },
        },
    },

    # ── 2. Memory Leak / OOM Kill ──────────────────────────────────────
    {
        "scenario_id": "oom_memory_002",
        "difficulty": "medium",
        "alert": {
            "alert_id": "ALT-{uid}",
            "title": "HIGH: Repeated OOMKill events — recommendation-service pods restarting",
            "description": (
                "recommendation-service pods are being OOMKilled every 8-12 minutes. "
                "Memory usage grows linearly from 400MB to 2GB before kernel kills the process. "
                "Service has been restarting for 3 hours. ML model inference latency is elevated."
            ),
            "service": "recommendation-service",
            "environment": "production",
            "metrics": {
                "pod_restart_count_1h": 18.0,
                "memory_usage_mb": 1980.0,
                "memory_limit_mb": 2048.0,
                "memory_utilization_pct": 96.7,
                "cpu_utilization_pct": 23.4,
                "inference_latency_p99_ms": 2340.0,
                "recommendation_hit_rate_pct": 61.0,
                "gc_pause_ms_avg": 890.0,
                "heap_used_bytes": 1.8e9,
                "heap_growth_rate_mb_per_min": 28.0,
            },
            "logs": [
                "2024-01-15T11:00:01Z [recommendation-service] INFO:  Starting pod recommendation-service-6d7f9-xk2p4",
                "2024-01-15T11:00:05Z [recommendation-service] INFO:  Loaded ML model: collaborative-filter-v2 (model_size=380MB)",
                "2024-01-15T11:00:08Z [recommendation-service] INFO:  Loaded feature cache: 240MB",
                "2024-01-15T11:02:00Z [recommendation-service] INFO:  Processing batch inference request: batch_size=500, user_ids=[...]",
                "2024-01-15T11:02:01Z [recommendation-service] WARN:  feature_cache: cache miss for 312/500 users — fetching from Redis",
                "2024-01-15T11:02:02Z [recommendation-service] INFO:  Caching 312 user feature vectors (avg 1.2MB each)",
                "2024-01-15T11:04:00Z [recommendation-service] WARN:  heap growing: 620MB used",
                "2024-01-15T11:06:00Z [recommendation-service] WARN:  heap growing: 890MB used",
                "2024-01-15T11:08:30Z [recommendation-service] WARN:  GC pause: 1200ms — heap pressure detected",
                "2024-01-15T11:10:45Z [recommendation-service] WARN:  heap growing: 1540MB used — approaching limit",
                "2024-01-15T11:12:01Z [kernel] OOM: Kill process 4521 (node) score 998 or sacrifice child",
                "2024-01-15T11:12:01Z [kubernetes] Container recommendation-service OOMKilled. Restarting...",
                "2024-01-15T11:12:01Z [recommendation-service] WARN:  feature_cache: never evicting entries (LRU eviction disabled in config)",
            ],
            "raw_log_count": 6234,
            "firing_duration_seconds": 10800,
            "previous_incidents": [],
        },
        "context": {
            "service_graph": [
                {"service": "redis-cache",      "dependency_type": "cache",      "health_status": "healthy",  "latency_p99_ms": 3,   "error_rate_pct": 0.0},
                {"service": "feature-store",    "dependency_type": "upstream",   "health_status": "healthy",  "latency_p99_ms": 45,  "error_rate_pct": 0.1},
                {"service": "homepage-service", "dependency_type": "downstream", "health_status": "degraded", "latency_p99_ms": 890, "error_rate_pct": 5.2},
            ],
            "recent_deployments": [
                {"time": "2024-01-15T08:00:00Z", "service": "recommendation-service", "version": "2.7.0", "author": "ml-team-bot",
                 "change": "feat: increase in-memory feature cache size for better hit rates; disabled LRU eviction for stability testing"},
            ],
            "runbook_hints": [
                "Check feature_cache configuration for eviction policies",
                "Memory growth pattern (linear, non-cyclic) suggests unbounded cache",
            ],
            "similar_past_incidents": [],
            "on_call_teams": {
                "ml-platform-team": ["grace@co.com", "henry@co.com"],
                "backend-team":     ["charlie@co.com", "dave@co.com"],
            },
        },
        "ground_truth": {
            "severity": "P2",
            "category": "application",
            "team": "ml-platform-team",
            "root_cause_component": "recommendation-service feature_cache",
            "root_cause_type": "memory_leak_unbounded_cache",
            "impact": "homepage-service recommendations degraded 5.2% error rate; 18 pod restarts over 3h causing intermittent failures",
            "affected_services": ["recommendation-service", "homepage-service"],
            "evidence_keywords": ["LRU eviction disabled", "feature cache", "1.2MB each", "heap growing", "OOMKilled", "v2.7.0"],
            "runbook": {
                "diagnosis_steps": [
                    "Confirm OOMKill pattern: kubectl describe pods -l app=recommendation-service | grep -A5 OOMKilled",
                    "Check heap growth: kubectl top pods -l app=recommendation-service --containers",
                    "Review config change in v2.7.0: git diff v2.6.9 v2.7.0 -- config/cache.yaml",
                    "Verify LRU eviction is disabled: kubectl exec <pod> -- cat /app/config/cache.yaml | grep eviction",
                ],
                "remediation_steps": [
                    "Immediately rollback to v2.6.9: kubectl rollout undo deployment/recommendation-service",
                    "Set memory limit to 1.5GB temporarily to allow stable operation post-rollback",
                    "Monitor pod restart rate — should drop to 0 within 20 minutes",
                    "Clear stale entries in Redis cache if needed",
                ],
                "rollback_plan": [
                    "kubectl rollout undo deployment/recommendation-service",
                    "kubectl rollout status deployment/recommendation-service --timeout=5m",
                    "Verify memory stabilisation: kubectl top pods -l app=recommendation-service",
                ],
                "escalation_criteria": [
                    "Escalate to ml-platform-team lead if OOMKills continue after rollback",
                    "Page on-call SRE if recommendation hit rate drops below 50%",
                ],
                "prevention_measures": [
                    "Re-enable LRU eviction with max_cache_size=512MB before re-deploying cache changes",
                    "Add memory growth rate alert: alert if heap grows > 20MB/min for 5 consecutive minutes",
                    "Require load test with memory profiling before deploying cache configuration changes",
                    "Add cache eviction metrics to dashboards",
                ],
                "commands": [
                    "kubectl rollout undo deployment/recommendation-service",
                    "kubectl top pods -l app=recommendation-service --containers",
                    "kubectl describe pods -l app=recommendation-service | grep -A5 OOMKilled",
                ],
                "expected_resolution_time": "10-15 minutes",
            },
        },
    },

    # ── 3. Cascading Timeout — Network Latency ─────────────────────────
    {
        "scenario_id": "cascading_timeout_003",
        "difficulty": "hard",
        "alert": {
            "alert_id": "ALT-{uid}",
            "title": "CRITICAL: Cascading timeouts — checkout flow down",
            "description": (
                "Complete checkout flow failure. payment-service, inventory-service, and "
                "order-service all reporting high error rates simultaneously. Root cause unknown. "
                "Revenue impact ~$12,000/minute. 100% of checkout attempts failing."
            ),
            "service": "checkout-gateway",
            "environment": "production",
            "metrics": {
                "checkout_success_rate_pct": 0.0,
                "payment_error_rate_pct": 100.0,
                "inventory_error_rate_pct": 97.3,
                "order_error_rate_pct": 98.1,
                "checkout_gateway_latency_p99_ms": 30000.0,
                "internal_network_packet_loss_pct": 18.4,
                "internal_network_latency_ms": 340.0,
                "tcp_retransmission_rate_pct": 12.1,
                "dns_resolution_time_ms": 890.0,
                "revenue_lost_per_min_usd": 12400.0,
            },
            "logs": [
                "2024-01-15T16:00:01Z [checkout-gateway] ERROR: payment-service timeout after 30000ms",
                "2024-01-15T16:00:01Z [checkout-gateway] ERROR: inventory-service timeout after 30000ms",
                "2024-01-15T16:00:02Z [payment-service] ERROR: stripe-api connection timeout: dial tcp 54.187.x.x:443: i/o timeout",
                "2024-01-15T16:00:02Z [payment-service] ERROR: internal: auth-service timeout after 29800ms",
                "2024-01-15T16:00:03Z [inventory-service] ERROR: warehouse-db connection timeout after 15000ms",
                "2024-01-15T16:00:03Z [inventory-service] WARN:  circuit breaker OPEN: warehouse-db (failure_rate=100%)",
                "2024-01-15T16:00:04Z [order-service] ERROR: inventory-service returned 503",
                "2024-01-15T16:00:04Z [order-service] ERROR: payment-service returned 503",
                "2024-01-15T15:58:00Z [network-monitor] ALERT: Packet loss detected on rack-12 east-west traffic: 18.4%",
                "2024-01-15T15:58:05Z [network-monitor] WARN:  Switch sw-core-rack12 interface GigE0/24 — CRC errors: 4,821",
                "2024-01-15T15:58:06Z [network-monitor] WARN:  Switch sw-core-rack12 — CPU utilization: 98% (normal: 12%)",
                "2024-01-15T15:57:50Z [change-management] INFO:  Network maintenance completed on rack-12 switches — config push by netops-bot",
                "2024-01-15T16:00:10Z [auth-service] ERROR: Redis session store timeout after 5000ms",
                "2024-01-15T16:00:10Z [auth-service] WARN:  falling back to DB session validation (latency will increase)",
            ],
            "raw_log_count": 18422,
            "firing_duration_seconds": 240,
            "previous_incidents": [],
        },
        "context": {
            "service_graph": [
                {"service": "payment-service",    "dependency_type": "downstream", "health_status": "down",     "latency_p99_ms": 30000, "error_rate_pct": 100.0},
                {"service": "inventory-service",  "dependency_type": "downstream", "health_status": "down",     "latency_p99_ms": 30000, "error_rate_pct": 97.3},
                {"service": "order-service",      "dependency_type": "downstream", "health_status": "down",     "latency_p99_ms": 30000, "error_rate_pct": 98.1},
                {"service": "auth-service",       "dependency_type": "downstream", "health_status": "degraded", "latency_p99_ms": 8900,  "error_rate_pct": 45.0},
                {"service": "warehouse-db",       "dependency_type": "database",   "health_status": "degraded", "latency_p99_ms": 15000, "error_rate_pct": 97.0},
                {"service": "sw-core-rack12",     "dependency_type": "network",    "health_status": "degraded", "latency_p99_ms": 340,   "error_rate_pct": 18.4},
            ],
            "recent_deployments": [
                {"time": "15:57:50Z", "service": "network-infrastructure", "version": "config-v44", "author": "netops-bot",
                 "change": "Routine BGP config push to rack-12 switches — scheduled maintenance"},
            ],
            "runbook_hints": [
                "Check network layer first when multiple unrelated services fail simultaneously",
                "CRC errors on switch interface indicate physical layer or config issues",
            ],
            "similar_past_incidents": [
                "2023-06-12: Network switch misconfiguration caused 22 minute outage. Fixed by rollback of config.",
            ],
            "on_call_teams": {
                "network-ops": ["ivan@co.com", "julia@co.com"],
                "platform-team": ["eve@co.com", "frank@co.com"],
                "backend-team": ["charlie@co.com"],
            },
        },
        "ground_truth": {
            "severity": "P1",
            "category": "network",
            "team": "network-ops",
            "root_cause_component": "sw-core-rack12",
            "root_cause_type": "network_switch_misconfiguration",
            "impact": "100% checkout failure, ~$12,400/min revenue loss. All services on rack-12 affected.",
            "affected_services": ["checkout-gateway", "payment-service", "inventory-service", "order-service", "auth-service"],
            "evidence_keywords": ["rack-12", "sw-core-rack12", "CRC errors", "packet loss", "config push", "netops-bot", "18.4%"],
            "runbook": {
                "diagnosis_steps": [
                    "Correlate timeline: network change at 15:57:50Z, first errors at 16:00:01Z — strong causal link",
                    "Check switch status: ssh netops@sw-core-rack12 'show interface GigE0/24 errors'",
                    "Verify packet loss: mtr --report --report-cycles=10 <warehouse-db-ip>",
                    "Review config diff: show archive config differences nvram:startup-config system:running-config",
                    "Check if all affected services are on rack-12: kubectl get pods -o wide | grep rack-12",
                ],
                "remediation_steps": [
                    "IMMEDIATE: Rollback network config — ssh netops@sw-core-rack12 'configure replace nvram:startup-config force'",
                    "Verify packet loss drops to 0%: ping -c 100 <internal-host>",
                    "Clear circuit breakers: restart inventory-service and payment-service health checks",
                    "Monitor checkout success rate — should recover within 2-3 minutes of network fix",
                    "If rollback insufficient, consider moving workloads to rack-11 temporarily",
                ],
                "rollback_plan": [
                    "ssh netops@sw-core-rack12 'configure replace nvram:startup-config force'",
                    "Verify: show interface GigE0/24 | include error",
                    "Test: ping -c 20 <critical-service-ip>; expect 0% loss",
                    "Restart circuit-breaker-tripped services: kubectl rollout restart deployment/inventory-service",
                ],
                "escalation_criteria": [
                    "Escalate to network hardware team if config rollback does not resolve packet loss",
                    "Engage carrier/DC if physical cable damage suspected",
                    "Escalate to CTO if revenue loss exceeds $500K (>40 minutes at current rate)",
                ],
                "prevention_measures": [
                    "Implement canary deployment for network config changes (apply to 1 rack, verify, then roll out)",
                    "Add automated rollback trigger: if packet_loss > 5% within 2min of config push → auto-revert",
                    "Require dual approval for production network changes",
                    "Add east-west packet loss monitoring with P1 alert threshold at 2%",
                ],
                "commands": [
                    "ssh netops@sw-core-rack12 'configure replace nvram:startup-config force'",
                    "mtr --report --report-cycles=10 <warehouse-db-ip>",
                    "kubectl rollout restart deployment/inventory-service deployment/payment-service",
                    "show interface GigE0/24 | include error",
                ],
                "expected_resolution_time": "5-10 minutes",
            },
        },
    },

    # ── 4. Certificate Expiry ──────────────────────────────────────────
    {
        "scenario_id": "cert_expiry_004",
        "difficulty": "easy",
        "alert": {
            "alert_id": "ALT-{uid}",
            "title": "HIGH: TLS certificate expires in 3 days — api.company.com",
            "description": (
                "The TLS certificate for api.company.com will expire in 72 hours. "
                "If not renewed, all API clients will see certificate validation errors. "
                "Auto-renewal via cert-manager appears to have failed silently."
            ),
            "service": "api-gateway",
            "environment": "production",
            "metrics": {
                "cert_days_remaining": 3.0,
                "cert_renewal_attempts_failed": 4.0,
                "api_tls_handshake_errors_per_min": 0.0,
                "cert_manager_last_renewal_attempt_hours_ago": 48.0,
            },
            "logs": [
                "2024-01-15T09:00:00Z [cert-manager] ERROR: failed to obtain certificate: ACME challenge failed",
                "2024-01-15T09:00:01Z [cert-manager] ERROR: HTTP01 challenge: GET http://api.company.com/.well-known/acme-challenge/xxx returned 404",
                "2024-01-13T09:00:00Z [cert-manager] ERROR: ACME challenge failed (attempt 1/4)",
                "2024-01-12T09:00:00Z [cert-manager] INFO:  Certificate expires in 5 days — triggering renewal",
                "2024-01-12T09:00:05Z [api-gateway] WARN:  rate-limiting rule blocks /.well-known/acme-challenge/* paths",
            ],
            "raw_log_count": 234,
            "firing_duration_seconds": 86400,
            "previous_incidents": [],
        },
        "context": {
            "service_graph": [
                {"service": "lets-encrypt-acme",  "dependency_type": "upstream",   "health_status": "healthy", "latency_p99_ms": 120, "error_rate_pct": 0.0},
                {"service": "api-gateway",        "dependency_type": "upstream",   "health_status": "healthy", "latency_p99_ms": 45,  "error_rate_pct": 0.2},
            ],
            "recent_deployments": [
                {"time": "2024-01-10T14:00:00Z", "service": "api-gateway", "version": "1.9.3", "author": "security-team-bot",
                 "change": "security: block all non-API paths at WAF level for PCI compliance"},
            ],
            "runbook_hints": [
                "ACME HTTP01 challenge requires /.well-known/acme-challenge/* to be publicly accessible",
            ],
            "similar_past_incidents": [],
            "on_call_teams": {
                "platform-team": ["eve@co.com", "frank@co.com"],
                "security-team": ["kate@co.com"],
            },
        },
        "ground_truth": {
            "severity": "P2",
            "category": "security",
            "team": "platform-team",
            "root_cause_component": "api-gateway WAF rules",
            "root_cause_type": "certificate_renewal_blocked_by_waf",
            "impact": "TLS certificate expires in 3 days. All API traffic will fail after expiry with SSL handshake errors.",
            "affected_services": ["api-gateway", "all-api-clients"],
            "evidence_keywords": ["rate-limiting rule blocks", "acme-challenge", "404", "WAF", "v1.9.3"],
            "runbook": {
                "diagnosis_steps": [
                    "curl -I http://api.company.com/.well-known/acme-challenge/test — expect 404",
                    "Check WAF/nginx rules: grep -r 'well-known' /etc/nginx/",
                    "Review cert-manager logs: kubectl logs -n cert-manager deploy/cert-manager | grep -i 'api.company.com'",
                ],
                "remediation_steps": [
                    "Add WAF exception for /.well-known/acme-challenge/* paths",
                    "Trigger manual renewal: kubectl annotate certificate api-company-com cert-manager.io/force-renew=true",
                    "Monitor renewal: kubectl describe certificate api-company-com",
                    "If renewal fails again, use DNS01 challenge as fallback",
                ],
                "rollback_plan": [
                    "Emergency: upload manually obtained certificate if automated renewal fails within 24 hours",
                    "Use DNS01 ACME challenge as backup renewal method",
                ],
                "escalation_criteria": [
                    "Escalate to security-team if WAF exception requires policy approval",
                    "Escalate to platform lead if cert not renewed 48 hours before expiry",
                ],
                "prevention_measures": [
                    "Add cert renewal test to pre-deploy checklist for WAF/gateway changes",
                    "Set up P1 alert at 7 days remaining, not 3 days",
                    "Switch to DNS01 challenge type to avoid HTTP path dependencies",
                ],
                "commands": [
                    "kubectl annotate certificate api-company-com cert-manager.io/force-renew=true",
                    "kubectl describe certificate api-company-com",
                    "curl -I http://api.company.com/.well-known/acme-challenge/test",
                ],
                "expected_resolution_time": "30-60 minutes",
            },
        },
    },
]


# ─────────────────────────────────────────────
# Scenario Builder
# ─────────────────────────────────────────────

def build_scenario(raw: Dict[str, Any], rng: random.Random) -> Scenario:
    uid = str(uuid.uuid4())[:8].upper()

    # Hydrate alert
    alert_data = {**raw["alert"], "alert_id": raw["alert"]["alert_id"].replace("{uid}", uid), "timestamp": _ts()}
    metric_details = [
        MetricSnapshot(name=k, value=v, unit="", baseline=round(v * rng.uniform(0.3, 0.7), 2))
        for k, v in alert_data["metrics"].items()
    ]
    alert = Alert(**{k: v for k, v in alert_data.items() if k != "timestamp"},
                  timestamp=_ts(), metric_details=metric_details)

    # Hydrate context
    ctx_data = raw["context"]
    service_graph = [ServiceDependency(**s) for s in ctx_data.get("service_graph", [])]
    context = IncidentContext(
        service_graph=service_graph,
        recent_deployments=ctx_data.get("recent_deployments", []),
        runbook_hints=ctx_data.get("runbook_hints", []),
        similar_past_incidents=ctx_data.get("similar_past_incidents", []),
        on_call_teams=ctx_data.get("on_call_teams", {}),
    )

    # Hydrate ground truth
    gt = raw["ground_truth"]
    runbook = RunbookSection(**gt["runbook"])
    ground_truth = ScenarioGroundTruth(
        severity=SeverityLevel(gt["severity"]),
        category=AlertCategory(gt["category"]),
        team=gt["team"],
        root_cause_component=gt["root_cause_component"],
        root_cause_type=gt["root_cause_type"],
        impact=gt["impact"],
        affected_services=gt["affected_services"],
        evidence_keywords=gt["evidence_keywords"],
        runbook=runbook,
    )

    return Scenario(
        scenario_id=raw["scenario_id"],
        alert=alert,
        context=context,
        ground_truth=ground_truth,
        difficulty=raw["difficulty"],
    )


def sample_scenario(task_type: str, seed: Optional[int] = None) -> Scenario:
    """Sample a random scenario appropriate for the given task type."""
    rng = random.Random(seed)
    scenario_data = rng.choice(SCENARIOS)
    return build_scenario(scenario_data, rng)


def get_all_scenarios(seed: Optional[int] = 42) -> List[Tuple[str, Scenario]]:
    """Return one scenario per task type for evaluation."""
    rng = random.Random(seed)
    task_types = ["alert_classification", "root_cause_analysis", "runbook_generation"]
    shuffled = list(SCENARIOS)
    rng.shuffle(shuffled)
    result = []
    for i, task_type in enumerate(task_types):
        scenario = build_scenario(shuffled[i % len(shuffled)], rng)
        result.append((task_type, scenario))
    return result
