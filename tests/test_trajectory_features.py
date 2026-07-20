from __future__ import annotations

from longrun_agent.evaluation.trajectory import TrajectoryFeatureExtractor


def test_trajectory_extracts_first_symptom_and_causal_divergence() -> None:
    features = TrajectoryFeatureExtractor().extract(
        [
            [
                {"event_id": "e1", "event_type": "task_completion_requested", "timestamp": "1"},
                {"event_id": "e2", "event_type": "integrity_violation_detected", "timestamp": "2"},
                {"event_id": "e3", "event_type": "verification_check_failed", "timestamp": "3"},
            ]
        ]
    )
    assert features.first_completion_request == "e1"
    assert features.first_integrity_violation == "e2"
    assert features.first_observable_symptom == "e3"
    assert features.first_causal_divergence == "e2"


def test_trajectory_treats_nonpassing_transition_as_failed_required_check() -> None:
    features = TrajectoryFeatureExtractor().extract(
        [[{"event_id": "transition", "event_type": "verification_transition_computed", "verdict": "P2F"}]]
    )
    assert features.first_failed_required_check == "transition"
