from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PolicyDecision:
    action: str
    service_time_ms: float
    compute_cost_ms: float
    queueable: bool
    metadata: Dict[str, Any]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {'1', 'true', 'yes', 'y'}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == '' or text.lower() == 'nan':
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _event_modality(event: Dict[str, Any]) -> str:
    return str(event.get('modality', 'camera')).strip().lower()


def _audio_confidence_hit(event: Dict[str, Any], config: Dict[str, Any]) -> bool:
    conf = _to_float(event.get('confidence'), -1.0)
    threshold = _to_float(config.get('audio_conf_threshold'), 0.7)
    if conf >= 0:
        return conf >= threshold
    event_id = int(event.get('_event_index', 0))
    fallback_probability = _to_float(config.get('audio_detection_probability'), 0.25)
    modulo = max(1, int(round(1.0 / max(fallback_probability, 1e-6))))
    return (event_id % modulo) == 0


def _simulate_camera_detection_hit(event: Dict[str, Any], config: Dict[str, Any]) -> bool:
    detection_probability = _to_float(config.get('detection_probability'), 0.20)
    path_text = str(event.get('path', '')).lower()
    keyword_hit = any(token in path_text for token in ['animal', 'deer', 'bird', 'fox', 'bear', 'elk', 'coyote'])
    if keyword_hit:
        return True
    event_id = int(event.get('_event_index', 0))
    modulo = max(1, int(round(1.0 / max(detection_probability, 1e-6))))
    return (event_id % modulo) == 0


def process_all_policy(event: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> PolicyDecision:
    modality = _event_modality(event)
    yolo_ms = _to_float(config.get('yolo_latency_ms'), 38.6)
    bioclip_ms = _to_float(config.get('bioclip_latency_ms'), 60.8)
    birdnet_ms = _to_float(config.get('birdnet_latency_ms'), 19057.5)

    if modality == 'audio':
        service_time_ms = birdnet_ms
        return PolicyDecision(
            action='process_now',
            service_time_ms=service_time_ms,
            compute_cost_ms=service_time_ms,
            queueable=True,
            metadata={
                'policy_stage': 'audio_full_pipeline',
                'policy_state': 'audio_full_pipeline',
                'classification_triggered': True,
                'detection_positive': True,
            },
        )

    service_time_ms = yolo_ms + bioclip_ms
    return PolicyDecision(
        action='process_now',
        service_time_ms=service_time_ms,
        compute_cost_ms=service_time_ms,
        queueable=True,
        metadata={
            'policy_stage': 'full_pipeline',
            'policy_state': 'full_pipeline',
            'classification_triggered': True,
            'detection_positive': True,
        },
    )


def detection_only_policy(event: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> PolicyDecision:
    modality = _event_modality(event)
    yolo_ms = _to_float(config.get('yolo_latency_ms'), 38.6)
    birdnet_ms = _to_float(config.get('birdnet_latency_ms'), 19057.5)

    if modality == 'audio':
        return PolicyDecision(
            action='skip',
            service_time_ms=0.0,
            compute_cost_ms=0.0,
            queueable=False,
            metadata={
                'policy_stage': 'audio_skipped',
                'policy_state': 'audio_skipped',
                'skip_reason': 'audio_disabled_under_detection_only',
                'classification_triggered': False,
                'detection_positive': False,
            },
        )

    return PolicyDecision(
        action='process_now',
        service_time_ms=yolo_ms,
        compute_cost_ms=yolo_ms,
        queueable=True,
        metadata={
            'policy_stage': 'detection_only',
            'policy_state': 'detection_only',
            'classification_triggered': False,
            'detection_positive': False,
        },
    )


def selective_classification_policy(event: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> PolicyDecision:
    modality = _event_modality(event)
    yolo_ms = _to_float(config.get('yolo_latency_ms'), 38.6)
    bioclip_ms = _to_float(config.get('bioclip_latency_ms'), 60.8)
    birdnet_ms = _to_float(config.get('birdnet_latency_ms'), 19057.5)

    if modality == 'audio':
        hit = _audio_confidence_hit(event, config)
        if not hit:
            return PolicyDecision(
                action='skip',
                service_time_ms=0.0,
                compute_cost_ms=0.0,
                queueable=False,
                metadata={
                    'policy_stage': 'audio_confidence_filter',
                    'policy_state': 'audio_low_confidence',
                    'skip_reason': 'audio_below_conf_threshold',
                    'classification_triggered': False,
                    'detection_positive': False,
                },
            )
        return PolicyDecision(
            action='process_now',
            service_time_ms=birdnet_ms,
            compute_cost_ms=birdnet_ms,
            queueable=True,
            metadata={
                'policy_stage': 'audio_confidence_filter',
                'policy_state': 'audio_high_confidence',
                'classification_triggered': True,
                'detection_positive': True,
            },
        )

    hit = _simulate_camera_detection_hit(event, config)
    service_time_ms = yolo_ms + bioclip_ms if hit else yolo_ms
    return PolicyDecision(
        action='process_now',
        service_time_ms=service_time_ms,
        compute_cost_ms=service_time_ms,
        queueable=True,
        metadata={
            'policy_stage': 'selective_classification',
            'policy_state': 'selective_classification',
            'classification_triggered': hit,
            'detection_positive': hit,
        },
    )


def event_triggered_policy(event: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> PolicyDecision:
    modality = _event_modality(event)
    if modality == 'camera' and _to_bool(event.get('is_timelapse', False)):
        return PolicyDecision(
            action='skip',
            service_time_ms=0.0,
            compute_cost_ms=0.0,
            queueable=False,
            metadata={
                'policy_stage': 'event_triggered',
                'policy_state': 'timelapse_filtered',
                'skip_reason': 'timelapse_filtered',
                'classification_triggered': False,
                'detection_positive': False,
            },
        )
    decision = selective_classification_policy(event, state, config)
    decision.metadata['policy_stage'] = 'event_triggered'
    return decision


def adaptive_policy(event: Dict[str, Any], state: Dict[str, Any], config: Dict[str, Any]) -> PolicyDecision:
    queue_length = int(state.get('queue_length', 0))
    recent_detection_count = int(state.get('recent_detection_count', 0))
    idle_duration_hours = _to_float(state.get('idle_duration_hours', 0.0), 0.0)

    backlog_threshold = int(config.get('backlog_threshold', 25))
    high_activity_threshold = int(config.get('high_activity_threshold', 10))
    idle_backlog_threshold_hours = _to_float(config.get('idle_backlog_threshold_hours', 2.0), 2.0)

    if queue_length >= backlog_threshold:
        decision = selective_classification_policy(event, state, config)
        decision.metadata['policy_stage'] = 'adaptive'
        decision.metadata['policy_state'] = 'overloaded'
        return decision
    if idle_duration_hours >= idle_backlog_threshold_hours:
        decision = process_all_policy(event, state, config)
        decision.metadata['policy_stage'] = 'adaptive'
        decision.metadata['policy_state'] = 'idle_backlog_flush'
        return decision
    if recent_detection_count >= high_activity_threshold:
        decision = process_all_policy(event, state, config)
        decision.metadata['policy_stage'] = 'adaptive'
        decision.metadata['policy_state'] = 'high_activity'
        return decision
    decision = event_triggered_policy(event, state, config)
    decision.metadata['policy_stage'] = 'adaptive'
    decision.metadata['policy_state'] = 'normal'
    return decision


POLICY_REGISTRY = {
    'process_all': process_all_policy,
    'detection_only': detection_only_policy,
    'selective_classification': selective_classification_policy,
    'event_triggered': event_triggered_policy,
    'adaptive': adaptive_policy,
}
