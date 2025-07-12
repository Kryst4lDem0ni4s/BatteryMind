"""
BatteryMind ‑ Federated-Learning Utility Package
================================================

This sub-package hosts low-level helpers that are shared by the client-side and
server-side federated-learning components.  All utilities are intentionally kept
framework-agnostic so they can be reused in gRPC, WebSocket or plain-HTTP
deployments.

Currently exposed helpers
-------------------------
communication      • Core message, serialization, encryption and transport logic
"""

from .communication import (
    MessageType,
    FLMessage,
    SecureSerializer,
    SecureCommsMixin,
    CommunicationClient,
    CommunicationServer,
    DEFAULT_PORT,
    DEFAULT_HOST,
)

__all__ = [
    "MessageType",
    "FLMessage",
    "SecureSerializer",
    "SecureCommsMixin",
    "CommunicationClient",
    "CommunicationServer",
    "DEFAULT_PORT",
    "DEFAULT_HOST",
]
