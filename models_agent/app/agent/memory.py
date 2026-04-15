"""会话级短期记忆。"""

from __future__ import annotations

from threading import Lock

from app.agent.state import AgentIntent, MissingInformationItem, SessionMemorySnapshot
from app.contracts import AgentHistoryItem


class ShortTermMemoryManager:
    """基于内存的轻量会话记忆。"""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[int, SessionMemorySnapshot] = {}

    def get(
        self,
        session_id: int,
        history: list[AgentHistoryItem],
    ) -> SessionMemorySnapshot:
        with self._lock:
            snapshot = self._sessions.get(session_id)
            if snapshot is None:
                snapshot = SessionMemorySnapshot(session_id=session_id)
                self._sessions[session_id] = snapshot

        if history and not snapshot.recent_questions:
            recent_user_questions = [
                item.content for item in history if item.role == "user"
            ]
            if recent_user_questions:
                snapshot = snapshot.model_copy(
                    update={"recent_questions": recent_user_questions[-3:]}
                )
                with self._lock:
                    self._sessions[session_id] = snapshot
        return snapshot

    def update(
        self,
        session_id: int,
        *,
        intent: AgentIntent,
        question: str,
        actions: list[str],
        missing_information: list[MissingInformationItem],
        active_goal: str | None,
    ) -> SessionMemorySnapshot:
        with self._lock:
            previous = self._sessions.get(session_id, SessionMemorySnapshot(session_id=session_id))
            recent_questions = (previous.recent_questions + [question])[-5:]
            recent_actions = (actions + previous.recent_actions)[:5]
            snapshot = previous.model_copy(
                update={
                    "last_intent": intent,
                    "active_goal": active_goal or previous.active_goal,
                    "recent_questions": recent_questions,
                    "recent_actions": recent_actions,
                    "pending_missing_information": missing_information[:3],
                }
            )
            self._sessions[session_id] = snapshot
            return snapshot

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

