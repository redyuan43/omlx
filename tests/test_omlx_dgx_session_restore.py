from __future__ import annotations

from omlx_dgx.session_restore import SessionRestoreSnapshot, SessionRestoreStore


def _snapshot(
    *,
    conversation_id: str,
    prefix_digest: str,
    estimated_prompt_tokens: int = 100,
    request_shape_digest: str = "shape-a",
    prompt_mode: str = "messages",
    last_access_at: float = 1.0,
) -> SessionRestoreSnapshot:
    return SessionRestoreSnapshot(
        conversation_id=conversation_id,
        model_id="qwen35-35b",
        slot_id=0,
        estimated_prompt_tokens=estimated_prompt_tokens,
        prefix_digest=prefix_digest,
        request_shape_digest=request_shape_digest,
        prompt_mode=prompt_mode,
        message_count=1,
        slot_message_count=2,
        save_filename=f"{conversation_id}.bin",
        state_payload={"conversation_id": conversation_id},
        saved_at=last_access_at,
        last_access_at=last_access_at,
    )


def test_session_restore_store_find_exact_prefix_digest(tmp_path):
    store = SessionRestoreStore(tmp_path)
    store.put(_snapshot(conversation_id="a", prefix_digest="same", last_access_at=1.0))
    store.put(_snapshot(conversation_id="b", prefix_digest="same", last_access_at=3.0))
    store.put(_snapshot(conversation_id="c", prefix_digest="other", last_access_at=2.0))

    matches = store.find_exact_prefix_digest(
        model_id="qwen35-35b",
        prefix_digest="same",
        request_shape_digest="shape-a",
        prompt_mode="messages",
        exclude_conversation_id="b",
    )

    assert [item.conversation_id for item in matches] == ["a"]


def test_session_restore_store_find_prefix_candidates_filters_and_orders(tmp_path):
    store = SessionRestoreStore(tmp_path)
    store.put(_snapshot(conversation_id="a", prefix_digest="p1", estimated_prompt_tokens=40, last_access_at=1.0))
    store.put(_snapshot(conversation_id="b", prefix_digest="p2", estimated_prompt_tokens=60, last_access_at=3.0))
    store.put(_snapshot(conversation_id="c", prefix_digest="p3", estimated_prompt_tokens=80, last_access_at=2.0))

    matches = store.find_prefix_candidates(
        model_id="qwen35-35b",
        request_shape_digest="shape-a",
        prompt_mode="messages",
        exclude_conversation_id="c",
        max_estimated_prompt_tokens=70,
    )

    assert [item.conversation_id for item in matches] == ["b", "a"]
