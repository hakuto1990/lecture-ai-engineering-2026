import json

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent


@pytest.mark.parametrize("approved", [True, False])
def test_learning_log_requires_approval(tmp_path, monkeypatch, approved):
    # 保存先をテスト専用の一時フォルダにする。
    # tools.py の保存先計算に使われる __file__ を一時的に差し替える。
    monkeypatch.setattr(
        tools,
        "__file__",
        str(tmp_path / "app" / "tools.py"),
    )
    log_dir = tmp_path / "learning_log"

    model = TestModel(call_tools=["save_learning_log"])

    # 保存を提案しても、承認前にはファイルを作らない。
    pending = agent.run_sync("学習内容を保存して", model=model)

    assert isinstance(pending.output, DeferredToolRequests)
    assert not log_dir.exists()

    call, = pending.output.approvals
    assert call.tool_name == "save_learning_log"

    # 承認または拒否の結果を渡して再開する。
    result = agent.run_sync(
        model=model,
        message_history=pending.all_messages(),
        deferred_tool_results=DeferredToolResults(
            approvals={call.tool_call_id: approved}
        ),
    )

    assert isinstance(result.output, str)

    if approved:
        # 承認した場合だけ、要求された内容が1件保存される。
        files = list(log_dir.glob("*.json"))
        assert len(files) == 1

        record = json.loads(files[0].read_text(encoding="utf-8"))
        args = call.args_as_dict()
        assert record["title"] == args["title"]
        assert record["content"] == args["content"]
    else:
        # 拒否した場合は、保存先フォルダも作られない。
        assert not log_dir.exists()