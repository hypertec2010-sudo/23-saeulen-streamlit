from modules.decision_engine import build_decision


def test_buy():
    d = build_decision({
        "action_clarity_pkg": {"label": "Kaufen", "summary": "Trigger bestätigt"},
        "timing_action_confidence_pkg": {"score": 82},
        "trigger_confluence_pkg": {"score": 78},
        "valid_trade_setup": True,
        "suggested_entry_zone": "100-103",
        "stop_used": 96,
        "tp1": 110,
    })
    assert d["decision"] == "BUY"
    assert d["traffic_light"] == "GREEN"
    assert d["state"] == "TRIGGER_ACTIVE"


def test_prepare():
    d = build_decision({"action_clarity_pkg": {"label": "Vorbereiten"}})
    assert d["decision"] == "PREPARE"
    assert d["state"] == "ARMED"


def test_position_exit_override():
    d = build_decision({"position_action": "Halten", "exit_score": 85}, position_mode=True)
    assert d["decision"] == "EXIT"
    assert d["traffic_light"] == "RED"


if __name__ == "__main__":
    test_buy()
    test_prepare()
    test_position_exit_override()
    print("decision_engine tests passed")
