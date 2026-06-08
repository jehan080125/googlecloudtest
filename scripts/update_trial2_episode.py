"""One-off script: rewrite disaster_epitaph trial 2 data (3 battles)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EPISODE_PATH = ROOT / "data" / "episodes" / "disaster_epitaph.json"


def main() -> None:
    data = json.loads(EPISODE_PATH.read_text(encoding="utf-8"))

    data.setdefault("scripted_trap", {})
    data["scripted_trap"]["description"] = (
        "1차 재판에서 이소은의 VX 독성·피부 흡수 거짓말을 파헤친다. "
        "2차 재판에서 CCTV 우회전·서버 로그 좌회전 모순으로 검찰 논리를 흔든 뒤 "
        "카카오 동기만으로는 유죄 불가 판정으로 휴정한다."
    )

    by_id = {e["id"]: e for e in data["evidences"]}
    by_id["ev_ep_cctv_car"].update(
        {
            "description": (
                "2030-07-04 14:08:22 교차로. 이소은 연행 호송차 급가속 후 급우회전, "
                "차량 측면이 가로등 기둥에 충돌한 경로가 지도에 표시됨."
            ),
            "fact": "실제 주행은 우회전. 서버 로그 「좌회전 명령」과 정면 배치.",
            "tags": ["CCTV", "차량", "우회전"],
        }
    )
    by_id["ev_ep_server_log"].update(
        {
            "description": (
                '[04/Jul/2030:14:07:11] <차량 0x1B6B8A9D 수신> 현재 위치 '
                '[36°31\'21.4"N, 127°14\'56.5"E] <차량 0x1B6B8A9D 로 코드 송신> '
                "//급가속 후 좌회전 명령"
            ),
            "fact": "송신 기록은 「좌회전」이다. CCTV 실제 우회전과 대조해 사용.",
        }
    )
    if "ev_ep_minsoo_opinion" not in by_id:
        kakao_idx = next(
            (i for i, e in enumerate(data["evidences"]) if e["id"] == "ev_ep_kakao"),
            len(data["evidences"]),
        )
        data["evidences"].insert(
            kakao_idx,
            {
                "id": "ev_ep_minsoo_opinion",
                "name": "임민수 LiDAR 소견서",
                "description": (
                    "라이다 센서 글리치로 좌·우 회전이 반대로 기록될 확률은 "
                    "극히 희박하다는 임민수 전문가 소견."
                ),
                "fact": (
                    "검찰은 이 소견으로 좌회전 코드·우회전 실주행을 설명하려 한다. "
                    "확률 희박성이 약점."
                ),
                "details": "2차 재판 Battle 2 — 검사 중간 제출.",
                "tags": ["라이다", "소견", "임민수"],
            },
        )
        by_id = {e["id"]: e for e in data["evidences"]}
    else:
        by_id["ev_ep_minsoo_opinion"].update(
            {
                "fact": (
                    "검찰은 이 소견으로 좌회전 코드·우회전 실주행을 설명하려 한다. "
                    "확률 희박성이 약점."
                ),
            }
        )

    clickable_ids = {o["id"] for o in data["clickable_objects"]}
    for object_id, evidence_id, label in (
        ("inv_ep_wiretap", "ev_ep_wiretap", "차고지 수색"),
        ("inv_ep_anthony_id", "ev_ep_anthony_id", "차고지 수색"),
        ("inv_ep_cctv_car", "ev_ep_cctv_car", "차고지 수색"),
        ("inv_ep_autodrive_log", "ev_ep_autodrive_log", "차고지 수색"),
    ):
        if object_id not in clickable_ids:
            data["clickable_objects"].append(
                {
                    "id": object_id,
                    "evidence_id": evidence_id,
                    "label": label,
                    "position": {"x": 0, "y": 0},
                }
            )
            clickable_ids.add(object_id)

    data["trial_exclude_evidence"] = {
        "trial_epitaph_2": [
            "ev_ep_autopsy",
            "ev_ep_medical",
            "ev_ep_vx_info",
            "ev_ep_glasses",
            "ev_ep_cctv_club",
            "ev_ep_club_flyer",
            "ev_ep_doctor_opinion",
        ],
        "trial_epitaph_3": [
            "ev_ep_autopsy",
            "ev_ep_medical",
            "ev_ep_vx_info",
            "ev_ep_glasses",
            "ev_ep_cctv_club",
            "ev_ep_club_flyer",
            "ev_ep_doctor_opinion",
        ],
    }
    data.setdefault("trial_skip_extra_evidence", {})
    data["trial_skip_extra_evidence"]["trial_epitaph_2"] = [
        "ev_ep_cctv_car",
        "ev_ep_autodrive_log",
        "ev_ep_wiretap",
        "ev_ep_anthony_id",
    ]
    data["trial_skip_extra_evidence"]["trial_epitaph_3"] = [
        "ev_ep_cctv_car",
        "ev_ep_autodrive_log",
        "ev_ep_laptop",
        "ev_ep_server_log",
        "ev_ep_kakao",
        "ev_ep_minsoo_opinion",
        "ev_ep_wiretap",
        "ev_ep_anthony_id",
    ]

    data["contradictions"] = [
        c
        for c in data["contradictions"]
        if c.get("rule_id") not in {"rule_ep_motive_not_guilt"}
    ]
    for rule in data["contradictions"]:
        if rule.get("rule_id") == "rule_ep_turn_direction":
            rule["breakdown_delta"] = 34
            rule["description"] = "CCTV 우회전과 서버 로그 좌회전이 충돌한다."
        if rule.get("rule_id") == "rule_ep_lidar_rare":
            rule["breakdown_delta"] = 33
            rule["description"] = "소견 희박성 + 살의면 우회전 코드 — 라이다 변명과 모순."
    data["contradictions"].append(
        {
            "rule_id": "rule_ep_motive_not_guilt",
            "required_evidence_id": "ev_ep_kakao",
            "target_statement_id": "counter_kakao_motive",
            "breakdown_delta": 33,
            "description": "카카오는 소호 보복 의심만 보여주며 이소은 살해 동기를 직접 증명하지 않는다.",
        }
    )

    trial2 = data["trials"][1]
    trial2["opening_lines"] = [
        {
            "speaker": "judge_001",
            "dialogue": "피고 앤서니는 범행 사실을 인정합니까?",
            "animation_tag": "think",
            "is_fixed": True,
        },
        {
            "speaker": "def_ep_002",
            "dialogue": (
                "Oh no~!! 말도 안 돼! 나는 해킹한 적이 없다고!! "
                "그 로그는 조작이야!!"
            ),
            "animation_tag": "serious",
            "is_fixed": True,
        },
    ]

    stage = trial2["stages"][0]
    stage["fixed_testimony_chain"] = [
        {
            "statement_id": "stmt_minsoo_hack_left",
            "text": (
                "앤서니가 회사 서버를 통해 연행 호송차에 급가속 후 우회전 명령을 보냈습니다. "
                "노트북·서버 로그가 그를 가리킵니다."
            ),
            "is_fixed": True,
            "weakness_id": "weak_turn_cctv",
            "required_evidence_ids": ["ev_ep_cctv_car", "ev_ep_server_log"],
            "required_logic_points": [
                "차량 CCTV 지도상 실제 주행은 교차로에서 우회전이다.",
                "서버 로그에는 급가속 후 좌회전 명령이 기록돼 있다.",
                "CCTV 우회전과 서버 로그 좌회전은 정면으로 모순된다.",
            ],
            "damage_on_success": 34,
            "life_loss_on_fail": 1,
            "counter_statement_id": "counter_minsoo_lidar",
        }
    ]
    stage["counter_statements"] = [
        {
            "statement_id": "counter_minsoo_lidar",
            "text": (
                "라이다 글리치로 회피 조작 시 좌·우가 반대로 나타날 수 있습니다. "
                "제 소견서에 그 확률이 적혀 있습니다."
            ),
            "is_fixed": True,
            "weakness_id": "weak_lidar_rare",
            "required_evidence_ids": ["ev_ep_minsoo_opinion", "ev_ep_server_log"],
            "required_logic_points": [
                "임민수 소견: 라이다 좌·우 반전 확률은 극히 희박하다.",
                "살해 의도라면 CCTV 우회전 경로에 맞춰 우회전 코드를 보냈을 것이다.",
                "로그는 좌회전 명령 — 의도적 살해와 맞지 않는다.",
            ],
            "damage_on_success": 33,
            "life_loss_on_fail": 1,
            "next_counter_statement_id": "counter_kakao_motive",
        },
        {
            "statement_id": "counter_kakao_motive",
            "text": (
                "앤서니와 임민수의 카카오 대화에는 바텐더 소호에 대한 보복·살해 계획이 담겨 있습니다. "
                "이것이 연행 호송차 해킹 살인의 명백한 동기입니다."
            ),
            "is_fixed": True,
            "weakness_id": "weak_motive_not_guilt",
            "required_evidence_ids": ["ev_ep_kakao"],
            "required_logic_points": [
                "카카오는 바텐더 소호에 대한 보복·의심만 보여준다.",
                "이소은 연행 호송차 살해에 대한 직접 동기는 증명되지 않는다.",
                "동기만으로는 유죄를 단정할 수 없다.",
            ],
            "damage_on_success": 33,
            "life_loss_on_fail": 1,
        },
    ]
    stage["prosecution_context"] = {
        "purpose": "앤서니가 연행 호송차를 해킹해 살해했다고 기소한다.",
        "opening_line": (
            "회사 서버 로그에는 사고 직전 연행 호송차에 급가속·좌회전 제어 코드가 송신된 기록이 있습니다. "
            "앤서니 노트북에는 그와 같은 시각 회사 서버에 접속한 흔적이 남아 있습니다."
        ),
        "opening_line_is_fixed": True,
        "fixed_prosecutor_submit_line": (
            "검찰은 회사 서버 로그 기록을 증거로 제출합니다. "
            "이어서 앤서니의 노트북 기록도 제출합니다."
        ),
        "fixed_judge_post_denial_line": "……검사, 증거에 대해 설명해 보십시오.",
        "fixed_prosecutor_explain_line": (
            "회사 서버 로그에는 사고 직전 연행 호송차에 급가속·좌회전 제어 코드가 송신된 기록이 있습니다. "
            "앤서니 노트북에는 그와 같은 시각 회사 서버에 접속한 흔적이 남아 있습니다."
        ),
        "fixed_prosecutor_battle2_line": "그럴 줄 알았습니다… 임민수 전문가 LiDAR 소견서를 제출합니다.",
        "fixed_anthony_battle2_line": "임민수?? 안 돼!! 왜…?! 거짓말이야!!!",
        "fixed_prosecutor_only_anthony_line": (
            "회사에서 원격 제어 코드를 보낼 수 있는 건 앤서니뿐입니다! 다른 용의자는 없습니다!"
        ),
        "fixed_anthony_forge_line": (
            "해킹 실력 있는 건 나뿐이지만, 로그랑 노트북 기록은 누구나 위조할 수 있어!"
        ),
        "fixed_judge_battle3_line": (
            "지금으로서는 증거만으로 유죄를 단정하기 어렵습니다. 변호인, 할 말 있습니까?"
        ),
        "fixed_prosecutor_adjourn_request_line": (
            "잠깐! 추가 수사가 필요합니다! 재판을 휴정해 주십시오!"
        ),
        "fixed_judge_adjourn_line": (
            "지금으로서는 유죄를 단정하기 어렵습니다. 재판을 휴정하겠습니다."
        ),
        "witness_claim": "서버·노트북 증거와 우회전 해킹.",
        "supports": ["앤서니 유죄", "임민수 전문가 신뢰"],
        "do_not_reveal": ["임민수 배신", "CCTV 우회전 모순", "휴정 후 3차 재판"],
        "model_answers": [
            "CCTV 우회전·서버로그 좌회전—방향 모순!",
            "소견 희박·살의면 우회전 코드 보냈을 것. 로그는 좌회전.",
            "카카오는 소호 보복 의심뿐. 이소은 살해 동기 증명 안 됨—아직 무죄.",
        ],
        "model_answer_hints": [
            "모범답안) CCTV+서버로그 우·좌 회전 모순",
            "모범답안) LiDAR 소견 희박 + 살의면 우회전",
            "모범답안) 카카오=소호 의심, 이소은 살해 동기 미증명",
        ],
        "fixed_prosecutor_post_testimony_line": (
            "임민수 증인의 증언은 바로 이 점을 보여줍니다. "
            "앤서니가 의도적으로 연행 호송차를 조종했다는 검찰의 주장을 확실히 뒷받침합니다."
        ),
    }
    stage["hints"] = [
        "CCTV 지도 — 실제 우회전 vs 서버 로그 좌회전을 비교하세요.",
        "임민수 소견 희박성 + 살의면 우회전 코드 — 서버 로그 좌회전과 대조하세요.",
        "카카오는 소호 보복 의심만 보여줍니다. 이소은 살해 동기를 직접 증명하지 못합니다.",
    ]
    stage["contradiction_helper_lines"] = [
        [
            "핵심 모순! CCTV는 우회전인데 서버 로그는 좌회전?",
            "차량 CCTV와 서버 로그를 함께 제시하세요!",
        ],
        [
            "라이다 글리치? 소견서 희박성 + 살의면 우회전!",
            "임민수 소견과 서버 로그를 함께 제시하세요.",
        ],
        [
            "카카오만으론 유죄 불가!",
            "카카오 대화를 제시해 소호 의심과 이소은 살해 동기를 구분하세요.",
        ],
    ]

    for claim in ("소호 차량", "소호차량", "소호의 차량"):
        if claim not in data["forbidden_claims"]:
            data["forbidden_claims"].append(claim)

    t1_stage = data["trials"][0]["stages"][0]
    ctx1 = t1_stage.setdefault("prosecution_context", {})
    ctx1["model_answers"] = [
        "VX 정보 50mg·술잔 20mg, 마시기만으론 치사량 미달. 독살 단정은 성급.",
        "오른손에서 VX물질이 발견된 이유에 대해서 설명해 주세요!",
        "증언 #1·#2: 먼저 기절 vs 양진혁이 쓰러질 때 손에 묻음. 순서 모순!",
        "춤 전 손에 치사량 독이 쏟아졌다면 즉사! VX정보 피부 치사량 10mg입니다!",
    ]
    ctx1["model_answer_hints"] = [
        "모범답안) VX 50mg·20mg 용량 + 증언 #1 독살 단정 성급",
        "모범답안) 오른손 VX 검출 경위 질문",
        "모범답안) #1·#2 순서 모순",
        "모범답안) 10mg 피부 + 춤 불가",
    ]
    ctx1.setdefault("fixed_judge_cross_exam_prompt", "변호인 할 말 있습니까?")
    ctx1.setdefault("fixed_defense_cross_exam_intent", "증인에게 물어보고 싶은 것이 있습니다.")
    ctx1.setdefault(
        "fixed_prosecutor_adjourn_line",
        "앗! 좀 더 수사가 필요할 것 같습니다! 재판을 멈춰주세요!",
    )
    ctx1.setdefault(
        "fixed_judge_adjourn_line",
        "알겠습니다. 수사가 완료될 때 까지 재판을 연기하도록 하죠.",
    )
    confession = ctx1.get("fixed_confession_line") or ""
    if confession and not confession.startswith("("):
        ctx1["fixed_confession_line"] = (
            "(머리를 감싸 쥐며 부르르 떤다) 으, 으으으... 기, 기건...! 내가... 내가 피부가 "
            "엄청나게 두꺼워서...! 아니, 장갑을 끼고 있어서...!! 윽 ...! 사실 다 거짓말이야. "
            "나를 용서해줘. 바텐더는 죄가 없어…. (죄를 시인한다.)"
        )

    EPISODE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Updated", EPISODE_PATH)


if __name__ == "__main__":
    main()
