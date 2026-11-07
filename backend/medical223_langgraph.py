from __future__ import annotations

import sys
from typing import Literal

from langgraph.graph import END, StateGraph
from medical223_core import (
    MedicalState,
    check_agent,
    chronic_specialist_agent,
    diagnosis_agent,
    dict_to_state,
    image_agent,
    infectious_specialist_agent,
    knowledge_agent,
    render_report,
    review_agent,
    route_after_image,
    self_feedback_agent,
    should_continue,
    symptom_agent,
)


def symptom_node(state: MedicalState | dict) -> dict:
    return dict_to_state(symptom_agent(dict_to_state(state))).__dict__


def image_node(state: MedicalState | dict) -> dict:
    return dict_to_state(image_agent(dict_to_state(state))).__dict__


def knowledge_node(state: MedicalState | dict) -> dict:
    return dict_to_state(knowledge_agent(dict_to_state(state))).__dict__


def route_node(state: MedicalState | dict) -> dict:
    return dict_to_state(route_after_image(dict_to_state(state))).__dict__


def chronic_node(state: MedicalState | dict) -> dict:
    return dict_to_state(chronic_specialist_agent(dict_to_state(state))).__dict__


def infectious_node(state: MedicalState | dict) -> dict:
    return dict_to_state(infectious_specialist_agent(dict_to_state(state))).__dict__


def diagnosis_node(state: MedicalState | dict) -> dict:
    s = dict_to_state(state)
    s.iteration += 1
    return diagnosis_agent(s).__dict__


def review_node(state: MedicalState | dict) -> dict:
    return review_agent(dict_to_state(state)).__dict__


def check_node(state: MedicalState | dict) -> dict:
    return check_agent(dict_to_state(state)).__dict__


def feedback_node(state: MedicalState | dict) -> dict:
    return self_feedback_agent(dict_to_state(state)).__dict__


def route_router(
    state: MedicalState | dict,
) -> Literal["diagnosis", "infectious_specialist", "chronic_specialist"]:
    s = dict_to_state(state)
    if s.route_decision == "infectious":
        return "infectious_specialist"
    if s.route_decision in {"chronic", "dual"}:
        return "chronic_specialist"
    return "diagnosis"


def after_chronic_router(
    state: MedicalState | dict,
) -> Literal["diagnosis", "infectious_specialist"]:
    s = dict_to_state(state)
    return "infectious_specialist" if s.route_decision == "dual" else "diagnosis"


def feedback_router(state: MedicalState | dict) -> Literal["feedback", "__end__"]:
    return "feedback" if should_continue(dict_to_state(state)) else "__end__"


def build_graph():
    graph = StateGraph(dict)
    graph.add_node("symptom", symptom_node)
    graph.add_node("image", image_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("route", route_node)
    graph.add_node("chronic_specialist", chronic_node)
    graph.add_node("infectious_specialist", infectious_node)
    graph.add_node("diagnosis", diagnosis_node)
    graph.add_node("review", review_node)
    graph.add_node("check", check_node)
    graph.add_node("feedback", feedback_node)

    graph.set_entry_point("symptom")
    graph.add_edge("symptom", "image")
    graph.add_edge("image", "knowledge")
    graph.add_edge("knowledge", "route")
    graph.add_conditional_edges(
        "route",
        route_router,
        {
            "diagnosis": "diagnosis",
            "infectious_specialist": "infectious_specialist",
            "chronic_specialist": "chronic_specialist",
        },
    )
    graph.add_conditional_edges(
        "chronic_specialist",
        after_chronic_router,
        {
            "diagnosis": "diagnosis",
            "infectious_specialist": "infectious_specialist",
        },
    )
    graph.add_edge("infectious_specialist", "diagnosis")
    graph.add_edge("diagnosis", "review")
    graph.add_edge("review", "check")
    graph.add_conditional_edges("check", feedback_router, {"feedback": "feedback", "__end__": END})
    graph.add_edge("feedback", "diagnosis")
    return graph.compile()


def run_medical223(symptom: str, image_path: str = "") -> MedicalState:
    graph = build_graph()
    initial = MedicalState(symptom=symptom.strip(), image_path=image_path.strip()).__dict__
    result = graph.invoke(initial)
    return dict_to_state(result)


def main() -> int:
    print("\n===== 医疗223：肺部异常辅助筛查系统（LangGraph版）=====\n")
    print("说明：该版本由原版完整复制而来，并在副本中新增医疗223三分类主模型入口。")
    symptom = input("请输入症状: ").strip()
    image_path = input("请输入影像路径(可留空): ").strip()
    if not symptom:
        print("症状不能为空。")
        return 1
    state = run_medical223(symptom, image_path)
    print()
    print(render_report(state))
    return 0


if __name__ == "__main__":
    sys.exit(main())
