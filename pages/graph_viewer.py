import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components


EXAMPLE_PATH = Path(__file__).resolve().parent.parent / "neo4j_graph_export.json"

NODE_COLORS = {
    "Document": "#64748b",
    "Triplet": "#f97316",
    "Entity": "#84cc16",
    "EntityConcept": "#0ea5e9",
    "RelationConcept": "#a855f7",
    "FrameOccurrence": "#f59e0b",
    "FrameNode": "#14b8a6",
    "GraphRoot": "#ef4444",
}


def load_graph(uploaded_file) -> dict[str, Any]:
    if uploaded_file is None:
        with EXAMPLE_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    return json.load(uploaded_file)


def validate_graph(graph: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(graph, dict):
        raise ValueError("JSON должен содержать объект верхнего уровня.")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("В JSON ожидаются списки nodes и edges.")

    return nodes, edges


def get_node_type(node: dict[str, Any]) -> str:
    labels = node.get("labels") or []
    return labels[0] if labels else "Node"


def get_node_label(node: dict[str, Any]) -> str:
    properties = node.get("properties") or {}
    for key in ("name", "text", "root_text", "source_name", "subject_text", "norm"):
        value = properties.get(key)
        if value not in (None, ""):
            return str(value)
    return f"{get_node_type(node)} #{node.get('neo4j_id', '?')}"


def get_node_search_text(node: dict[str, Any]) -> str:
    return json.dumps(node, ensure_ascii=False, default=str).lower()


def get_edge_label(edge: dict[str, Any]) -> str:
    properties = edge.get("properties") or {}
    predicate = properties.get("predicate")
    if predicate:
        return str(predicate)
    return str(edge.get("type", "EDGE"))


def select_subgraph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_types: set[str],
    edge_types: set[str],
    search_query: str,
    hops: int,
    node_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes_by_id = {str(node.get("id")): node for node in nodes}
    allowed_node_ids = {
        node_id
        for node_id, node in nodes_by_id.items()
        if get_node_type(node) in node_types
    }
    filtered_edges = [
        edge
        for edge in edges
        if edge.get("type") in edge_types
        and str(edge.get("source")) in allowed_node_ids
        and str(edge.get("target")) in allowed_node_ids
    ]

    adjacency: dict[str, set[str]] = {}
    for edge in filtered_edges:
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    query = search_query.strip().lower()
    if query:
        seed_ids = [
            node_id
            for node_id in allowed_node_ids
            if query in get_node_search_text(nodes_by_id[node_id])
        ]
    else:
        connected_ids = set(adjacency)
        seed_ids = sorted(connected_ids or allowed_node_ids)

    selected_ids: set[str] = set()
    queue = deque((node_id, 0) for node_id in seed_ids)
    while queue and len(selected_ids) < node_limit:
        node_id, depth = queue.popleft()
        if node_id in selected_ids:
            continue
        selected_ids.add(node_id)
        if depth >= hops:
            continue
        for neighbor_id in sorted(adjacency.get(node_id, set())):
            if neighbor_id not in selected_ids:
                queue.append((neighbor_id, depth + 1))

    selected_nodes = [nodes_by_id[node_id] for node_id in selected_ids]
    selected_edges = [
        edge
        for edge in filtered_edges
        if str(edge.get("source")) in selected_ids
        and str(edge.get("target")) in selected_ids
    ]
    return selected_nodes, selected_edges


def as_title(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def deduplicate_edges_for_render(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_edges = {}
    for edge in edges:
        properties = edge.get("properties") or {}
        key = (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("type", "EDGE")),
            str(properties.get("predicate", "")),
        )
        if key not in unique_edges:
            unique_edges[key] = {
                **edge,
                "merged_edges": [edge],
            }
        else:
            unique_edges[key]["merged_edges"].append(edge)
    return list(unique_edges.values())


def render_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    show_edge_labels: bool,
) -> None:
    vis_nodes = []
    for node in nodes:
        node_type = get_node_type(node)
        vis_nodes.append(
            {
                "id": str(node.get("id")),
                "label": get_node_label(node),
                "group": node_type,
                "color": NODE_COLORS.get(node_type, "#94a3b8"),
                "title": as_title(node),
            }
        )

    vis_edges = []
    for edge in deduplicate_edges_for_render(edges):
        merged_edges = edge["merged_edges"]
        edge_title: Any = edge
        if len(merged_edges) > 1:
            edge_title = {
                "merged_count": len(merged_edges),
                "edges": merged_edges,
            }
        vis_edges.append(
            {
                "id": str(edge.get("id")),
                "from": str(edge.get("source")),
                "to": str(edge.get("target")),
                "label": get_edge_label(edge) if show_edge_labels else "",
                "title": as_title(edge_title),
                "arrows": "to",
            }
        )

    nodes_json = json.dumps(vis_nodes, ensure_ascii=False).replace("</", "<\\/")
    edges_json = json.dumps(vis_edges, ensure_ascii=False).replace("</", "<\\/")
    graph_html = f"""
    <div id="graph" style="height: 760px; border: 1px solid #d1d5db; border-radius: 8px;"></div>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script>
      const nodes = new vis.DataSet({nodes_json});
      const edges = new vis.DataSet({edges_json});
      const container = document.getElementById("graph");
      const options = {{
        interaction: {{ hover: true, navigationButtons: true, keyboard: true }},
        nodes: {{ shape: "dot", size: 18, font: {{ size: 14 }} }},
        edges: {{
          color: {{ color: "#94a3b8", highlight: "#334155" }},
          font: {{ size: 11, align: "middle" }},
          smooth: {{ type: "dynamic" }}
        }},
        physics: {{
          stabilization: {{ iterations: 180 }},
          barnesHut: {{ gravitationalConstant: -3500, springLength: 130 }}
        }}
      }};
      new vis.Network(container, {{ nodes, edges }}, options);
    </script>
    """
    components.html(graph_html, height=790)


st.set_page_config(page_title="Просмотр графа", layout="wide")
st.title("Просмотр графа из JSON")
st.caption("Загрузите экспорт Neo4j или используйте пример из проекта.")

uploaded_file = st.file_uploader("JSON-файл графа", type=["json"])
use_example = st.checkbox(
    "Использовать neo4j_graph_export.json из проекта",
    value=uploaded_file is None,
    disabled=not EXAMPLE_PATH.exists(),
)

if uploaded_file is None and not use_example:
    st.info("Загрузите JSON-файл, чтобы построить граф.")
    st.stop()

try:
    graph = load_graph(uploaded_file)
    nodes, edges = validate_graph(graph)
except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
    st.error(f"Не удалось прочитать граф: {exc}")
    st.stop()

node_type_counts = Counter(get_node_type(node) for node in nodes)
edge_type_counts = Counter(str(edge.get("type", "EDGE")) for edge in edges)

metric_nodes, metric_edges = st.columns(2)
metric_nodes.metric("Узлов в файле", len(nodes))
metric_edges.metric("Связей в файле", len(edges))

default_node_types = [
    node_type
    for node_type in ("EntityConcept", "RelationConcept")
    if node_type in node_type_counts
]
default_edge_types = [
    edge_type
    for edge_type in ("RELATION_INSTANCE",)
    if edge_type in edge_type_counts
]

with st.sidebar:
    st.header("Фильтры")
    selected_node_types = st.multiselect(
        "Типы узлов",
        options=sorted(node_type_counts),
        default=default_node_types or sorted(node_type_counts),
        format_func=lambda item: f"{item} ({node_type_counts[item]})",
    )
    selected_edge_types = st.multiselect(
        "Типы связей",
        options=sorted(edge_type_counts),
        default=default_edge_types or sorted(edge_type_counts),
        format_func=lambda item: f"{item} ({edge_type_counts[item]})",
    )
    search_query = st.text_input("Поиск узла", placeholder="Например: твхп")
    hops = st.slider("Глубина окружения", min_value=0, max_value=4, value=1)
    node_limit = st.slider("Максимум узлов на экране", min_value=20, max_value=1000, value=250, step=10)
    show_edge_labels = st.checkbox("Показывать подписи связей", value=False)

selected_nodes, selected_edges = select_subgraph(
    nodes=nodes,
    edges=edges,
    node_types=set(selected_node_types),
    edge_types=set(selected_edge_types),
    search_query=search_query,
    hops=hops,
    node_limit=node_limit,
)

shown_nodes, shown_edges = st.columns(2)
shown_nodes.metric("Показано узлов", len(selected_nodes))
shown_edges.metric("Показано связей", len(selected_edges))

if not selected_nodes:
    st.warning("По выбранным фильтрам узлы не найдены.")
    st.stop()

render_graph(selected_nodes, selected_edges, show_edge_labels)

with st.expander("Данные выбранного подграфа"):
    st.subheader("Узлы")
    st.dataframe(selected_nodes, use_container_width=True)
    st.subheader("Связи")
    st.dataframe(selected_edges, use_container_width=True)
