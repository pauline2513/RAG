from llama_cpp import Llama
from scripts.graph_analytics import get_neo4j_driver

MAX_DEPTH = 4
TOP_LIST = 5
MODEL_NAME = "Qwen/Qwen3-8B-GGUF"

SYSTEM_PROMPT_EXTRACTING_MAIN_ENTITIES = """
Ты - LLM для работы с извлечением информацией для пользователя. Тебе будут поступать запросы о металлургии и металах. 
По запросу тебе необходимо выделить несколько ключевых слов и вывести их в формате строки с разделением КАЖДОГО СЛОВА через точку с запятой.
Например:
Q: Сколько градусов необходимо установить в печи чтобы расплавить медь?
A: градусов;печи;расплавить;медь;
"""

SYSTEM_PROMPT_EXTRACTION_RANGING = f"""
Ты - LLM для работы с извлечением информацией для пользователя.
Тебе поступил результат извлечения информации из графа по ключевым словам запроса пользователя в формате разделения названий вершин через точку с запятой. 
Тебе необходимо оценить {TOP_LIST} лучших релевантынх запросу сущностей или путей из тех что тебе поступили. 
Выведи их в том формате в котором они поступили.
"""

SYSTEM_PROPT_EXTRACTION_EVALUATION = """
Ты - LLM для работы с извлечением информацией для пользователя. Тебе необходимо ответить на вопрос достаточно ли информации для ответа на запрос пользователя.
В ответе укажи только True или False. Больше ничего не пиши.
"""
SYSTEM_PROMPT_NO_RAG = """
Ты - модель для ответа на пользовательские вопросы о металлургии и прокатной плавки.
На вход будут поступать вопросы, тебе надо на них ответить четко и коротко. 
"""

SYSTEM_PROMPT_EXTRACT_WITH_RAG = """
Ты - модель для ответа на пользовательские вопросы о металлургии и прокатной плавки.
Ниже представлен список путей в графе, которые помогут ответить на вопрос. 
"""

TEST_USER_PROMPT = "При скольких градусах проводится ТВХП для стали 08г2мб?"

neo_uri = "bolt://localhost:7687"
neo_user = "neo4j"
neo_password = "neo4jpass"
neo_cfg = {"uri": neo_uri, "user": neo_user, "password": neo_password}
DRIVER = get_neo4j_driver(neo_cfg)


llm = Llama.from_pretrained(repo_id="t-tech/T-lite-it-2.1-GGUF",
                            filename="*Q5_K_M.gguf",
                            n_gpu_layers=-1,
                            n_ctx=4096)
PATHS = []


def find_start_concepts(driver, entities, limit=20):
    query = f"""
    UNWIND $entities AS search_entity
    CALL (search_entity) {{
        MATCH (concept:EntityConcept)
        WHERE concept.norm = search_entity
           OR (concept.norm CONTAINS search_entity)
        WITH DISTINCT concept.name AS candidate_name,
            concept.norm AS candidate_norm
        RETURN candidate_name, candidate_norm
        ORDER BY
            CASE WHEN candidate_norm = search_entity THEN 0 ELSE 1 END,
            abs(size(candidate_norm) - size(search_entity)),
            candidate_norm
        LIMIT $limit
    }}
    RETURN search_entity,
        candidate_name,
        candidate_norm,
        candidate_norm = search_entity AS exact_match
    ORDER BY search_entity, exact_match DESC, candidate_norm
    """

    params = {
        "entities": [e.strip().lower() for e in entities if e.strip()],
        "limit": limit
    }

    with driver.session() as session:
        return list(session.run(query, params))
    


def find_relations(driver, paths, limit=20):

    query = f"""
    UNWIND $paths AS path
    MATCH (current:EntityConcept)-[rel:RELATION_INSTANCE]-(neighbor:EntityConcept)
    WHERE current.norm = path.current
      AND NOT neighbor.norm IN path.nodes
    RETURN DISTINCT
        path.start AS start,
        current.name AS current_name,
        current.norm AS current_norm,
        neighbor.name AS neighbor_name,
        neighbor.norm AS neighbor_norm,
        rel.predicate AS predicate,
        rel.sentence AS sentence,
        rel.triplet_id AS triplet_id,
        rel.document_id AS document_id,
        startNode(rel).norm = current.norm AS outgoing
    ORDER BY start, current_norm, neighbor_norm, predicate
    LIMIT $limit
    """

    params = {
        "paths": [{
            "start": path["start"].strip().lower(),
            "current": path["current"].strip().lower(),
            "nodes": [
                node.strip().lower()
                for node in path.get("nodes", [path["start"], *[step["end"] for step in path.get("paths", [])]])
            ],
            }
            for path in paths
        ],
        "limit": limit
    }

    with driver.session() as session:
        return list(session.run(query, params))



def deduplicate_records(records):
    cleaned_records = []
    sett = set()
    for record in records:
        key = (
            record["start"],
            record["current_norm"],
            record["neighbor_norm"],
            record["triplet_id"],
        )
        if key in sett:
            continue
        sett.add(key)
        cleaned_records.append(record)
    return cleaned_records


def update_paths(paths, clean_records):
    extends = []
    for record in clean_records:
        start = record["start"]
        current = record["current_norm"]
        predicate = record["predicate"]
        next_entity = record["neighbor_norm"]

        for path in paths:
            if path["start"] != start or path["current"] != current:
                continue

            nodes = path.get(
                "nodes",
                [path["start"], *[step["end"] for step in path["paths"]]],
            )
            if next_entity in nodes:
                continue

            extends.append({
                "start": start,
                "current": next_entity,
                "nodes": nodes + [next_entity],
                "paths": path["paths"] + [{
                    "start": current,
                    "predicate": predicate,
                    "end": next_entity,
                }],
            })
    return extends


def create_path_prompt_for_eval(PATHS):
    prompt = ""
    for path in PATHS:
        query = f"{path["start"]}"
        for steps in path["paths"]:
           query += f" -> {steps["predicate"]} -> {steps["end"]}"
        prompt += f"\n{query};"
    return prompt

def extract_start_points(user_prompt):
    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_EXTRACTING_MAIN_ENTITIES},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )["choices"][0]["message"]["content"] # type: ignore
    try:
        main_entities = [e for e in list(response.split(';')) if e != '']
    except:
        main_entities = ['0']
    return main_entities

def string_to_paths_select(paths, answer):
    formated_path = []
    lines = [a.strip() for a in answer.split(';') if a.strip()]
    starts_ends_pairs = [
        (line.split('->')[0].strip().lower(), line.split('->')[-1].strip().lower())
        for line in lines
    ]
    print(starts_ends_pairs)
    for path in paths:
        start = path["start"]
        current = path["current"]
        for start_end in starts_ends_pairs:
            if start == start_end[0] and current == start_end[1]:
                formated_path.append(path)
    return formated_path


def general_search(user_prompt, PATHS):
    depth = 0
    enough_knowledge_flag = False
    while depth < MAX_DEPTH and not enough_knowledge_flag:
        depth += 1
        records = find_relations(DRIVER, PATHS, limit=20)
        clean_records = deduplicate_records(records)
        candidate_paths = update_paths(PATHS, clean_records)
        if not candidate_paths:
            break
        prompt_for_checking_if_enough = create_path_prompt_for_eval(candidate_paths)
        eval_response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": SYSTEM_PROPT_EXTRACTION_EVALUATION},
                {"role": "user", "content": f"Пользовательский запрос: {user_prompt}\nПуть:\n{prompt_for_checking_if_enough}"} # type: ignore
            ]
        )["choices"][0]["message"]["content"]
        if 'true' in eval_response.strip().lower(): # type: ignore
            enough_knowledge_flag = True
            answer = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": f"{SYSTEM_PROMPT_EXTRACT_WITH_RAG}\n{prompt_for_checking_if_enough}"}, # type: ignore
                    {"role": "user", "content": user_prompt}
                ]
            )["choices"][0]["message"]["content"]
            return answer
        else:
            response_range_paths = llm.create_chat_completion(
                messages = [
                    {"role": "system", "content": f"{SYSTEM_PROMPT_EXTRACTION_RANGING}\nЗапрос пользователя:{user_prompt}"},
                    {"role": "user", "content": prompt_for_checking_if_enough} # type: ignore
                ]
            )["choices"][0]["message"]["content"]
            PATHS = string_to_paths_select(candidate_paths, response_range_paths)
            if not PATHS:
                break
    return "Извините, не смогли найти точную информацию"


def answer(user_prompt):
    main_entities = extract_start_points(user_prompt)
    PATHS = []
    for entity in main_entities:
        if entity != '':
            normalized_entity = entity.strip().lower()
            PATHS.append({
                "start": normalized_entity,
                "current": normalized_entity,
                "nodes": [normalized_entity],
                "paths": [],
            })
    print(main_entities)
    answer = general_search(user_prompt, PATHS)
    return answer

def answer_no_rag(user_prompt):
    response = llm.create_chat_completion(
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NO_RAG},
            {"role": "user", "content": user_prompt}
        ]
    )["choices"][0]["message"]["content"] # type: ignore
    return response
# print(answer(TEST_USER_PROMPT))


