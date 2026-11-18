"""Live answer-quality check: fixed questions through the real RAG pipeline and the real LLM.

pytest cannot pin down what a model does, so this runs a handful of questions several times and
reports, per run, which tools were called and whether the answer is grounded in the index. It
costs a few LLM calls per run and is NOT part of the pytest suite.

    cd backend
    uv run python eval_answers.py [--repeat 3] [--only cross_course] [-v]

Exit code 1 if any run fails a check. Checks are heuristics, not proofs: the invented-course
detector looks for Title-Case phrases in bold/quotes that are not a course or lesson title.
"""
import argparse
import os
import re
import sys

import truststore
truststore.inject_into_ssl()

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # config.CHROMA_PATH is relative to backend/

import litellm  # noqa: E402

import ai_generator  # noqa: E402
from config import config  # noqa: E402
from rag_system import RAGSystem  # noqa: E402


def search_calls(run):
    return [a for n, a in run["tools"] if n == "search_course_content"]


def called(run, name):
    return any(n == name for n, _ in run["tools"])


def mentions_nothing_found(answer):
    return bool(re.search(r"\b(no|not|n't|nothing|none|neither|only)\b", answer, re.I))


# Each check: (description, predicate(run) -> bool). `run` has: answer, sources, tools, titles.
CASES = {
    "cross_course": {
        "question": "Find a course that covers the same topic as lesson 4 of the MCP course",
        "checks": [
            ("looked for other courses (a search without course_name, or with exclude_course)",
             lambda r: any(not a.get("course_name") or a.get("exclude_course") for a in search_calls(r))),
            ("did not search only inside the MCP course",
             lambda r: not search_calls(r) or any(a.get("course_name") is None for a in search_calls(r))),
            ("answer names no course outside the catalog", lambda r: not r["invented"]),
        ],
    },
    "off_topic_course": {
        "question": "Which course teaches Italian pasta cooking?",
        "checks": [
            ("searched the materials", lambda r: called(r, "search_course_content")),
            ("answer names no course outside the catalog", lambda r: not r["invented"]),
            ("answer says there is nothing", lambda r: mentions_nothing_found(r["answer"])),
        ],
    },
    "content_in_named_course": {
        "question": "What does the MCP course say about creating an MCP server?",
        "checks": [
            ("searched with course_name MCP",
             lambda r: any("mcp" in (a.get("course_name") or "").lower() for a in search_calls(r))),
            ("answer is about servers", lambda r: "server" in r["answer"].lower()),
            ("answer names no course outside the catalog", lambda r: not r["invented"]),
        ],
    },
    "outline": {
        "question": "What lessons does the Advanced Retrieval for AI with Chroma course have?",
        "checks": [
            ("used get_course_outline", lambda r: called(r, "get_course_outline")),
            ("did not use content search", lambda r: not called(r, "search_course_content")),
            ("lists a real lesson", lambda r: "query expansion" in r["answer"].lower()),
        ],
    },
    "general_knowledge": {
        "question": "What is the capital of France?",
        "checks": [
            ("no tool call", lambda r: not r["tools"]),
            ("answers Paris", lambda r: "paris" in r["answer"].lower()),
        ],
    },
}


def catalog(rag):
    titles = set(rag.vector_store.get_existing_course_titles())
    lessons = {l["title"] if "title" in l else l.get("lesson_title", "")
               for c in rag.vector_store.get_all_courses_metadata() for l in c.get("lessons", [])}
    return {t.lower() for t in titles}, {t.lower() for t in lessons if t}


def invented_titles(answer, course_titles, lesson_titles):
    """Title-Case phrases in bold or quotes, next to the word "course", that are neither a course
    nor a lesson title. (Bold is also used for ordinary headings, so "course" must be nearby.)"""
    out = []
    for m in re.finditer(r'\*\*(.+?)\*\*|"(.+?)"|“(.+?)”', answer):
        phrase = next(g for g in m.groups() if g).strip().strip(".,:;")
        words = re.findall(r"[A-Za-z][\w'-]*", phrase)
        if len(words) < 2 or sum(w[0].isupper() for w in words) / len(words) < 0.6:
            continue  # not title-like ("server-side development", "Lesson 4", ...)
        around = answer[max(0, m.start() - 60):m.end() + 40].lower()
        if "course" not in around:
            continue
        low = phrase.lower()
        known = any(low in t or t in low for t in course_titles | lesson_titles)
        if not known:
            out.append(phrase)
    return out


def run_once(rag, question, course_titles, lesson_titles):
    tools = []
    orig_exec = rag.tool_manager.execute_tool

    def traced(name, **kw):
        tools.append((name, kw))
        return orig_exec(name, **kw)

    rag.tool_manager.execute_tool = traced
    try:
        answer, sources = rag.query(question)
    finally:
        rag.tool_manager.execute_tool = orig_exec
    return {"answer": answer, "sources": sources, "tools": tools,
            "invented": invented_titles(answer, course_titles, lesson_titles)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeat", type=int, default=3, help="runs per question (the model is not deterministic)")
    ap.add_argument("--only", help="run only cases whose name contains this text")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    args = ap.parse_args()

    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    rag = RAGSystem(config)
    course_titles, lesson_titles = catalog(rag)
    print(f"model={config.LLM_MODEL}  courses={len(course_titles)}  repeat={args.repeat}\n")

    failures = 0
    for name, case in CASES.items():
        if args.only and args.only not in name:
            continue
        print(f"== {name}: {case['question']}")
        for i in range(1, args.repeat + 1):
            try:
                run = run_once(rag, case["question"], course_titles, lesson_titles)
            except Exception as e:  # provider error etc.: count as a failed run, keep going
                failures += 1
                print(f"  run {i}: ERROR {type(e).__name__}: {e}")
                continue
            failed = [d for d, ok in case["checks"] if not ok(run)]
            failures += bool(failed)
            calls = ", ".join(f"{n}({', '.join(f'{k}={v!r}' for k, v in kw.items())})" for n, kw in run["tools"])
            print(f"  run {i}: {'FAIL' if failed else 'ok  '}  tools: {calls or '-'}")
            for d in failed:
                print(f"           x {d}")
            if run["invented"]:
                print(f"           ! titles not in the catalog: {run['invented']}")
            if args.verbose or failed:
                print(f"           answer: {run['answer'][:300]!r}")
        print()

    print("FAILED runs:" if failures else "All runs passed.", failures or "")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
