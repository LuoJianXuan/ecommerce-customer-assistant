"""Verify frozen artifacts and recompute recorded metrics with the Python standard library."""
from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def equal(actual, expected):
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9), (actual, expected)
    else:
        assert actual == expected, (actual, expected)

def main():
    current = load(ROOT/"evals/current.json")
    for kind in ["rag", "topic"]:
        directory = (ROOT/current[kind]["directory"]).resolve()
        assert directory.is_relative_to(ROOT)
        for name, expected in load(directory/"checksums.json").items():
            path = (directory/name).resolve()
            assert path.is_relative_to(directory)
            equal(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    directory = ROOT/current["rag"]["directory"]
    report = load(directory/"report.json")
    cases = rows(directory/"eval_cases.jsonl")
    by_id = {r["id"]:r for r in cases}
    assert len(cases) == len(by_id) == 100
    equal(dict(Counter(r["answerability"] for r in cases)), {"full":70,"partial":3,"absent":27})
    equal(dict(Counter(r["bucket"] for r in cases)), report["manifest"]["bucket_counts"])
    equal(report["run_config"]["rerank_candidates"], 6)
    equal(load(directory/"run_config.json"), report["run_config"])
    expected_gt = load(directory/"ground_truth_ids.json")
    active = load(directory/"knowledge_snapshot.json")
    assert len(active) == 39 and all(r["governance_status"] == "active" for r in active)
    active_ids = {r["id"] for r in active}
    vector_rows = load(directory/"milvus_text_snapshot.json")
    assert len(vector_rows) == 39 and {r["chunk_id"] for r in vector_rows} == active_ids
    vector_text = {r["chunk_id"]:r["text"] for r in vector_rows}
    for r in active:
        equal(vector_text[r["id"]], f"{r['category']}\n{r['questions']}\n{r['answer']}")
    for key, name in [("corrected_cases_sha256","eval_cases.jsonl"),("knowledge_sha256","knowledge_snapshot.json"),("milvus_sha256","milvus_text_snapshot.json")]:
        equal(hashlib.sha256((directory/name).read_bytes()).hexdigest(), report["manifest"][key])

    metrics = defaultdict(lambda:defaultdict(list))
    retrieval = rows(directory/"retrieval_cases.jsonl")
    assert len(retrieval) == len({(r["id"],r["strategy"]) for r in retrieval}) == 292
    for r in retrieval:
        case = by_id[r["id"]]
        assert case["answerability"] != "absent"
        gt = set(expected_gt[r["id"]])
        assert gt == set(r["gt_ids"]) and gt and gt <= active_ids
        ranked = [h["chunk_id"] for h in r["hits"]]
        assert set(ranked) <= active_ids
        values = {"r5":len(set(ranked[:5])&gt)/len(gt),"r10":len(set(ranked[:10])&gt)/len(gt),
                  "mrr":next((1/i for i,cid in enumerate(ranked,1) if cid in gt),0.0)}
        for key,value in values.items():equal(value,r[key])
        metrics[r["strategy"]][case["bucket"]].append(values)
    for strategy,buckets in metrics.items():
        buckets["ALL"] = [v for values in buckets.values() for v in values]
        for bucket,values in buckets.items():
            expected=report["retrieval"][strategy][bucket]
            equal(len(values), expected["n"])
            for key in ["r5","r10","mrr"]:equal(statistics.mean(v[key] for v in values), expected[key])
    equal(load(directory/"retrieval_summary.json"), report["retrieval"])

    generated = rows(directory/"generation_cases.jsonl")
    assert len(generated) == len({r["id"] for r in generated}) == 100
    assert {r["id"] for r in generated} == set(by_id)
    for r in generated:
        equal(r["bucket"],by_id[r["id"]]["bucket"])
        equal(r["query"],by_id[r["id"]]["query"])
        assert all(e["chunk_id"] in active_ids for e in r["evidence"])
        valid=isinstance(r.get("verdict"),dict) and all(isinstance(r["verdict"].get(k),bool) for k in ["faithful","refused","answerable_from_evidence"])
        equal(valid,r["judge_valid"])
    a=[r for r in generated if r["answerability"]!="absent"]
    d=[r for r in generated if r["answerability"]=="absent"]
    judged=[r for r in a if r["judge_valid"]]
    g=report["generation"]
    for key,value in {"judged":len(judged),"judge_unknown":len(a)-len(judged),
                      "fabricated":sum(not r["verdict"]["faithful"] for r in judged),
                      "answerable_refused":sum(r["verdict"]["refused"] for r in judged),
                      "refused_correct_semantic":sum(r["judge_valid"] and r["verdict"]["refused"] for r in d),
                      "refused_correct_prefix":sum(r["prefix_refused"] for r in d)}.items():equal(value,g[key])
    assert len(a)==73 and len(d)==27
    before={r["id"]:r for r in rows(directory/"generation_cases_before_judge_retry.jsonl")}
    retried={r["id"] for r in load(directory/"judge_retry.json")}
    assert retried=={"C19"}
    for r in generated:
        equal(r["answer"],before[r["id"]]["answer"])
        equal(r["evidence"],before[r["id"]]["evidence"])
        if r["id"] not in retried:equal(r,before[r["id"]])
    assert not before["C19"]["judge_valid"]
    gate=load(directory/"gate_calibration.json")
    samples=gate["samples"]
    assert len(samples)==len({r["id"] for r in samples})==100
    threshold=gate["current"]["threshold"]
    equal(sum(r["answerable"] and r["confidence"]>=threshold for r in samples),55)
    equal(sum(not r["answerable"] and r["confidence"]<threshold for r in samples),27)
    equal(gate["current"], report["gate"]["current"])

    topic=load(ROOT/current["topic"]["directory"]/"report.json")
    for key,value in {"n":510,"macro_f1":0.9778,"micro_f1":0.9761,"exact_match":0.9608}.items():equal(topic["test"][key],value)
    equal(topic["threshold"],0.4)
    for name,expected in load(ROOT/current["topic"]["directory"]/"provenance.json")["dataset_files"].items():
        equal(hashlib.sha256((ROOT/"training/data"/name).read_bytes()).hexdigest(),expected)
    print("PASS: artifact checksums; 100 labels; 39 active knowledge blocks; 292 retrieval records; 100 generation records; gate counts; frozen 510-row topic report.")
    print("Offline record verification only: no model, database, API request or new accuracy measurement.")

if __name__ == "__main__":
    main()
