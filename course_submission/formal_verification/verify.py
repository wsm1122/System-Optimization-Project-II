"""Bounded integer-index abstraction; no numerical or CUDA correctness claim."""
import datetime
import json
from pathlib import Path
import sys
import uuid

import z3


def main():
    folder = Path(__file__).parent / 'results' / (
        datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    neighbor, label = z3.Ints('neighbor label')
    size, vocab = 1798000, 4233
    guard = z3.And(neighbor >= 0, neighbor < size)
    label_contract = z3.And(label >= 0, label < vocab)
    queries = [
        ('id_guard_safe', [guard, z3.Or(neighbor < 0, neighbor >= size)], 'unsat'),
        ('conditional_label_safe', [guard, label_contract, z3.Or(label < 0, label >= vocab)], 'unsat'),
        ('missing_label_contract', [guard, label < 0], 'sat'),
        ('missing_id_lower_bound', [neighbor < size, neighbor < 0], 'sat'),
        ('relaxed_id_upper_bound', [neighbor >= 0, neighbor <= size, neighbor == size], 'sat'),
        ('id_minus_one_rejected', [guard, neighbor == -1], 'unsat'),
        ('id_N_rejected', [guard, neighbor == size], 'unsat'),
        ('label_minus_one_actual', [guard, label == -1], 'sat'),
        ('label_V_actual', [guard, label == vocab], 'sat'),
        ('label_minus_one_contract', [label_contract, label == -1], 'unsat'),
        ('label_V_contract', [label_contract, label == vocab], 'unsat'),
        ('blank_valid', [guard, label_contract, label == 0], 'sat'),
    ]
    results = []
    for name, constraints, expected in queries:
        solver = z3.Solver()
        solver.set(timeout=10000)
        solver.add(*constraints)
        actual = str(solver.check())
        row = {'name': name, 'expected': expected, 'actual': actual, 'passed': actual == expected}
        if actual == 'sat':
            model = solver.model()
            row['witness'] = {'id': model.eval(neighbor, model_completion=True).as_long(),
                              'label': model.eval(label, model_completion=True).as_long()}
        (folder / (name + '.smt2')).write_text(solver.to_smt2())
        results.append(row)
        print(json.dumps(row), flush=True)
    report = {'status': 'PASS' if all(row['passed'] for row in results) else 'FAIL',
              'z3_version': z3.get_version_string(), 'N': size, 'V': vocab, 'queries': results,
              'scope': 'bounded integer abstraction; actual ID guard plus explicit label data contract',
              'unconditional_scatter_safety': 'NOT PROVED; SAT counterexamples',
              'empty_inputs': 'N=0 and K=0 excluded by positive-size/nonempty guard preconditions',
              'source_mapping': '../../retrieval_guards.py and ../../gpu_aggregation.py',
              'label_contract_evidence': '../../audit/assets.json',
              'does_not_prove': 'FAISS, floating point, PyTorch/CUDA, ASR correctness'}
    (folder / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Result:', folder / 'result.json')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
