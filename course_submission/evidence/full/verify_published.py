"""Verify the evidence subset actually published on GitHub, without a GPU."""

import csv
import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = 'combine_20261122T120238Z_cacaf324'


def read(path):
    data = path.read_bytes()
    return gzip.decompress(data) if path.suffix == '.gz' else data


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for row, reference_token in enumerate(reference, 1):
        current = [row]
        for column, prediction_token in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (reference_token != prediction_token)))
        previous = current
    return previous[-1]


def main():
    manifest = json.loads((ROOT / 'PUBLISHED_MANIFEST.json').read_text())
    for item in manifest['records']:
        path = ROOT / item['file']
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['stored_sha256'], item['file']
        assert hashlib.sha256(read(path)).hexdigest() == item['uncompressed_sha256'], item['file']
    run = ROOT / 'r8b' / 'results' / RUN
    canonical = None
    unique_scores = {}
    totals = None
    for block in range(3):
        for variant in ('E1', 'E2', 'E3', 'E4'):
            directory = run / ('block_' + str(block) + '_' + variant) / 'dev'
            predictions = read(directory / 'predictions.txt.gz')
            lines = predictions.decode().splitlines()
            identifiers = [line.split(maxsplit=1)[0] for line in lines]
            assert len(lines) == len(set(identifiers)) == 14326
            if canonical is None:
                canonical = predictions
            assert predictions == canonical
            score_bytes = read(directory / 'score.log.gz')
            score_sha = hashlib.sha256(score_bytes).hexdigest()
            if score_sha not in unique_scores:
                log = score_bytes.decode()
                matches = re.findall(r'^WER: .* N=(\d+) C=(\d+) S=(\d+) D=(\d+) I=(\d+)', log, re.M)
                assert len(matches) == 14326
                counts = dict(zip(('N', 'C', 'S', 'D', 'I'), map(sum, zip(*(map(int, match) for match in matches)))))
                references = re.findall(r'^lab: (.*)$', log, re.M)
                hypotheses = re.findall(r'^rec: (.*)$', log, re.M)
                assert len(references) == len(hypotheses) == 14326
                errors = sum(edit_distance(reference.split(), hypothesis.split()) for reference, hypothesis in zip(references, hypotheses))
                assert errors == counts['S'] + counts['D'] + counts['I']
                unique_scores[score_sha] = counts
            totals = unique_scores[score_sha]
            expected = json.loads(read(directory / 'summary.json'))['scores']
            assert all(totals[key] == expected[key] for key in totals)
    with (run / 'process_summary.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    means = {variant: statistics.fmean(float(row['mean_ms']) for row in rows if row['variant'] == variant) for variant in ('E1', 'E2', 'E3', 'E4')}
    differences = [float(next(row for row in rows if int(row['block']) == block and row['variant'] == 'E2')['mean_ms']) - float(next(row for row in rows if int(row['block']) == block and row['variant'] == 'E1')['mean_ms']) for block in range(3)]
    center = statistics.fmean(differences)
    radius = 4.302652729911275 * statistics.stdev(differences) / math.sqrt(3)
    print(json.dumps({'status': 'PASS', 'scope': 'published file integrity and complete R8-B prediction/scoring evidence', 'new_gpu_run': False,
                      'r8b_prediction_files': 12, 'r8b_audio_per_file': 14326, 'r8b_predictions_byte_identical': True,
                      'r8b_cer_percent': 100 * (totals['S'] + totals['D'] + totals['I']) / totals['N'],
                      'r8b_process_mean_request_ms': means, 'r8b_E2_minus_E1_process_t95_ci_ms': [center-radius, center+radius],
                      'r9_complete_prediction_evidence': 'INCOMPLETE: only the first test prediction file is published',
                      'raw_request_logs': 'NOT PUBLISHED', 'paired_audio_tables': 'NOT PUBLISHED'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
