"""실행 로그(Colab 출력)에서 조건별 평가 결과를 추출한다."""
import re, csv, sys, collections
FILES = sys.argv[1:]
rows = {}
for F in FILES:
    cur = None
    for line in open(F, encoding="utf-8", errors="replace"):
        m = re.search(r"eval_robustness\.py --model runs/(\w+)/(best(?:_norecal)?)\.keras .*--tag (\w+)", line)
        if m:
            cur = m.group(3); continue
        m = re.match(r"\s*(?:\d+\s+)?(clean|gaussian_noise|gaussian_blur|contrast|jpeg|motion_blur)\s+s=(\d)\s+acc=([\d.]+)\s+f1=([\d.]+)\s+auc=([\d.]+)\s+rN=([\d.]+)\s+rP=([\d.]+)", line)
        if m and cur:
            tag, seed = cur.rsplit("_s", 1)
            rows[(tag, int(seed), m.group(1), int(m.group(2)))] = [float(x) for x in m.groups()[2:]]
w = csv.writer(open("per_condition.csv", "w", newline=""))
w.writerow(["model", "seed", "corruption", "severity", "accuracy", "f1", "auc", "recall_normal", "recall_pneumonia"])
for k in sorted(rows):
    w.writerow(list(k) + rows[k])
cnt = collections.Counter((k[0], k[1]) for k in rows)
for k in sorted(cnt): print(k, cnt[k])
