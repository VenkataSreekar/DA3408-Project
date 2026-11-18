#!/usr/bin/env bash
# System tests on the running Docker stack (test report, section 4).
# Run from the repository root with the stack up:  scripts/system_tests.sh | tee docs/system_test_run.txt
# Automates ST-2, ST-3, ST-4, ST-5, ST-6, ST-9 and ST-14 at the API level.
# The page-level parts (what the browser shows) are listed at the end for a quick manual check.
set -uo pipefail
cd "$(dirname "$0")/.."

API=http://localhost:8000
PROM=http://localhost:9090
HANDLE=${HANDLE:-Venkata_Sreekar}
PASS=0; FAIL=0
ok()   { echo "  PASS: $1"; PASS=$((PASS + 1)); }
bad()  { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
code() { curl -s -o /dev/null -w '%{http_code}' --max-time "${2:-120}" "$1"; }
json() { python3 -c "import json,sys; d=json.load(sys.stdin); print($1)"; }

wait_backend() {  # wait until the backend answers /health
  for _ in $(seq 1 30); do
    [ "$(code "$API/health" 5)" = "200" ] && return 0
    sleep 2
  done
  return 1
}

echo "System test run: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Commit: $(git rev-parse --short HEAD)"
docker compose ps --format 'table {{.Service}}\t{{.State}}\t{{.Status}}'
echo

echo "ST-2  Health check"
H=$(curl -s --max-time 10 "$API/health")
echo "  response: $H"
[ "$H" = '{"status":"ok","model_server":"up"}' ] && ok "health ok, model server up" || bad "unexpected /health"

echo "ST-3  Recommendations for a real user ($HANDLE)"
R=$(curl -s --max-time 120 "$API/recommend?handle=$HANDLE&k=10")
if echo "$R" | json "d['recommendations']" >/dev/null 2>&1; then
  echo "$R" | json "f\"  rating {d['rating']}, weak tags {[t['tag'] for t in d['weak_tags']]}, {d['in_zone']} in zone, {len(d['recommendations'])} returned\""
  IN_ZONE=$(echo "$R" | json "int(all(0.5 <= r['p_solve'] <= 0.7 for r in d['recommendations']))")
  N=$(echo "$R" | json "len(d['recommendations'])")
  LINKS=$(echo "$R" | json "int(all(r['url'].startswith('https://codeforces.com/problemset/problem/') for r in d['recommendations']))")
  [ "$N" -ge 1 ] && [ "$N" -le 10 ] && [ "$IN_ZONE" = 1 ] && [ "$LINKS" = 1 ] \
    && ok "$N problems, all with solve chance in 50-70%, all linked to Codeforces" \
    || bad "recommendations out of range or missing links"
else
  bad "no recommendations: $R"
fi

echo "ST-4  Unknown handle"
C=$(code "$API/recommend?handle=zzz_nobody_here_987")
[ "$C" = "404" ] && ok "404 for an unknown handle" || bad "expected 404, got $C"

echo "ST-5  Invalid handle"
C=$(code "$API/recommend?handle=ab")
[ "$C" = "400" ] && ok "400 for 'ab' (the page also blocks it before sending)" || bad "expected 400, got $C"

echo "ST-9  Monitoring records traffic"
sleep 20   # let Prometheus scrape at least once after the requests above
TARGETS=$(curl -s "$PROM/api/v1/targets" | json "' '.join(sorted(t['labels']['job'] + '=' + t['health'] for t in d['data']['activeTargets']))")
echo "  targets: $TARGETS"
[ "$TARGETS" = "backend=up grafana=up prometheus=up" ] && ok "all three scrape targets up" || bad "a target is not up"
SERVICES=$(curl -s "$PROM/api/v1/query?query=np_service_up" | json "' '.join(sorted(r['metric']['service'] + '=' + r['value'][1] for r in d['data']['result']))")
echo "  np_service_up: $SERVICES"
[ "$SERVICES" = "frontend=1 model_server=1" ] && ok "model server and frontend reported up" || bad "service probes not both up"
ERRS=$(curl -s "$PROM/api/v1/query?query=np_errors_total" | json "' '.join(sorted(r['metric']['type'] for r in d['data']['result'] if float(r['value'][1]) > 0))")
echo "  error types seen: $ERRS"
[[ "$ERRS" == *handle_not_found* && "$ERRS" == *invalid_handle* ]] && ok "errors by type recorded" || bad "error counters missing"
REQS=$(curl -s "$PROM/api/v1/query?query=sum(np_requests_total)" | json "d['data']['result'][0]['value'][1]")
echo "  total requests recorded: $REQS"
DRIFT=$(curl -s "$PROM/api/v1/query?query=np_feature_mean_shift" | json "len(d['data']['result'])")
[ "$DRIFT" -ge 1 ] && ok "drift gauges have values" || bad "no drift values"

echo "ST-6  Backend unavailable"
docker compose stop backend >/dev/null 2>&1
C=$(code "$API/health" 5)
[ "$C" = "000" ] && ok "backend unreachable while stopped (the page shows 'Cannot reach the recommendation service')" || bad "backend still answering ($C)"
docker compose start backend >/dev/null 2>&1
wait_backend && ok "backend answers again after restart" || bad "backend did not come back"

echo "ST-14 Outage raises an alert (takes about 2-4 minutes)"
docker compose stop model >/dev/null 2>&1
HD=$(curl -s --max-time 10 "$API/health"); echo "  /health while model is stopped: $HD"
FIRED=0
for i in $(seq 1 16); do   # up to 4 minutes: 1 min 'for' + scrape and evaluation intervals
  sleep 15
  STATE=$(curl -s "$PROM/api/v1/alerts" | json "next((a['state'] for a in d['data']['alerts'] if a['labels']['alertname']=='ModelServerDown'), 'inactive')")
  echo "  after $((i * 15)) s: ModelServerDown $STATE"
  [ "$STATE" = "firing" ] && { FIRED=1; break; }
done
[ "$FIRED" = 1 ] && ok "ModelServerDown fired" || bad "ModelServerDown did not fire"
docker compose start model >/dev/null 2>&1
CLEARED=0
for i in $(seq 1 16); do
  sleep 15
  STATE=$(curl -s "$PROM/api/v1/alerts" | json "next((a['state'] for a in d['data']['alerts'] if a['labels']['alertname']=='ModelServerDown'), 'inactive')")
  [ "$STATE" = "inactive" ] && { CLEARED=1; echo "  cleared $((i * 15)) s after restart"; break; }
done
[ "$CLEARED" = 1 ] && ok "alert cleared after the model server restarted" || bad "alert did not clear"
echo "  /health now: $(curl -s --max-time 10 "$API/health")"

echo
echo "RESULT: $PASS checks passed, $FAIL failed"
echo
echo "Quick manual checks in the browser (http://localhost:8080):"
echo "  ST-3  search $HANDLE: weak topics and a table appear; a problem name opens Codeforces"
echo "  ST-4  search zzz_nobody_here_987: 'Handle ... was not found on Codeforces.'"
echo "  ST-5  search ab: invalid-handle message appears immediately"
echo "  ST-6  run 'docker compose stop backend', search: 'Cannot reach the recommendation service...'; then 'docker compose start backend'"
echo "  ST-9  Grafana (http://localhost:3001/d/next-problem): request count, errors by type, drift, Service status all UP"
[ "$FAIL" = 0 ]
